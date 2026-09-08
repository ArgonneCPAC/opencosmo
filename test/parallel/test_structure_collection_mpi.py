import numpy as np
import pytest
from opencosmo.mpi import get_comm_world

import opencosmo as oc


@pytest.fixture
def halos_600_path(test_data):
    return test_data.lightcone.step(600).halos


@pytest.fixture
def galaxies_600_path(test_data):
    return test_data.lightcone.step(600).galaxies


@pytest.fixture
def halos_601_path(test_data):
    return test_data.lightcone.step(601).halos


@pytest.fixture
def galaxies_601_path(test_data):
    return test_data.lightcone.step(601).galaxies


@pytest.fixture
def lightcone_files(test_data):
    """Map a component name to the per-step files that provide it."""

    step_600 = test_data.lightcone.step(600)
    step_601 = test_data.lightcone.step(601)
    return {
        name: [getattr(step_600, name), getattr(step_601, name)]
        for name in (
            "halo_properties",
            "halo_particles",
            "halo_profiles",
            "galaxy_properties",
            "galaxy_particles",
        )
    }


# Each entry is the set of components combined into a lightcone structure
# collection and the dataset keys we expect the resulting collection to expose.
LIGHTCONE_COMBINATIONS = {
    "halo_particles": (
        ["halo_properties", "halo_particles"],
        {
            "agn_particles",
            "dm_particles",
            "gas_particles",
            "star_particles",
            "halo_properties",
        },
    ),
    "halo_profiles": (
        ["halo_properties", "halo_profiles"],
        {"halo_profiles", "halo_properties"},
    ),
    "halo_particles_profiles": (
        ["halo_properties", "halo_particles", "halo_profiles"],
        {
            "agn_particles",
            "dm_particles",
            "gas_particles",
            "star_particles",
            "halo_profiles",
            "halo_properties",
        },
    ),
    "halo_particles_profiles_galaxy_properties": (
        ["halo_properties", "halo_particles", "halo_profiles", "galaxy_properties"],
        {
            "agn_particles",
            "dm_particles",
            "gas_particles",
            "star_particles",
            "halo_profiles",
            "galaxy_properties",
            "halo_properties",
        },
    ),
    "halo_galaxy_properties": (
        ["halo_properties", "galaxy_properties"],
        {"galaxy_properties", "halo_properties"},
    ),
    "halo_galaxy_properties_particles": (
        ["halo_properties", "galaxy_properties", "galaxy_particles"],
        {"galaxies", "halo_properties"},
    ),
    "galaxy_properties_particles": (
        ["galaxy_properties", "galaxy_particles"],
        {"star_particles", "galaxy_properties"},
    ),
}

COMBINATION_PARAMS = [
    pytest.param(c, k, id=name) for name, (c, k) in LIGHTCONE_COMBINATIONS.items()
]


def reduce_for_write(collection, components):
    """Reduce a collection to a manageable size before writing.

    Particles only exist for halos above ~10**13.5, so filtering particle
    collections by mass loses no linked data while keeping the write small.
    Collections without particles (e.g. halo profiles, which only exist for a
    sparse subset of halos) are written in full so that the sparse idx-based
    links are exercised -- a mass filter would keep only massive halos, which
    all have profiles, and would hide bugs in how sparse links are written.
    """
    if "halo_properties" not in collection.keys():
        # Galaxy-only collection: no sparse profile links to preserve, so a
        # plain subset keeps the write small.
        return collection.take(1000)
    has_particles = any("particles" in component for component in components)
    if has_particles:
        return collection.filter(oc.col("fof_halo_mass") > 10**13.5)
    return collection


def verify_halo(halo):
    gravity_particle_tags = (
        halo["dm_particles"].select("fof_halo_tag").get_data("numpy")
    )
    assert np.all(gravity_particle_tags == halo["halo_properties"]["fof_halo_tag"])
    halo_bin_tags = halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")
    assert np.all(halo_bin_tags == halo["halo_properties"]["fof_halo_tag"])
    if "galaxy" not in halo:
        return
    for galaxy in halo["galaxies"].galaxies():
        assert (
            galaxy["galaxy_properties"]["fof_halo_tag"]
            == halo["halo_properties"]["fof_halo_tag"]
        )

        if "star_particles" not in galaxy:
            continue
        tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
        assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])


def verify_structure_links(structure):
    """Verify every linked dataset in a structure points back to its host."""
    host_tag = structure["halo_properties"]["fof_halo_tag"]
    for name, linked in structure.items():
        if name == "halo_properties":
            continue
        if name == "halo_profiles":
            tags = linked.select("fof_halo_bin_tag").get_data("numpy")
            assert np.all(tags == host_tag)
        elif name == "galaxy_properties":
            tags = linked.select("fof_halo_tag").get_data("numpy")
            assert np.all(tags == host_tag)
        elif name == "galaxies":
            for galaxy in linked.galaxies():
                assert galaxy["galaxy_properties"]["fof_halo_tag"] == host_tag
        elif "particles" in name:
            tags = linked.select("fof_halo_tag").get_data("numpy")
            assert np.all(tags == host_tag)


def verify_collection_links(collection, n=10):
    """Verify links across a structure collection of halos or galaxies."""
    if "halo_properties" in collection.keys():
        subset = collection.filter(oc.col("sod_halo_mass") > 1e13).take(n)
        for structure in subset.halos():
            verify_structure_links(structure)
    else:
        for galaxy in collection.take(50).galaxies():
            if "star_particles" not in galaxy:
                continue
            tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
            assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_open_lightcone_structure_combinations(
    lightcone_files, components, expected_keys
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = oc.open(*paths)

    assert isinstance(collection, oc.StructureCollection)
    assert set(collection.keys()) == expected_keys

    verify_collection_links(collection)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_write_lightcone_structure_combinations(
    lightcone_files, components, expected_keys, per_test_dir
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = reduce_for_write(oc.open(*paths), components)

    output = per_test_dir / "collection.hdf5"
    oc.write(output, collection)
    reopened = oc.open(output)

    assert isinstance(reopened, oc.StructureCollection)
    assert set(reopened.keys()) == expected_keys

    # Every linked dataset must survive the write unchanged, including sparse
    # idx-based links like halo profiles. Lengths are compared globally since
    # each rank holds only its partition. Iterate in a deterministic (sorted)
    # order so the collective allgather calls line up across ranks.
    comm = get_comm_world()
    for name in sorted(expected_keys):
        if name in ("halo_properties", "galaxy_properties", "galaxies"):
            continue
        reopened_total = sum(comm.allgather(len(reopened[name])))
        original_total = sum(comm.allgather(len(collection[name])))
        assert reopened_total == original_total

    verify_collection_links(reopened)


@pytest.mark.parallel(nprocs=4)
def test_write_lightcone_structure(halos_600_path, halos_601_path, per_test_dir):
    comm = get_comm_world()
    ds = (
        oc.open(
            *halos_600_path,
            *halos_601_path,
        )
        .filter(oc.col("fof_halo_mass") > 1e14)
        .take(1000)
    )
    halo_tags_start = set()
    halo_tags_end = set()
    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos():
        halo_tags_start.add(halo["halo_properties"]["fof_halo_tag"])
        verify_halo(halo)

    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos():
        halo_tags_end.add(halo["halo_properties"]["fof_halo_tag"])
        verify_halo(halo)
    all_halos_read = set(
        np.concatenate(
            comm.allgather(
                ds["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    oc.write(per_test_dir / "halos.hdf5", ds)
    ds_new = oc.open(per_test_dir / "halos.hdf5")

    all_halos = set(
        np.concatenate(
            comm.allgather(
                ds_new["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        verify_halo(halo)

    assert halo_tags_start.issubset(all_halos)
    assert halo_tags_end.issubset(all_halos)
    assert all_halos_read == all_halos


@pytest.mark.parallel(nprocs=4)
def test_write_lightcone_structure_with_galaxies(
    halos_600_path, halos_601_path, galaxies_600_path, galaxies_601_path, per_test_dir
):
    comm = get_comm_world()
    ds = (
        oc.open(
            *halos_600_path, *halos_601_path, *galaxies_600_path, *galaxies_601_path
        )
        .filter(oc.col("fof_halo_mass") > 1e14)
        .take(1000)
    )
    halo_tags_start = set()
    halo_tags_end = set()
    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos():
        halo_tags_start.add(halo["halo_properties"]["fof_halo_tag"])
        verify_halo(halo)

    for halo in ds.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos():
        halo_tags_end.add(halo["halo_properties"]["fof_halo_tag"])
        verify_halo(halo)
    all_halos_read = set(
        np.concatenate(
            comm.allgather(
                ds["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    oc.write(per_test_dir / "halos.hdf5", ds)
    ds_new = oc.open(per_test_dir / "halos.hdf5")

    all_halos = set(
        np.concatenate(
            comm.allgather(
                ds_new["halo_properties"].select("fof_halo_tag").get_data("numpy")
            )
        )
    )

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        verify_halo(halo)

    assert halo_tags_start.issubset(all_halos)
    assert halo_tags_end.issubset(all_halos)
    assert all_halos_read == all_halos
