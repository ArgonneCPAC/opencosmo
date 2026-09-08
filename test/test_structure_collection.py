import numpy as np
import pytest

import opencosmo as oc
from opencosmo.collection.structure.handler import link_slot_values


def _link_slot(collection, link_name):
    """Per-source-row link slot values (idx with -1, or chunk sizes).

    Reads straight off the collection's match sets -- the same primitive the
    library itself uses to decide which structures are empty.
    """
    handler = collection._StructureCollection__handler
    source = collection._StructureCollection__source
    values, _ = link_slot_values(handler.match_sets, source, link_name)
    return values


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
        "halo_properties": [step_600.halo_properties, step_601.halo_properties],
        "halo_particles": [step_600.halo_particles, step_601.halo_particles],
        "halo_profiles": [step_600.halo_profiles, step_601.halo_profiles],
        "galaxy_properties": [step_600.galaxy_properties, step_601.galaxy_properties],
        "galaxy_particles": [step_600.galaxy_particles, step_601.galaxy_particles],
    }


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
        n_checked = 0
        for structure in subset.halos():
            verify_structure_links(structure)
            n_checked += 1
        assert n_checked > 0
    else:
        # Galaxy-only structure collection
        n_checked = 0
        for galaxy in collection.take(50).galaxies():
            if "star_particles" not in galaxy:
                continue
            tags = galaxy["star_particles"].select("gal_tag").get_data("numpy")
            assert np.all(tags == galaxy["galaxy_properties"]["gal_tag"])
            n_checked += 1
        assert n_checked > 0


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


@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_open_lightcone_structure_combinations(
    lightcone_files, components, expected_keys
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = oc.open(*paths)

    assert isinstance(collection, oc.StructureCollection)
    assert set(collection.keys()) == expected_keys

    verify_collection_links(collection)


@pytest.mark.parametrize("components,expected_keys", COMBINATION_PARAMS)
def test_write_lightcone_structure_combinations(
    lightcone_files, components, expected_keys, tmp_path
):
    paths = [p for component in components for p in lightcone_files[component]]
    collection = reduce_for_write(oc.open(*paths), components)

    output = tmp_path / "collection.hdf5"
    oc.write(output, collection)
    reopened = oc.open(output)

    assert isinstance(reopened, oc.StructureCollection)
    assert set(reopened.keys()) == expected_keys

    # Every linked dataset must survive the write unchanged, including sparse
    # idx-based links like halo profiles.
    for name in expected_keys:
        if name in ("halo_properties", "galaxy_properties", "galaxies"):
            continue
        assert len(reopened[name]) == len(collection[name])

    verify_collection_links(reopened)


def test_lightcone_ignore_empty(lightcone_files):
    """
    On a lightcone structure collection, ignore_empty (the default) should drop
    halos that are empty in the opened linked datasets, considering only the
    datasets that were actually opened. ignore_empty=False keeps every halo.
    """
    paths = [
        p
        for component in ("halo_properties", "halo_profiles", "galaxy_properties")
        for p in lightcone_files[component]
    ]

    kept_all = oc.open(*paths, ignore_empty=False)
    kept_nonempty = oc.open(*paths)

    # Compute the expected kept set: halos with both a profile and a galaxy,
    # since both datasets were opened.
    has_profile = _link_slot(kept_all, "halo_profiles") != -1
    has_galaxy = _link_slot(kept_all, "galaxy_properties") != 0
    expected = int((has_profile & has_galaxy).sum())

    assert len(kept_nonempty) == expected
    assert len(kept_nonempty) < len(kept_all)

    # Every remaining halo must actually have both links populated.
    for halo in kept_nonempty.take(20).halos():
        assert "halo_profiles" in halo and len(halo["halo_profiles"]) > 0
        assert "galaxy_properties" in halo and len(halo["galaxy_properties"]) > 0


def test_lightcone_ignore_empty_only_considers_opened(lightcone_files):
    """
    Halos empty in an unopened dataset must not be dropped. Opening only halo
    profiles should keep every halo that has a profile, regardless of whether
    those halos have particles or galaxies (which are not opened here).
    """
    paths = [
        p
        for component in ("halo_properties", "halo_profiles")
        for p in lightcone_files[component]
    ]
    collection = oc.open(*paths)

    kept_all = oc.open(*paths, ignore_empty=False)
    expected = int((_link_slot(kept_all, "halo_profiles") != -1).sum())

    assert len(collection) == expected
    assert len(collection["halo_profiles"]) == expected


def test_evaluate_into_galaxy_properties(halos_600_path, galaxies_600_path):
    def offset(halo_properties, galaxy_properties):
        dx = galaxy_properties["gal_com_x"] - halo_properties["fof_halo_center_x"]
        dy = galaxy_properties["gal_com_y"] - halo_properties["fof_halo_center_y"]
        dz = galaxy_properties["gal_com_z"] - halo_properties["fof_halo_center_z"]
        dist2 = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist2)
        offset = dist / halo_properties["sod_halo_RVir"]  # divide by virial radius
        return {"gal_offset": offset}

    ds = oc.open(*halos_600_path, galaxies_600_path[0])
    ds = ds.evaluate(
        offset,
        dataset="galaxy_properties",
        insert=True,
        format="numpy",
        halo_properties=["fof_halo_center_*", "sod_halo_RVir"],
        galaxy_properties=["gal_com_*"],
    )
    for halo in ds.halos():
        _ = halo["galaxy_properties"].select("gal_offset").get_data()


def test_write_lightcone_structure(halos_600_path, halos_601_path, tmp_path):
    ds = (
        oc.open(*halos_600_path, *halos_601_path)
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
    oc.write(tmp_path / "halos.hdf5", ds)
    ds_new = oc.open(tmp_path / "halos.hdf5")

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        assert halo["halo_properties"]["fof_halo_tag"] in halo_tags_start
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        assert halo["halo_properties"]["fof_halo_tag"] in halo_tags_end
        verify_halo(halo)


def test_write_lightcone_structure_with_galaxies(
    halos_600_path, halos_601_path, galaxies_600_path, galaxies_601_path, tmp_path
):
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
    oc.write(tmp_path / "halos.hdf5", ds)
    ds_new = oc.open(tmp_path / "halos.hdf5")

    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="start").halos()
    ):
        assert halo["halo_properties"]["fof_halo_tag"] in halo_tags_start
        verify_halo(halo)
    for halo in (
        ds_new.filter(oc.col("sod_halo_mass") > 1e14).take(10, at="end").halos()
    ):
        assert halo["halo_properties"]["fof_halo_tag"] in halo_tags_end
        verify_halo(halo)


def test_data_link_sort_write_lightcone(halos_600_path, halos_601_path, tmp_path):
    collection = oc.open(*halos_600_path, *halos_601_path)
    collection = collection.filter(oc.col("sod_halo_mass") > 10**14).sort_by(
        "fof_halo_mass"
    )
    output = tmp_path / "halos.hdf5"
    oc.write(output, collection)
    new_collection = oc.open(output).take(10)
    assert np.all(
        collection["halo_properties"].select("sod_halo_mass").get_data("numpy") > 10**14
    )
    for halo in new_collection.objects(("halo_profiles",)):
        assert np.all(
            halo["halo_properties"]["fof_halo_tag"]
            == halo["halo_profiles"].select("fof_halo_bin_tag").get_data("numpy")[0]
        )


def test_redshift_bound(halos_600_path, halos_601_path, tmp_path):
    collection = oc.open(*halos_600_path, *halos_601_path)
    collection = collection.with_redshift_range(0.038, 0.039)

    collection = collection.filter(oc.col("sod_halo_mass") > 10**14)
    for halo in collection.halos():
        redshift = halo["halo_properties"]["redshift"]
        assert redshift > 0.038 and redshift < 0.039
        verify_halo(halo)
