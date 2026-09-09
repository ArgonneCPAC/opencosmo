import os

import astropy.units as u
import numpy as np
import pytest
from opencosmo.io.discover import discover_file

import opencosmo as oc


@pytest.fixture
def input_path(test_data):
    return test_data.snapshot.primary.halo_properties


@pytest.fixture
def particle_path(test_data):
    return test_data.snapshot.primary.halo_particles


@pytest.fixture
def max_mass(input_path):
    ds = oc.open(input_path)
    sod_mass = ds.get_data()["sod_halo_mass"]
    return 0.95 * sod_mass.max()


def test_descriptions(input_path):
    ds = oc.open(input_path)
    assert isinstance(ds.descriptions, dict)


def test_dataset_uuid_matches_discovery_and_survives_transformations(input_path):
    """Test that dataset identity is preserved from opening through transforms."""
    dataset = oc.open(input_path)
    discovered_uuid = discover_file(input_path).groups[0].uuid

    assert dataset.uuid == discovered_uuid
    assert dataset.uuid == dataset.filter(oc.col("sod_halo_mass") > 0).uuid
    assert dataset.uuid == dataset.select("fof_halo_mass").uuid
    assert dataset.uuid == dataset.take(10).uuid


def test_filter_oom(input_path, max_mass):
    # Assuming test_open worked, this is the only
    # thing that needs to be directly tested

    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0, oc.col("sod_halo_mass") < max_mass)
        data = ds.get_data()
    assert data["sod_halo_mass"].min() > 0


def test_filter_to_numpy(input_path, max_mass):
    # Assuming test_open worked, this is the only
    # thing that needs to be directly tested

    with oc.open(input_path) as f:
        ds = f.filter(oc.col("sod_halo_mass") > 0, oc.col("sod_halo_mass") < max_mass)
        data = ds.get_data(format="numpy")
    assert isinstance(data, dict)
    for val in data.values():
        assert isinstance(val, np.ndarray)
    assert data["sod_halo_mass"].min() > 0


def test_take_oom(input_path):
    with oc.open(input_path) as f:
        ds = f.take(10)
        data = ds.get_data()
    assert len(data) == 10


def test_take_sorted(input_path):
    n = 150
    dataset = oc.open(input_path)
    fof_masses = dataset.select("fof_halo_mass").get_data("numpy")
    toolkit_sorted_fof_masses = (
        dataset.sort_by("fof_halo_mass")
        .take(150, at="start")
        .select("fof_halo_mass")
        .get_data("numpy")
    )
    manually_sorted_fof_masses = np.sort(fof_masses)
    assert np.all(manually_sorted_fof_masses[:n] == toolkit_sorted_fof_masses)
    assert fof_masses.min() == toolkit_sorted_fof_masses[0]


def test_take_sorted_inverted(input_path):
    n = 150
    dataset = oc.open(input_path)
    fof_masses = dataset.select("fof_halo_mass").get_data("numpy")
    toolkit_sorted_fof_masses = (
        dataset.sort_by("fof_halo_mass", invert=True)
        .take(150, at="start")
        .select("fof_halo_mass")
        .get_data("numpy")
    )
    manually_sorted_fof_masses = -np.sort(-fof_masses)
    assert np.all(manually_sorted_fof_masses[:n] == toolkit_sorted_fof_masses)
    assert fof_masses.max() == toolkit_sorted_fof_masses[0]


def test_sort_by_derived(input_path):
    n = 150
    ds = oc.open(input_path)
    dx = oc.col("fof_halo_com_x") - oc.col("sod_halo_com_x")
    dy = oc.col("fof_halo_com_y") - oc.col("sod_halo_com_y")
    dz = oc.col("fof_halo_com_z") - oc.col("sod_halo_com_z")
    dr = (dx**2 + dy**2 + dz**2) ** (0.5)
    xoff = dr / oc.col("sod_halo_radius")

    ds = ds.with_new_columns(xoff=xoff)
    xoff = ds.select(("xoff", "fof_halo_tag")).get_data("numpy")
    toolkit_sorted_xoff = (
        ds.sort_by("xoff")
        .take(n, at="start")
        .select(("fof_halo_tag", "xoff"))
        .get_data("numpy")
    )
    idxs = np.argsort(xoff["xoff"])
    assert np.all(xoff["xoff"][idxs][:n] == toolkit_sorted_xoff["xoff"])
    assert xoff["xoff"].min() == toolkit_sorted_xoff["xoff"][0]
    assert np.all(xoff["fof_halo_tag"][idxs][:n] == toolkit_sorted_xoff["fof_halo_tag"])


def test_sort_by_unknown(input_path):
    dataset = oc.open(input_path)
    with pytest.raises(ValueError):
        dataset.sort_by("random_column")


def test_select_wildcard(input_path):
    ds = oc.open(input_path)
    fof_columns = set(col for col in ds.columns if col.startswith("fof"))
    ds = ds.select("fof*")
    assert set(ds.columns) == fof_columns


def test_select_multiple_wildcard_one_pattern(input_path):
    ds = oc.open(input_path)
    fof_columns = set(col for col in ds.columns if "com" in col)
    ds = ds.select("*com*")
    assert set(ds.columns) == fof_columns


def test_select_multiple_wildcard(input_path):
    ds = oc.open(input_path)
    fof_columns = set(
        col for col in ds.columns if col.startswith("fof") or col.startswith("sod")
    )
    ds = ds.select(("fof*", "sod*"))
    assert set(ds.columns) == fof_columns


def test_select_wildcard_and_exact(input_path):
    ds = oc.open(input_path)
    fof_columns = set(col for col in ds.columns if col.startswith("fof"))
    fof_columns.add("sod_halo_com_x")
    ds = ds.select(("fof*", "sod_halo_com_x"))
    assert set(ds.columns) == fof_columns


def test_drop_wildcard_and_exact(input_path):
    ds = oc.open(input_path)
    fof_columns = set(col for col in ds.columns if col.startswith("fof"))
    fof_columns.add("sod_halo_com_x")
    ds = ds.drop(("fof*", "sod_halo_com_x"))
    assert not set(ds.columns).intersection(fof_columns)


def test_drop(input_path):
    with oc.open(input_path) as ds:
        data = ds.get_data()
        cols = list(data.columns)
        # select 10 columns at random
        dropped_cols = np.random.choice(cols, 10, replace=False)
        selected = ds.drop(dropped_cols)
        selected_data = selected.get_data()

    dropped_cols = set(dropped_cols)
    remaining_cols = set(selected_data.colnames)
    assert not dropped_cols.intersection(remaining_cols)


def test_drop_single(input_path):
    with oc.open(input_path) as ds:
        data = ds.get_data()
        cols = list(data.columns)
        # select 10 columns at random
        dropped_col = np.random.choice(cols)
        remaining = ds.drop(dropped_col)
        remaining_data = remaining.get_data()

    assert dropped_col not in remaining_data.colnames


def test_visit_vectorize_single(input_path):
    ds = oc.open(input_path)

    def fof_total(fof_halo_mass):
        return np.cumsum(fof_halo_mass)

    ds = ds.evaluate(fof_total, vectorize=True, insert=True)
    assert "fof_total" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_total"]).get_data("numpy")
    assert np.all(data["fof_total"] == np.cumsum(data["fof_halo_mass"]))


def test_visit_vectorize_multiple(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


class Counter:
    def __init__(self):
        self.__count = 0

    def increment(self):
        self.__count += 1

    @property
    def count(self):
        return self.__count


def test_visit_batched(input_path):
    ds = oc.open(input_path)
    batch_size = 10_000

    def fof_total(fof_halo_mass, counter):
        counter.increment()
        return np.cumsum(fof_halo_mass)

    counter = Counter()
    fof_total = ds.evaluate(
        fof_total,
        vectorize=True,
        insert=False,
        batch_size=batch_size,
        counter=counter,
        format="numpy",
    )["fof_total"]
    assert counter.count == len(ds) // batch_size + 1  # +1 for endpoint
    halo_mass = ds.select("fof_halo_mass").get_data("numpy")
    split_points = np.append(np.arange(batch_size, len(ds), batch_size), len(ds))
    split_halo_masses = np.array_split(halo_mass, split_points)
    split_fof_total = np.array_split(fof_total, split_points)
    for fof_total_split, halo_mass_split in zip(split_fof_total, split_halo_masses):
        assert np.all(fof_total_split == np.cumsum(halo_mass_split))


def test_visit_batched_astropy(input_path):
    ds = oc.open(input_path)
    batch_size = 10_000

    def fof_total(fof_halo_mass, counter):
        counter.increment()
        return np.cumsum(fof_halo_mass)

    counter = Counter()
    fof_total = ds.evaluate(
        fof_total,
        vectorize=True,
        insert=False,
        batch_size=batch_size,
        counter=counter,
        format="astropy",
    )["fof_total"]
    assert counter.count == len(ds) // batch_size + 1  # +1 for endpoint
    halo_mass = ds.select("fof_halo_mass").get_data("astropy")
    split_points = np.append(np.arange(batch_size, len(ds), batch_size), len(ds))
    split_halo_masses = np.array_split(halo_mass, split_points)
    split_fof_total = np.array_split(fof_total, split_points)
    for fof_total_split, halo_mass_split in zip(split_fof_total, split_halo_masses):
        assert np.all(fof_total_split == np.cumsum(halo_mass_split))


def test_visit_batched_larger(input_path):
    ds = oc.open(input_path).take(5000)
    batch_size = 10_000

    def fof_total(fof_halo_mass, counter):
        counter.increment()
        return np.cumsum(fof_halo_mass)

    counter = Counter()
    fof_total = ds.evaluate(
        fof_total,
        vectorize=True,
        insert=False,
        batch_size=batch_size,
        counter=counter,
        format="numpy",
    )
    assert counter.count == len(ds) // batch_size + 1  # +1 for endpoint
    halo_mass = ds.select("fof_halo_mass").get_data("numpy")
    split_points = np.append(np.arange(batch_size, len(ds), batch_size), len(ds))
    split_halo_masses = np.array_split(halo_mass, split_points)
    split_fof_total = np.array_split(fof_total["fof_total"], split_points)
    for fof_total_split, halo_mass_split in zip(split_fof_total, split_halo_masses):
        assert np.all(fof_total_split == np.cumsum(halo_mass_split))


def test_visit_batched_lazy(input_path):
    ds = oc.open(input_path)
    batch_size = 10_000

    def fof_total(fof_halo_mass, counter):
        counter.increment()
        return np.cumsum(fof_halo_mass)

    counter = Counter()
    ds = ds.evaluate(
        fof_total,
        insert=True,
        batch_size=batch_size,
        counter=counter,
        format="numpy",
    )
    data = ds.select(("fof_halo_mass", "fof_total")).get_data("numpy")
    assert (
        counter.count == len(ds) // batch_size + 2
    )  # 1 for endpoint, 1 for verification step
    split_points = np.append(np.arange(batch_size, len(ds), batch_size), len(ds))
    split_halo_masses = np.array_split(data["fof_halo_mass"], split_points)
    split_fof_total = np.array_split(data["fof_total"], split_points)
    for fof_total_split, halo_mass_split in zip(split_fof_total, split_halo_masses):
        assert np.all(fof_total_split == np.cumsum(halo_mass_split))


def test_visit_with_default(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_val=5):
        return fof_halo_mass * fof_halo_com_vx * random_val

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"] * 5)


def test_visit_with_return_none(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_val=5):
        return None

    with pytest.raises(ValueError):
        ds.evaluate(fof_px, vectorize=True, insert=True)


def test_visit_multiple_with_kwargs_numpy(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx, random_value):
        return fof_halo_mass * fof_halo_com_vx + random_value

    random_values = np.random.randint(0, 100, len(ds))
    result = ds.evaluate(
        fof_px, insert=False, format="numpy", vectorize=True, random_value=random_values
    )
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx"]).get_data("numpy")
    assert np.all(
        result["fof_px"]
        == data["fof_halo_mass"] * data["fof_halo_com_vx"] + random_values
    )


def test_visit_vectorize_multiple_noinsert(input_path):
    ds = oc.open(input_path)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        assert len(fof_halo_mass) == len(
            ds
        )  # checks that we skip the verification step
        return fof_halo_mass * fof_halo_com_vx

    result = ds.evaluate(fof_px, vectorize=True, insert=False)
    data = ds.select(("fof_halo_mass", "fof_halo_com_vx")).get_data()

    assert np.all(result["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_visit_rows_nfw(input_path):
    ds = oc.open(input_path).filter(oc.col("sod_halo_cdelta") > 0)

    def nfw(sod_halo_radius, sod_halo_cdelta, sod_halo_mass):
        r_ = np.logspace(-2, 0, 50)
        A = np.log(1 + sod_halo_cdelta) - sod_halo_cdelta / (1 + sod_halo_cdelta)

        halo_density = sod_halo_mass / (4 / 3 * np.pi * sod_halo_radius**3)
        profile = halo_density / (3 * A * r_) / (1 / sod_halo_cdelta + r_) ** 2
        return {"nfw_radius": r_ * sod_halo_radius, "nfw_profile": profile}

    ds = ds.evaluate(nfw, insert=True)

    assert "nfw_radius" in ds.columns
    assert "nfw_profile" in ds.columns
    profile = ds.select("nfw_profile").get_data("numpy")
    assert profile.shape == (len(ds), 50)


def test_visit_rows_multiple(input_path):
    ds = oc.open(input_path).take(100)

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=False, insert=True)
    assert "fof_px" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_halo_com_vx", "fof_px"]).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_visit_rows_single(input_path):
    ds = oc.open(input_path).take(100)

    def fof_random(fof_halo_mass):
        return fof_halo_mass * np.random.randint(0, 100)

    ds = ds.evaluate(fof_random, vectorize=False, insert=True)
    assert "fof_random" in ds.columns
    data = ds.select(["fof_halo_mass", "fof_random"]).get_data()
    assert data["fof_random"].unit == u.solMass
    factor = data["fof_random"] / data["fof_halo_mass"]
    factor = factor.value
    assert np.all(np.isclose(factor, np.around(factor)))


def test_visit_with_sort(input_path, tmp_path):
    ds = oc.open(input_path).take(1000).sort_by("fof_halo_mass")

    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    ds = ds.evaluate(fof_px, vectorize=True, insert=True)
    oc.write(tmp_path / "data.hdf5", ds)
    oc.open(tmp_path / "data.hdf5")

    data = ds.select(("fof_halo_mass", "fof_halo_com_vx", "fof_px")).get_data("numpy")
    assert np.all(data["fof_px"] == data["fof_halo_mass"] * data["fof_halo_com_vx"])


def test_select_oom(input_path):
    with oc.open(input_path) as ds:
        data = ds.get_data()
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols, 10, replace=False)
        selected = ds.select(selected_cols)
        selected_data = selected.get_data()

    for col in selected_cols:
        assert np.all(data[col] == selected_data[col])
    assert set(selected_cols) == set(selected_data.columns)


def test_select_single(input_path):
    with oc.open(input_path) as ds:
        for column in ds.columns:
            selected = ds.select(column)
            selected_data = selected.get_data()
            assert isinstance(selected_data, (u.Quantity, np.ndarray))


def test_select_single_numpy(input_path):
    with oc.open(input_path) as ds:
        data = ds.get_data()
        cols = list(data.columns)
        # select 10 columns at random
        selected_cols = np.random.choice(cols)
        selected = ds.select(selected_cols)
        selected_data = selected.get_data("numpy")
    assert isinstance(selected_data, np.ndarray)


def test_write_after_filter(input_path, tmp_path):
    with oc.open(input_path) as ds:
        ds = ds.filter(oc.col("sod_halo_mass") > 0)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.get_data()

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        filtered_data = new_ds.get_data()
        for col in filtered_data.columns:
            assert np.all(filtered_data[col] == data[col])


def test_write_after_derive(input_path, tmp_path):
    with oc.open(input_path) as ds:
        col = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
        ds = ds.with_new_columns(fof_halo_px=col)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.select("fof_halo_px").get_data()

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        written_data = new_ds.select("fof_halo_px").get_data()
        assert np.all(np.isclose(data, written_data))


def test_write_after_evaluate(input_path, tmp_path):
    def fof_px(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    with oc.open(input_path) as ds:
        ds = ds.evaluate(fof_px, insert=True, vectorize=True)

        oc.write(tmp_path / "haloproperties.hdf5", ds)

        data = ds.select("fof_px").get_data()

    with oc.open(tmp_path / "haloproperties.hdf5") as new_ds:
        written_data = new_ds.select("fof_px").get_data()
        assert np.all(np.isclose(data, written_data))


def test_sort_after_filter(input_path):
    dataset = oc.open(input_path)
    dataset = dataset.filter(oc.col("fof_halo_mass") > 1e13)
    dataset = dataset.sort_by("sod_halo_mass")
    data = dataset.select(("fof_halo_mass", "sod_halo_mass")).get_data("numpy")
    assert np.all(data["fof_halo_mass"] > 1e13)
    assert np.all(data["sod_halo_mass"][:-1] <= data["sod_halo_mass"][1:])


def test_sort_rows(input_path):
    dataset = oc.open(input_path)
    dataset = dataset.sort_by("sod_halo_mass")
    dataset = dataset.take(100)
    fof_masses = dataset.select("fof_halo_mass").get_data("numpy")
    for i, row in enumerate(dataset.rows()):
        assert row["fof_halo_mass"].value == fof_masses[i]


def test_rows_cache(input_path):
    dataset = oc.open(input_path)
    dataset = dataset.sort_by("sod_halo_mass")
    dataset = dataset.take(100)
    fof_px = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    dataset = dataset.with_new_columns(fof_px=fof_px)

    for i, row in enumerate(dataset.rows()):
        assert row["fof_px"] == row["fof_halo_mass"] * row["fof_halo_com_vx"]

    # After iterating rows(), derived columns should be cached.
    state = dataset._Dataset__state
    cache = state.cache
    columns = state.column_map
    assert "fof_px" in cache.columns
    pairs = {(columns["fof_px"], "fof_px")}
    uuid_data = cache.get_data(pairs)
    flat = {name: arr for d in uuid_data.values() for name, arr in d.items()}
    assert len(flat["fof_px"]) == 100


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cache(input_path):
    from time import time

    dataset = oc.open(input_path)

    start1 = time()
    dataset.get_data()
    end1 = time()
    dt1 = end1 - start1

    start2 = time()
    dataset.get_data()
    end2 = time()
    dt2 = end2 - start2
    assert (dt2 / dt1) < 0.2


def test_cache_select(input_path):
    dataset = oc.open(input_path)
    _ = dataset.get_data()
    cache = dataset._Dataset__state.cache
    assert set(dataset.columns) == cache.columns
    columns = np.random.choice(dataset.columns, 5, replace=False)
    dataset2 = dataset.select(columns)

    del dataset
    from gc import collect

    collect()  # force a collection

    assert cache.columns == set(dataset2.columns)


def test_cache_filter(input_path):
    dataset = oc.open(input_path)
    dataset = dataset.filter(oc.col("fof_halo_mass") > 1e14)

    cache = dataset._Dataset__state.cache

    assert (
        len(cache.columns) > 0 or cache._ColumnCache__parent() is not None
    )  # just to be safe


def test_cache_column_conversion(input_path):
    dataset = oc.open(input_path)

    data = dataset.get_data()
    columns = set()
    for col in data.columns:
        if data[col].unit == u.Mpc:
            columns.add(col)

    dataset2 = dataset.with_units(conversions={u.Mpc: u.lyr})

    cache = dataset2._Dataset__state.cache

    data2 = dataset2.get_data()
    assert len(cache.columns) == len(dataset2.columns)
    for col in columns:
        assert data2[col].unit == u.lyr


def test_cache_change_units(input_path):
    dataset = oc.open(input_path)
    dataset.get_data()
    dataset2 = dataset.with_units("scalefree")

    cache = dataset2._Dataset__state.cache

    assert (
        len(cache.columns) == 0 and cache._ColumnCache__parent is None
    )  # just to be safe


def test_cache_conversion_propogation(input_path):
    dataset = oc.open(input_path)
    dataset2 = dataset.with_units(conversions={u.Mpc: u.lyr}, fof_halo_center_x=u.km)
    dataset2.get_data()

    state = dataset._Dataset__state
    state2 = dataset2._Dataset__state
    cache = state.cache
    cache2 = state2.cache
    col_to_uuid = state.column_map
    pairs = {(uuid, name) for name, uuid in col_to_uuid.items()}

    assert len(cache.columns) == len(dataset2.columns)  # just to be safe
    flat = {
        name: arr for d in cache.get_data(pairs).values() for name, arr in d.items()
    }
    flat2 = {
        name: arr for d in cache2.get_data(pairs).values() for name, arr in d.items()
    }
    for col in flat.values():
        if isinstance(col, u.Quantity):
            assert col.unit not in [u.lyr, u.km]
    for col in flat2.values():
        if isinstance(col, u.Quantity):
            assert col.unit != u.Mpc


def test_write_after_sorted(input_path, tmp_path):
    dataset = oc.open(input_path)
    dataset = dataset.sort_by("fof_halo_mass", invert=True)
    halo_tags = dataset.select("fof_halo_tag").get_data("numpy")
    oc.write(tmp_path / "test.hdf5", dataset)
    new_dataset = oc.open(tmp_path / "test.hdf5").sort_by("fof_halo_mass", invert=True)
    to_check = new_dataset.select(("fof_halo_mass", "fof_halo_tag")).get_data("numpy")
    assert np.all(to_check["fof_halo_mass"][:-1] >= to_check["fof_halo_mass"][1:])
    assert np.all(to_check["fof_halo_tag"] == halo_tags)


def test_descriptions_after_insert(input_path, tmp_path):
    ds = oc.open(input_path)
    p = tmp_path / "test.hdf5"
    random_data = np.random.randint(0, 100, len(ds))
    ds = ds.with_new_columns(random_data=random_data)
    assert ds.descriptions["random_data"] == "None"
    oc.write(p, ds)
    ds = oc.open(p)
    descriptions = ds.descriptions
    assert descriptions["random_data"] == "None"


def test_description_with_insert(input_path, tmp_path):
    ds = oc.open(input_path)
    p = tmp_path / "test.hdf5"
    random_data = np.random.randint(0, 100, len(ds))
    ds = ds.with_new_columns(
        random_data=random_data, descriptions="random data for a test"
    )
    assert ds.descriptions["random_data"] == "random data for a test"
    oc.write(p, ds)
    ds = oc.open(p)
    descriptions = ds.descriptions
    assert descriptions["random_data"] == "random data for a test"


def test_description_with_insert_multiple(input_path, tmp_path):
    ds = oc.open(input_path)
    p = tmp_path / "test.hdf5"
    random_data = np.random.randint(0, 100, len(ds))
    fof_halo_px = oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
    ds = ds.with_new_columns(
        random_data=random_data,
        halo_px=fof_halo_px,
        descriptions={
            "random_data": "random data for a test",
            "halo_px": "halo x momentum",
        },
    )
    assert ds.descriptions["random_data"] == "random data for a test"
    assert ds.descriptions["halo_px"] == "halo x momentum"
    oc.write(p, ds)
    ds = oc.open(p)
    descriptions = ds.descriptions
    assert descriptions["random_data"] == "random data for a test"
    assert descriptions["halo_px"] == "halo x momentum"


def test_with_new_columns_overwrite(input_path):
    ds = oc.open(input_path)
    log_mass = oc.col("fof_halo_mass").log10()

    with pytest.raises(ValueError):
        ds.with_new_columns(fof_halo_mass=log_mass)

    ds_overwritten = ds.with_new_columns(fof_halo_mass=log_mass, allow_overwrite=True)
    assert "fof_halo_mass" in ds_overwritten.columns

    original = ds.select("fof_halo_mass").get_data("numpy")
    overwritten = ds_overwritten.select("fof_halo_mass").get_data("numpy")
    assert np.all(np.isclose(overwritten, np.log10(original)))


def test_evaluate_overwrite(input_path):
    ds = oc.open(input_path)

    def fof_halo_mass(fof_halo_mass, fof_halo_com_vx):
        return fof_halo_mass * fof_halo_com_vx

    with pytest.raises(ValueError):
        ds.evaluate(fof_halo_mass, vectorize=True, insert=True)

    ds_overwritten = ds.evaluate(
        fof_halo_mass, vectorize=True, insert=True, allow_overwrite=True
    )
    assert "fof_halo_mass" in ds_overwritten.columns

    data = ds.select(["fof_halo_mass", "fof_halo_com_vx"]).get_data("numpy")
    overwritten = ds_overwritten.select("fof_halo_mass").get_data("numpy")
    assert np.all(
        np.isclose(overwritten, data["fof_halo_mass"] * data["fof_halo_com_vx"])
    )
