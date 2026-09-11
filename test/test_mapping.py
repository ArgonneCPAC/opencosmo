from __future__ import annotations

import shutil
from itertools import combinations
from typing import TYPE_CHECKING
from uuid import uuid4

import h5py
import numpy as np
import opencosmo.collection.simulation.io as simulation_io
import opencosmo.collection.simulation.simulation as simulation_module
import opencosmo.mapping.write as mapping_write
import pytest
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.serial import allocate
from opencosmo.io.verify import verify_structure
from opencosmo.mapping.mapping import DatasetMatchSet, rebuild_single_with_new_source
from opencosmo.utils import normalize_kwarg_name

import opencosmo as oc
from opencosmo.index import into_array

if TYPE_CHECKING:
    from pathlib import Path


REFERENCE = normalize_kwarg_name("SCIDAC_128_GO")
SIMULATION_A = normalize_kwarg_name(
    "KAPPA_2.222_EGW_0.759_SEED_7.810e5_VKIN_5889_EPS_5.257"
)
SIMULATION_B = normalize_kwarg_name(
    "KAPPA_2.984_EGW_0.682_SEED_6e5_VKIN_7286_EPS_4.883"
)
SIMULATION_C = normalize_kwarg_name(
    "KAPPA_2.444_EGW_1_SEED_6.667e5_VKIN_4841_EPS_6.006"
)
TARGET_SIMULATIONS = (SIMULATION_A, SIMULATION_B, SIMULATION_C)
MAPPED_SIMULATIONS = (REFERENCE, *TARGET_SIMULATIONS)


@pytest.fixture
def mapped_paths(test_data):
    return {
        REFERENCE: test_data.snapshot.mapping_reference,
        SIMULATION_A: test_data.snapshot.scidac(0).halo_properties,
        SIMULATION_B: test_data.snapshot.scidac(1).halo_properties,
        SIMULATION_C: test_data.snapshot.scidac(2).halo_properties,
    }


def _open_mapped(mapped_paths, mapping_path, simulations=MAPPED_SIMULATIONS):
    return oc.open(*(mapped_paths[name] for name in simulations), mapping_path)


def _expected_pairwise_maps(mapping_path: Path, mapped_paths):
    uuids = {}
    lengths = {}
    for name, path in mapped_paths.items():
        with h5py.File(path) as file:
            uuids[name] = str(file["data"].attrs["uuid"])
            lengths[name] = len(file["data/fof_halo_tag"])

    with h5py.File(mapping_path) as file:
        group = file["map"]
        reference_uuid = str(group.attrs["reference"])
        assert uuids[REFERENCE] == reference_uuid

        primary = {
            name: group[f"primary/{uuids[name]}/index"][:]
            for name in TARGET_SIMULATIONS
        }

    pairwise = {}
    for target, target_primary in primary.items():
        pairwise[(REFERENCE, target)] = target_primary
        inverse = np.full(lengths[target], -1, dtype=np.int64)
        reference_rows = np.flatnonzero(target_primary >= 0)
        inverse[target_primary[reference_rows]] = reference_rows
        pairwise[(target, REFERENCE)] = inverse

    with h5py.File(mapping_path) as file:
        auxiliary_group = file["map/auxiliary"]
        for source, target in combinations(TARGET_SIMULATIONS, 2):
            source_uuid = uuids[source]
            target_uuid = uuids[target]
            if source_uuid < target_uuid:
                pair = auxiliary_group[f"{source_uuid}__{target_uuid}"]
                auxiliary_source = pair["source"][:]
                auxiliary_target = pair["target"][:]
            else:
                pair = auxiliary_group[f"{target_uuid}__{source_uuid}"]
                auxiliary_source = pair["target"][:]
                auxiliary_target = pair["source"][:]

            source_to_target = np.full(lengths[source], -1, dtype=np.int64)
            primary_rows = np.flatnonzero(
                (primary[source] >= 0) & (primary[target] >= 0)
            )
            source_to_target[primary[source][primary_rows]] = primary[target][
                primary_rows
            ]
            source_to_target[auxiliary_source] = auxiliary_target
            pairwise[(source, target)] = source_to_target

            target_to_source = np.full(lengths[target], -1, dtype=np.int64)
            target_to_source[primary[target][primary_rows]] = primary[source][
                primary_rows
            ]
            target_to_source[auxiliary_target] = auxiliary_source
            pairwise[(target, source)] = target_to_source
    return pairwise


def _assert_matches_mapping(before, matched, source, pairwise):
    source_index = into_array(before[source].index)
    rows_to_keep = np.ones(len(source_index), dtype=bool)

    for target in before.keys() - {source}:
        mapped_rows = pairwise[(source, target)][source_index]
        rows_to_keep &= mapped_rows >= 0
        rows_to_keep &= np.isin(mapped_rows, into_array(before[target].index))

    np.testing.assert_array_equal(
        into_array(matched[source].index), source_index[rows_to_keep]
    )
    for target in before.keys() - {source}:
        np.testing.assert_array_equal(
            into_array(matched[target].index),
            pairwise[(source, target)][source_index[rows_to_keep]],
        )


def _column(dataset, name):
    return np.asarray(dataset.select(name).get_data(format="numpy"))


def _absolute_rows_from_ids(dataset, ids, identifier="fof_halo_tag"):
    dataset_ids = _column(dataset, identifier)
    dataset_index = into_array(dataset.index)
    rows_by_id = dict(zip(dataset_ids.tolist(), dataset_index.tolist(), strict=True))
    return np.asarray([rows_by_id[value] for value in ids.tolist()], dtype=np.int64)


def _expected_source_driven_ids(
    original, expected_source, source, pairwise, identifier="fof_halo_tag"
):
    expected_ids = {source: _column(expected_source, identifier)}
    source_rows = _absolute_rows_from_ids(
        original[source], expected_ids[source], identifier
    )
    for target in original.keys() - {source}:
        target_rows = pairwise[(source, target)][source_rows]
        assert np.all(target_rows >= 0)

        target_ids = _column(original[target], identifier)
        target_index = into_array(original[target].index)
        ids_by_row = dict(zip(target_index.tolist(), target_ids.tolist(), strict=True))
        expected_ids[target] = np.asarray(
            [ids_by_row[row] for row in target_rows.tolist()], dtype=target_ids.dtype
        )
    return expected_ids


def _assert_source_driven_result(
    original, expected_source, actual, source, pairwise, identifier="fof_halo_tag"
):
    """Assert logical row order in every catalog follows ``expected_source``."""
    expected_ids = _expected_source_driven_ids(
        original, expected_source, source, pairwise, identifier
    )
    for name, ids in expected_ids.items():
        np.testing.assert_array_equal(_column(actual[name], identifier), ids)


def _assert_mapping_equal(before, after, identifier="fof_halo_tag"):
    """Assert two collections describe the same matches using stable row IDs."""
    assert set(before) == set(after)
    names = tuple(sorted(before))

    for source in names:
        before_matched = before.match(source)
        after_matched = after.match(source)

        before_ids = [
            np.asarray(before_matched[name].select(identifier).get_data(format="numpy"))
            for name in names
        ]
        after_ids = [
            np.asarray(after_matched[name].select(identifier).get_data(format="numpy"))
            for name in names
        ]

        before_pairs = np.column_stack(before_ids)
        after_pairs = np.column_stack(after_ids)
        before_order = np.lexsort(before_pairs.T[::-1])
        after_order = np.lexsort(after_pairs.T[::-1])
        np.testing.assert_array_equal(
            before_pairs[before_order],
            after_pairs[after_order],
            err_msg=f"Mapping differs for source {source!r}",
        )


def _assert_ordered_mapping_equal(
    before, after, source, identifier="fof_halo_tag", sort_by=None
):
    """Assert exact matched row order survives persistence."""
    after = after.match(source)
    if sort_by is not None:
        after = after.sort_by(*sort_by).take_range(0, len(after[source]))
    assert tuple(before.keys()) == tuple(after.keys())
    for name in before.keys():
        np.testing.assert_array_equal(
            _column(after[name], identifier),
            _column(before[name], identifier),
            err_msg=name,
        )


@pytest.mark.parametrize("source", MAPPED_SIMULATIONS)
def test_match_aligns_rows_for_each_source(source, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(source)

    assert isinstance(matched, oc.SimulationCollection)
    assert set(matched) == set(MAPPED_SIMULATIONS)
    _assert_matches_mapping(collection, matched, source, pairwise)


@pytest.mark.parametrize("source", TARGET_SIMULATIONS)
def test_match_without_reference(source, mapped_paths, test_data):
    simulations = TARGET_SIMULATIONS
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(source)

    assert set(matched) == set(simulations)
    _assert_matches_mapping(collection, matched, source, pairwise)


def test_match_honors_existing_row_selection(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).take_range(
        10_000, 100_000
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(REFERENCE)

    _assert_matches_mapping(collection, matched, REFERENCE, pairwise)


@pytest.mark.parametrize(
    "filtered_simulations",
    (None, (REFERENCE,), (SIMULATION_A,)),
    ids=("all", "source-only", "target-only"),
)
def test_match_honors_filters(filtered_simulations, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    original_indices = {
        name: into_array(dataset.index) for name, dataset in collection.items()
    }
    collection = collection.filter(
        oc.col("fof_halo_mass") > 1e14,
        datasets=filtered_simulations,
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    expected_filtered = set(filtered_simulations or MAPPED_SIMULATIONS)
    for name, dataset in collection.items():
        if name in expected_filtered:
            assert len(dataset) < len(original_indices[name])
        else:
            np.testing.assert_array_equal(
                into_array(dataset.index), original_indices[name]
            )

    matched = collection.match(REFERENCE)

    _assert_matches_mapping(collection, matched, REFERENCE, pairwise)


@pytest.mark.parametrize("source", MAPPED_SIMULATIONS)
def test_matched_take_range_is_driven_by_active_source(source, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(source)
    expected_source = matched[source].take_range(7, 31)

    result = matched.take_range(7, 31)

    _assert_source_driven_result(original, expected_source, result, source, pairwise)


def test_matched_filter_is_evaluated_only_on_active_source(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    threshold = np.median(_column(matched[REFERENCE], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > threshold
    expected_source = matched[REFERENCE].filter(mask)

    result = matched.filter(mask)

    assert 0 < len(expected_source) < len(matched[REFERENCE])
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parametrize("invert", (False, True), ids=("ascending", "descending"))
def test_matched_sort_uses_active_source_order(invert, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = matched[REFERENCE].sort_by("fof_halo_mass", invert=invert)
    result = matched.sort_by("fof_halo_mass", invert=invert)

    masses = _column(result[REFERENCE], "fof_halo_mass")
    differences = np.diff(masses)
    assert np.all(differences <= 0 if invert else differences >= 0)
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parametrize("at", ("start", "end"))
def test_matched_take_is_driven_by_active_source(at, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = matched[REFERENCE].take(23, at=at)

    result = matched.take(23, at=at)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_random_take_preserves_source_order(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)

    result = matched.take(37, at="random")
    selected_source = result[REFERENCE]

    assert len(selected_source) == 37
    selected_ids = _column(selected_source, "fof_halo_tag")
    matched_ids = _column(matched[REFERENCE], "fof_halo_tag")
    positions = {value: position for position, value in enumerate(matched_ids.tolist())}
    selected_positions = np.asarray(
        [positions[value] for value in selected_ids.tolist()]
    )
    assert np.all(np.diff(selected_positions) > 0)
    _assert_source_driven_result(original, selected_source, result, REFERENCE, pairwise)


def test_matched_bound_is_evaluated_on_active_source(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).with_units("scalefree")
    coordinate_names = tuple(f"fof_halo_center_{axis}" for axis in "xyz")
    coordinates = matched[REFERENCE].select(coordinate_names).get_data(format="numpy")
    lower = tuple(
        float(np.quantile(coordinates[name], 0.3)) for name in coordinate_names
    )
    upper = tuple(
        float(np.quantile(coordinates[name], 0.7)) for name in coordinate_names
    )
    region = oc.make_box(lower, upper)
    expected_source = matched[REFERENCE].bound(region)

    result = matched.bound(region)

    assert 0 < len(expected_source) < len(matched[REFERENCE])
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_operation_intersects_pre_filtered_target(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    target_threshold = np.median(_column(original[SIMULATION_A], "fof_halo_mass"))
    original = original.filter(
        oc.col("fof_halo_mass") > target_threshold, datasets=SIMULATION_A
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    source_threshold = np.median(_column(matched[REFERENCE], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > source_threshold
    expected_source = matched[REFERENCE].filter(mask)

    result = matched.filter(mask)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_chained_index_operations_remain_aligned(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(SIMULATION_A)
    threshold = np.median(_column(matched[SIMULATION_A], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > threshold
    expected_source = (
        matched[SIMULATION_A]
        .filter(mask)
        .sort_by("fof_halo_mass", invert=True)
        .take_range(3, 29)
    )

    result = (
        matched.filter(mask).sort_by("fof_halo_mass", invert=True).take_range(3, 29)
    )

    _assert_source_driven_result(
        original, expected_source, result, SIMULATION_A, pairwise
    )


@pytest.mark.parametrize(
    "operation",
    ("filter", "sort", "take-start", "take-end", "take-random", "range", "bound"),
)
def test_clear_match_rebuilds_pending_targets(operation, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)

    if operation == "filter":
        result = matched.filter(oc.col("fof_halo_mass") > 1e14)
    elif operation == "sort":
        result = matched.sort_by("fof_halo_mass", invert=True)
    elif operation.startswith("take-"):
        result = matched.take(23, at=operation.removeprefix("take-"))
    elif operation == "range":
        result = matched.take_range(7, 31)
    else:
        matched = matched.with_units("scalefree")
        result = matched.bound(oc.make_box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)))

    expected_source = result[REFERENCE]
    cleared = result.clear_match()

    _assert_source_driven_result(
        original, expected_source, cleared, REFERENCE, pairwise
    )
    assert {len(dataset) for dataset in cleared.values()} == {len(expected_source)}


def test_mapped_collection_context_manager_exits_cleanly(mapped_paths, test_data):
    with (
        _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
        .match(REFERENCE)
        .take_range(7, 31) as matched
    ):
        assert {len(dataset) for dataset in matched.values()} == {24}


def test_matched_targets_are_rebuilt_at_most_once(mapped_paths, test_data, monkeypatch):
    calls = 0
    prepare = simulation_module.prepare_matched_datasets

    def counting_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(simulation_module, "prepare_matched_datasets", counting_prepare)
    matched = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )

    list(matched.values())
    list(matched.items())
    repr(matched)

    assert calls == 1

    pending = matched.take_range(7, 31)
    list(pending.values())
    list(pending.values())

    assert calls == 2


@pytest.mark.parametrize("accessor", ("getitem", "values", "items"))
def test_matched_dataset_access_rebuilds_targets(accessor, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).take_range(7, 31)
    expected_source = matched[REFERENCE]
    expected_ids = _expected_source_driven_ids(
        original, expected_source, REFERENCE, pairwise
    )

    if accessor == "getitem":
        accessed = {name: matched[name] for name in matched.keys()}
    elif accessor == "values":
        accessed = dict(zip(matched.keys(), matched.values(), strict=True))
    else:
        accessed = dict(matched.items())

    for name, dataset in accessed.items():
        np.testing.assert_array_equal(
            _column(dataset, "fof_halo_tag"), expected_ids[name]
        )


def test_matched_evaluate_rebuilds_targets_before_evaluation(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).take_range(7, 31)
    expected_source = matched[REFERENCE]
    expected_ids = _expected_source_driven_ids(
        original, expected_source, REFERENCE, pairwise
    )

    def evaluated_tag(fof_halo_tag):
        return fof_halo_tag

    result = matched.evaluate(
        evaluated_tag, vectorize=True, insert=False, format="numpy"
    )

    for name, dataset in matched.items():
        np.testing.assert_array_equal(
            _column(dataset, "fof_halo_tag"), expected_ids[name]
        )
        np.testing.assert_array_equal(
            result[name]["evaluated_tag"], _column(dataset, "fof_halo_tag")
        )


@pytest.mark.parametrize(
    "transform",
    (
        pytest.param(
            lambda collection: collection.select("fof_halo_tag", "fof_halo_mass"),
            id="column-selection",
        ),
        pytest.param(
            lambda collection: collection.with_units("scalefree"),
            id="unit-conversion",
        ),
        pytest.param(
            lambda collection: collection.with_new_columns(
                doubled_mass=oc.col("fof_halo_mass") * 2
            ),
            id="derived-column",
        ),
    ),
)
def test_non_index_operation_preserves_active_match_source(
    transform, mapped_paths, test_data
):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = transform(matched[REFERENCE]).take_range(4, 19)

    result = transform(matched).take_range(4, 19)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parametrize("datasets", (REFERENCE, [REFERENCE], (REFERENCE,)))
def test_matched_filter_accepts_active_source_dataset_forms(
    datasets, mapped_paths, test_data
):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    collection = original.match(REFERENCE)
    mask = oc.col("fof_halo_mass") > 1e14
    expected_source = collection[REFERENCE].filter(mask)

    result = collection.filter(mask, datasets=datasets)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


def test_matched_filter_rejects_non_source_datasets(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )
    mask = oc.col("fof_halo_mass") > 1e14

    with pytest.raises(ValueError, match="active source"):
        collection.filter(mask, datasets=SIMULATION_A)
    with pytest.raises(ValueError, match="active source"):
        collection.filter(mask, datasets=(REFERENCE, SIMULATION_A))


@pytest.mark.parametrize(
    ("paths", "message"),
    (
        (("primary", "mapping_reference"), "different simulations"),
        (("primary", "alternate_step"), "KAPPA_2_EGW_0.568_SEED_1.048e6"),
    ),
)
def test_open_without_connecting_mapping_raises(paths, message, test_data):
    snapshot = test_data.snapshot
    first = (
        snapshot.primary.halo_properties
        if paths[0] == "primary"
        else getattr(snapshot, paths[0])
    )
    second = getattr(snapshot, paths[1])

    with pytest.raises(ValueError, match=message):
        oc.open(first, second)


def test_mapping_file_alone_raises(test_data):
    with pytest.raises(ValueError, match="Cannot open a dataset mapping on its own"):
        oc.open(test_data.snapshot.halo_mapping)


def test_open_multiple_mapping_files_raises(mapped_paths, test_data, tmp_path):
    second_mapping = tmp_path / "second_mapping.hdf5"
    shutil.copy(test_data.snapshot.halo_mapping, second_mapping)

    with pytest.raises(ValueError, match="multiple dataset mapping files"):
        oc.open(
            mapped_paths[REFERENCE],
            mapped_paths[SIMULATION_A],
            test_data.snapshot.halo_mapping,
            second_mapping,
        )


def test_primary_mapping_length_must_match_reference(mapped_paths, test_data, tmp_path):
    mapping = tmp_path / "invalid_length_mapping.hdf5"
    shutil.copy(test_data.snapshot.halo_mapping, mapping)
    with h5py.File(mapping, "a") as file:
        primary = file["map/primary"]
        target = next(iter(primary))
        slot = primary[target]
        values = slot["index"][:-1]
        del slot["index"]
        slot.create_dataset("index", data=values)

    with pytest.raises(ValueError, match="reference dataset length"):
        oc.open(
            mapped_paths[REFERENCE],
            mapped_paths[SIMULATION_A],
            mapping,
        )


def test_match_requires_mapping(test_data):
    collection = oc.open(test_data.snapshot.multi_simulation)

    with pytest.raises(ValueError, match="does not contain matching information"):
        collection.match("scidac1")


def test_match_requires_known_source(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)

    with pytest.raises(ValueError, match="does not have a simulation named unknown"):
        collection.match("unknown")


def test_mapping_write(mapped_paths, test_data, tmp_path):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


def test_mapping_write_unfiltered(mapped_paths, test_data, tmp_path):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


def test_mapping_write_without_reference(mapped_paths, test_data, tmp_path):
    simulations = TARGET_SIMULATIONS
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


@pytest.mark.parametrize(
    ("transform", "sort_after_read"),
    (
        pytest.param(
            lambda collection: collection.filter(oc.col("fof_halo_mass") > 1e14),
            None,
            id="filter",
        ),
        pytest.param(
            lambda collection: collection.sort_by(
                "fof_halo_mass", invert=True
            ).take_range(7, 31),
            ("fof_halo_mass", True),
            id="sort-and-range",
        ),
        pytest.param(
            lambda collection: collection.take(37, at="random"),
            None,
            id="random-take",
        ),
    ),
)
def test_active_match_write_preserves_order(
    transform, sort_after_read, mapped_paths, test_data, tmp_path
):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )
    collection = transform(collection)
    path = tmp_path / "test.hdf5"

    oc.write(path, collection)
    written = oc.open(path)

    _assert_ordered_mapping_equal(
        collection, written, REFERENCE, sort_by=sort_after_read
    )


def test_active_match_write_restores_canonical_spatial_order(
    mapped_paths, test_data, tmp_path
):
    collection = (
        _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
        .match(REFERENCE)
        .take_range(7, 31)
    )
    coordinate_names = tuple(f"fof_halo_center_{axis}" for axis in "xyz")
    indices = {name: into_array(dataset.index) for name, dataset in collection.items()}
    assert any(np.any(np.diff(index) < 0) for index in indices.values())

    expected = {}
    for name, index in indices.items():
        with h5py.File(mapped_paths[name]) as file:
            expected[name] = {
                column: file[f"data/{column}"][:][np.sort(index)]
                for column in coordinate_names
            }

    path = tmp_path / "test.hdf5"
    oc.write(path, collection)
    with h5py.File(path) as file:
        for name, coordinates in expected.items():
            for column, values in coordinates.items():
                np.testing.assert_array_equal(file[f"{name}/data/{column}"][:], values)

    _assert_mapping_equal(collection, oc.open(path))


def test_mapping_write_with_empty_target(mapped_paths, test_data, tmp_path):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    collection = collection.filter(
        oc.col("fof_halo_mass") < 0,
        datasets=SIMULATION_C,
    )
    assert len(collection[SIMULATION_C]) == 0

    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


def test_mapping_write_reference_and_one_target(mapped_paths, test_data, tmp_path):
    simulations = (REFERENCE, SIMULATION_A)
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    oc.write(tmp_path / "test.hdf5", collection)
    written = oc.open(tmp_path / "test.hdf5")

    _assert_mapping_equal(collection, written)


def test_make_schema_retains_raw_primary_and_auxiliary_endpoints(tmp_path):
    reference, first, second = (uuid4() for _ in range(3))
    new_reference, new_first, new_second = (uuid4() for _ in range(3))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_first = file.create_dataset("first", data=[10, 11, 12])
        primary_second = file.create_dataset("second", data=[20, 21, 22])
        auxiliary_first = file.create_dataset("aux_first", data=[50])
        auxiliary_second = file.create_dataset("aux_second", data=[60])
        match_set = DatasetMatchSet(
            reference,
            {first: primary_first, second: primary_second},
            {(first, second): (auxiliary_first, auxiliary_second)},
            {"reference": reference, "first": first, "second": second},
        )

        schema = match_set.make_schema(
            {"reference": new_reference, "first": new_first, "second": new_second},
            {
                "reference": np.array([2, 0]),
                "first": np.array([10]),
                "second": np.array([20]),
            },
        )

    primary = schema.children["primary"].children
    np.testing.assert_array_equal(
        primary[str(new_first)].columns["index"].data, [10, 12]
    )
    np.testing.assert_array_equal(
        primary[str(new_second)].columns["index"].data, [20, 22]
    )
    auxiliary = schema.children["auxiliary"].children[f"{new_first}__{new_second}"]
    np.testing.assert_array_equal(auxiliary.columns["source"].data, [50])
    np.testing.assert_array_equal(auxiliary.columns["target"].data, [60])


def test_unresolved_mapping_schema_is_rejected_by_generic_verification(tmp_path):
    schema = make_schema(
        "/",
        FileEntry.SIMULATION_COLLECTION,
        children={
            "map": make_schema(
                "map",
                FileEntry.METADATA,
                attributes={},
            )
        },
    )

    with pytest.raises(
        ValueError, match="Unresolved raw-coordinate simulation mapping"
    ):
        verify_structure(schema)
    with h5py.File(tmp_path / "unresolved.hdf5", "w") as file:
        with pytest.raises(
            ValueError, match="Unresolved raw-coordinate simulation mapping"
        ):
            allocate(file, schema)


def test_lowered_mapping_schema_is_accepted_by_generic_verification():
    schema = make_schema(
        "/",
        FileEntry.SIMULATION_COLLECTION,
        children={"map": make_schema("map", FileEntry.METADATA, attributes={"": {}})},
    )

    verify_structure(schema)


def test_lower_primary_mapping_preserves_unmatched_and_missing_targets():
    writer = mapping_write.ColumnWriter.from_numpy_array(np.array([8, -1, 3, 99]))

    lowered = mapping_write.__lower_primary_writer(writer, {3: 0, 8: 1})

    np.testing.assert_array_equal(lowered.data, [1, -1, 0, -1])


def test_lower_auxiliary_mapping_filters_and_sorts_pairs():
    source = mapping_write.ColumnWriter.from_numpy_array(
        np.array([20, 10, 20, 99]), attrs={"source": "attribute"}
    )
    target = mapping_write.ColumnWriter.from_numpy_array(
        np.array([7, 9, 8, 7]), attrs={"target": "attribute"}
    )

    lowered_source, lowered_target = mapping_write.__lower_auxiliary_writers(
        source, target, {10: 0, 20: 1}, {7: 1, 8: 0}
    )

    np.testing.assert_array_equal(lowered_source.data, [1, 1])
    np.testing.assert_array_equal(lowered_target.data, [0, 1])
    assert lowered_source.attrs == {"source": "attribute"}
    assert lowered_target.attrs == {"target": "attribute"}


def test_lower_mapping_rejects_duplicate_raw_ids():
    with pytest.raises(ValueError, match="duplicate output raw row IDs"):
        mapping_write.__make_output_position_lookup(np.array([3, 1, 3]))


def test_mpi_dataset_output_plan_is_stable_and_balanced():
    canonical, lookup = simulation_io.__plan_dataset_output(
        np.array([8, 2, 9, 1, 5, 4, 7]), 3
    )

    np.testing.assert_array_equal(canonical, [1, 2, 4, 5, 7, 8, 9])
    assert dict(lookup.output_positions) == {
        1: 0,
        2: 1,
        4: 2,
        5: 3,
        7: 4,
        8: 5,
        9: 6,
    }
    assert dict(lookup.writer_ranks) == {
        1: 0,
        2: 0,
        4: 0,
        5: 1,
        7: 1,
        8: 2,
        9: 2,
    }


def test_mpi_dataset_output_plan_allows_empty_writer_intervals():
    _, lookup = simulation_io.__plan_dataset_output(np.array([3, 1]), 4)

    assert dict(lookup.output_positions) == {1: 0, 3: 1}
    assert dict(lookup.writer_ranks) == {1: 0, 3: 1}


def test_mpi_dataset_output_plan_rejects_duplicate_raw_ids():
    with pytest.raises(ValueError, match="duplicate raw row IDs"):
        simulation_io.__plan_dataset_output(np.array([3, 1, 3]), 2)


def test_rebuild_with_new_source_folds_auxiliary_into_primary(tmp_path):
    old_reference, old_source, old_target = (uuid4() for _ in range(3))
    new_source, new_target = (uuid4() for _ in range(2))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[-1])
        primary_target = file.create_dataset("target", data=[-1])
        auxiliary_source = file.create_dataset("aux_source", data=[2])
        auxiliary_target = file.create_dataset("aux_target", data=[3])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_target: primary_target},
            {(old_source, old_target): (auxiliary_source, auxiliary_target)},
            {"source": old_source, "target": old_target},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "target": new_target},
            {"source": np.array([2]), "target": np.array([3])},
            "source",
        )

    np.testing.assert_array_equal(primary[new_target], [3])
    assert auxiliary == {}


def test_rebuild_with_new_source_omits_fully_routed_auxiliary_pair(tmp_path):
    old_reference, old_source, old_a, old_b = (uuid4() for _ in range(4))
    new_source, new_a, new_b = (uuid4() for _ in range(3))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[0, 1])
        primary_a = file.create_dataset("a", data=[10, 11])
        primary_b = file.create_dataset("b", data=[20, 21])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_a: primary_a, old_b: primary_b},
            {},
            {"source": old_source, "a": old_a, "b": old_b},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "a": new_a, "b": new_b},
            {
                "source": np.array([0, 1]),
                "a": np.array([10, 11]),
                "b": np.array([20, 21]),
            },
            "source",
        )

    np.testing.assert_array_equal(primary[new_a], [10, 11])
    np.testing.assert_array_equal(primary[new_b], [20, 21])
    assert auxiliary == {}


def test_rebuild_with_new_source_no_surviving_correspondence(tmp_path):
    old_reference, old_source, old_target = (uuid4() for _ in range(3))
    new_source, new_target = (uuid4() for _ in range(2))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[0, 1])
        primary_target = file.create_dataset("target", data=[10, 11])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_target: primary_target},
            {},
            {"source": old_source, "target": old_target},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "target": new_target},
            {
                "source": np.array([0, 1]),
                "target": np.array([20, 21]),
            },
            "source",
        )

    np.testing.assert_array_equal(primary[new_target], [10, 11])
    assert auxiliary == {}


def test_rebuild_with_new_source_emits_only_residual_auxiliary_pairs(tmp_path):
    old_reference, old_source, old_a, old_b = (uuid4() for _ in range(4))
    new_source, new_a, new_b = (uuid4() for _ in range(3))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[0, -1])
        primary_a = file.create_dataset("a", data=[0, 1])
        primary_b = file.create_dataset("b", data=[0, 1])
        auxiliary_a = file.create_dataset("aux_a", data=[2])
        auxiliary_b = file.create_dataset("aux_b", data=[2])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_a: primary_a, old_b: primary_b},
            {(old_a, old_b): (auxiliary_a, auxiliary_b)},
            {"source": old_source, "a": old_a, "b": old_b},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "a": new_a, "b": new_b},
            {
                "source": np.array([0]),
                "a": np.array([0, 1, 2, 3]),
                "b": np.array([0, 1, 2]),
            },
            "source",
        )

    np.testing.assert_array_equal(primary[new_a], [0])
    np.testing.assert_array_equal(primary[new_b], [0])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][0], [1, 2])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][1], [1, 2])


def test_rebuild_with_new_source_honors_asymmetric_selections(tmp_path):
    old_reference, old_source, old_a, old_b = (uuid4() for _ in range(4))
    new_source, new_a, new_b = (uuid4() for _ in range(3))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[0, 1, 2])
        primary_a = file.create_dataset("a", data=[10, 11, 12])
        primary_b = file.create_dataset("b", data=[20, 21, 22])
        auxiliary_a = file.create_dataset("aux_a", data=[13, 14, 15])
        auxiliary_b = file.create_dataset("aux_b", data=[23, 24, 25])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_a: primary_a, old_b: primary_b},
            {(old_a, old_b): (auxiliary_a, auxiliary_b)},
            {"source": old_source, "a": old_a, "b": old_b},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "a": new_a, "b": new_b},
            {
                "source": np.array([2, 0, 1]),
                "a": np.array([14, 10, 13, 12]),
                "b": np.array([25, 23, 20, 22]),
            },
            "source",
        )

    np.testing.assert_array_equal(primary[new_a], [10, 11, 12])
    np.testing.assert_array_equal(primary[new_b], [20, 21, 22])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][0], [13, 14])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][1], [23, 24])


def test_rebuild_with_new_source_preserves_old_reference_pair_as_auxiliary(tmp_path):
    old_reference, old_source, old_a, old_b = (uuid4() for _ in range(4))
    new_source, new_a, new_b = (uuid4() for _ in range(3))
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_source = file.create_dataset("source", data=[0, -1])
        primary_a = file.create_dataset("a", data=[-1, 0])
        primary_b = file.create_dataset("b", data=[-1, 0])
        match_set = DatasetMatchSet(
            old_reference,
            {old_source: primary_source, old_a: primary_a, old_b: primary_b},
            {},
            {"source": old_source, "a": old_a, "b": old_b},
        )

        primary, auxiliary = rebuild_single_with_new_source(
            match_set,
            {"source": new_source, "a": new_a, "b": new_b},
            {
                "source": np.array([0]),
                "a": np.array([0]),
                "b": np.array([0]),
            },
            "source",
        )

    np.testing.assert_array_equal(primary[new_a], [-1])
    np.testing.assert_array_equal(primary[new_b], [-1])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][0], [0])
    np.testing.assert_array_equal(auxiliary[(new_a, new_b)][1], [0])
