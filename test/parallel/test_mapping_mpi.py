from __future__ import annotations

import shutil
from itertools import combinations
from typing import TYPE_CHECKING

import h5py
import numpy as np
import opencosmo.collection.simulation.io as simulation_io
import pytest
from mpi4py import MPI
from opencosmo.io.schema import FileEntry, Schema
from opencosmo.utils import normalize_kwarg_name
from pytest_mpi import parallel_assert

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
ALL_PAIRS = list(combinations(MAPPED_SIMULATIONS, 2))


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


@pytest.mark.parallel(nprocs=4)
def test_mpi_dataset_output_lookup_assigns_uneven_runs():
    local_raw_ids = (
        np.array([7, 1, 8]),
        np.array([4]),
        np.empty(0, dtype=np.int64),
        np.array([2, 6]),
    )[MPI.COMM_WORLD.Get_rank()]

    lookup, target_ranks = simulation_io.__get_dataset_output_lookup(
        local_raw_ids, MPI.COMM_WORLD
    )

    parallel_assert(
        dict(lookup.output_positions) == {1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5}
    )
    parallel_assert(dict(lookup.writer_ranks) == {1: 0, 2: 0, 4: 1, 6: 1, 7: 2, 8: 3})
    expected_targets = (
        np.array([2, 0, 3]),
        np.array([1]),
        np.empty(0, dtype=np.int64),
        np.array([0, 1]),
    )[MPI.COMM_WORLD.Get_rank()]
    parallel_assert(np.array_equal(target_ranks, expected_targets))


@pytest.mark.parallel(nprocs=4)
def test_mpi_dataset_redistribution_validates_indices_collectively(monkeypatch):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    child = Schema("catalog", FileEntry.DATASET, {}, {}, {})
    schema = Schema(
        "/",
        FileEntry.SIMULATION_COLLECTION,
        {"catalog": child} if rank < 2 else {},
        {},
        {},
    )

    monkeypatch.setattr(
        simulation_io,
        "get_dataset_schema_index",
        lambda _: None if rank == 0 else np.asarray([rank], dtype=np.int64),
    )

    with pytest.raises(
        ValueError, match="Dataset 'catalog' has no output raw row index"
    ):
        simulation_io.redistribute_simulation_collection_data(schema, comm)


@pytest.mark.parallel(nprocs=4)
def test_mpi_primary_lowering_routes_remote_target_to_source_owner():
    comm = MPI.COMM_WORLD
    lookup = simulation_io.__make_dataset_output_lookup(
        np.arange(comm.Get_size(), dtype=np.int64), comm.Get_size()
    )
    source_raw_id = np.asarray(
        [(comm.Get_rank() + 1) % comm.Get_size()], dtype=np.int64
    )
    raw_target = np.asarray([(source_raw_id[0] + 1) % comm.Get_size()], dtype=np.int64)

    lowered = simulation_io.__lower_primary_values(
        source_raw_id, raw_target, lookup, lookup, comm
    )

    expected_source = comm.Get_rank()
    parallel_assert(np.array_equal(lowered, [(expected_source + 1) % comm.Get_size()]))


@pytest.mark.parallel(nprocs=4)
def test_mpi_auxiliary_lowering_routes_remote_target_to_source_owner():
    comm = MPI.COMM_WORLD
    lookup = simulation_io.__make_dataset_output_lookup(
        np.arange(comm.Get_size(), dtype=np.int64), comm.Get_size()
    )
    raw_source = (
        np.asarray([1], dtype=np.int64)
        if comm.Get_rank() == 0
        else np.empty(0, dtype=np.int64)
    )
    raw_target = (
        np.asarray([2], dtype=np.int64)
        if comm.Get_rank() == 0
        else np.empty(0, dtype=np.int64)
    )

    source, target = simulation_io.__lower_auxiliary_values(
        raw_source, raw_target, lookup, lookup, comm
    )

    expected_source = (
        np.asarray([1], dtype=np.int64)
        if comm.Get_rank() == 1
        else np.empty(0, dtype=np.int64)
    )
    expected_target = (
        np.asarray([2], dtype=np.int64)
        if comm.Get_rank() == 1
        else np.empty(0, dtype=np.int64)
    )
    parallel_assert(np.array_equal(source, expected_source))
    parallel_assert(np.array_equal(target, expected_target))


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


def _assert_matches_mapping_mpi(before, matched, source, pairwise, mapped_paths):
    """
    Under MPI each rank owns a partition of the *source* rows only. Matching
    pulls the GLOBAL rows of every other dataset that the local source rows map
    to, so the resulting target indices are generally NOT a subset of the rows
    that rank held before the match.

    A source row survives when it has a valid mapping into every target AND that
    mapped row still exists somewhere in the global state. The second condition
    only bites for chained matches, where an earlier match may already have
    dropped the target row from every rank.
    """
    comm = MPI.COMM_WORLD
    targets = sorted(before.keys() - {source})

    global_target_index = {
        target: np.concatenate(comm.allgather(into_array(before[target].index)))
        for target in targets
    }

    source_index = into_array(before[source].index)
    rows_to_keep = np.ones(len(source_index), dtype=bool)
    for target in targets:
        mapped = pairwise[(source, target)][source_index]
        rows_to_keep &= mapped >= 0
        rows_to_keep &= np.isin(mapped, global_target_index[target])

    expected_source = source_index[rows_to_keep]
    matched_source = into_array(matched[source].index)
    parallel_assert(
        np.array_equal(matched_source, expected_source),
        msg=f"source rows differ on rank {comm.Get_rank()}",
    )

    for target in targets:
        expected_target = pairwise[(source, target)][expected_source]
        matched_target = into_array(matched[target].index)
        parallel_assert(
            np.array_equal(matched_target, expected_target),
            msg=f"target rows for {target!r} differ on rank {comm.Get_rank()}",
        )

        # The rebuilt handler must actually read the global rows it now points
        # at, including rows that were never part of this rank's partition.
        with h5py.File(mapped_paths[target]) as file:
            expected_tags = file["data/fof_halo_tag"][:][expected_target]
        parallel_assert(
            np.array_equal(_column(matched[target], "fof_halo_tag"), expected_tags),
            msg=f"target data for {target!r} differs on rank {comm.Get_rank()}",
        )

    # Globally, the matched source rows must partition the full set of matchable
    # rows: every rank contributes a disjoint slice and nothing is lost. Target
    # rows may repeat globally, because the mapping is not injective.
    global_source_before = np.concatenate(comm.allgather(source_index))
    global_valid = np.ones(len(global_source_before), dtype=bool)
    for target in targets:
        mapped = pairwise[(source, target)][global_source_before]
        global_valid &= mapped >= 0
        global_valid &= np.isin(mapped, global_target_index[target])
    global_expected_source = np.sort(global_source_before[global_valid])

    global_matched_source = np.sort(np.concatenate(comm.allgather(matched_source)))
    parallel_assert(np.array_equal(global_matched_source, global_expected_source))

    for target in targets:
        global_matched_target = np.concatenate(
            comm.allgather(into_array(matched[target].index))
        )
        parallel_assert(
            np.array_equal(
                np.sort(global_matched_target),
                np.sort(pairwise[(source, target)][global_expected_source]),
            )
        )


def _column(dataset, name):
    return np.asarray(dataset.select(name).get_data(format="numpy"))


def _absolute_rows_from_ids(dataset, ids, identifier="fof_halo_tag"):
    comm = MPI.COMM_WORLD
    dataset_ids = np.concatenate(comm.allgather(_column(dataset, identifier)))
    dataset_index = np.concatenate(comm.allgather(into_array(dataset.index)))
    rows_by_id = dict(zip(dataset_ids.tolist(), dataset_index.tolist(), strict=True))
    return np.asarray([rows_by_id[value] for value in ids.tolist()], dtype=np.int64)


def _expected_source_driven_ids(
    original, expected_source, source, pairwise, identifier="fof_halo_tag"
):
    expected_ids = {source: _column(expected_source, identifier)}
    source_rows = _absolute_rows_from_ids(
        original[source], expected_ids[source], identifier
    )
    for target in sorted(original.keys() - {source}):
        target_rows = pairwise[(source, target)][source_rows]
        assert np.all(target_rows >= 0)

        comm = MPI.COMM_WORLD
        target_ids = np.concatenate(
            comm.allgather(_column(original[target], identifier))
        )
        target_index = np.concatenate(
            comm.allgather(into_array(original[target].index))
        )
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
        parallel_assert(
            np.array_equal(_column(actual[name], identifier), ids),
            msg=f"source-driven rows differ for {name!r}",
        )


def _assert_mapping_equal(before, after, identifier="fof_halo_tag"):
    """Assert two collections describe the same matches using stable row IDs."""
    comm = MPI.COMM_WORLD
    parallel_assert(set(before) == set(after))
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

        before_pairs = np.concatenate(
            comm.allgather(np.column_stack(before_ids)), axis=0
        )
        after_pairs = np.concatenate(comm.allgather(np.column_stack(after_ids)), axis=0)
        before_order = np.lexsort(before_pairs.T[::-1])
        after_order = np.lexsort(after_pairs.T[::-1])
        parallel_assert(
            np.array_equal(before_pairs[before_order], after_pairs[after_order]),
            msg=f"mapping differs for source {source!r}",
        )


def _assert_ordered_mapping_equal(
    before, after, source, identifier="fof_halo_tag", sort_by=None
):
    """Assert exact matched row order survives persistence."""
    after = after.match(source)
    if sort_by is not None:
        after = after.sort_by(*sort_by).take_range(0, len(after[source]))
    parallel_assert(tuple(before.keys()) == tuple(after.keys()))
    for name in before.keys():
        before_values = np.concatenate(
            MPI.COMM_WORLD.allgather(_column(before[name], identifier))
        )
        after_values = np.concatenate(
            MPI.COMM_WORLD.allgather(_column(after[name], identifier))
        )
        parallel_assert(
            np.array_equal(after_values, before_values),
            msg=f"ordered mapping differs for {name!r}",
        )


def _primary_stable_pairs(path, source, target):
    """Read one written primary slot as stable-ID pairs."""
    with h5py.File(path) as file:
        source_data = file[f"{source}/data"]
        target_data = file[f"{target}/data"]
        source_tags = source_data["fof_halo_tag"][:]
        target_tags = target_data["fof_halo_tag"][:]
        target_uuid = str(target_data.attrs["uuid"])
        primary = file[f"map/primary/{target_uuid}/index"][:]
    return {
        int(source_tag): int(target_tags[target_position])
        for source_tag, target_position in zip(source_tags, primary, strict=True)
        if target_position >= 0
    }


def _remote_primary_candidate(collection, pairwise, target):
    """Return a stable-ID primary pair whose endpoints begin on different ranks."""
    comm = MPI.COMM_WORLD
    source_rows = into_array(collection[REFERENCE].index)
    target_rows = into_array(collection[target].index)
    source_owners = {
        int(row): rank
        for rank, rows in enumerate(comm.allgather(source_rows))
        for row in rows
    }
    target_owners = {
        int(row): rank
        for rank, rows in enumerate(comm.allgather(target_rows))
        for row in rows
    }
    candidates = sorted(
        (row, int(pairwise[(REFERENCE, target)][row]))
        for row in source_owners
        if pairwise[(REFERENCE, target)][row] >= 0
        and int(pairwise[(REFERENCE, target)][row]) in target_owners
        and source_owners[row] != target_owners[int(pairwise[(REFERENCE, target)][row])]
    )
    assert candidates
    source_row, target_row = candidates[0]
    source_tags = np.concatenate(
        comm.allgather(_column(collection[REFERENCE], "fof_halo_tag"))
    )
    source_indices = np.concatenate(comm.allgather(source_rows))
    target_tags = np.concatenate(
        comm.allgather(_column(collection[target], "fof_halo_tag"))
    )
    target_indices = np.concatenate(comm.allgather(target_rows))
    return (
        int(dict(zip(source_indices, source_tags, strict=True))[source_row]),
        int(dict(zip(target_indices, target_tags, strict=True))[target_row]),
    )


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("source", MAPPED_SIMULATIONS)
def test_match_aligns_rows_for_each_source(source, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(source)

    parallel_assert(isinstance(matched, oc.SimulationCollection))
    parallel_assert(set(matched) == set(MAPPED_SIMULATIONS))
    _assert_matches_mapping_mpi(collection, matched, source, pairwise, mapped_paths)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("source", TARGET_SIMULATIONS)
def test_match_without_reference(source, mapped_paths, test_data):
    simulations = TARGET_SIMULATIONS
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(source)

    assert set(matched) == set(simulations)
    _assert_matches_mapping_mpi(collection, matched, source, pairwise, mapped_paths)


@pytest.mark.parallel(nprocs=4)
def test_match_honors_existing_row_selection(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).take_range(
        10_000, 100_000, mode="global"
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(REFERENCE)

    _assert_matches_mapping_mpi(collection, matched, REFERENCE, pairwise, mapped_paths)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_match_chain(pair, mapped_paths, test_data):
    simulations = MAPPED_SIMULATIONS
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    # Chaining narrows twice: the second match starts from the already-matched
    # collection, so its baseline is the first match's output, not `collection`.
    first = collection.match(pair[0])
    matched = first.match(pair[1])

    parallel_assert(set(matched) == set(simulations))
    _assert_matches_mapping_mpi(first, matched, pair[1], pairwise, mapped_paths)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize(
    "filtered_simulations",
    (None, (REFERENCE,), (SIMULATION_A,)),
    ids=("all", "source-only", "target-only"),
)
def test_match_honors_filters(filtered_simulations, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    collection = collection.filter(
        oc.col("fof_halo_mass") > 1e14, datasets=filtered_simulations
    )
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)

    matched = collection.match(REFERENCE)

    _assert_matches_mapping_mpi(collection, matched, REFERENCE, pairwise, mapped_paths)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("source", TARGET_SIMULATIONS)
def test_match_redistributes_cache(source, mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    new_id = {}
    for name, ds in collection.items():
        tag = ds.select("fof_halo_tag").get_data()
        new_id[name] = tag + 1

    collection = collection.with_new_columns(new_id=new_id).match(source)
    for name, ds in collection.items():
        data = ds.select("fof_halo_tag", "new_id").get_data("numpy")
        assert np.all(data["new_id"] == data["fof_halo_tag"] + 1)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("source", MAPPED_SIMULATIONS)
def test_matched_take_range_is_driven_by_active_source(source, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(source)

    expected_source = matched[source].take_range(7, 31)

    result = matched.take_range(7, 31)

    _assert_source_driven_result(original, expected_source, result, source, pairwise)


@pytest.mark.parallel(nprocs=4)
def test_matched_filter_is_evaluated_only_on_active_source(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    threshold = np.median(_column(matched[REFERENCE], "fof_halo_mass"))
    mask = oc.col("fof_halo_mass") > threshold
    expected_source = matched[REFERENCE].filter(mask)

    result = matched.filter(mask)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("invert", (False, True), ids=("ascending", "descending"))
def test_matched_sort_uses_active_source_order(invert, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = matched[REFERENCE].sort_by("fof_halo_mass", invert=invert)

    result = matched.sort_by("fof_halo_mass", invert=invert)

    masses = _column(result[REFERENCE], "fof_halo_mass")
    differences = np.diff(masses)
    parallel_assert(np.all(differences <= 0 if invert else differences >= 0))
    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("at", ("start", "end"))
def test_matched_take_is_driven_by_active_source(at, mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)
    expected_source = matched[REFERENCE].take(23, at=at)

    result = matched.take(23, at=at)

    _assert_source_driven_result(original, expected_source, result, REFERENCE, pairwise)


@pytest.mark.parallel(nprocs=4)
def test_matched_random_take_preserves_source_order(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE)

    result = matched.take(37, at="random")
    selected_source = result[REFERENCE]

    parallel_assert(len(selected_source) == 37)
    selected_ids = _column(selected_source, "fof_halo_tag")
    matched_ids = _column(matched[REFERENCE], "fof_halo_tag")
    positions = {value: position for position, value in enumerate(matched_ids.tolist())}
    selected_positions = np.asarray(
        [positions[value] for value in selected_ids.tolist()]
    )
    parallel_assert(np.all(np.diff(selected_positions) > 0))
    _assert_source_driven_result(original, selected_source, result, REFERENCE, pairwise)


@pytest.mark.parallel(nprocs=4)
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


@pytest.mark.parallel(nprocs=4)
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
        parallel_assert(
            np.array_equal(_column(dataset, "fof_halo_tag"), expected_ids[name])
        )


@pytest.mark.parallel(nprocs=4)
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


@pytest.mark.parallel(nprocs=4)
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


@pytest.mark.parallel(nprocs=4)
def test_matched_filter_rejects_non_source_datasets(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )
    mask = oc.col("fof_halo_mass") > 1e14

    with pytest.raises(ValueError, match="active source"):
        collection.filter(mask, datasets=SIMULATION_A)
    with pytest.raises(ValueError, match="active source"):
        collection.filter(mask, datasets=(REFERENCE, SIMULATION_A))


@pytest.mark.parallel(nprocs=4)
def test_matched_evaluate_rebuilds_targets_before_evaluation(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    matched = original.match(REFERENCE).take_range(7, 31)
    expected_ids = _expected_source_driven_ids(
        original, matched[REFERENCE], REFERENCE, pairwise
    )

    def evaluated_tag(fof_halo_tag):
        return fof_halo_tag

    result = matched.evaluate(
        evaluated_tag, vectorize=True, insert=False, format="numpy"
    )

    for name, dataset in matched.items():
        parallel_assert(
            np.array_equal(_column(dataset, "fof_halo_tag"), expected_ids[name])
        )
        parallel_assert(
            np.array_equal(result[name]["evaluated_tag"], expected_ids[name])
        )


@pytest.mark.parallel(nprocs=4)
def test_clear_match_rebuilds_pending_targets(mapped_paths, test_data):
    original = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    result = original.match(REFERENCE).take_range(7, 31)
    expected_source = result[REFERENCE]

    cleared = result.clear_match()

    _assert_source_driven_result(
        original, expected_source, cleared, REFERENCE, pairwise
    )
    parallel_assert(
        {len(dataset) for dataset in cleared.values()} == {len(expected_source)}
    )


@pytest.mark.parallel(nprocs=4)
def test_mapped_collection_context_manager_exits_cleanly(mapped_paths, test_data):
    with (
        _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
        .match(REFERENCE)
        .take_range(7, 31) as matched
    ):
        parallel_assert({len(dataset) for dataset in matched.values()} == {24})


@pytest.mark.parallel(nprocs=4)
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


@pytest.mark.parallel(nprocs=4)
def test_mapping_file_alone_raises(test_data):
    with pytest.raises(ValueError, match="Cannot open a dataset mapping on its own"):
        oc.open(test_data.snapshot.halo_mapping)


@pytest.mark.parallel(nprocs=4)
def test_open_multiple_mapping_files_raises(mapped_paths, test_data, per_test_dir):
    comm = MPI.COMM_WORLD
    second_mapping = per_test_dir / "second_mapping.hdf5"
    if comm.Get_rank() == 0:
        shutil.copy(test_data.snapshot.halo_mapping, second_mapping)
    comm.Barrier()

    with pytest.raises(ValueError, match="multiple dataset mapping files"):
        oc.open(
            mapped_paths[REFERENCE],
            mapped_paths[SIMULATION_A],
            test_data.snapshot.halo_mapping,
            second_mapping,
        )


@pytest.mark.parallel(nprocs=4)
def test_primary_mapping_length_must_match_reference(
    mapped_paths, test_data, per_test_dir
):
    comm = MPI.COMM_WORLD
    mapping = per_test_dir / "invalid_length_mapping.hdf5"
    if comm.Get_rank() == 0:
        shutil.copy(test_data.snapshot.halo_mapping, mapping)
        with h5py.File(mapping, "a") as file:
            primary = file["map/primary"]
            target = next(iter(primary))
            slot = primary[target]
            values = slot["index"][:-1]
            del slot["index"]
            slot.create_dataset("index", data=values)
    comm.Barrier()

    with pytest.raises(ValueError, match="reference dataset length"):
        oc.open(
            mapped_paths[REFERENCE],
            mapped_paths[SIMULATION_A],
            mapping,
        )


@pytest.mark.parallel(nprocs=4)
def test_match_requires_mapping(test_data):
    collection = oc.open(test_data.snapshot.multi_simulation)

    with pytest.raises(ValueError, match="does not contain matching information"):
        collection.match("scidac1")


@pytest.mark.parallel(nprocs=4)
def test_match_requires_known_source(mapped_paths, test_data):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)

    with pytest.raises(ValueError, match="does not have a simulation named unknown"):
        collection.match("unknown")


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize(
    ("simulations", "filtered"),
    (
        (MAPPED_SIMULATIONS, False),
        (MAPPED_SIMULATIONS, True),
        (TARGET_SIMULATIONS, True),
        ((REFERENCE, SIMULATION_A), False),
    ),
    ids=("unfiltered", "filtered", "without-reference", "one-target"),
)
def test_mapping_write_round_trip(
    simulations, filtered, mapped_paths, test_data, per_test_dir
):
    collection = _open_mapped(
        mapped_paths, test_data.snapshot.halo_mapping, simulations
    )
    if filtered:
        collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
    path = per_test_dir / "mapping.hdf5"

    oc.write(path, collection)
    written = oc.open(path)

    _assert_mapping_equal(collection, written)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize("filtered", (False, True), ids=("unfiltered", "filtered"))
def test_mapping_write_preserves_remote_primary_entry(
    filtered, mapped_paths, test_data, per_test_dir
):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    pairwise = _expected_pairwise_maps(test_data.snapshot.halo_mapping, mapped_paths)
    source_tag, target_tag = _remote_primary_candidate(
        collection, pairwise, SIMULATION_A
    )
    if filtered:
        collection = collection.filter(
            oc.col("fof_halo_tag") == source_tag, datasets=REFERENCE
        )
    path = per_test_dir / "mapping.hdf5"

    oc.write(path, collection)
    MPI.COMM_WORLD.Barrier()
    pairs = (
        _primary_stable_pairs(path, REFERENCE, SIMULATION_A)
        if MPI.COMM_WORLD.Get_rank() == 0
        else None
    )
    pairs = MPI.COMM_WORLD.bcast(pairs, root=0)

    parallel_assert(pairs.get(source_tag) == target_tag)


@pytest.mark.parallel(nprocs=4)
def test_mapping_write_with_empty_target(mapped_paths, test_data, per_test_dir):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping)
    collection = collection.filter(oc.col("fof_halo_mass") < 0, datasets=SIMULATION_C)
    parallel_assert(len(collection[SIMULATION_C]) == 0)
    path = per_test_dir / "mapping.hdf5"

    oc.write(path, collection)
    written = oc.open(path)

    _assert_mapping_equal(collection, written)


@pytest.mark.parallel(nprocs=4)
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
    ),
)
def test_active_match_write_preserves_order(
    transform, sort_after_read, mapped_paths, test_data, per_test_dir
):
    collection = _open_mapped(mapped_paths, test_data.snapshot.halo_mapping).match(
        REFERENCE
    )
    collection = transform(collection)
    path = per_test_dir / "mapping.hdf5"

    oc.write(path, collection)
    written = oc.open(path)

    _assert_ordered_mapping_equal(
        collection, written, REFERENCE, sort_by=sort_after_read
    )
