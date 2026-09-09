from __future__ import annotations

import pytest
from opencosmo.io.discover import discover_all, discover_file
from opencosmo.mpi import get_comm_world
from pytest_mpi.parallel_assert import parallel_assert


def _layout_signature(layout) -> tuple:
    """Compute a hashable signature of a FileLayout for comparison."""
    return (
        str(layout.path),
        tuple((g.path, g.header_path) for g in layout.groups),
        layout.error,
    )


@pytest.mark.parallel(nprocs=4)
def test_discover_all_byte_identical_across_ranks(test_data):
    """Test that discover_all returns byte-identical layouts on all ranks."""
    comm = get_comm_world()
    if comm is None:
        pytest.skip("MPI not available")

    # Use lightcone files: step_600 and step_601, each with 5 files
    all_paths = sorted(
        test_data.lightcone.step(600).all + test_data.lightcone.step(601).all
    )

    # All ranks discover together
    layouts = discover_all(all_paths, comm=comm)

    # Gather all layouts on each rank
    all_layouts_per_rank = comm.allgather(layouts)

    # All ranks should have identical layouts (by signature)
    rank_0_sigs = [_layout_signature(fl) for fl in all_layouts_per_rank[0]]

    for rank_idx, rank_layouts in enumerate(all_layouts_per_rank):
        rank_sigs = [_layout_signature(fl) for fl in rank_layouts]
        parallel_assert(
            rank_sigs == rank_0_sigs,
            f"Rank {rank_idx} layouts differ from rank 0",
        )


@pytest.mark.parallel(nprocs=4)
def test_discover_all_round_robin_coverage(test_data):
    """Test that round-robin coverage finds all files."""
    comm = get_comm_world()
    if comm is None:
        pytest.skip("MPI not available")

    all_paths = sorted(
        test_data.lightcone.step(600).all + test_data.lightcone.step(601).all
    )
    expected_count = len(all_paths)

    # Discover all
    layouts = discover_all(all_paths, comm=comm)

    # Should have discovered all files
    discovered_paths = {str(fl.path) for fl in layouts}
    expected_paths = {str(p) for p in all_paths}

    parallel_assert(
        discovered_paths == expected_paths,
        f"Discovered paths differ from expected. Discovered: {discovered_paths}, Expected: {expected_paths}",
    )

    parallel_assert(
        len(layouts) == expected_count,
        f"Expected {expected_count} files, got {len(layouts)}",
    )


@pytest.mark.parallel(nprocs=4)
def test_discover_all_fewer_files_than_ranks(test_data):
    """Test that discovery works when there are fewer files than ranks."""
    comm = get_comm_world()
    if comm is None:
        pytest.skip("MPI not available")

    # Use only 2 files across (potentially) 4 ranks
    paths = [
        test_data.lightcone.step(600).halo_properties,
        test_data.lightcone.step(601).halo_properties,
    ]

    layouts = discover_all(paths, comm=comm)

    # All ranks should get both layouts
    parallel_assert(
        len(layouts) == 2,
        f"Expected 2 layouts, got {len(layouts)}",
    )

    # Gather results to verify they're identical across ranks
    all_layouts_per_rank = comm.allgather(layouts)
    rank_0_sigs = [_layout_signature(fl) for fl in all_layouts_per_rank[0]]

    for rank_idx, rank_layouts in enumerate(all_layouts_per_rank):
        rank_sigs = [_layout_signature(fl) for fl in rank_layouts]
        parallel_assert(
            rank_sigs == rank_0_sigs,
            f"Rank {rank_idx} layouts differ from rank 0",
        )


@pytest.mark.parallel(nprocs=4)
def test_discover_all_single_file_collective_free(test_data):
    """Test that single-file discovery doesn't need allgather (remains collective-free)."""
    comm = get_comm_world()
    if comm is None:
        pytest.skip("MPI not available")

    # Single file should not trigger allgather (every rank should independently discover it)
    single_file = [test_data.lightcone.step(600).halo_properties]

    layouts = discover_all(single_file, comm=comm)

    # Should have one layout
    parallel_assert(
        len(layouts) == 1,
        f"Expected 1 layout, got {len(layouts)}",
    )

    # All ranks should have the same layout
    all_layouts_per_rank = comm.allgather(layouts)
    rank_0_sigs = [_layout_signature(fl) for fl in all_layouts_per_rank[0]]

    for rank_idx, rank_layouts in enumerate(all_layouts_per_rank):
        rank_sigs = [_layout_signature(fl) for fl in rank_layouts]
        parallel_assert(
            rank_sigs == rank_0_sigs,
            f"Rank {rank_idx} layouts differ from rank 0",
        )


@pytest.mark.parallel(nprocs=4)
def test_discover_all_sorted_determinism(test_data):
    """Test that results are sorted by path (deterministic order)."""
    comm = get_comm_world()
    if comm is None:
        pytest.skip("MPI not available")

    all_paths = sorted(
        test_data.lightcone.step(600).all + test_data.lightcone.step(601).all
    )

    layouts = discover_all(all_paths, comm=comm)

    # Verify paths are sorted
    path_strs = [str(fl.path) for fl in layouts]
    sorted_path_strs = sorted(path_strs)

    parallel_assert(
        path_strs == sorted_path_strs,
        f"Paths not sorted. Got: {path_strs}",
    )


@pytest.mark.parallel(nprocs=4)
def test_dataset_uuids_agree_across_ranks(test_data):
    """Test that each rank computes identical synthesized dataset identities."""
    comm = get_comm_world()
    if comm is None:
        pytest.skip("MPI not available")

    all_paths = sorted(
        test_data.lightcone.step(600).all + test_data.lightcone.step(601).all
    )
    layouts = discover_all(all_paths, comm=comm)
    gathered_layout_uuids = comm.allgather(
        [group.uuid for layout in layouts for group in layout.groups]
    )

    for rank, uuids in enumerate(gathered_layout_uuids):
        parallel_assert(uuids == gathered_layout_uuids[0], f"Rank {rank} UUIDs differ")

    # discover_all allgathers layouts, so this direct call is the load-bearing check:
    # it makes each rank compute an identity for a file owned by another rank.
    direct_uuids = [group.uuid for group in discover_file(all_paths[1]).groups]
    gathered_direct_uuids = comm.allgather(direct_uuids)

    for rank, uuids in enumerate(gathered_direct_uuids):
        parallel_assert(
            uuids == gathered_direct_uuids[0], f"Rank {rank} direct UUIDs differ"
        )
