import numpy as np

from opencosmo.index import coalesce_chunks


def test_coalesce_merges_adjacent():
    starts, sizes = coalesce_chunks(np.array([0, 1, 2]), np.array([1, 1, 1]))
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(sizes, [3])


def test_coalesce_keeps_non_adjacent_split():
    starts, sizes = coalesce_chunks(np.array([0, 5]), np.array([1, 1]))
    np.testing.assert_array_equal(starts, [0, 5])
    np.testing.assert_array_equal(sizes, [1, 1])


def test_coalesce_mixed_runs():
    # [0,1,2] contiguous -> one chunk; gap; [10,11] contiguous -> one chunk
    starts, sizes = coalesce_chunks(
        np.array([0, 1, 2, 10, 11]), np.array([1, 1, 1, 1, 1])
    )
    np.testing.assert_array_equal(starts, [0, 10])
    np.testing.assert_array_equal(sizes, [3, 2])


def test_coalesce_varied_sizes():
    # 0->3->8->10 is fully contiguous, so all three fuse into one chunk.
    starts, sizes = coalesce_chunks(np.array([0, 3, 8]), np.array([3, 5, 2]))
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(sizes, [10])

    # A gap after the first run keeps it separate.
    starts, sizes = coalesce_chunks(np.array([0, 3, 20]), np.array([3, 5, 2]))
    np.testing.assert_array_equal(starts, [0, 20])
    np.testing.assert_array_equal(sizes, [8, 2])


def test_coalesce_empty():
    starts, sizes = coalesce_chunks(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    assert len(starts) == 0
    assert len(sizes) == 0


def test_coalesce_single():
    starts, sizes = coalesce_chunks(np.array([7]), np.array([4]))
    np.testing.assert_array_equal(starts, [7])
    np.testing.assert_array_equal(sizes, [4])


def test_coalesce_preserves_array_order():
    # Locally contiguous but not globally ascending: merging must not reorder rows.
    starts, sizes = coalesce_chunks(np.array([10, 11, 0, 1]), np.array([1, 1, 1, 1]))
    np.testing.assert_array_equal(starts, [10, 0])
    np.testing.assert_array_equal(sizes, [2, 2])


def test_coalesce_output_dtype_is_int64():
    starts, sizes = coalesce_chunks(
        np.array([0, 1], dtype=np.int32), np.array([1, 1], dtype=np.int32)
    )
    assert starts.dtype == np.int64
    assert sizes.dtype == np.int64
