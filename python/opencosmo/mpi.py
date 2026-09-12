from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Generator, Optional

import numpy as np

from opencosmo.index import coalesce_chunks, into_array, sort

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None  # type: ignore


def has_mpi() -> bool:
    return get_comm_world() is not None


@cache
def get_comm_world() -> Optional["MPI.Comm"]:
    if MPI is None or MPI.COMM_WORLD.Get_size() == 1:
        return None
    return MPI.COMM_WORLD.Dup()


def get_mpi():
    return MPI


def parallel_assert(condition: bool, comm: MPI.Comm | None = None):
    comm = comm or get_comm_world()
    assert comm is not None
    is_bool = comm.allgather(isinstance(condition, (np.bool, bool)))
    if not all(is_bool):
        raise ValueError("Expected a boolean condition on all ranks")
    passes = comm.allgather(condition)
    failed = [i for i in range(len(passes)) if not passes[i]]
    if failed:
        raise ValueError(f"Parallel assertion failed on ranks {failed}")


def parallel_assert_is_simple_index(value, comm: MPI.Comm | None = None):
    parallel_assert(isinstance(value, np.ndarray), comm)
    parallel_assert(value.ndim == 1, comm)
    parallel_assert(value.dtype == np.int64, comm)


def parallel_assert_is_chunked_index(value: tuple, comm: MPI.Comm | None = None):
    has_shape = (
        len(value) == 2
        and isinstance(value[0], np.ndarray)
        and isinstance(value[1], np.ndarray)
    )

    parallel_assert(has_shape, comm)
    parallel_assert(value[0].ndim == 1 and value[1].ndim == 1, comm)
    parallel_assert(value[0].dtype == np.int64 and value[1].dtype == np.int64, comm)


def parallel_assert_same_dtype(value: np.ndarray, comm: MPI.Comm | None = None):
    parallel_assert(isinstance(value, np.ndarray), comm)
    assert comm is not None
    all_dtypes = comm.allgather(value.dtype)
    parallel_assert(len(set(all_dtypes)) == 1, comm)


def parallel_assert_compatible_shapes(value: np.ndarray, comm: MPI.Comm | None = None):
    parallel_assert(isinstance(value, np.ndarray), comm)
    assert comm is not None
    shapes = [s[1:] for s in comm.allgather(value.shape)]
    parallel_assert(len(set(shapes)) == 1, comm)


def parallel_assert_can_stack(value: np.ndarray, comm: MPI.Comm | None = None):
    parallel_assert_same_dtype(value, comm)
    parallel_assert_compatible_shapes(value, comm)


def get_subcom(include: list[bool], comm: MPI.Comm):
    group = comm.Get_group()
    new_group = group.Incl(np.flatnonzero(include).tolist())
    new_comm = comm.Create(new_group)
    group.free()
    return new_comm, new_group


def gather_index(
    index: np.ndarray | tuple[np.ndarray, np.ndarray],
    comm: MPI.Comm,
    all: bool = False,
    sorted=False,
):
    if isinstance(index, tuple):
        parallel_assert_is_chunked_index(index, comm)
    else:
        parallel_assert_is_simple_index(index, comm)

    rank_is_chunked = comm.allgather(isinstance(index, tuple))

    if np.all(rank_is_chunked):
        starts = gather_data(index[0], comm, all=all)
        sizes = gather_data(index[1], comm, all=all)
        if not all and comm.Get_rank() != 0:
            return None

        if sorted:
            start, sizes = sort((starts, sizes))

        return coalesce_chunks(starts, sizes)

    result = gather_data(into_array(index), comm, all=all)
    if not all and comm.Get_rank() != 0:
        return None

    return sort(result) if sorted else result


def sum_scatter(data: np.ndarray, comm: MPI.Comm):
    parallel_assert_can_stack(data, comm)
    row_counts = np.asarray(comm.allgather(len(data)), dtype=np.int64)
    parallel_assert(len(np.unique(row_counts)) == 1, comm)
    parallel_assert(data.ndim == 1, comm)

    dtype = comm.bcast(data.dtype if data is not None else None)
    assert dtype is not None

    counts_per_rank = len(data) // comm.Get_size()
    counts_per = np.full(comm.Get_size(), counts_per_rank)

    counts_per[0] += len(data) % comm.Get_size()

    recvbuf = np.empty(counts_per[comm.Get_rank()], dtype)

    comm.Reduce_scatter(
        data,
        recvbuf,
        counts_per,  # type: ignore
    )
    offsets = np.cumsum(counts_per)
    offset = 0 if comm.Get_rank() == 0 else offsets[comm.Get_rank() - 1]

    return (recvbuf, offset)


def gather_data(data: np.ndarray, comm: MPI.Comm, all: bool = False):
    from mpi4py.util import dtlib

    parallel_assert_can_stack(data, comm)

    row_counts = np.asarray(comm.allgather(len(data)), dtype=np.int64)
    row_size = int(np.prod(data.shape[1:], dtype=np.int64))
    counts = row_counts * row_size
    if comm.Get_rank() == 0 or all:
        displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]
        recvbuf = np.empty(int(np.sum(counts)), dtype=data.dtype)
        recv_counts = counts
    else:
        displacements = None
        recvbuf = None
        recv_counts = None

    mpi_dtype = dtlib.from_numpy_dtype(data.dtype)
    if all:
        comm.Allgatherv(
            sendbuf=data.reshape(-1),
            recvbuf=[recvbuf, recv_counts, displacements, mpi_dtype],
        )
    else:
        comm.Gatherv(
            sendbuf=data.reshape(-1),
            recvbuf=[recvbuf, recv_counts, displacements, mpi_dtype],
            root=0,
        )
    if recvbuf is None:
        return None
    return recvbuf.reshape((int(np.sum(row_counts)), *data.shape[1:]))


def scatter_index(index: np.ndarray | None, length: int, comm: MPI.Comm):
    counts = comm.allgather(length)
    index_length = comm.bcast(len(index) if index is not None else None)
    parallel_assert(np.sum(counts) == index_length, comm)

    if comm.Get_rank() == 0:
        displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]
        send_counts = counts

    else:
        displacements = None
        send_counts = None

    recvbuf = np.empty(length, np.int64)
    comm.Scatterv(
        [index, send_counts, displacements, MPI.INT64_T],
        recvbuf,
        root=0,
    )
    return recvbuf
    # 4. Perform the variable-length gather operation
    # Note the uppercase 'G' which indicates a buffer/array optimization wrapper


def scatter_data(data: np.ndarray | None, length: int, comm: MPI.Comm):
    from mpi4py.util import dtlib

    row_counts = np.asarray(comm.allgather(length), dtype=np.int64)
    data_length = comm.bcast(len(data) if data is not None else None)
    parallel_assert(np.sum(row_counts) == data_length, comm)
    dtype = comm.bcast(data.dtype if data is not None else None)
    trailing_shape = comm.bcast(data.shape[1:] if data is not None else None)
    assert dtype is not None
    assert trailing_shape is not None
    row_size = int(np.prod(trailing_shape, dtype=np.int64))
    counts = row_counts * row_size

    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

    recvbuf = np.empty(length * row_size, dtype)
    comm.Scatterv(
        [
            None if data is None else data.reshape(-1),
            counts,
            displacements,
            dtlib.from_numpy_dtype(dtype),
        ],
        recvbuf,
        root=0,
    )
    return recvbuf.reshape((length, *trailing_shape))


def redistribute_data(data: np.ndarray, target_rank: np.ndarray, comm: MPI.Comm):
    from mpi4py.util import dtlib

    parallel_assert_is_simple_index(target_rank, comm)
    parallel_assert(len(target_rank) == len(data), comm)
    parallel_assert_can_stack(data, comm)

    rank_distribution = [np.where(target_rank == i)[0] for i in range(comm.Get_size())]
    row_lengths = np.asarray([len(tr) for tr in rank_distribution], dtype=np.int64)
    parallel_assert(sum(row_lengths) == len(data), comm)
    recv_row_lengths = np.empty(comm.Get_size(), dtype=np.int64)
    comm.Alltoall(row_lengths, recv_row_lengths)

    row_size = int(np.prod(data.shape[1:], dtype=np.int64))
    lengths = row_lengths * row_size
    recv_lengths = recv_row_lengths * row_size

    send_displs = np.zeros(comm.Get_size(), dtype="i")
    send_displs[1:] = np.cumsum(lengths)[:-1]

    recv_displs = np.zeros(comm.Get_size(), dtype="i")
    recv_displs[1:] = np.cumsum(recv_lengths)[:-1]

    # --- Build the actual send buffer ---
    # Total data this rank sends = sum(send_counts)
    sendbuf = np.concat(
        [data[rank_distribution[i]] for i in range(comm.Get_size())]
    ).reshape(-1)

    # --- Allocate the receive buffer ---
    total_recv = int(np.sum(recv_lengths))
    recvbuf = np.empty(total_recv, dtype=data.dtype)

    # --- The actual call ---
    dtype = dtlib.from_numpy_dtype(data.dtype)
    comm.Alltoallv(
        [sendbuf, lengths, send_displs, dtype],
        [recvbuf, recv_lengths, recv_displs, dtype],
    )
    return recvbuf.reshape((int(np.sum(recv_row_lengths)), *data.shape[1:]))


def get_all_keys[T: SupportsRichComparison](
    data: dict[T, Any], comm: Optional[MPI.Comm]
) -> list[T]:
    """
    Return all keys in the dictionary across all ranks, sorted
    alphabetically. When defining the file structure, we have to iterate
    through the schemas in the same order across all ranks, including
    when one rank doesn't have a given child.
    """
    data_names: set[T] = set(data.keys())
    if comm is None:
        return sorted(list(data_names))

    all_data_names: set[T] = data_names.union(*comm.allgather(data_names))
    return sorted(all_data_names)


def get_all_entries[T: (SupportsRichComparison), U](
    data: dict[T, U], comm: Optional[MPI.Comm]
) -> Generator[tuple[T, U | None]]:
    for key in get_all_keys(data, comm):
        yield key, data.get(key)
