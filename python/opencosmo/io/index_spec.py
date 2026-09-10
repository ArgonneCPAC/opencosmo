from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from opencosmo.index.build import empty, from_range

if TYPE_CHECKING:
    from mpi4py import MPI

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.spatial.tree import Tree


class ResolvedIndex(NamedTuple):
    """Result of resolving an index spec: the row index for this rank (None means
    'no restriction — read the whole dataset on this rank') plus the possibly
    partition-updated spatial region to thread back into dataset state."""

    index: Optional[DataIndex]  # type: ignore[assignment]  # shadows tuple.index
    region: Any  # a spatial Region; opaque here, threaded through unchanged by full/empty_ref


if TYPE_CHECKING:
    # An index spec answers one question — "what rows does this rank read for this
    # node?" — and feeds back the possibly partition-updated region. It carries no
    # state, so each spec is just a function of this shape, not a class.
    IndexSpec = Callable[
        [
            Optional[MPI.Comm],
            OpenCosmoHeader,
            DatasetTarget,
            Optional[Tree],
            int,
            object,
        ],
        ResolvedIndex,
    ]


def spatial(
    comm: Optional[MPI.Comm],
    header: OpenCosmoHeader,
    target: DatasetTarget,
    tree: Optional[Tree],
    ds_length: int,
    sim_region: object,
) -> ResolvedIndex:
    """Source in spatial mode: MPI-partition this rank's rows, feeding back the
    partition's region. Holds the partition() call, the KeyError even-chunk
    fallback, and lightcone-region expansion."""
    # Serial opens (no comm) behave as `full` — no row restriction.
    if comm is None:
        return ResolvedIndex(None, sim_region)

    # Lazy imports to avoid a circular import with iopen.
    from opencosmo.dataset.mpi import partition
    from opencosmo.io.iopen import _expand_lightcone_region

    try:
        ds_group = target.group
        part = partition(comm, header, ds_group["index"], ds_group["data"], tree)
        if part is None:
            index = empty()
        else:
            index = part.idx  # type: ignore[assignment]
            sim_region = part.region if part.region is not None else sim_region
        if header.file.is_lightcone:
            sim_region = _expand_lightcone_region(sim_region, tree)
        return ResolvedIndex(index, sim_region)
    except KeyError:
        n_ranks = comm.Get_size()
        n_per = ds_length // n_ranks
        chunk_boundaries = [i * n_per for i in range(n_ranks + 1)]
        chunk_boundaries[-1] = ds_length
        rank = comm.Get_rank()
        return ResolvedIndex(
            from_range(chunk_boundaries[rank], chunk_boundaries[rank + 1]),
            sim_region,
        )


def full(
    comm: Optional[MPI.Comm],
    header: OpenCosmoHeader,
    target: DatasetTarget,
    tree: Optional[Tree],
    ds_length: int,
    sim_region: object,
) -> ResolvedIndex:
    """Linked children and redshift-split nodes: no index restriction — the whole
    dataset is read on this rank (matches the old bypass_mpi=True path)."""
    return ResolvedIndex(None, sim_region)


def empty_ref(
    comm: Optional[MPI.Comm],
    header: OpenCosmoHeader,
    target: DatasetTarget,
    tree: Optional[Tree],
    ds_length: int,
    sim_region: object,
) -> ResolvedIndex:
    """Empty reference rank: read nothing (matches the old index_override=empty())."""
    return ResolvedIndex(empty(), sim_region)


def index_spec_for(
    index_kind: str,
    is_empty_ref: bool,
    *,
    is_source: bool,
) -> IndexSpec:
    """Resolve the appropriate index spec for a node's role and distribution mode.

    Semantic map (from retire-open-kwargs-indexspec.md):
    - Source in spatial mode → spatial (MPI-partitioned, with region feedback)
    - Linked children, redshift-split nodes → full (whole dataset on this rank)
    - Empty reference ranks → empty_ref (read nothing)
    """
    if is_empty_ref:
        return empty_ref
    if index_kind == "spatial" and is_source:
        return spatial
    return full
