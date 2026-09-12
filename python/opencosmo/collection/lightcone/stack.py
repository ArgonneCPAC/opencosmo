from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Iterable, Optional

import healpy as hp
import numpy as np

from opencosmo import dataset as ds
from opencosmo.io.schema import FileEntry, add_metadata, make_schema
from opencosmo.mpi import get_all_keys, get_comm_world
from opencosmo.spatial.check import find_coordinates_2d

if TYPE_CHECKING:
    from opencosmo.io.schema import Schema
    from opencosmo.mpi import MPI


def update_order(data: np.ndarray, comm: Optional[MPI.Comm], order: np.ndarray):
    if comm is not None:
        return update_global_order_mpi(data, comm, order)

    return data[order]


def _global_inverse_order_mpi(order: np.ndarray, comm) -> np.ndarray:
    """
    Build the global inverse permutation across all MPI ranks.

    Each rank's `order` (possibly referencing rows on other ranks) is combined
    into a single global permutation of length N. The inverse maps global input
    position -> global output position in the written file.
    """
    sizes = comm.allgather(len(order))
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)
    N = int(starts[-1])

    # global_order[i] is the global input position that lands at global output i
    global_order = order + starts[comm.Get_rank()]
    all_global_orders = np.concatenate(comm.allgather(global_order))

    global_inv = np.empty(N, dtype=np.int64)
    global_inv[all_global_orders] = np.arange(N)
    return global_inv


def update_top_host_idx(
    data: np.ndarray,
    comm: Optional[MPI.Comm],
    order: np.ndarray,
    slice_sizes: list[int],
):
    result = data.copy()

    # Add per-slice global offsets so local row references become global
    offset = 0
    if comm is not None:
        offsets = np.cumsum(comm.allgather(len(data)))
        rank = comm.Get_rank()
        if rank > 0:
            offset = offsets[rank - 1]

    pos = 0
    for size in slice_sizes:
        segment = result[pos : pos + size]
        segment[segment >= 0] += offset
        offset += size
        pos += size

    # Reorder row positions (same as all other columns)
    if comm is not None:
        result = update_global_order_mpi(result, comm, order)
        # After a cross-rank shuffle, result[valid] may contain global indices
        # (0..N-1). np.argsort(order) only covers this rank's local portion, so
        # we need the full global inverse permutation instead.
        inverse_order = _global_inverse_order_mpi(order, comm)
    else:
        result = result[order]
        inverse_order = np.argsort(order)

    # Remap stored row references through the inverse permutation
    output = np.full_like(result, -1)
    valid = result >= 0
    output[valid] = inverse_order[result[valid]]
    return output


def update_global_order_mpi(data, comm, order):
    needs_global_reordering = comm.allgather(np.any((order < 0) | (order > len(order))))
    if not np.any(needs_global_reordering):
        return data[order]

    ends = np.cumsum(comm.allgather(len(order)))
    starts = np.insert(ends, 0, 0)
    global_order = order + starts[comm.Get_rank()]
    all_data = comm.allgather(data)
    return np.concat(all_data)[global_order]


def sync_metadata(dataset_schemas: list[Schema], skip: list[str] = []):
    metadata = [schema.attributes for schema in dataset_schemas]
    identity = {"uuid", "main_uuid"}
    to_compare = []
    for md in metadata:
        to_compare.append({k: v for k, v in md.items() if k not in identity})
    if not all(am == to_compare[0] for am in to_compare[1:]):
        raise ValueError("Datasets don't have the same metadata!")

    child_names = set(frozenset(schema.children.keys()) for schema in dataset_schemas)
    if len(child_names) > 1:
        raise ValueError("Datasets don't have the same metadata!")
    for child in list(child_names)[0]:
        if child in skip:
            continue
        schemas = [sc.children[child] for sc in dataset_schemas]
        sync_metadata(schemas)


def sync_headers(datasets: list[ds.Dataset], redshift_range):
    if not datasets and (comm := get_comm_world()) is not None:
        comm.allgather(0)
        comm.allgather(-1)
        comm.allgather((1000, -1))
        return

    steps = (
        dataset.header.file.step
        for dataset in datasets
        if dataset.header.file.step is not None
    )
    redshifts = (
        dataset.header.file.redshift
        for dataset in datasets
        if dataset.header.file.redshift is not None
    )
    step = max(steps)
    redshift = max(redshifts)

    if (comm := get_comm_world()) is not None:
        step = np.max(comm.allgather(step))
        redshift = max(comm.allgather(redshift))
        z_ranges = comm.allgather(redshift_range)
        z_min = min(zr[0] for zr in z_ranges)
        z_max = max(zr[1] for zr in z_ranges)
        redshift_range = (z_min, z_max)

    # lightcones are identified by their upper redshift slice
    header_schema = datasets[0].header.dump()
    header_schema = add_metadata(
        "file", header_schema, {"redshift": redshift, "step": step}
    )
    header_schema = add_metadata(
        "lightcone", header_schema, {"z_range": redshift_range}
    )
    return header_schema


def stack_lightcone_datasets_in_schema(
    datasets: dict[str, list[ds.Dataset]],
    path: str,
    redshift_range: Optional[tuple[float, float]],
    no_stack: bool = False,
):
    name = path.split("/")[-1]
    n_datasets = sum(len(lst) for lst in datasets.values())
    if n_datasets == 1 and get_comm_world() is None:
        dataset_list = next(iter(datasets.values()))
        dataset_name = next(iter(datasets.keys()))

        schema = dataset_list[0].make_schema(path=path)
        header = sync_headers(dataset_list, redshift_range)
        schema.children["header"] = header
        return {dataset_name: schema}

    schema_children = {}
    ds_groups = get_all_keys(datasets, get_comm_world())
    for ds_group in ds_groups:
        schema_name = ds_group if len(datasets) > 1 else name
        ds_list = datasets.get(ds_group, [])
        ds_list = list(filter(lambda ds: len(ds) > 0, ds_list))
        if len(ds_list) == 0:
            if no_stack:
                continue
            get_stacked_lightcone_order([], -1)
            sync_headers(ds_list, None)
            continue
        schemas = [ds.make_schema(path=path) for ds in ds_list]
        index_names = list(schemas[0].children["index"].children.keys())
        index_names.sort()
        max_level = int(index_names[-1][-1])

        assert all(isinstance(dataset, ds.Dataset) for dataset in ds_list)
        if no_stack:
            assert len(schemas) == 1
            schema_children[schema_name] = schemas[0]
            continue
        new_data_group = stack_data_groups(
            [schema.children["data"] for schema in schemas]
        )
        order = get_stacked_lightcone_order(ds_list, max_level)
        updater = partial(update_order, order=order)
        slice_sizes = [len(dataset) for dataset in ds_list]
        top_host_idx_updater = partial(
            update_top_host_idx, order=order, slice_sizes=slice_sizes
        )

        for col_name, column in new_data_group.columns.items():
            if col_name == "top_host_idx":
                column.set_transformation(top_host_idx_updater)
            else:
                column.set_transformation(updater)

        new_index_group = stack_index_groups(
            [schema.children["index"] for schema in schemas]
        )
        header_schema = sync_headers(ds_list, redshift_range)
        additional_metadata = sync_metadata(schemas, skip=["header"])

        children = schemas[0].children | {
            "data": new_data_group,
            "index": new_index_group,
            "header": header_schema,
        }
        if all("data_linked" in c.children for c in schemas):
            children["data_linked"] = stack_data_groups(
                [schema.children["data_linked"] for schema in schemas]
            )

        assert schema_name is not None
        schema_children[schema_name] = make_schema(
            schema_name,
            FileEntry.LIGHTCONE,
            children=children,
            attributes=additional_metadata,
        )

    return schema_children


def stack_index_groups(schemas: list[Schema]):
    base_schema = schemas[0]
    new_children = {}
    for index_level in base_schema.children.keys():
        all_level_schemas = [schema.children[index_level] for schema in schemas]
        new_children[index_level] = stack_data_groups(all_level_schemas)
    return make_schema("index", FileEntry.COLUMNS, new_children)


def stack_data_groups(schemas: list[Schema]):
    if len(schemas) == 1:
        return schemas[0]
    base_schema = schemas[0]
    new_writers = {}
    for name, column_writer in base_schema.columns.items():
        other_writers = [schema.columns[name] for schema in schemas[1:]]
        new_writer = column_writer.combine(other_writers)
        new_writers[name] = new_writer

    new_schema = make_schema(
        base_schema.name,
        base_schema.type,
        children=base_schema.children,
        columns=new_writers,
        attributes=base_schema.attributes,
    )
    return new_schema


def get_order_mpi(pixels, comm):
    pixel_order = np.argsort(pixels)
    if len(pixels) > 0:
        pixel_ranges = comm.allgather((pixels[pixel_order[0]], pixels[pixel_order[-1]]))

    else:
        pixel_ranges = comm.allgather(None)

    pixel_ranges = [pr for pr in pixel_ranges if pr is not None]
    for i in range(len(pixel_ranges) - 1):
        if pixel_ranges[i][1] > pixel_ranges[i + 1][0]:
            break
    else:
        return pixel_order

    all_pixels = np.concat(comm.allgather(pixels))
    new_order = np.argsort(all_pixels)
    bounds = np.cumsum(comm.allgather(len(pixels)))
    bounds = np.insert(bounds, 0, 0)
    rank = comm.Get_rank()
    return new_order[bounds[rank] : bounds[rank + 1]] - bounds[rank]


def get_stacked_lightcone_order(datasets: Iterable[ds.Dataset], max_index_depth: int):
    datasets = list(datasets)
    nside = 2**max_index_depth
    coordinates = [find_coordinates_2d(dataset._state) for dataset in datasets]
    coordinates = list(filter(lambda coord_list: len(coord_list) > 0, coordinates))

    if datasets:
        pixels = np.concatenate(
            [
                hp.ang2pix(
                    nside, coords.ra.value, coords.dec.value, lonlat=True, nest=True
                )
                for coords in coordinates
            ]
        )

    else:
        pixels = np.array([])

    if (comm := get_comm_world()) is not None:
        return get_order_mpi(pixels, comm)

    return np.argsort(pixels)
