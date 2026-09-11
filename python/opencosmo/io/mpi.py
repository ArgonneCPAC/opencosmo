from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import h5py
import hdf5plugin
import numpy as np
from h5py import h5fd, h5p, h5s

from opencosmo.io.schema import FileEntry, Schema, make_schema
from opencosmo.io.verify import schema_data_length, verify_structure
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter
from opencosmo.mpi import MPI, get_all_keys, get_comm_world, get_subcom, sum_scatter

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.io.schema import Schema


"""
When working with MPI, datasets are chunked across ranks. Here we combine the schemas
from several ranks into a single schema that can be allocated by rank 0. Each 
rank will then write its own data to the specific section of the file 
it is responsible for.

When writing data with MPI, there are basically 3 things we have to verify in order to 
determine if everything is valid.

1. Is the top-level file structure the same for all ranks (e.g. lightcone? dataset?).
2. Do all columns that are going to be written to by two or more ranks have the same data type and compatible shapes?
3. Is metadata consistent across ranks? If not, are there rules in place to combine/update the fields?

If all three of these checks pass, it is guaranteed we can create a schema that can accomodate the data being written.

File schemas are simply collections of columns and metadata. Columns contain:

1. A reference to the underying data that will be written (either an h5py dataset or a numpy array)
2. An index which tells us which elements in 1 we are going to actually write
3. Possibly an output index, which tells us where in the output we are going to actually write to.
3. Possibly a function to update those values before writing. For example, a spatial index should be summed across ranks rather than concatenated.

In order to avoid MPI deadlocks, we always sort columns in alphabetical order before performing operations on the file.
"""


def collective_write(dset, start, data=None):
    """Collective write where any rank may contribute nothing.
    All ranks in the file's communicator must call this."""
    dxpl = h5p.create(h5p.DATASET_XFER)
    dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)

    fspace = dset.id.get_space()
    n = 0 if data is None else len(data)

    if n:
        arr = np.ascontiguousarray(data, dtype=dset.dtype)
        mspace = h5s.create_simple(arr.shape)
        fspace.select_hyperslab((start,) + (0,) * (dset.ndim - 1), arr.shape)
    else:
        arr = np.zeros((1,) + dset.shape[1:], dtype=dset.dtype)
        mspace = h5s.create_simple(arr.shape)
        mspace.select_none()
        fspace.select_none()

    dset.id.write(mspace, fspace, arr, dxpl=dxpl)


class CombineState(Enum):
    VALID = 1
    ZERO_LENGTH = 2
    INVALID = 3


def write_parallel(file: Path, file_schema: Schema):
    """
    Main entry point for writing data in parallel. Proceeds
    in three steps:

    1. Verify the file schemas are valid separately
    2. Recursively verify the schemas can be made consistent across ranks
    3. Recursively write the data.
    """
    comm = get_comm_world()
    if comm is None:
        raise ValueError("Got a null comm!")
    paths = set(comm.allgather(file))
    if len(paths) != 1:
        raise ValueError("Different ranks recieved a different path to output to!")

    if schema_data_length(file_schema) == 0:
        results = comm.allgather(CombineState.ZERO_LENGTH)
    else:
        try:
            verify_structure(
                file_schema, allow_unresolved_maps=True
            )  # Tier 1: structural correctness
            # Tier 2: does this rank actually contribute any rows?
            results = comm.allgather(CombineState.VALID)
        except ValueError:
            results = comm.allgather(CombineState.INVALID)
            raise
    if any(rs == CombineState.INVALID for rs in results):
        raise ValueError("One or more ranks recieved invalid schemas!")
    has_data = [i for i, state in enumerate(results) if state == CombineState.VALID]
    if len(has_data) == 0:
        raise ValueError("No ranks have any data to write!")

    group = comm.Get_group()
    new_group = group.Incl(has_data)
    new_comm = comm.Create(new_group)
    group.Free()
    if new_comm == MPI.COMM_NULL:
        return cleanup_mpi(comm, new_comm, new_group)

    file_schema = sync_schemas(file_schema, new_comm)
    __verify_structure_collective(file_schema, new_comm)
    offsets = __get_all_offsets(file_schema, new_comm, "")

    if new_comm.Get_rank() == 0:
        with h5py.File(file, "w") as f:
            __allocate(file_schema, f, new_comm)
    else:
        __allocate(file_schema, None, new_comm)
    try:
        with h5py.File(file, "a", driver="mpio", comm=new_comm) as f:
            __write_parallel(file_schema, f, offsets, new_comm)

    except ValueError:  # parallell hdf5 not available
        schema = __replace_writers_with_updates(file_schema, new_comm)
        __write_serial(schema, file, offsets, new_comm)
    cleanup_mpi(comm, new_comm, new_group)


def cleanup_mpi(comm_world: MPI.Comm, comm_write: MPI.Comm, group_write: MPI.Group):
    comm_world.Barrier()
    if comm_write != MPI.COMM_NULL:
        comm_write.Free()
    group_write.Free()


def sync_schemas(schema: Schema, comm: MPI.Comm) -> Schema:
    from opencosmo.collection.simulation.io import resort_simulation_collection_mpi

    schema = sync_uuids(schema, comm, {})
    if schema.type == FileEntry.SIMULATION_COLLECTION:
        schema = resort_simulation_collection_mpi(schema, comm)
    return verify_schemas(schema, comm)


def __verify_structure_collective(schema: Schema, comm: MPI.Comm) -> None:
    """Validate final schemas without allowing one rank to skip a collective."""
    try:
        verify_structure(schema)
        message = None
    except ValueError as error:
        message = str(error)
    messages = comm.allgather(message)
    invalid_message = next((value for value in messages if value is not None), None)
    if invalid_message is not None:
        raise ValueError(invalid_message)


def sync_uuids(schema: Schema, comm: MPI.Comm, uuid_map: dict[str, str]) -> Schema:
    """Synchronize dataset identities and update mapping schemas.

    A rank may have an empty schema for a child that is present on another rank.  Keep
    those ranks in the collectives here instead of creating a sub-communicator: they
    still need the UUID mappings when a map is present locally.
    """

    def collect(current: Schema) -> None:
        if current.name == "data":
            local_uuid = current.attributes.get("main_uuid")
            all_uuids = comm.allgather(local_uuid)
            canonical_uuid = next(
                (value for value in all_uuids if value is not None), None
            )
            if canonical_uuid is not None:
                for old_uuid in all_uuids:
                    if old_uuid is not None:
                        uuid_map[str(old_uuid)] = str(canonical_uuid)

        for cname in get_all_keys(current.children, comm):
            child_schema = current.children.get(cname)
            collect(child_schema or make_schema(cname, FileEntry.EMPTY))

    def rewrite(current: Schema) -> Schema:
        rewritten_children = {}
        for cname in get_all_keys(current.children, comm):
            child_schema = current.children.get(cname)
            rewritten_child = rewrite(
                child_schema or make_schema(cname, FileEntry.EMPTY)
            )
            if child_schema is not None:
                rewritten_children[cname] = rewritten_child

        attributes = current.attributes
        if current.name == "data":
            new_uuid = uuid_map.get(str(current.attributes.get("main_uuid")))
            if new_uuid is not None:
                attributes = current.attributes | {
                    "uuid": new_uuid,
                    "main_uuid": new_uuid,
                }

        if current.name == "map":
            if "reference" in current.attributes:
                reference = str(current.attributes["reference"])
                attributes = current.attributes | {
                    "reference": uuid_map.get(reference, reference)
                }

            for parent_name in ("primary", "auxiliary"):
                parent = rewritten_children.get(parent_name)
                if parent is None:
                    continue
                renamed_children = {}
                for child_name, child_schema in parent.children.items():
                    if parent_name == "primary":
                        new_name = uuid_map.get(child_name, child_name)
                    else:
                        endpoints = child_name.split("__")
                        new_name = "__".join(
                            uuid_map.get(endpoint, endpoint) for endpoint in endpoints
                        )
                    renamed_children[new_name] = child_schema._replace(name=new_name)
                rewritten_children[parent_name] = parent._replace(
                    children=renamed_children
                )

        return current._replace(children=rewritten_children, attributes=attributes)

    collect(schema)
    schema = rewrite(schema)

    return schema


def verify_schemas(schema: Schema, comm: MPI.Comm) -> Schema:
    """
    By this stage, we know that all the ranks that are participating have a valid
    file schema. We now need to verify that they can be made consistent across ranks.

    This is done recursively. For each schema, we verify its columns and its attributes.
    Then we proceed to its children. Ranks that do not have data for a given child
    are simply excluded from the verification process.

    """

    file_types = set(comm.allgather(schema.type))
    if len(file_types) > 1:
        raise ValueError(
            "Unable to combine file schemas, as they do not have the same type!"
        )

    verify_columns(schema.columns, comm)
    new_attributes = sync_attributes(schema.attributes, schema.name, comm)
    schema = schema._replace(attributes=new_attributes)

    all_child_names = get_all_keys(schema.children if schema is not None else {}, comm)

    for child_name in all_child_names:
        has_child = comm.allgather(child_name in schema.children)
        if all(has_child):
            new_comm = comm
            new_group = None
        else:
            ranks_to_include = [i for i in range(len(has_child)) if has_child[i]]
            group = comm.Get_group()
            new_group = group.Incl(ranks_to_include)
            new_comm = comm.Create(new_group)
            group.Free()
        if child_name in schema.children:
            new_schema = verify_schemas(schema.children[child_name], new_comm)
            schema.children[child_name] = new_schema
        # Free only the sub-communicator/group we created; never the parent comm.
        if new_group is not None:
            if new_comm != MPI.COMM_NULL:
                new_comm.Free()
            new_group.Free()
    return schema


def verify_columns(columns: dict[str, ColumnWriter], comm: MPI.Comm):
    """
    Verify a group of columns in a schema. Note, the single-threaded verification
    process already ensures that all columns in the data group have the same
    length in any given rank, which guarantees this will also be the case when
    we combine columns across ranks. All we have to check here is:

    1. Are the shapes consistent
    2. Are the data types consistent
    3. Are the column attributes consistent

    """
    all_column_names = get_all_keys(columns, comm)
    for colname in all_column_names:
        if colname not in columns:
            # Simply ignore a rank that doesn't have this column
            colmeta = comm.allgather(None)
        else:
            column = columns[colname]
            data_to_send = (
                column.combine_strategy,
                column.shape,
                column.dtype,
                column.attrs,
            )
            colmeta = comm.allgather(data_to_send)

        combine_strategies = set([cm[0] for cm in colmeta if cm is not None])
        if len(combine_strategies) > 1:
            raise ValueError("Combine strategy must be the same accross ranks!")

        shapes = set([cm[1][1:] for cm in colmeta if cm is not None])
        if len(shapes) > 1:
            raise ValueError(
                f"Column {colname} did not have consistent shapes across ranks!"
            )
        dtype_kinds = set([cm[2].kind for cm in colmeta if cm is not None])
        if len(dtype_kinds) > 1:
            # We allow type promotion, though there are very few cases where
            # It would be necessary.
            raise ValueError(
                f"Column {colname} did not have consistent dtypes across ranks!"
            )
        attrs = [cm[3] for cm in colmeta if cm is not None]
        if any(attr_set != attrs[0] for attr_set in attrs[1:]):
            raise ValueError("Metadata was not consistent across ranks!")


def sync_attributes(metadata: dict[str, Any], group_name: str, comm: MPI.Comm):
    all_metadata = comm.allgather(metadata)
    for md in all_metadata[1:]:
        if md != all_metadata[0]:
            raise ValueError(
                f"Not all ranks recieved the same metadata in {group_name}"
            )
    return metadata


def __write_parallel(
    schema: Schema,
    group: h5py.File | h5py.Group,
    offsets: dict,
    comm: MPI.Comm | None,
):
    """
    Used with both the parallel and serial version, though the later passes through
    __write_serial.
    """
    __write_columns(schema, group, offsets, comm)
    all_child_names = get_all_keys(schema.children, comm)
    for cn in all_child_names:
        child_schema = schema.children.get(cn, make_schema(cn, FileEntry.EMPTY))
        new_group = group[cn]
        __write_parallel(child_schema, new_group, offsets, comm)


def __write_serial(schema: Schema, file_path: Path, offsets: dict, comm: MPI.Comm):
    """
    We do NOT have parallel hdf5, so we have to write one rank at a time.
    """
    for i in range(comm.Get_size()):
        if i == comm.Get_rank():
            with h5py.File(file_path, "a") as f:
                __write_parallel(schema, f, offsets, None)

        comm.Barrier()


def __replace_writers_with_updates(schema: Schema, comm: MPI.Comm):
    """
    For columns that require updates, compute the update and replace. The most common form
    of updated is "start/size" indexes, since "start" must be updated consistently across
    ranks.

    """
    colnames = get_all_keys(schema.columns, comm)
    for cn in colnames:
        colwriter = schema.columns.get(cn)

        participating = comm.allgather(colwriter is not None)
        if all(participating):
            new_comm = comm
            new_group = None
        else:
            participating_ranks = [
                i for i in range(len(participating)) if participating[i]
            ]
            group = comm.Get_group()
            new_group = group.Incl(participating_ranks)
            new_comm = comm.Create(new_group)
            group.Free()

        try:
            if colwriter is None:
                continue

            has_update = new_comm.allgather(
                colwriter is not None and colwriter.has_transformation
            )
            if any(has_update) and not all(has_update):
                raise ValueError("Update was not consistent across ranks!")
            elif not any(has_update):
                continue
            assert colwriter is not None
            data = colwriter.get_data(new_comm)
            new_writer = ColumnWriter.from_numpy_array(
                data, colwriter.combine_strategy, colwriter.attrs
            )
            schema.columns[cn] = new_writer
        finally:
            # Free only the sub-communicator/group we created; never the parent.
            if new_group is not None:
                if new_comm != MPI.COMM_NULL:
                    new_comm.Free()
                new_group.Free()

    child_names = get_all_keys(schema.children, comm)
    for cn in child_names:
        child_schema = schema.children.get(cn, make_schema(cn, FileEntry.EMPTY))
        new_child_schema = __replace_writers_with_updates(child_schema, comm)
        schema.children[cn] = new_child_schema

    return schema


def __get_all_offsets(schema: Schema, comm: MPI.Comm, name: str):
    """
    Get the rank-wise offset for every column.
    """
    output = {}
    all_column_names = get_all_keys(schema.columns, comm)
    for colname in all_column_names:
        writer = schema.columns.get(colname)
        offset = get_column_offset(writer, comm)
        key = "/".join([name, colname])
        output[key] = offset

    all_child_names = get_all_keys(schema.children, comm)
    for cn in all_child_names:
        key = "/".join([name, cn])
        child_schema = schema.children.get(cn, make_schema(cn, FileEntry.EMPTY))
        child_offsets = __get_all_offsets(child_schema, comm, key)
        output.update(child_offsets)
    return output


def __allocate(schema: Schema, group: Optional[h5py.File | h5py.Group], comm: MPI.Comm):
    """
    Allocate the file.
    """
    all_column_names = get_all_keys(schema.columns, comm)
    for column_name in all_column_names:
        column_writer = schema.columns.get(column_name)
        __allocate_column(column_name, column_writer, group, comm)

    __write_metadata(schema, group, comm)

    all_child_names = get_all_keys(schema.children, comm)
    for cn in all_child_names:
        if group is not None:
            new_group = group.require_group(cn)
        else:
            new_group = None

        child_schema = schema.children.get(cn, make_schema(cn, FileEntry.EMPTY))

        __allocate(child_schema, new_group, comm)


def get_column_allocation_metadata(column: Optional[ColumnWriter], comm: MPI.Comm):
    """
    Determine how to allocate the column. The most important thing this does is
    determine the overal shape. Keep in mind we have already done verification
    at this point to make sure everything is consistent.
    """
    strategy = None if column is None else column.combine_strategy
    strategies = list(filter(lambda strat: strat is not None, comm.allgather(strategy)))
    strategy = strategies[0]

    meta = None if column is None else (column.shape, column.dtype, column.attrs)
    all_meta = list(filter(lambda cm: cm is not None, comm.allgather(meta)))
    reference_meta = all_meta[0]

    if strategy == ColumnCombineStrategy.CONCAT:
        total_length = sum(m[0][0] for m in all_meta)
    else:
        total_length = reference_meta[0][0]

    # Note, we have already verified that all data types have the same
    # kind, so we simply need to pick the highest-precision one.
    all_dtypes = [m[1] for m in all_meta]
    all_dtypes.sort(key=lambda dt: dt.itemsize, reverse=True)

    shape = (total_length,) + reference_meta[0][1:]
    return shape, all_dtypes[0], all_meta[0][2]


def get_column_offset(column: Optional[ColumnWriter], comm: MPI.Comm):
    """
    Determine the offset for a given column on this rank.
    """
    if column is None or column.combine_strategy == ColumnCombineStrategy.SUM:
        length = 0
    else:
        length = len(column)
    all_lengths = comm.allgather(length)
    offsets = np.insert(np.cumsum(all_lengths), 0, 0)
    return offsets[comm.Get_rank()]


def __write_metadata(
    schema: Schema, group: Optional[h5py.File | h5py.Group], comm: MPI.Comm
):
    """
    Write metadata-only groups.
    """
    if schema.type == FileEntry.EMPTY:
        attrs = comm.allgather(None)
    else:
        attrs = comm.allgather(schema.attributes)

    attrs_to_write = list(filter(lambda at: at is not None, attrs))
    if not attrs_to_write:
        return
    if group is not None:
        group.attrs.update(attrs_to_write[0])


def __allocate_column(
    name: str,
    column_writer: Optional[ColumnWriter],
    group: Optional[h5py.Group | h5py.File],
    comm: MPI.Comm,
) -> Optional[h5py.Dataset]:
    """
    Allocate a single column
    """

    shape, dtype, attrs = get_column_allocation_metadata(column_writer, comm)
    if group is not None:
        ds = group.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            compression=hdf5plugin.Blosc(
                cname="blosclz", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        ds.attrs.update(attrs)
        return ds
    return None


def __write_columns(
    schema: Schema,
    group: h5py.File | h5py.Group,
    offsets: dict,
    comm: MPI.Comm | None,
):
    all_column_names = get_all_keys(schema.columns, comm)
    for cn in all_column_names:
        writer = schema.columns.get(cn)
        ds = group[cn]
        offset = offsets[ds.name]
        assert ds is not None
        __write_column(writer, ds, offset, comm)


def __write_column_serial(writer, offset, ds):
    data = writer.get_data()
    match writer.combine_strategy:
        case ColumnCombineStrategy.CONCAT:
            ds[offset : offset + len(data)] = data
        case ColumnCombineStrategy.SUM:
            data += ds[:]
            ds[:] = data


def __write_column(
    writer: Optional[ColumnWriter],
    ds: h5py.Dataset,
    offset: int,
    write_comm: MPI.Comm | None,
):
    if write_comm is None:
        return __write_column_serial(writer, offset, ds)

    strategy = None if writer is None else writer.combine_strategy
    strategies = list(
        filter(lambda strat: strat is not None, write_comm.allgather(strategy))
    )
    strategy = strategies[0]
    data_comm, new_group = get_subcom(
        write_comm.allgather(writer is not None), write_comm
    )

    data: np.ndarray | None
    match strategy:
        case ColumnCombineStrategy.CONCAT:
            if writer is not None:
                data = writer.get_data(data_comm)
            else:
                shape = (0,) + ds.shape[1:]
                data = np.empty(shape, dtype=ds.dtype)

        case ColumnCombineStrategy.SUM:
            if writer is None:
                data = np.zeros(ds.shape, ds.dtype)
            else:
                data = writer.get_data(data_comm)
            data, offset = sum_scatter(data, write_comm)

        case _:
            data = None

    if data is None or len(data) == 0:
        shape = (0,) + ds.shape[1:]
        data = np.empty(shape, dtype=ds.dtype)

    collective_write(ds, offset, data)

    # Free only the sub-communicator/group we created; never the parent comm.
    # Ranks excluded from the sub-communicator get COMM_NULL (which must not be
    # freed) but still own a group handle that must be released.
    if data_comm != MPI.COMM_NULL:
        data_comm.Free()

    new_group.Free()
    write_comm.Barrier()

    ds.file.flush()
