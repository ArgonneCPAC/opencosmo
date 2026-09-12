from __future__ import annotations

from typing import TYPE_CHECKING

import hdf5plugin

if TYPE_CHECKING:
    import h5py

    from opencosmo.io.schema import Schema


def allocate(group: h5py.File | h5py.Group, schema: Schema):
    for column_name, column_writer in schema.columns.items():
        group.require_dataset(
            column_name,
            column_writer.shape,
            column_writer.dtype,
            compression=hdf5plugin.Blosc(
                cname="blosclz", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
            ),
        )
    for child_name, child_schema in schema.children.items():
        child_group = group.require_group(child_name)
        allocate(child_group, child_schema)


def write_columns(group: h5py.File | h5py.Group, schema: Schema):
    for column_path, column_writer in schema.columns.items():
        group[column_path][:] = column_writer.data
        group[column_path].attrs.update(column_writer.attrs)
    for child_name, child_schema in schema.children.items():
        write_columns(group[child_name], child_schema)


def write_metadata(group: h5py.File | h5py.Group | h5py.Dataset, schema: Schema):
    if schema.attributes:
        group.attrs.update(schema.attributes)

    for child_name, child_schema in schema.children.items():
        write_metadata(group[child_name], child_schema)
