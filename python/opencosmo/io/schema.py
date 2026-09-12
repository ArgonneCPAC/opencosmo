from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

import numpy as np

from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter, Hdf5Source

if TYPE_CHECKING:
    from uuid import UUID

    from opencosmo.index import SimpleIndex
    from opencosmo.mpi import MPI


class FileEntry(Enum):
    DATASET = "dataset"
    MULTI_DATASET = "multi_dataset"
    STRUCTURE_COLLECTION = "structure_collection"
    SIMULATION_COLLECTION = "simulation_collection"
    LIGHTCONE = "lightcone"
    LIGHCONE_MAP = "lightcone_map"
    HEALPIX_MAP = "healpix_map"
    COLUMNS = "columns"
    METADATA = "metadata"
    EMPTY = "empty"


class Schema(NamedTuple):
    name: str
    type: FileEntry
    children: dict[str, Schema]
    columns: dict[str, ColumnWriter]
    attributes: dict[str, str | int | UUID]
    updater: Callable[[Schema, MPI.Comm | None], Schema] | None = None


def dataset_schema_length(schema: Schema) -> Optional[int]:
    if schema.type != FileEntry.DATASET:
        return None

    column = next(iter(schema.children["data"].columns.values()))
    return column.shape[0]


def get_dataset_schema_index(schema: Schema):
    if schema.type != FileEntry.DATASET:
        return None
    data_schema = schema.children["data"]
    index = None
    for name, column in data_schema.columns.items():
        column_sources = column.sources
        if len(column_sources) == 1 and isinstance(column_sources[0], Hdf5Source):
            column_index = column_sources[0].index
        else:
            continue
        if index is None:
            index = column_index
        elif len(column_index) != len(index):
            raise ValueError("Inconsistent indices found!")

    return index


def reorder_dataset_schema(schema: Schema, index: SimpleIndex):
    assert "data" in schema.children
    data_columns = schema.children["data"].columns
    assert len(data_columns) > 0

    new_columns = {}
    for name, column in data_columns.items():
        new_column = column.reorder(index)
        new_columns[name] = new_column

    new_data_schema = schema.children["data"]._replace(columns=new_columns)

    return schema._replace(children=schema.children | {"data": new_data_schema})


def empty_schema(name: str, type_: FileEntry) -> Schema:
    return Schema(name, type_, {}, {}, {})


def make_schema(
    name: str,
    type_: FileEntry,
    children: Optional[dict] = None,
    columns: Optional[dict] = None,
    attributes: Optional[dict] = None,
    updater: Callable | None = None,
):
    if children is None:
        children = {}
    if columns is None:
        columns = {}
    if attributes is None:
        attributes = {}
    return Schema(name, type_, children, columns, attributes, updater=updater)


def add_metadata(
    path: str,
    schema: Schema,
    metadata: dict[str, Any],
    overrides: dict[str, ColumnCombineStrategy] | None = None,
):
    overrides = overrides or {}
    writers = {}
    if not path:
        metadata_names = list(metadata.keys())
        for name in metadata_names:
            if isinstance(metadata[name], list):
                column_data = np.array(metadata.pop(name))
                writer = ColumnWriter.from_numpy_array(
                    column_data, overrides.get(name, ColumnCombineStrategy.CONCAT)
                )
                writers[name] = writer

        new_schema = schema._replace(
            attributes=schema.attributes | metadata, columns=schema.columns | writers
        )
        return new_schema
    sep_index = path.find("/")
    local_path = path[:sep_index] if sep_index != -1 else path
    remainder = path[sep_index + 1 :] if sep_index != -1 else ""

    child_schema = schema.children.get(local_path)
    if child_schema is None:
        child_schema = empty_schema(local_path, FileEntry.METADATA)

    child_schema = add_metadata(remainder, child_schema, metadata, overrides)
    return schema._replace(children=schema.children | {local_path: child_schema})


def combine_with_cached_schema(raw_data_schema, cached_schema):
    if raw_data_schema is None or raw_data_schema.type == FileEntry.EMPTY:
        return cached_schema
    elif cached_schema is None or cached_schema.type == FileEntry.EMPTY:
        return raw_data_schema

    for cname, column in cached_schema.columns.items():
        if cname not in raw_data_schema.columns:
            raw_data_schema.columns[cname] = cached_schema.columns[cname]

    return raw_data_schema
