from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Optional

import astropy.units as u

from opencosmo.column.column import RawColumn
from opencosmo.io.schema import (
    FileEntry,
    combine_with_cached_schema,
    make_schema,
)
from opencosmo.io.writer import ColumnCombineStrategy, ColumnWriter, NumpySource

if TYPE_CHECKING:
    from uuid import UUID

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region
    from opencosmo.spatial.tree import Tree


def get_derived_column_names(
    producers: list[ConstructedColumn], columns: set[str]
) -> set[str]:
    all_derived: set[str] = reduce(
        lambda acc, col: acc.union(
            col.produces if not isinstance(col, RawColumn) else set()
        ),
        producers,
        set(),
    )
    return all_derived.intersection(columns)


def build_derived_writers(
    producers: list[ConstructedColumn],
    derived_data: dict,
    data_schema: Schema,
    cached_data_schema: Schema,
) -> None:
    """Add ColumnWriter entries to data_schema for each non-raw, non-cached producer."""
    for producer in producers:
        if (
            isinstance(producer, RawColumn)
            or producer.produces.issubset(cached_data_schema.columns.keys())
            or not producer.produces.issubset(derived_data.keys())
        ):
            continue
        coldata = {name: derived_data[name] for name in producer.produces}
        units = {
            name: str(cd.unit) if isinstance(cd, u.Quantity) else ""
            for name, cd in coldata.items()
        }
        coldata = {
            name: cd.value if isinstance(cd, u.Quantity) else cd
            for name, cd in coldata.items()
        }
        for name, cd in coldata.items():
            attrs = {"unit": units[name], "description": producer.description or "None"}
            source = NumpySource(cd)
            writer = ColumnWriter([source], ColumnCombineStrategy.CONCAT, attrs=attrs)
            data_schema.columns[name] = writer


def make_dataset_schema(
    producers: list[ConstructedColumn],
    raw_data_handler: DataHandler,
    cache: DataCache,
    columns_to_uuid: dict[str, UUID],
    header: OpenCosmoHeader,
    tree: Tree | None,
    region: Region,
    raw_index: DataIndex,
    derived_data: dict,
    dataset_uuid: UUID,
    name: Optional[str] = None,
) -> Schema:
    columns = set(columns_to_uuid.keys())
    # header = header.with_region(region)
    raw_columns = columns.intersection(raw_data_handler.columns)
    data_schema = raw_data_handler.make_schema(raw_columns)

    cached_data_schema = cache.make_schema(columns_to_uuid)

    build_derived_writers(producers, derived_data, data_schema, cached_data_schema)

    attributes = {}
    if (load_conditions := raw_data_handler.load_conditions) is not None:
        attributes["load/if"] = load_conditions

    data_schema = combine_with_cached_schema(
        data_schema,
        cached_data_schema,
    )

    new_data_attributes = data_schema.attributes.get("", {}) | {
        "uuid": str(dataset_uuid),
        "main_uuid": str(dataset_uuid),
    }
    new_attributes = data_schema.attributes
    new_attributes[""] = new_data_attributes
    data_schema = data_schema._replace(attributes=new_attributes)

    children = {"data": data_schema}
    if name is None:
        name = ""

    if tree is not None:
        tree = tree.apply_index(raw_index)
        tree_schema = tree.make_schema()
        children["index"] = tree_schema
        header = header.with_region(tree.get_region())

    header_schema = header.dump()
    children["header"] = header_schema

    return make_schema(
        name, FileEntry.DATASET, children=children, attributes=attributes
    )
