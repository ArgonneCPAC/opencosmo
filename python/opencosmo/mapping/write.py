from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
from opencosmo.collection.structure.handler import LINK_ALIASES
from opencosmo.io.schema import FileEntry, get_dataset_schema_index
from opencosmo.io.writer import ColumnWriter

from opencosmo.index import into_array

if TYPE_CHECKING:
    from opencosmo.io.schema import Schema


def __make_output_position_lookup(raw_ids: np.ndarray) -> dict[int, int]:
    """Map unique raw row IDs to their positions in an output dataset."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    unique_ids, counts = np.unique(raw_ids, return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    if len(duplicate_ids) > 0:
        raise ValueError(
            "Cannot lower mapping with duplicate output raw row IDs: "
            f"{duplicate_ids.tolist()}"
        )
    return {int(raw_id): position for position, raw_id in enumerate(raw_ids)}


def __lower_primary_writer(
    writer: ColumnWriter, target_positions: Mapping[int, int]
) -> ColumnWriter:
    raw_targets = writer.data
    output_targets = np.full(len(raw_targets), -1, dtype=np.int64)
    for position, raw_target in enumerate(raw_targets):
        if raw_target >= 0:
            output_targets[position] = target_positions.get(int(raw_target), -1)
    return ColumnWriter.from_numpy_array(
        output_targets, writer.combine_strategy, writer.attrs
    )


def __lower_auxiliary_writers(
    source_writer: ColumnWriter,
    target_writer: ColumnWriter,
    source_positions: Mapping[int, int],
    target_positions: Mapping[int, int],
) -> tuple[ColumnWriter, ColumnWriter]:
    raw_source = source_writer.data
    raw_target = target_writer.data
    if len(raw_source) != len(raw_target):
        raise ValueError("Auxiliary mapping source and target have different lengths")

    source = np.fromiter(
        (source_positions.get(int(raw_id), -1) for raw_id in raw_source),
        dtype=np.int64,
        count=len(raw_source),
    )
    target = np.fromiter(
        (target_positions.get(int(raw_id), -1) for raw_id in raw_target),
        dtype=np.int64,
        count=len(raw_target),
    )
    retained = (source >= 0) & (target >= 0)
    source = source[retained]
    target = target[retained]
    reorder = np.lexsort((target, source))
    return (
        ColumnWriter.from_numpy_array(
            source[reorder], source_writer.combine_strategy, source_writer.attrs
        ),
        ColumnWriter.from_numpy_array(
            target[reorder], target_writer.combine_strategy, target_writer.attrs
        ),
    )


def __dataset_positions(schema: Schema) -> dict[str, dict[int, int]]:
    positions_by_uuid: dict[str, dict[int, int]] = {}
    for child_name, child in schema.children.items():
        if child_name == "map" or child.type != FileEntry.DATASET:
            continue
        raw_index = get_dataset_schema_index(child)
        if raw_index is None:
            raise ValueError(f"Dataset '{child_name}' has no output raw row index")
        uuid = child.children["data"].attributes["main_uuid"]
        positions_by_uuid[str(uuid)] = __make_output_position_lookup(
            into_array(raw_index)
        )
    return positions_by_uuid


def __lower_simulation_maps(schema: Schema) -> Schema:
    if "map" not in schema.children:
        return schema

    positions_by_uuid = __dataset_positions(schema)
    map_schema = schema.children["map"]
    if map_schema.map_coordinates is None:
        return schema
    reference = str(map_schema.attributes["reference"])
    if reference not in positions_by_uuid:
        raise ValueError(
            "Simulation mapping reference dataset is not in the output schema"
        )

    primary_children = {}
    for target_uuid, child in map_schema.children["primary"].children.items():
        try:
            target_positions = positions_by_uuid[target_uuid]
        except KeyError as error:
            raise ValueError(
                f"Primary mapping target dataset {target_uuid} is not in the output schema"
            ) from error
        primary_children[target_uuid] = child._replace(
            columns={
                "index": __lower_primary_writer(
                    child.columns["index"], target_positions
                )
            }
        )

    auxiliary_children = {}
    for pair_name, child in map_schema.children["auxiliary"].children.items():
        source_uuid, target_uuid = pair_name.split("__")
        try:
            source_positions = positions_by_uuid[source_uuid]
            target_positions = positions_by_uuid[target_uuid]
        except KeyError as error:
            raise ValueError(
                f"Auxiliary mapping endpoint for {pair_name} is not in the output schema"
            ) from error
        source, target = __lower_auxiliary_writers(
            child.columns["source"],
            child.columns["target"],
            source_positions,
            target_positions,
        )
        if len(source) > 0:
            auxiliary_children[pair_name] = child._replace(
                columns={"source": source, "target": target}
            )

    lowered_map = map_schema._replace(
        children=map_schema.children
        | {
            "primary": map_schema.children["primary"]._replace(
                children=primary_children
            ),
            "auxiliary": map_schema.children["auxiliary"]._replace(
                children=auxiliary_children
            ),
        },
        map_coordinates=None,
    )
    return schema._replace(children=schema.children | {"map": lowered_map})


def __step_name(name: str | int) -> str:
    name = str(name)
    return (
        name.split("_", 1)[1]
        if "_" in name and name.split("_", 1)[0].isdigit()
        else name
    )


def __target_schema(source: Schema, root: Schema, prefix: str) -> Schema:
    alias = LINK_ALIASES[prefix]
    source_name = __step_name(source.name)
    candidates: list[Schema] = []

    def collect(schema: Schema) -> None:
        if schema.type == FileEntry.DATASET and __step_name(schema.name) == alias:
            candidates.append(schema)
        for child in schema.children.values():
            collect(child)

    collect(root)
    source_schema_name = str(source.name)
    step = source_schema_name.split("_", 1)[0] if "_" in source_schema_name else None
    if step is not None:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.name.startswith(f"{step}_")
        ]
    if source_name == "galaxy_properties":
        nested = [
            candidate for candidate in candidates if "galaxies_" in candidate.name
        ]
        if nested:
            candidates = nested
    elif len(candidates) > 1:
        direct = [
            candidate for candidate in candidates if "galaxies_" not in candidate.name
        ]
        if direct:
            candidates = direct
    if len(candidates) != 1:
        raise ValueError(
            f"Unable to resolve output target for structure link '{prefix}'"
        )
    return candidates[0]


def __lower_structure_dataset(
    source: Schema, root: Schema, available_prefixes: frozenset[str]
) -> Schema:
    linked = source.children.get("data_linked")
    if linked is None:
        return source
    columns = dict(linked.columns)
    for name, writer in linked.columns.items():
        if name.endswith("_idx"):
            prefix = name.removesuffix("_idx")
        elif name.endswith("_start"):
            prefix = name.removesuffix("_start")
        else:
            continue
        if prefix not in available_prefixes:
            continue
        try:
            target = __target_schema(source, root, prefix)
        except ValueError:
            columns[name] = (
                __lower_idx_writer(writer)
                if name.endswith("_idx")
                else __lower_start_writer(writer, linked.columns[f"{prefix}_size"])
            )
            continue
        raw_index = get_dataset_schema_index(target)
        if raw_index is None:
            raise ValueError(
                f"Structure link target '{prefix}' has no output raw row index"
            )
        columns[name] = __lower_primary_writer(
            writer, __make_output_position_lookup(into_array(raw_index))
        )
    return source._replace(
        children=source.children | {"data_linked": linked._replace(columns=columns)}
    )


def __lower_idx_writer(writer: ColumnWriter) -> ColumnWriter:
    data = writer.data
    valid = data >= 0
    result = np.full(len(data), -1, dtype=np.int64)
    result[valid] = np.arange(valid.sum(), dtype=np.int64)
    return ColumnWriter.from_numpy_array(result, writer.combine_strategy, writer.attrs)


def __lower_start_writer(
    writer: ColumnWriter, size_writer: ColumnWriter
) -> ColumnWriter:
    starts = np.insert(np.cumsum(size_writer.data), 0, 0)[:-1]
    return ColumnWriter.from_numpy_array(starts, writer.combine_strategy, writer.attrs)


def __lower_structure_links(
    schema: Schema, root: Schema, available_prefixes: frozenset[str]
) -> Schema:
    children = {
        name: __lower_structure_links(child, root, available_prefixes)
        for name, child in schema.children.items()
    }
    schema = schema._replace(children=children)
    if schema.type == FileEntry.DATASET:
        return __lower_structure_dataset(schema, root, available_prefixes)
    return schema


def lower_collection_coordinates(
    schema: Schema, *, canonical_raw_order: bool
) -> Schema:
    """Lower raw link coordinates after the collection output order is fixed.

    ``canonical_raw_order`` is explicit because simulation collections are reordered
    into canonical raw order, while structure collections retain source file order
    so their spatial indices remain valid.
    """
    if canonical_raw_order:
        if schema.type != FileEntry.SIMULATION_COLLECTION:
            raise ValueError(
                "Canonical raw-order lowering requires a simulation collection"
            )
        return __lower_simulation_maps(schema)
    if schema.type != FileEntry.STRUCTURE_COLLECTION:
        raise ValueError("Structure-link lowering requires a structure collection")
    target_names = set(schema.children)
    available_prefixes = frozenset(
        prefix
        for prefix, alias in LINK_ALIASES.items()
        if alias in target_names
        or any(name.endswith(f"_{alias}") for name in target_names)
        or (alias == "galaxy_properties" and "galaxies" in target_names)
    )
    return __lower_structure_links(schema, schema, available_prefixes)
