from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

from opencosmo.index import into_array
from opencosmo.io.schema import (
    FileEntry,
    Schema,
    get_dataset_schema_index,
    reorder_dataset_schema,
)
from opencosmo.io.writer import ColumnWriter
from opencosmo.mpi import (
    gather_index,
    get_all_keys,
    get_subcom,
    redistribute_data,
    scatter_index,
)

if TYPE_CHECKING:
    from opencosmo.mpi import MPI


def resort_simulation_collection(schema):
    """Sort local dataset rows into their canonical raw-index order."""
    if "map" not in schema.children:
        return schema
    new_children = {}
    for name, child in schema.children.items():
        if name == "map":
            continue
        index = get_dataset_schema_index(child)
        reorder = np.argsort(into_array(index))
        new_children[name] = reorder_dataset_schema(child, reorder)

    new_children["map"] = schema.children["map"]
    return schema._replace(children=new_children)


@dataclass(frozen=True)
class DatasetOutputLookup:
    """Global output coordinates and writer ownership for one dataset."""

    output_positions: Mapping[int, int]
    writer_ranks: Mapping[int, int]


def __make_dataset_output_lookup(
    canonical_raw_ids: np.ndarray, nranks: int
) -> DatasetOutputLookup:
    """Build immutable output lookups from globally sorted, unique raw IDs."""
    output_positions = {
        int(raw_id): position for position, raw_id in enumerate(canonical_raw_ids)
    }
    lengths = np.full(nranks, len(canonical_raw_ids) // nranks, dtype=np.int64)
    lengths[: len(canonical_raw_ids) % nranks] += 1
    writer_ranks = {
        int(raw_id): rank
        for rank, raw_ids in enumerate(
            np.split(canonical_raw_ids, np.cumsum(lengths)[:-1])
        )
        for raw_id in raw_ids
    }
    return DatasetOutputLookup(
        MappingProxyType(output_positions), MappingProxyType(writer_ranks)
    )


def __plan_dataset_output(
    raw_ids: np.ndarray, nranks: int
) -> tuple[np.ndarray, DatasetOutputLookup]:
    """Return canonical raw IDs and their deterministic writer assignment."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    canonical_raw_ids = raw_ids[np.argsort(raw_ids, kind="stable")]
    duplicate = np.flatnonzero(np.diff(canonical_raw_ids) == 0)
    if len(duplicate) > 0:
        duplicate_ids = canonical_raw_ids[duplicate]
        raise ValueError(
            "Cannot write simulation dataset with duplicate raw row IDs: "
            f"{duplicate_ids.tolist()}"
        )
    return canonical_raw_ids, __make_dataset_output_lookup(canonical_raw_ids, nranks)


def __get_dataset_output_lookup(
    raw_ids: np.ndarray, comm: MPI.Comm
) -> tuple[DatasetOutputLookup, np.ndarray]:
    """Collectively plan output ownership and return this rank's destinations."""
    gathered_raw_ids = gather_index(raw_ids, comm)
    if comm.Get_rank() == 0:
        try:
            canonical_raw_ids, lookup = __plan_dataset_output(
                gathered_raw_ids, comm.Get_size()
            )
            target_ranks = np.fromiter(
                (lookup.writer_ranks[int(raw_id)] for raw_id in gathered_raw_ids),
                dtype=np.int64,
                count=len(gathered_raw_ids),
            )
            payload: tuple[str | None, np.ndarray | None, np.ndarray | None] = (
                None,
                canonical_raw_ids,
                target_ranks,
            )
        except ValueError as error:
            payload = (str(error), None, None)
    else:
        payload = (None, None, None)

    message, canonical_raw_ids, target_ranks = comm.bcast(payload, root=0)
    if message is not None:
        raise ValueError(message)
    assert canonical_raw_ids is not None
    assert target_ranks is not None
    lookup = __make_dataset_output_lookup(canonical_raw_ids, comm.Get_size())
    return lookup, scatter_index(target_ranks, len(raw_ids), comm)


def redistribute_simulation_collection_data(
    schema: Schema, comm: MPI.Comm
) -> tuple[Schema, dict[str, DatasetOutputLookup]]:
    """Redistribute dataset rows and retain their global output lookups.

    Map children are intentionally not considered here.  Their raw-coordinate
    values are lowered in the subsequent MPI mapping phase.
    """
    new_children = {}
    output_lookups = {}
    for child_name in get_all_keys(schema.children, comm):
        if child_name == "map":
            continue
        rank_has_child = child_name in schema.children
        all_has_child = comm.allgather(rank_has_child)
        child = schema.children.get(child_name)
        local_raw_ids = None if child is None else get_dataset_schema_index(child)
        __collective_error(
            f"Dataset '{child_name}' has no output raw row index"
            if child is not None and local_raw_ids is None
            else None,
            comm,
        )

        subcom, subgroup = get_subcom(all_has_child, comm)
        try:
            if child is None:
                continue
            assert local_raw_ids is not None
            lookup, target_ranks = __get_dataset_output_lookup(local_raw_ids, subcom)
            received_raw_ids = redistribute_data(local_raw_ids, target_ranks, subcom)
            new_children[child_name] = update_dataset_schema_with_redistribute(
                child,
                target_ranks,
                received_raw_ids,
                subcom,
            )
            # ``lookup`` is planned on the dataset subcommunicator.  Primary map
            # lowering routes on ``comm``, so translate its ranks back to the
            # parent communicator explicitly.
            active_ranks = np.flatnonzero(all_has_child)
            output_lookups[child_name] = DatasetOutputLookup(
                lookup.output_positions,
                MappingProxyType(
                    {
                        raw_id: int(active_ranks[rank])
                        for raw_id, rank in lookup.writer_ranks.items()
                    }
                ),
            )
        finally:
            if rank_has_child:
                subcom.Free()
            subgroup.Free()
            comm.Barrier()
    return schema._replace(children=new_children), output_lookups


def __collective_error(message: str | None, comm: MPI.Comm) -> None:
    """Raise the first local validation error on every rank."""
    messages = comm.allgather(message)
    error = next((value for value in messages if value is not None), None)
    if error is not None:
        raise ValueError(error)


def __lower_primary_values(
    source_raw_ids: np.ndarray,
    raw_targets: np.ndarray,
    reference_lookup: DatasetOutputLookup,
    target_lookup: DatasetOutputLookup,
    comm: MPI.Comm,
) -> np.ndarray:
    """Route one raw-coordinate primary slot to its source output owners."""
    source_raw_ids = np.asarray(source_raw_ids, dtype=np.int64)
    raw_targets = np.asarray(raw_targets, dtype=np.int64)
    message = None
    if len(source_raw_ids) != len(raw_targets):
        message = "Primary mapping length does not match the local reference dataset"

    source_positions = np.empty(len(source_raw_ids), dtype=np.int64)
    source_ranks = np.empty(len(source_raw_ids), dtype=np.int64)
    if message is None:
        try:
            for position, raw_id in enumerate(source_raw_ids):
                source_positions[position] = reference_lookup.output_positions[
                    int(raw_id)
                ]
                source_ranks[position] = reference_lookup.writer_ranks[int(raw_id)]
        except KeyError as error:
            message = f"Primary mapping source raw row ID is not in the output: {error}"
    __collective_error(message, comm)

    output_targets = np.full(len(raw_targets), -1, dtype=np.int64)
    for position, raw_target in enumerate(raw_targets):
        if raw_target >= 0:
            output_targets[position] = target_lookup.output_positions.get(
                int(raw_target), -1
            )

    received_positions = redistribute_data(source_positions, source_ranks, comm)
    received_targets = redistribute_data(output_targets, source_ranks, comm)
    reorder = np.argsort(received_positions, kind="stable")
    received_positions = received_positions[reorder]
    received_targets = received_targets[reorder]

    expected_positions = np.asarray(
        sorted(
            position
            for raw_id, position in reference_lookup.output_positions.items()
            if reference_lookup.writer_ranks[raw_id] == comm.Get_rank()
        ),
        dtype=np.int64,
    )
    message = None
    if not np.array_equal(received_positions, expected_positions):
        message = (
            "Primary mapping entries do not cover this rank's contiguous reference "
            "output interval"
        )
    __collective_error(message, comm)
    return received_targets


def __lower_auxiliary_values(
    raw_source: np.ndarray,
    raw_target: np.ndarray,
    source_lookup: DatasetOutputLookup,
    target_lookup: DatasetOutputLookup,
    comm: MPI.Comm,
) -> tuple[np.ndarray, np.ndarray]:
    """Route one raw-coordinate auxiliary pair to its source output owners."""
    raw_source = np.asarray(raw_source, dtype=np.int64)
    raw_target = np.asarray(raw_target, dtype=np.int64)
    __collective_error(
        None
        if len(raw_source) == len(raw_target)
        else "Auxiliary mapping source and target have different lengths",
        comm,
    )

    source_positions = np.fromiter(
        (source_lookup.output_positions.get(int(raw_id), -1) for raw_id in raw_source),
        dtype=np.int64,
        count=len(raw_source),
    )
    target_positions = np.fromiter(
        (target_lookup.output_positions.get(int(raw_id), -1) for raw_id in raw_target),
        dtype=np.int64,
        count=len(raw_target),
    )
    retained = (source_positions >= 0) & (target_positions >= 0)
    source_positions = source_positions[retained]
    target_positions = target_positions[retained]
    source_ranks = np.fromiter(
        (source_lookup.writer_ranks[int(raw_id)] for raw_id in raw_source[retained]),
        dtype=np.int64,
        count=len(source_positions),
    )

    received_source = redistribute_data(source_positions, source_ranks, comm)
    received_target = redistribute_data(target_positions, source_ranks, comm)
    reorder = np.lexsort((received_target, received_source))
    return received_source[reorder], received_target[reorder]


def __dataset_names_and_lookups(
    schema: Schema, output_lookups: dict[str, DatasetOutputLookup], comm: MPI.Comm
) -> tuple[dict[str, str], dict[str, DatasetOutputLookup]]:
    """Collect UUID-to-name and lookup maps from asymmetric dataset children."""
    local_uuids = {}
    for name, child in schema.children.items():
        if child.type == FileEntry.DATASET:
            local_uuids[str(child.children["data"].attributes["main_uuid"])] = name
    uuid_to_name: dict[str, str] = {}
    for values in comm.allgather(local_uuids):
        for uuid, name in values.items():
            existing = uuid_to_name.setdefault(uuid, name)
            if existing != name:
                raise ValueError(f"Output dataset UUID {uuid} has multiple names")

    serialized_lookups = {
        name: (dict(lookup.output_positions), dict(lookup.writer_ranks))
        for name, lookup in output_lookups.items()
    }
    lookups_by_name: dict[str, DatasetOutputLookup] = {}
    for values in comm.allgather(serialized_lookups):
        for name, (positions, ranks) in values.items():
            lookups_by_name.setdefault(
                name,
                DatasetOutputLookup(
                    MappingProxyType(positions), MappingProxyType(ranks)
                ),
            )
    return uuid_to_name, lookups_by_name


def resort_simulation_collection_mpi(schema: Schema, comm: MPI.Comm):
    """Redistribute dataset rows and lower maps to output coordinates."""
    rank_has_map = "map" in schema.children
    has_map = comm.allgather(rank_has_map)
    if not any(has_map):
        return schema

    data_schema, output_lookups = redistribute_simulation_collection_data(schema, comm)
    uuid_to_name, lookups_by_name = __dataset_names_and_lookups(
        schema, output_lookups, comm
    )

    local_map = schema.children.get("map")
    map_attributes = next(
        (
            value
            for value in comm.allgather(local_map.attributes if local_map else None)
            if value
        ),
        None,
    )
    assert map_attributes is not None
    reference_uuid = str(map_attributes["reference"])
    reference_name = uuid_to_name.get(reference_uuid)
    __collective_error(
        None
        if reference_name in lookups_by_name
        else "Simulation mapping reference dataset is not in the output schema",
        comm,
    )
    assert reference_name is not None
    reference_lookup = lookups_by_name[reference_name]

    local_primary = local_map.children.get("primary") if local_map is not None else None
    primary_children = {}
    for target_uuid in get_all_keys(
        local_primary.children if local_primary is not None else {}, comm
    ):
        target_name = uuid_to_name.get(target_uuid)
        __collective_error(
            None
            if target_name in lookups_by_name
            else f"Primary mapping target dataset {target_uuid} is not in the output schema",
            comm,
        )
        assert target_name is not None
        target_lookup = lookups_by_name[target_name]
        local_slot = (
            local_primary.children.get(target_uuid)
            if local_primary is not None
            else None
        )
        reference_child = schema.children.get(reference_name)
        source_raw_ids = (
            into_array(get_dataset_schema_index(reference_child))
            if reference_child is not None
            else np.empty(0, dtype=np.int64)
        )
        raw_targets = (
            local_slot.columns["index"].data
            if local_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        lowered = __lower_primary_values(
            source_raw_ids, raw_targets, reference_lookup, target_lookup, comm
        )

        slot_metadata = comm.allgather(
            None
            if local_slot is None
            else (
                local_slot.columns["index"].combine_strategy,
                local_slot.columns["index"].attrs,
                local_slot.attributes,
            )
        )
        strategy, attrs, attributes = next(
            value for value in slot_metadata if value is not None
        )
        writer = ColumnWriter.from_numpy_array(lowered, strategy, attrs)
        if local_slot is not None:
            primary_children[target_uuid] = local_slot._replace(
                columns={"index": writer}
            )
        elif len(lowered) > 0:
            primary_children[target_uuid] = Schema(
                target_uuid, FileEntry.COLUMNS, {}, {"index": writer}, attributes
            )

    primary_schema = Schema("primary", FileEntry.COLUMNS, primary_children, {}, {})
    local_auxiliary = (
        local_map.children.get("auxiliary") if local_map is not None else None
    )
    auxiliary_children = {}
    for pair_name in get_all_keys(
        local_auxiliary.children if local_auxiliary is not None else {}, comm
    ):
        try:
            source_uuid, target_uuid = pair_name.split("__")
        except ValueError:
            __collective_error(
                f"Auxiliary mapping pair name is invalid: {pair_name}", comm
            )
            raise RuntimeError("unreachable")
        source_name = uuid_to_name.get(source_uuid)
        target_name = uuid_to_name.get(target_uuid)
        __collective_error(
            None
            if source_name in lookups_by_name
            else f"Auxiliary mapping source dataset {source_uuid} is not in the output schema",
            comm,
        )
        __collective_error(
            None
            if target_name in lookups_by_name
            else f"Auxiliary mapping target dataset {target_uuid} is not in the output schema",
            comm,
        )
        assert source_name is not None
        assert target_name is not None

        local_slot = (
            local_auxiliary.children.get(pair_name)
            if local_auxiliary is not None
            else None
        )
        message = None
        if local_slot is not None:
            if "source" not in local_slot.columns or "target" not in local_slot.columns:
                message = (
                    f"Auxiliary mapping pair {pair_name} has no source or target column"
                )
            elif len(local_slot.columns["source"].data) != len(
                local_slot.columns["target"].data
            ):
                message = "Auxiliary mapping source and target have different lengths"
        __collective_error(message, comm)

        raw_source = (
            local_slot.columns["source"].data
            if local_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        raw_target = (
            local_slot.columns["target"].data
            if local_slot is not None
            else np.empty(0, dtype=np.int64)
        )
        source_child = schema.children.get(source_name)
        local_source_ids = (
            None if source_child is None else get_dataset_schema_index(source_child)
        )
        __collective_error(
            f"Dataset '{source_name}' has no output raw row index"
            if source_child is not None and local_source_ids is None
            else None,
            comm,
        )
        if local_source_ids is not None:
            local_source_ids = into_array(local_source_ids)
            local_pairs = np.isin(raw_source, local_source_ids)
            raw_source = raw_source[local_pairs]
            raw_target = raw_target[local_pairs]
        source, target = __lower_auxiliary_values(
            raw_source,
            raw_target,
            lookups_by_name[source_name],
            lookups_by_name[target_name],
            comm,
        )
        if not any(comm.allgather(len(source))):
            continue

        slot_metadata = comm.allgather(
            None
            if local_slot is None
            else (
                local_slot.columns["source"].combine_strategy,
                local_slot.columns["source"].attrs,
                local_slot.columns["target"].combine_strategy,
                local_slot.columns["target"].attrs,
                local_slot.attributes,
            )
        )
        (
            source_strategy,
            source_attrs,
            target_strategy,
            target_attrs,
            attributes,
        ) = next(value for value in slot_metadata if value is not None)
        columns = {
            "source": ColumnWriter.from_numpy_array(
                source, source_strategy, source_attrs
            ),
            "target": ColumnWriter.from_numpy_array(
                target, target_strategy, target_attrs
            ),
        }
        auxiliary_children[pair_name] = (
            local_slot._replace(columns=columns)
            if local_slot is not None
            else Schema(pair_name, FileEntry.COLUMNS, {}, columns, attributes)
        )

    auxiliary_schema = Schema(
        "auxiliary", FileEntry.COLUMNS, auxiliary_children, {}, {}
    )
    map_schema = Schema(
        "map",
        FileEntry.METADATA,
        {"primary": primary_schema, "auxiliary": auxiliary_schema},
        {},
        map_attributes,
    )
    return data_schema._replace(children=data_schema.children | {"map": map_schema})


def update_dataset_schema_with_redistribute(
    schema: Schema, target_rank, rank_local_index, comm
):
    if "data" not in schema.children:
        raise ValueError("Dataset schema has no data child")
    new_columns = {}
    argsort_local = np.argsort(rank_local_index, kind="stable")
    sorted_rank_local_index = rank_local_index[argsort_local]
    for name in get_all_keys(schema.children["data"].columns, comm):
        # This should alwyas be the same for all
        writer = (
            schema.children["data"]
            .columns[name]
            .redistribute(
                rank_local_index,
                sorted_rank_local_index,
                argsort_local,
                target_rank,
                comm,
            )
        )
        new_columns[name] = writer
    new_data = schema.children["data"]._replace(columns=new_columns)
    return schema._replace(children=schema.children | {"data": new_data})
