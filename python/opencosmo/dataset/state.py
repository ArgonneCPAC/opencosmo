from __future__ import annotations

import dataclasses
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Generator, Optional
from weakref import finalize

import astropy.units as u
import numpy as np

from opencosmo.column.cache import ColumnCache
from opencosmo.column.column import EvaluatedColumn, RawColumn
from opencosmo.column.select import MissingColumnError, get_column_selection
from opencosmo.dataset.columns import add_columns, resort
from opencosmo.dataset.graph import get_all_required_pairs
from opencosmo.dataset.instantiate import instantiate_dataset
from opencosmo.dataset.output import get_derived_column_names, make_dataset_schema
from opencosmo.handler.empty import EmptyHandler
from opencosmo.handler.hdf5 import Hdf5Handler
from opencosmo.index import from_size, reindex_column, single_chunk
from opencosmo.index.mask import into_array
from opencosmo.mpi import gather_index, get_comm_world
from opencosmo.plugins.contexts import (
    DatasetInstantiateCtx,
    HookPoint,
    IndexUpdateCtx,
    PostSortCtx,
)
from opencosmo.plugins.hook import fold
from opencosmo.units.handler import (
    make_unit_handler_from_unit_strings,
    make_unit_handler_from_units,
)
from opencosmo.uuid import (
    get_in_memory_dataset_uuid,
    get_raw_column_uuid,
    get_string_uuid,
)

if TYPE_CHECKING:
    from uuid import UUID

    from astropy import table
    from astropy.cosmology import Cosmology

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.handler.protocols import DataCache, DataHandler
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region
    from opencosmo.spatial.tree import Tree
    from opencosmo.units import UnitConvention
    from opencosmo.units.handler import UnitHandler


def deregister_state(id: int, cache: DataCache):
    cache.deregister_column_group(id)


def sort_data(
    data: dict[str, np.ndarray],
    sort_by: tuple[str, bool, bool] | None,
    state: DatasetState,
):
    if sort_by is None:
        return data
    sort_column = data[sort_by[0]]
    order = np.argsort(sort_column)
    if sort_by[1]:
        order = order[::-1]

    data = {key: value[order] for key, value in data.items()}
    if sort_by[2] and set(data.keys()) != set(sort_by[0]):
        data.pop(sort_by[0])
    return fold(HookPoint.PostSort, PostSortCtx(state, data, np.argsort(order))).data


@dataclass(frozen=True)
class DatasetState:
    """
    Main state container for the Dataset. Functions for manipulating it can be found below. The dataclass
    itself only exposes basic lookup operations.
    """

    uuid: UUID
    producers: dict[UUID, ConstructedColumn]
    raw_data_handler: DataHandler
    cache: DataCache
    unit_handler: UnitHandler
    header: OpenCosmoHeader
    tree: Tree | None
    column_map: dict[str, UUID]
    open_kwargs: dict[str, Any]
    sort_key: Optional[tuple[str, bool, bool]]

    def __post_init__(self):
        self.cache.register_column_group(id(self), self.column_map)
        finalize(self, deregister_state, id(self), self.cache)

    @property
    def columns(self) -> list[str]:
        sort_to_drop: int | str = -1
        if self.sort_key is not None and self.sort_key[2]:
            sort_to_drop = self.sort_key[0]

        return [c for c in self.column_map if c != sort_to_drop]

    @property
    def descriptions(self):
        all_descriptions = {}
        for producer in self.producers.values():
            update = {name: producer.description for name in producer.produces}
            all_descriptions |= update
        all_descriptions |= self.cache.descriptions

        return {
            name: description
            for name, description in all_descriptions.items()
            if name in self.columns
        }

    @property
    def region(self):
        if self.tree is None:
            return None
        return self.tree.get_region()

    @property
    def kwargs(self):
        return self.open_kwargs

    @property
    def raw_index(self):
        if (si := get_sorted_index(self)) is not None:
            ni = into_array(self.raw_data_handler.index)
            return ni[si]
        return self.raw_data_handler.index

    @property
    def units(self):
        units = self.unit_handler.current_units
        return {name: units[name] for name in self.columns}

    @property
    def convention(self):
        return self.unit_handler.current_convention

    def __len__(state: DatasetState) -> int:
        if isinstance(state.raw_data_handler, EmptyHandler):
            return len(state.cache)
        return len(state.raw_data_handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory functions (replace classmethods)
# ---------------------------------------------------------------------------


def state_from_target(
    target: DatasetTarget,
    unit_convention: UnitConvention,
    region: Region,
    open_kwargs: dict[str, Any],
    index: Optional[DataIndex] = None,
    tree: Tree | None = None,
) -> DatasetState:
    handler = Hdf5Handler(
        target.data_group,
        {cn: None for cn in target.column_names},
        index if index is not None else from_size(target.row_count),
        target.load_conditions,
        descriptions=target.column_descriptions,
    )
    unit_handler = make_unit_handler_from_unit_strings(
        target.column_units, target.header, unit_convention
    )
    descriptions = target.column_descriptions
    uuids = target.column_uuids

    raw_producers = [
        RawColumn(
            cname,
            descriptions.get(cname, "None"),
            _uuid=uuid,
            on_disk=True,
        )
        for cname, uuid in uuids.items()
    ]
    column_map = {p.name: p.uuid for p in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {p.uuid: p for p in raw_producers}
    cache = ColumnCache.empty()
    return DatasetState(
        uuid=target.uuid,
        producers=producers,
        raw_data_handler=handler,
        cache=cache,
        unit_handler=unit_handler,
        header=target.header,
        tree=tree,
        column_map=column_map,
        open_kwargs=open_kwargs,
        sort_key=None,
    )


def state_in_memory(
    data_columns: dict,
    header: OpenCosmoHeader,
    unit_convention: UnitConvention,
    region: Region,
    open_kwargs: dict[str, Any],
    descriptions: Optional[dict[str, str]] = None,
    index: Optional[DataIndex] = None,
    tree: Tree | None = None,
) -> DatasetState:
    descriptions = descriptions or {}

    all_columns = dict(data_columns)
    raw_producers = [
        RawColumn(
            cname, descriptions.get(cname, "None"), get_raw_column_uuid(cname, set())
        )
        for cname in all_columns.keys()
    ]
    column_map = {p.name: p.uuid for p in raw_producers}
    producers: dict[UUID, ConstructedColumn] = {p.uuid: p for p in raw_producers}

    cache = ColumnCache.empty()
    if all_columns:
        uuid_data = {p.uuid: {p.name: all_columns[p.name]} for p in raw_producers}
        cache.add_data(uuid_data, descriptions)

    units: dict[str, u.Unit] = {}
    for name, column in all_columns.items():
        units[name] = None
        if isinstance(column, u.Quantity):
            units[name] = column.unit

    unit_handler = make_unit_handler_from_units(units, header, unit_convention)

    return DatasetState(
        uuid=get_in_memory_dataset_uuid(data_columns),
        producers=producers,
        raw_data_handler=EmptyHandler(),
        cache=cache,
        unit_handler=unit_handler,
        header=header,
        tree=tree,
        column_map=column_map,
        open_kwargs=open_kwargs,
        sort_key=None,
    )


# ---------------------------------------------------------------------------
# Standalone functions (replace methods)
# ---------------------------------------------------------------------------


def exit_state(state: DatasetState, *exec_details):
    return None


def get_data(
    state: DatasetState,
    ignore_sort: bool = False,
    unit_kwargs: dict = {},
) -> dict:
    """
    Use a State to get the associated data. Most of the logic can be found in the
    instantiate_dataset method.
    """
    state = fold(HookPoint.DatasetInstantiate, DatasetInstantiateCtx(state)).state
    data = instantiate_dataset(
        list(state.producers.values()),
        state.column_map,
        state.raw_data_handler,
        state.cache,
        state.unit_handler,
        unit_kwargs,
        None if (ignore_sort or state.sort_key is None) else state.sort_key[0],
    )

    if missing := set(state.columns).difference(data.keys()):
        raise RuntimeError(
            f"Some columns are missing from the output! This is likely a bug. Please report it on GitHub. Missing: {missing}"
        )

    if not ignore_sort:
        data = sort_data(data, state.sort_key, state)

    new_order = list(state.columns)
    if state.sort_key is not None and not new_order:
        new_order = [state.sort_key[0]]

    return {name: data[name] for name in new_order}


def iter_rows(
    state: DatasetState,
    unit_kwargs: dict = {},
) -> Generator:
    """
    Iterate over the rows of a given DatasetState
    """
    derived_to_collect = (
        set(state.columns)
        .difference(state.cache.columns)
        .difference(state.raw_data_handler.columns)
    )
    derived_storage: dict[str, list[np.ndarray]] = {
        name: [] for name in derived_to_collect
    }
    total_length = len(state)
    chunk_ranges = [
        (i, min(i + 1000, total_length)) for i in range(0, total_length, 1000)
    ]
    if not chunk_ranges:
        raise StopIteration

    try:
        for start, end in chunk_ranges:
            chunk = take_rows(state, single_chunk(start, end - start))
            data = get_data(chunk, unit_kwargs=unit_kwargs)
            for name in derived_to_collect:
                derived_storage[name].append(data[name])

            for i in range(len(chunk)):
                yield {name: column[i] for name, column in data.items()}
        all_derived = {
            name: np.concatenate(arr) for name, arr in derived_storage.items()
        }
        derived_storage = resort(all_derived, get_sorted_index(state))
        if derived_storage:
            uuid_keyed: dict = {}
            for name, arr in derived_storage.items():
                uuid = state.column_map[name]
                uuid_keyed.setdefault(uuid, {})[name] = arr
            state.cache.add_data(uuid_keyed, {})
    except GeneratorExit:
        pass
    except BaseException:
        raise


def make_schema(state: DatasetState, path: str) -> Schema:
    producers = list(state.producers.values())
    columns = set(state.column_map.keys())
    derived_names = get_derived_column_names(producers, columns)
    if derived_names:
        selected = select(state, derived_names)
        converted = with_units(
            selected, state.unit_handler.base_convention, {}, {}, None, None
        )
        derived_data = get_data(converted, ignore_sort=True)
    else:
        derived_data = {}

    column_map = copy(state.column_map)
    if state.sort_key is not None and state.sort_key[2]:
        column_map.pop(state.sort_key[0])

    name = path.split("/")[-1]

    return make_dataset_schema(
        producers,
        state.raw_data_handler,
        state.cache,
        column_map,
        state.header,
        state.tree,
        state.region,
        state.raw_index,
        derived_data,
        get_string_uuid(path),
        name,
    )


def with_new_columns(
    state: DatasetState,
    descriptions: dict[str, str] = {},
    allow_overwrite: bool = False,
    **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
) -> DatasetState:
    """
    Add columns to a given state
    """
    new_producers_list, new_column_map, new_unit_handler = add_columns(
        list(state.producers.values()),
        state.unit_handler,
        state.cache,
        state.column_map,
        set(state.producers.keys()),
        get_sorted_index(state),
        descriptions,
        new_columns,
        len(state),
        allow_overwrite=allow_overwrite,
    )
    producers = {}
    for producer in new_producers_list:
        assert producer.uuid is not None
        producers[producer.uuid] = producer

    return dataclasses.replace(
        state,
        producers=producers,
        column_map=new_column_map,
        unit_handler=new_unit_handler,
    )


def select(state: DatasetState, columns: set[str], drop: bool = False) -> DatasetState:
    """
    Select a set of columns
    """
    selections, missing = get_column_selection(state.columns, columns)
    if (
        len(columns) > 1
        and state.sort_key is not None
        and state.sort_key[2]
        and state.sort_key[0] in columns
    ):
        missing.add(state.sort_key[0])
    elif (
        len(columns) == 1
        and state.sort_key is not None
        and missing == set([state.sort_key[0]])
    ):
        selections = columns
        missing = set()

    if missing:
        raise MissingColumnError(
            f"Columns are included that are not in this dataset: {missing}"
        )
    elif not selections and columns:
        raise MissingColumnError("No columns matched the provided wildcards!")

    if drop:
        selections = set(state.columns) - selections

    new_sort_key = state.sort_key
    if state.sort_key is not None and state.sort_key[0] not in selections:
        selections.add(state.sort_key[0])
        new_sort_key = (state.sort_key[0], state.sort_key[1], True)

    new_column_map = {n: state.column_map[n] for n in selections}
    return dataclasses.replace(state, column_map=new_column_map, sort_key=new_sort_key)


def get_sorted_index(state: DatasetState) -> np.ndarray | None:
    if state.sort_key is not None:
        column = get_data(select(state, {state.sort_key[0]}), ignore_sort=True)[
            state.sort_key[0]
        ]
        sorted_idx = np.argsort(column)
        if state.sort_key[1]:
            sorted_idx = sorted_idx[::-1]
    else:
        sorted_idx = None

    return sorted_idx


def take_rows(state: DatasetState, rows: DataIndex) -> DatasetState:
    """
    Take a set of rows. The associated "take" functions in the
    dataset all delegate to this function.
    """
    if len(state) == 0:
        return state
    rows = fold(HookPoint.IndexUpdate, IndexUpdateCtx(state, rows)).index
    sorted_idx = get_sorted_index(state)
    if sorted_idx is not None:
        rows = np.sort(sorted_idx[into_array(rows)])
    new_handler = state.raw_data_handler.take(rows)
    new_cache = state.cache.take(rows)
    return dataclasses.replace(state, raw_data_handler=new_handler, cache=new_cache)


def with_units(
    state: DatasetState,
    convention: UnitConvention,
    conversions: dict[u.Unit, u.Unit],
    columns: dict[str, u.Unit],
    cosmology: Cosmology,
    redshift: float | table.Column,
) -> DatasetState:
    """
    Update the units of a given state.
    """
    new_handler = state.unit_handler.with_convention(convention).with_conversions(
        conversions, columns
    )

    if convention == state.unit_handler.current_convention:
        cache = state.cache.create_child()
    else:
        all_derived_names: set[str] = set()
        all_derived_names = reduce(
            lambda acc, col: acc.union(
                col.produces if not isinstance(col, RawColumn) else set()
            ),
            state.producers.values(),
            all_derived_names,
        ).intersection(state.columns)
        columns_to_drop = all_derived_names.union(state.raw_data_handler.columns)
        cache = state.cache.drop(columns_to_drop)
    new_header = state.header.with_units(convention)

    return dataclasses.replace(
        state, unit_handler=new_handler, cache=cache, header=new_header
    )


def redistribute(state: DatasetState, rows):
    all_required_producers = get_all_required_pairs(
        list(state.producers.values()), state.column_map
    )
    cached_columns_to_keep = defaultdict(list)
    for uuid, name in all_required_producers:
        producer = state.producers[uuid]
        if isinstance(producer, EvaluatedColumn) or (
            isinstance(producer, RawColumn) and not producer.on_disk
        ):
            cached_columns_to_keep[uuid].append(name)

    comm = get_comm_world()
    assert comm is not None

    if cached_columns_to_keep:
        reorder_map = None
        new_index = gather_index(rows, comm)
        original_index = gather_index(state.raw_index, comm)
        if comm.Get_rank() == 0:
            reorder_map = reindex_column(original_index, new_index)
        new_cache = state.cache.redistribute(
            reorder_map, len(rows), cached_columns_to_keep, comm
        )
    else:
        new_cache = state.cache.empty()
    new_handler = state.raw_data_handler.with_index(rows)
    return dataclasses.replace(state, cache=new_cache, raw_data_handler=new_handler)
