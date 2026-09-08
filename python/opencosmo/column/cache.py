from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Optional
from uuid import UUID
from weakref import finalize, ref

import astropy.units as u
import numpy as np

from opencosmo.index import DataIndex
from opencosmo.index.build import from_size
from opencosmo.index.get import get_data
from opencosmo.index.take import take
from opencosmo.index.unary import get_length, get_range
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import ColumnWriter
from opencosmo.mpi import (
    gather_data,
    get_all_entries,
    parallel_assert,
    scatter_data,
)

if TYPE_CHECKING:
    from opencosmo.index import DataIndex
    from opencosmo.io.schema import Schema

# (producer_uuid, column_name) — the unambiguous key for a cached column.
CacheKey = tuple[UUID, str]

ColumnUpdater = Callable[[np.ndarray | u.Quantity], np.ndarray | u.Quantity]


def finish(
    cached_data: dict[CacheKey, np.ndarray],
    index: Optional[DataIndex],
    cache_ref: ref[ColumnCache],
):
    cache = cache_ref()
    if cache is None:
        return

    # pylint: disable=protected-access
    if index is None:
        index = from_size(len(cache))
    pairs_to_add = cache.registered_pairs.intersection(cached_data.keys()) - set(
        cache.keys()
    )
    data = {key[0]: {key[1]: get_data(cached_data[key], index)} for key in pairs_to_add}
    if data:
        cache.add_data(data)


def check_length(cache: ColumnCache, data: dict[UUID, dict[str, np.ndarray]]):
    lengths = {len(arr) for uuid_data in data.values() for arr in uuid_data.values()}
    if len(lengths) > 1:
        raise ValueError(
            "When adding data to the cache, all columns must be the same length"
        )
    elif (length := len(cache)) > 0 and lengths and length != lengths.pop():
        raise ValueError(
            "When adding data to the cache, the columns must be the same length as the columns currently in the cache"
        )


class ColumnCache:
    """
    A column cache is used to persist data that is read from an hdf5 file. Caches can get data in one of two ways:
    1. They are explicitly given data that has been recently read from disk or
    2. They take data from a previous cache

    ColumnCaches break some of the rules that most other things follow in this library, notably that they have internal
    state (which can change). This mutability is required for two reasons.

    1. If the parent cache is garbage collected, the child cache needs to be able to copy over any data it needs
    2. If a new cache is created by adding columns, we need to signal the child to update their parent to the new
       cache. This allows us to preserve the standard "operations create new objects" pattern that is present
       throughout the library.

    Internal storage uses (producer_uuid, column_name) tuples as keys so that multiple
    producers that happen to produce a column with the same name are kept separate.
    """

    def __init__(
        self,
        cached_data: dict[CacheKey, np.ndarray],
        registered_column_groups: dict[int, set[CacheKey]],
        column_descriptions: dict[str, str],
        derived_index: Optional[DataIndex],
        parent: Optional[ref[ColumnCache]],
        children: Optional[list[ref[ColumnCache]]],
    ):
        self.__cached_data = cached_data
        self.__registered_column_groups = registered_column_groups
        self.__descriptions = column_descriptions
        self.__derived_index = derived_index
        self.__parent = parent
        if children is None:
            children = []
        self.__children = children
        self.__finalizer = None

        if parent is not None and (p := parent()) is not None:
            self.__finalizer = finalize(
                p,
                finish,
                p.__cached_data,
                derived_index,
                ref(self),
            )
            self.__finalizer.atexit = False  # type: ignore

    @classmethod
    def empty(cls):
        return ColumnCache({}, {}, {}, None, None, [])

    @property
    def columns(self) -> set[str]:
        return {name for (_, name) in self.__cached_data.keys()}

    def keys(self) -> set[CacheKey]:
        return set(self.__cached_data.keys())

    @property
    def descriptions(self) -> dict[str, str]:
        return self.__descriptions

    @property
    def registered_pairs(self) -> set[CacheKey]:
        if not self.__registered_column_groups:
            return set()
        return set().union(*self.__registered_column_groups.values())

    def create_child(self) -> ColumnCache:
        return ColumnCache({}, {}, {}, None, ref(self), [])

    def make_schema(self, columns_to_uuid: dict[str, UUID]) -> Schema:
        data = {}

        cached = self.get_data({(uuid, name) for name, uuid in columns_to_uuid.items()})
        for name, coldata in _flatten(cached).items():
            if isinstance(coldata, u.Quantity):
                column_data = coldata.value
                unit_str = str(coldata.unit)
            else:
                column_data = coldata
                unit_str = ""
            attrs = {
                "unit": unit_str,
                "description": self.__descriptions.get(name, "None"),
            }
            writer = ColumnWriter.from_numpy_array(column_data, attrs=attrs)
            data[name] = writer

        if not data:
            return make_schema("data", FileEntry.EMPTY)

        return make_schema("data", FileEntry.COLUMNS, columns=data)

    def redistribute(
        self, reorder_map, length, columns_to_keep: dict[UUID, list[str]], comm
    ):
        new_data = {}
        keys_to_keep = set()
        for uuid, cols in columns_to_keep.items():
            if not cols:
                continue
            keys_to_keep.update([(uuid, col) for col in cols])

        data_to_keep = self.get_data(keys_to_keep)
        for uuid, columns in get_all_entries(data_to_keep, comm):
            parallel_assert(columns is not None, comm)
            assert columns is not None
            for name, data in get_all_entries(columns, comm):
                parallel_assert(data is not None, comm)
                assert data is not None
                if name not in columns_to_keep[uuid]:
                    continue
                all_data = gather_data(data, comm)
                if comm.Get_rank() == 0:
                    all_data = all_data[reorder_map]
                new_data[(uuid, name)] = scatter_data(all_data, length, comm)
        return ColumnCache(new_data, {}, self.__descriptions, None, None, None)

    def __push_down(self, data: dict[CacheKey, np.ndarray]):
        pairs_to_keep = self.registered_pairs.intersection(data.keys()).difference(
            self.__cached_data.keys()
        )
        cached_data = {key: data[key] for key in pairs_to_keep}
        if self.__derived_index is not None:
            cached_data = {
                key: get_data(cd, self.__derived_index)
                for key, cd in cached_data.items()
            }
        self.__cached_data |= cached_data

    def __push_up(self, data: dict[CacheKey, np.ndarray]):
        assert len(self) == 0 or all(len(d) == len(self) for d in data.values())
        pairs_to_keep = self.registered_pairs.intersection(data.keys()).difference(
            self.__cached_data.keys()
        )
        self.__cached_data |= {key: data[key] for key in pairs_to_keep}

    def register_column_group(self, state_id: int, columns: dict[str, UUID]):
        assert state_id not in self.__registered_column_groups
        self.__registered_column_groups[state_id] = {
            (uuid, name) for name, uuid in columns.items()
        }

    def deregister_column_group(self, state_id: int):
        assert state_id in self.__registered_column_groups
        pairs = self.__registered_column_groups.pop(state_id)
        remaining = (
            set().union(*self.__registered_column_groups.values())
            if self.__registered_column_groups
            else set()
        )
        to_drop = pairs.difference(remaining)
        cached_data = {
            key: self.__cached_data.pop(key)
            for key in to_drop
            if key in self.__cached_data
        }
        if not cached_data:
            return
        for child_ in self.__children:
            if (child := child_()) is None:
                continue
            child.__push_down(cached_data)

    def __update_parent(self, parent: ColumnCache):
        assert self.__parent is not None
        assert self.__finalizer is not None
        self.__finalizer.detach()
        self.__parent = ref(parent)
        self.__finalizer = finalize(
            parent, finish, parent.__cached_data, self.__derived_index, ref(self)
        )
        self.__finalizer.atexit = False  # type: ignore

    def __len__(self):
        if not self.__cached_data and self.__derived_index is None:
            return 0
        elif self.__derived_index is not None:
            return get_length(self.__derived_index)
        elif self.__cached_data:
            return len(next(iter(self.__cached_data.values())))
        elif self.__parent is not None and (p := self.__parent()) is not None:
            return len(p)
        return 0

    def add_data(
        self,
        data: dict[UUID, dict[str, np.ndarray]],
        descriptions: dict[str, str] = {},
        push_up: bool = True,
    ):
        """Add UUID-keyed column data to the cache."""

        if not data:
            return
        check_length(self, data)
        self.__descriptions |= descriptions

        flat: dict[CacheKey, np.ndarray] = {
            (uuid, name): arr
            for uuid, uuid_data in data.items()
            for name, arr in uuid_data.items()
        }
        if (
            push_up
            and self.__derived_index is None
            and self.__parent is not None
            and (p := self.__parent()) is not None
        ):
            p.__push_up(flat)

        self.__cached_data |= flat

    def drop(self, column_names: Iterable[str]) -> ColumnCache:
        names_to_drop = set(column_names)
        data = {
            key: val
            for key, val in self.__cached_data.items()
            if key[1] not in names_to_drop
        }
        descriptions = {
            name: desc
            for name, desc in self.__descriptions.items()
            if name not in names_to_drop
        }
        return ColumnCache(data, {}, descriptions, None, None, [])

    def request(
        self, pairs: set[CacheKey], index: Optional[DataIndex]
    ) -> dict[CacheKey, np.ndarray]:
        pairs_in_cache = pairs.intersection(self.__cached_data.keys())
        data = {key: self.__cached_data[key] for key in pairs_in_cache}
        if index is not None:
            data = {key: get_data(cd, index) for key, cd in data.items()}

        if self.__parent is None or pairs == pairs_in_cache:
            return data

        parent = self.__parent()
        if parent is None:
            return data

        match (index, self.__derived_index):
            case (None, None):
                new_index = None
            case (_, None):
                new_index = index
            case (None, _):
                new_index = self.__derived_index
            case _:
                assert self.__derived_index is not None and index is not None
                new_index = take(self.__derived_index, index)

        return data | parent.request(pairs - pairs_in_cache, new_index)

    def take(self, index: DataIndex) -> ColumnCache:
        if len(self) == 0 and not self.columns:
            return ColumnCache.empty()
        if get_range(index)[1] > len(self):
            raise ValueError(
                "Tried to take more elements than the length of the cache!"
            )
        new_cache = ColumnCache({}, {}, {}, index, ref(self), [])
        self.__children.append(ref(new_cache))
        return new_cache

    def get_data(self, pairs: set[CacheKey]) -> dict[UUID, dict[str, np.ndarray]]:
        """
        Retrieve data for the requested (uuid, name) pairs. Returns a
        UUID-keyed dict so callers can look up each producer's contribution.
        """
        pairs_in_cache = pairs.intersection(self.__cached_data.keys())
        missing_pairs = pairs - pairs_in_cache
        flat = {key: self.__cached_data[key] for key in pairs_in_cache}
        flat |= self.__get_derived_pairs(missing_pairs)
        return _unflatten(flat)

    def __get_derived_pairs(self, pairs: set[CacheKey]) -> dict[CacheKey, np.ndarray]:
        if self.__parent is None:
            return {}
        parent = self.__parent()
        if parent is None:
            return {}
        result = parent.request(pairs, self.__derived_index)
        self.__cached_data |= result
        return result


def _flatten(uuid_data: dict[UUID, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Collapse UUID-keyed data to a flat name-keyed dict. Last writer wins."""
    return {name: arr for d in uuid_data.values() for name, arr in d.items()}


def _unflatten(flat: dict[CacheKey, np.ndarray]) -> dict[UUID, dict[str, np.ndarray]]:
    """Group (uuid, name) → array back into {uuid: {name: array}}."""
    result: dict[UUID, dict[str, np.ndarray]] = {}
    for (uuid, name), arr in flat.items():
        result.setdefault(uuid, {})[name] = arr
    return result
