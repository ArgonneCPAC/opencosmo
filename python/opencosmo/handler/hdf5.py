from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.io.writer import (
    ColumnWriter,
)
from opencosmo.uuid import get_hdf5_column_uuid

from opencosmo.index import (
    SimpleIndex,
    get_data,
    get_length,
    into_array,
    take,
)

if TYPE_CHECKING:
    from uuid import UUID

    import h5py
    from opencosmo.io.schema import Schema

    from opencosmo.index import DataIndex


class Hdf5Handler:
    """
    Handler for opencosmo.Dataset

    Holds the live ``/data`` group rather than pre-opened column datasets. A column's
    ``h5py.Dataset`` is acquired the first time it is actually read, which keeps an
    open proportional to the columns a caller touches instead of to the columns a file
    contains. On a network filesystem each acquisition is a round trip, so this matters
    most for wide files and for collections that open many nodes and read few.

    ``column_names`` comes from the discovered ``GroupLayout``, which also guarantees
    every column shares ``row_count``.
    """

    def __init__(
        self,
        data_group: h5py.Group,
        columns: dict[str, h5py.Dataset | None],
        index: DataIndex,
        load_conditions: Optional[dict[str, bool]] = None,
        descriptions: Optional[dict[str, str | None]] = None,
        uuids: Optional[dict[str, UUID]] = None,
    ):
        self.__data_group = data_group
        self.__columns = columns
        self.__index = index
        self.__load_conditions = load_conditions
        self.__descriptions = descriptions
        self.__uuids = uuids
        # Shared by reference with every handler derived from this one. All of them
        # address the same columns in the same file, so a select/filter/take chain
        # must not re-open the same datasets once per link.

    def __handle(self, name: str) -> h5py.Dataset:
        handle = self.__columns.get(name)
        if handle is None:
            handle = self.__data_group[name]
            self.__columns[name] = handle
        return handle

    def __len__(self):
        return get_length(self.__index)

    def with_index(self, index: DataIndex) -> Hdf5Handler:
        return self.__derive(index)

    def __derive(self, index: DataIndex) -> Hdf5Handler:
        return Hdf5Handler(
            self.__data_group,
            self.__columns,
            index,
            self.__load_conditions,
            descriptions=self.__descriptions,
            uuids=self.__uuids,
        )

    @property
    def load_conditions(self) -> Optional[dict[str, bool]]:
        return self.__load_conditions

    def get_uuids(self) -> dict[str, UUID]:
        if self.__uuids is not None:
            return self.__uuids
        return {
            name: get_hdf5_column_uuid(self.__handle(name)) for name in self.__columns
        }

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return self.__derive(other)

        if sorted is not None:
            return self.__take_sorted(other, sorted)

        return self.__derive(take(self.__index, other))

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if get_length(sorted) != get_length(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)

        new_raw_index = into_array(self.__index)[new_indices]
        new_index = np.sort(new_raw_index)

        return self.__derive(new_index)

    @property
    def index(self):
        return self.__index

    @cached_property
    def columns(self):
        return list(self.__columns.keys())

    @property
    def descriptions(self):
        if self.__descriptions is not None:
            return self.__descriptions
        return {
            name: self.__handle(name).attrs.get("description")
            for name in self.__columns
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def make_schema(
        self,
        columns: Iterable[str],
    ) -> Schema:
        columns = set(columns)
        data_writers = {}
        for column_name in columns:
            column = self.__handle(column_name)
            data_writers[column_name] = ColumnWriter.from_h5_dataset(
                column, self.__index, attrs=dict(column.attrs)
            )
        data_schema = make_schema("data", FileEntry.COLUMNS, columns=data_writers)

        return data_schema

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """
        # Ensure order is preserved
        return {name: get_data(self.__handle(name), self.__index) for name in columns}

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
