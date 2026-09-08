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
    from_size,
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
    """

    def __init__(
        self,
        columns: dict[str, h5py.Dataset],
        index: DataIndex,
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        self.__index = index
        self.__columns = columns
        self.__in_memory = next(iter(columns.values())).file.driver == "core"
        self.__load_conditions = load_conditions

    @classmethod
    def from_columns(
        cls,
        columns: list[h5py.Dataset],
        index: Optional[DataIndex] = None,
        load_conditions: Optional[dict[str, bool]] = None,
    ):
        all_columns = {
            col.name.split("/")[-1]: col
            for col in columns
            if col.name.split("/")[-2] == "data"
        }

        lengths = set(len(col) for col in all_columns.values())
        if len(lengths) > 1:
            raise ValueError("Not all columns are the same length!")

        if index is None:
            index = from_size(lengths.pop())

        return Hdf5Handler(all_columns, index, load_conditions)

    def __len__(self):
        return get_length(self.__index)

    def with_index(self, index: DataIndex) -> Hdf5Handler:
        return Hdf5Handler(self.__columns, index, self.__load_conditions)

    @property
    def in_memory(self) -> bool:
        return self.__in_memory

    @property
    def load_conditions(self) -> Optional[dict[str, bool]]:
        return self.__load_conditions

    def get_uuids(self) -> dict[str, UUID]:
        return {name: get_hdf5_column_uuid(col) for name, col in self.__columns.items()}

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None):
        if len(other) == 0:
            return Hdf5Handler(self.__columns, other, self.__load_conditions)

        if sorted is not None:
            return self.__take_sorted(other, sorted)

        new_index = take(self.__index, other)
        return Hdf5Handler(self.__columns, new_index, self.__load_conditions)

    def __take_sorted(self, other: DataIndex, sorted: np.ndarray):
        if get_length(sorted) != get_length(self.__index):
            raise ValueError("Sorted index has the wrong length!")
        new_indices = get_data(other, sorted)

        new_raw_index = into_array(self.__index)[new_indices]
        new_index = np.sort(new_raw_index)

        return Hdf5Handler(self.__columns, new_index, self.__load_conditions)

    @property
    def data(self):
        return next(iter(self.__columns.values())).parent

    @property
    def index(self):
        return self.__index

    @cached_property
    def columns(self):
        return list(self.__columns.keys())

    @cached_property
    def descriptions(self):
        return {
            colname: column.attrs.get("description")
            for colname, column in self.__columns.items()
        }

    def mask(self, mask):
        idx = SimpleIndex(np.where(mask)[0])
        return self.take(idx)

    def __enter__(self):
        return self

    def __exit__(self, *exec_details):
        self.__columns = None
        return self.__file.close()

    def make_schema(
        self,
        columns: Iterable[str],
    ) -> Schema:
        columns = set(columns)
        data_writers = {}
        for column_name in columns:
            column = self.__columns[column_name]
            data_writers[column_name] = ColumnWriter.from_h5_dataset(
                column, self.__index, attrs=dict(column.attrs)
            )
        data_schema = make_schema("data", FileEntry.COLUMNS, columns=data_writers)

        return data_schema

    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """
        if self.__columns is None:
            raise ValueError("This file has already been closed")
        data = {}

        for colname in columns:
            data[colname] = get_data(self.__columns[colname], self.__index)
        # Ensure order is preserved
        return {name: data[name] for name in columns}

    def take_range(self, start: int, end: int, indices: np.ndarray) -> np.ndarray:
        if start < 0 or end > len(indices):
            raise ValueError("Indices out of range")
        return indices[start:end]
