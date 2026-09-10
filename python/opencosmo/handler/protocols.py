from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Protocol, Self

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np
    from opencosmo.io.schema import Schema
    from opencosmo.mpi import MPI

    from opencosmo.index import DataIndex, SimpleIndex


class DataHandler(Protocol):
    def get_data(self, columns: Iterable[str]) -> dict[str, np.ndarray]:
        """ """

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None) -> Self: ...

    def make_schema(
        self,
        columns: Iterable[str],
    ) -> Schema: ...

    """

    """

    @property
    def columns(self) -> Iterable[str]: ...
    @property
    def load_conditions(self) -> Optional[dict]: ...

    @property
    def index(self) -> DataIndex: ...

    def with_index(self, index: DataIndex) -> Self: ...


class DataCache(Protocol):
    def add_data(
        self,
        data: dict[UUID, dict[str, np.ndarray]],
        descriptions: dict[str, str],
        push_up: bool = True,
    ): ...

    def get_data(
        self, pairs: set[tuple[UUID, str]]
    ) -> dict[UUID, dict[str, np.ndarray]]: ...

    def __len__(self) -> int: ...

    def take(self, index: DataIndex) -> Self: ...

    def drop(self, columns: Iterable[str]) -> Self: ...

    def register_column_group(
        self, state_id: int, columns: dict[str, UUID]
    ) -> None: ...

    def deregister_column_group(self, state_id: int) -> None: ...

    def create_child(self) -> Self: ...

    def redistribute(
        self,
        reorder_map: SimpleIndex | None,
        length: int,
        columns_to_keep: dict[UUID, list[str]],
        comm: MPI.Comm,
    ) -> Self: ...

    @classmethod
    def empty(cls) -> Self: ...

    @property
    def columns(self) -> set[str]: ...

    @property
    def descriptions(self) -> dict[str, str]: ...

    def make_schema(self, columns: dict[str, UUID]) -> Schema: ...
