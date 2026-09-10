from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Self

from opencosmo.io.schema import FileEntry, make_schema

from opencosmo.index import empty

if TYPE_CHECKING:
    import numpy as np

    from opencosmo.index import DataIndex


class EmptyHandler:
    def get_data(self, *args):
        return {}

    def take(self, other: DataIndex, sorted: Optional[np.ndarray] = None) -> Self:
        return self

    def with_index(self, index):
        return self

    def __len__(self) -> int:
        return 0

    def make_schema(self, *args, **kwargs):
        return make_schema("data", FileEntry.EMPTY)

    @property
    def columns(self) -> Iterable[str]:
        return set()

    @property
    def load_conditions(self):
        return None

    @property
    def descriptions(self):
        return {}

    @property
    def index(self) -> DataIndex:
        return empty()
