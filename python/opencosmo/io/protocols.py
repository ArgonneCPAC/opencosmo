from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import h5py

    from opencosmo.io.schema import Schema

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore


class DataWriter(Protocol):
    """
    Because DataSchemas are responsible for allocating files and producing the
    structure, a writer can always assume that the appropriate group or dataset already
    exists. If it doesn't exist, that is an error elsewhere that shoulds be resolved.
    """

    def write(self, group: h5py.File | h5py.Group): ...


class Writeable(Protocol):
    """
    In order to be writeable, an object must define a single method.
    """

    def make_schema(self, path: str) -> Schema: ...
