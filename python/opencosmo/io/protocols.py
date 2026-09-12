from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    import h5py

    from opencosmo.io.schema import Schema

try:
    from mpi4py import MPI
except ImportError:
    MPI = None  # type: ignore


class DataSchema(Protocol):
    """
    A DataSchema describes the layout of a file. Like HDF5, the schemas are organized
    hierarchicaly. As a result most schemas are only responsible for holding children
    and performing any verification that must be done across multiple children. For
    example, a DatasetSchema holds ColumnSchemas, and is responsible for verifying that
    all of its columns have the same length.

    A schema most be capable of transforming into a writer. As such, some schemas such
    as the ColumnSchema must hold reference to the data that wthey will write, as well
    as an index into the data of the elements that should be included.

    Schemas also must allocate their respective structures, i.e. by creating groups or
    datasets.
    """

    def add_child(self, child: "DataSchema", child_id: Any): ...
    def allocate(self, group: h5py.File | h5py.Group): ...
    def verify(self): ...
    def into_writer(self, comm: Optional["MPI.Comm"]): ...


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
