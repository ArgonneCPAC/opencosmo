from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import h5py

from opencosmo.file import FileExistance, resolve_path
from opencosmo.io.iopen import open_files
from opencosmo.io.serial import allocate, write_columns, write_metadata
from opencosmo.mpi import get_comm_world

if TYPE_CHECKING:
    from types import ModuleType

    import opencosmo as oc
    from opencosmo import collection

    from .protocols import Writeable

    mpiio: Optional[ModuleType]
    partition: Optional[Callable]

if get_comm_world() is not None:
    from opencosmo.dataset.mpi import partition
    from opencosmo.io import mpi as mpiio
else:
    mpiio = None
    partition = None

    """
    This module defines the main user-facing io functions: open and write

    open can take any number of file paths, and will always construct a single object
    (either a dataset or a collection).

    write takes exactly one path and exactly one opencosmo dataset or collection

    open works in the following way:

    1. Read headers and get dataset names and types for all files passed
    2. If there is only a single dataset, simply open it as such
    3. If there are multiple datasets, user the headers to determine
       if the dataset are compatible (i.e. capabale of existing together in
       a collection)
    4. Open all datasets individually
    5. Call the merge functionality for the appropriate collection.
    """


class COLLECTION_TYPE(Enum):
    LIGHTCONE = 0
    STRUCTURE_COLLECTION = 1
    SIMULATION_COLLECTION = 2


class MpiMode(Enum):
    """
    Enumeration of MPI decomposition modes for lightcone opens.

    SPATIAL: every rank opens every file and partitions data spatially (default, current behavior).
    REDSHIFT: distribute files across ranks by redshift step (one file per rank, if possible).
    """

    SPATIAL = "spatial"
    REDSHIFT = "redshift"


def open(
    *files: str | Path, mpi_mode: str = "spatial", **open_kwargs: bool
) -> oc.Dataset | collection.Collection:
    """
    Open a dataset or data collection from one or more opencosmo files.

    If you open a file with this function, you should generally close it
    when you're done

    .. code-block:: python

        import opencosmo as oc
        ds = oc.open("path/to/file.hdf5")
        # do work
        ds.close()

    Alternatively you can use a context manager, which will close the file
    automatically when you are done with it.

    .. code-block:: python

        import opencosmo as oc
        with oc.open("path/to/file.hdf5") as ds:
            # do work

    When you have multiple files that can be combined into a collection,
    you can use the following.

    .. code-block:: python

        import opencosmo as oc
        ds = oc.open("haloproperties.hdf5", "haloparticles.hdf5")


    Parameters
    ----------
    *files: str or pathlib.Path
        The path(s) to the file(s) to open.

    mpi_mode: str, default = "spatial"
        MPI decomposition mode for lightcone opens. One of:

        - "spatial": every rank opens every file and partitions data spatially
          (default; unchanged behavior).
        - "redshift": distribute lightcone files across ranks by data volume so
          each file is opened by exactly one rank. Ranks that receive no files
          still hold a full-schema, zero-length dataset, so ``select``/``filter``
          and scalar reductions behave identically on every rank. Nested
          (Diffsky step-then-type) lightcones fall back to spatial distribution.

        With no MPI communicator, ``mpi_mode`` is a no-op and files open normally
        on the single process.

    **open_kwargs: bool
        True/False flags that can be used to only load certain datasets from
        the files. Check the documentation for the data type you are working
        with for available flags. Will be ignored if only one file is passed
        and the file only contains a single dataset.

    Returns
    -------
    dataset : oc.Dataset or oc.Collection
        The dataset or collection opened from the file.

    Raises
    ------
    ValueError
        If mpi_mode is not one of the valid options, or if mpi_mode="redshift"
        is used with a non-lightcone dataset.

    """
    # Normalize and validate mpi_mode
    try:
        mode = MpiMode(mpi_mode)
    except ValueError:
        raise ValueError(
            f"Invalid mpi_mode '{mpi_mode}'. Must be one of: "
            f"{', '.join(m.value for m in MpiMode)}"
        )

    if len(files) == 1 and isinstance(files[0], list):
        file_list = files[0]
    else:
        file_list = list(files)
    file_list.sort()
    paths = [Path(fp).expanduser().resolve() for fp in file_list]
    return open_files(paths, open_kwargs, mpi_mode=mode)

    # For now the only way to open multiple files is with a StructureCollection


def write(path: Path, dataset: Writeable, overwrite=False, **schema_kwargs) -> None:
    """
    Write a dataset or collection to the file at the sepecified path.

    Parameters
    ----------
    file : str or pathlib.Path
        The path to the file to write to.
    dataset : oc.Dataset
        The dataset to write.
    overwrite : bool, default = False
        If the file already exists, overwrite it


    Raises
    ------
    FileExistsError
        If the file at the specified path already exists and overwrite is False
    FileNotFoundError
        If the parent folder of the ouput file does not exist
    """

    existance_requirement = FileExistance.MUST_NOT_EXIST
    if overwrite:
        existance_requirement = FileExistance.EITHER

    path = resolve_path(path, existance_requirement)

    schema = dataset.make_schema(**schema_kwargs)

    if mpiio is not None:
        return mpiio.write_parallel(path, schema)

    from opencosmo.io.schema import FileEntry
    from opencosmo.mapping.write import lower_collection_coordinates

    if schema.type == FileEntry.SIMULATION_COLLECTION:
        schema = lower_collection_coordinates(schema, canonical_raw_order=True)
    elif schema.type == FileEntry.STRUCTURE_COLLECTION:
        schema = lower_collection_coordinates(schema, canonical_raw_order=False)
    from opencosmo.io.verify import verify_structure

    verify_structure(schema)
    file = h5py.File(path, "w")
    allocate(file, schema)
    write_metadata(file, schema)
    write_columns(file, schema)
