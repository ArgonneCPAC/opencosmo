from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from uuid import UUID, uuid5

import numpy as np

if TYPE_CHECKING:
    import h5py

NAMESPACE = UUID("02af60eb-68cc-43f2-817b-eb0967ebfde4")


def coerce_to_uuid(value: str | bytes | np.bytes_ | UUID | None) -> UUID | None:
    """
    Coerce a value to UUID, handling multiple input formats.

    Returns None if the value is None, not a UUID-like type, or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return UUID(value)
        except (ValueError, AttributeError):
            return None
    return None


def get_path_uuid(path: Path) -> UUID:
    return uuid5(NAMESPACE, str(path))


def get_dataset_uuid(group: h5py.Group) -> UUID:
    """
    Return the runtime identity of a dataset's /data group.

    Uses the on-disk ``main_uuid`` attribute when it is present and parseable.
    Otherwise synthesizes one from the resolved file path and the group path, so
    that repeated opens and independent MPI ranks agree without communication.
    An unparseable ``main_uuid`` is treated as an absent one.
    """
    main_uuid = coerce_to_uuid(group.attrs.get("main_uuid"))
    if main_uuid is not None:
        return main_uuid

    file_path = Path(group.file.filename).resolve()
    return uuid5(NAMESPACE, f"{file_path}::{group.name}")


def get_in_memory_dataset_uuid(data_names: Iterable[str]) -> UUID:
    """
    Mint the identity of a dataset built from in-memory columns.

    Deterministic over the sorted column names so that independent MPI ranks
    building the same dataset agree without communication. Two structurally
    identical in-memory datasets therefore share an identity.
    """
    id = f"in_memory::data={','.join(sorted(data_names))}"
    return uuid5(NAMESPACE, id)


def get_hdf5_column_uuid(column: h5py.Dataset) -> UUID:
    file_name = column.file.filename
    path = column.name

    id = f"{file_name}::{path}"

    return uuid5(NAMESPACE, id)


def get_derived_column_uuid(
    dep_uuids: Iterable[UUID], output_names: Iterable[str], all_known_uuids: set[UUID]
):
    dep_uuids_str = [str(uuid) for uuid in sorted(dep_uuids)]
    output_names = sorted(output_names)
    id = f"{'+'.join(dep_uuids_str)}->{'+'.join(output_names)}"
    uuid = uuid5(NAMESPACE, id)
    return _get_free(uuid, all_known_uuids)


def get_raw_column_uuid(output_name: str, all_known_uuids: set[UUID]):
    id = f"raw::{output_name}"

    uuid = uuid5(NAMESPACE, id)
    return _get_free(uuid, all_known_uuids)


def _get_free(uuid: UUID, all_known_uuids: set[UUID]):
    while uuid in all_known_uuids:
        uuid = uuid5(NAMESPACE, str(uuid))
    return uuid
