from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional

import h5py
import numpy as np
from pydantic import ValidationError

from opencosmo.dtypes import read_map_header
from opencosmo.header import read_header
from opencosmo.io.cache import cache_layouts, get_cached_layouts
from opencosmo.uuid import coerce_to_uuid, get_dataset_uuid

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from uuid import UUID

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.mpi import MPI


@dataclass(frozen=True)
class GroupLayout:
    """Frozen layout of a single /data group within a file."""

    path: str
    """In-file path of the group holding /data, e.g. "/" or "/scidac1" or "/halo_properties"."""

    header_path: str
    """In-file path of the group's governing header (nearest enclosing /header)."""

    header: OpenCosmoHeader
    """The header object read from the governing header group."""

    column_names: tuple[str, ...]
    """Sorted tuple of column names under this group's /data."""

    column_dtypes: tuple[str, ...]
    """str(dtype) for each column, same order as column_names."""

    column_units: tuple[str | None, ...]
    """``unit`` attribute of each column, index-aligned with column_names. None when
    the attribute is absent."""

    column_descriptions: tuple[str | None, ...]
    """``description`` attribute of each column, index-aligned with column_names. None
    when the attribute is absent."""

    row_count: int
    """Number of rows in the first column, or 0 if no /data group."""

    has_index: bool
    """Whether an /index group exists as a sibling of /data."""

    linked_target_names: tuple[str, ...]
    """Sorted tuple of linked target name prefixes (from /data_linked _start/_size suffixes)."""

    uuid: UUID
    """Identity of this dataset's /data group. Read from the group's ``main_uuid``
    attribute when present, otherwise deterministically synthesized from the resolved
    file path and the group path so every MPI rank agrees without communication."""

    has_persistent_uuid: bool
    """Whether ``uuid`` came from an on-disk ``main_uuid`` attribute. Synthesized
    UUIDs must never satisfy a /map endpoint check, so consumers that resolve
    mapping files filter on this rather than on ``uuid``."""

    link_layout: LinkLayout | None = None
    """Frozen /data_linked layout when this group has one, otherwise None."""


class LinkSlotKind(StrEnum):
    """The index representation used by a /data_linked slot."""

    CHUNKED = "chunked"
    SIMPLE = "simple"


@dataclass(frozen=True)
class LinkSlot:
    """Frozen layout of one prefix-keyed /data_linked slot."""

    prefix: str
    """The slot prefix, retained verbatim from the on-disk dataset names."""

    kind: LinkSlotKind
    """Whether this slot uses a start/size pair or a simple idx array."""

    dataset_names: tuple[str, ...]
    """Verbatim on-disk dataset names, in representation order."""

    length: int
    """Number of rows in this slot's index representation."""


@dataclass(frozen=True)
class LinkLayout:
    """Frozen layout of a /data_linked group for structure links.

    Slots are keyed by their verbatim prefixes because /data_linked stores flat
    sibling datasets rather than UUID-named slot groups. The recorded names let a
    later reader re-open the exact datasets without reconstructing their spelling.
    """

    path: str
    """In-file path of the /data_linked group, e.g. "/data_linked"."""

    slots: tuple[LinkSlot, ...]
    """Sorted prefix-keyed link slots in this group."""


@dataclass(frozen=True)
class MapLayout:
    """Frozen layout of a /map group for dataset mapping.

    Records the endpoints (which datasets this map can connect) and the verbatim
    on-disk slot names needed to re-open them. Dataset mappings support only simple
    one-to-one ``index`` arrays; chunked ``start``/``size`` indexes belong to other
    OpenCosmo indexing use cases and are rejected during discovery.
    """

    path: str
    """In-file path of the /map group, e.g. "/map"."""

    reference: UUID
    """UUID of the reference dataset, from the /map group's 'reference' attribute."""

    primary_slots: tuple[tuple[str, UUID], ...]
    """Sorted (on-disk slot name, target UUID) pairs under /primary — the reference->target maps.

    The slot name is kept verbatim because nothing in this repo writes maps, so the
    on-disk spelling of a UUID is not guaranteed to match ``str(UUID)``. The reader
    indexes back into the live group by name; it must never reconstruct one.
    """

    primary_lengths: tuple[tuple[UUID, int], ...]
    """Logical length of each primary map, keyed by target UUID."""

    aux_slots: tuple[tuple[str, UUID, UUID], ...]
    """Sorted (on-disk slot name, uuid_a, uuid_b) triples under /auxiliary.

    Name kept verbatim for the same reason as ``primary_slots``.
    """

    @property
    def endpoints(self) -> frozenset[UUID]:
        """Every dataset UUID this map mentions, reference included."""
        return frozenset(
            (self.reference,)
            + tuple(target for _, target in self.primary_slots)
            + tuple(u for _, uuid_a, uuid_b in self.aux_slots for u in (uuid_a, uuid_b))
        )


@dataclass(frozen=True)
class FileLayout:
    """Frozen layout of a complete file."""

    path: Path
    """Path to the file."""

    groups: tuple[GroupLayout, ...]
    """Sorted tuple of GroupLayout objects found in this file."""

    error: Optional[str] = None
    """Error message if discovery failed, None if successful."""

    maps: tuple[MapLayout, ...] = ()
    """Sorted tuple of MapLayout objects found in this file."""


def _make_group_map(
    group: h5py.File | h5py.Group, prefix: str = ""
) -> dict[str, h5py.File | h5py.Group | h5py.Dataset]:
    """
    Build a flat map of all h5py objects in a file/group tree.
    Keys are in-file paths (e.g., "/header", "/scidac1/data").
    """
    index = {}
    for key, item in group.items():
        path = f"{prefix}/{key}"
        index[path] = item
        if isinstance(item, h5py.Group):
            index.update(_make_group_map(item, path))
    return index


def _normalize_attr(value: Any) -> str | None:
    """Normalize an HDF5 attribute value to a plain str, preserving absence.

    h5py may return attribute values as ``bytes``, ``np.bytes_``, ``str``, or
    ``np.str_`` (including as a 0-d numpy object). ``None`` (attribute absent)
    is passed through unchanged rather than collapsed with an empty string.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, (str, np.str_)):
        return str(value)
    return str(value)


def _header_scope(header_path: str) -> str:
    """Group path a "/header" governs (its parent): "/header" -> "/"."""
    return header_path.rsplit("/header", 1)[0] or "/"


def _iter_ancestors(group_path: str) -> Iterator[str]:
    """Yield ``group_path`` then each ancestor up to the root, deepest first."""
    yield group_path
    while group_path != "/":
        group_path = group_path.rsplit("/", 1)[0] or "/"
        yield group_path


def _verify_map_array(array: h5py.Dataset | None, where: str) -> int:
    """Validate a direct, simple mapping array without reading its values."""
    if not isinstance(array, h5py.Dataset):
        raise ValueError(f"{where}: expected a dataset")
    if array.ndim != 1:
        raise ValueError(f"{where}: mapping arrays must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{where}: mapping arrays must have an integer dtype")
    return array.shape[0]


def _verify_slot(slot_group: h5py.Group, where: str) -> int:
    """
    Check that one map slot is well formed, keeping nothing.

    Dataset mappings support only a simple ``index`` dataset. The point of the walk is
    to turn a malformed or unsupported slot into ``FileLayout.error`` during discovery
    rather than a raise during matching. Raises ValueError; ``discover_file`` converts
    it.
    """
    keys = set(slot_group.keys())
    if "start" in keys or "size" in keys:
        raise ValueError(
            f"{where}: dataset mappings require a simple 'index' array; chunked "
            "'start'/'size' mappings are not supported"
        )
    if "index" not in keys:
        raise ValueError(f"{where}: dataset mapping slot is missing its 'index' array")

    array = slot_group["index"]
    if not isinstance(array, h5py.Dataset):
        raise ValueError(f"{where}: index is not a dataset")
    if array.ndim != 1:
        raise ValueError(f"{where}: mapping arrays must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{where}: mapping arrays must have an integer dtype")
    return array.shape[0]


def _read_link_layout(
    link_path: str,
    link_group: h5py.Group,
) -> LinkLayout:
    """Resolve one /data_linked group into a frozen LinkLayout.

    Link slots are flat sibling datasets, unlike the group-based /map slots. Reads
    names, dtypes, and shapes only. Raises ValueError on malformed structure;
    ``discover_file`` converts it to ``FileLayout.error`` inside the collective.
    """
    slot_datasets: dict[str, dict[str, str]] = {}
    for name, item in link_group.items():
        suffix = next(
            (suffix for suffix in ("_start", "_size", "_idx") if name.endswith(suffix)),
            None,
        )
        where = f"{link_path}/{name}"
        if suffix is None:
            raise ValueError(
                f"{where}: link datasets must end in _start, _size, or _idx"
            )
        prefix = name.removesuffix(suffix)
        if not prefix:
            raise ValueError(f"{where}: link dataset prefix must not be empty")
        if not isinstance(item, h5py.Dataset):
            raise ValueError(f"{where}: expected a dataset")
        slot_datasets.setdefault(prefix, {})[suffix] = name

    slots: list[LinkSlot] = []
    for prefix, names in sorted(slot_datasets.items()):
        where = f"{link_path}/{prefix}"
        has_start = "_start" in names
        has_size = "_size" in names
        has_idx = "_idx" in names
        if has_start != has_size:
            missing = "_size" if has_start else "_start"
            raise ValueError(f"{where}: missing matching {missing} dataset")
        if has_idx and has_start:
            raise ValueError(
                f"{where}: cannot contain both start/size and idx datasets"
            )
        if has_start:
            start_name = names["_start"]
            size_name = names["_size"]
            start_length = _verify_map_array(
                link_group[start_name], f"{link_path}/{start_name}"
            )
            size_length = _verify_map_array(
                link_group[size_name], f"{link_path}/{size_name}"
            )
            if start_length != size_length:
                raise ValueError(
                    f"{where}: start and size arrays must have the same length"
                )
            slots.append(
                LinkSlot(
                    prefix=prefix,
                    kind=LinkSlotKind.CHUNKED,
                    dataset_names=(start_name, size_name),
                    length=start_length,
                )
            )
        elif has_idx:
            idx_name = names["_idx"]
            slots.append(
                LinkSlot(
                    prefix=prefix,
                    kind=LinkSlotKind.SIMPLE,
                    dataset_names=(idx_name,),
                    length=_verify_map_array(
                        link_group[idx_name], f"{link_path}/{idx_name}"
                    ),
                )
            )
        else:
            raise ValueError(f"{where}: link slot has no index datasets")

    return LinkLayout(path=link_path, slots=tuple(slots))


def _read_map_layout(
    map_path: str,
    map_group: h5py.Group,
) -> MapLayout:
    """
    Resolve one /map group into a frozen MapLayout.

    Reads attrs, names, and shapes only — never column values — so discovery stays
    within its existing single-allgather budget. Raises ValueError on any malformed
    structure; ``discover_file`` turns that into ``FileLayout.error`` rather than
    letting it escape, since discovery runs inside a collective.
    """
    attrs = dict(map_group.attrs)

    reference = coerce_to_uuid(attrs.get("reference"))
    if reference is None:
        raise ValueError(
            f"Malformed map at {map_path}: missing or invalid 'reference' attribute"
        )

    format_version = attrs.get("format_version")
    if format_version is None:
        raise ValueError(
            f"Malformed map at {map_path}: missing 'format_version' attribute"
        )
    if not isinstance(format_version, (int, np.integer)) or int(format_version) != 1:
        raise ValueError(
            f"Malformed map at {map_path}: unsupported format_version "
            f"{format_version!r}; only version 1 is supported"
        )

    # /primary/<target_uuid> — the reference->target maps.
    # Slot names are kept verbatim: on-disk UUID spelling is not guaranteed to match
    # str(UUID), so the reader must index back by the recorded name, never reconstruct.
    primary_slots: list[tuple[str, UUID]] = []
    primary_lengths: list[tuple[UUID, int]] = []
    known_primaries: set[UUID] = set()
    if isinstance(primary_group := map_group.get("primary"), h5py.Group):
        for name, slot in primary_group.items():
            where = f"{map_path}/primary/{name}"
            target = coerce_to_uuid(name)
            if target is None:
                raise ValueError(f"{where}: group name is not a UUID")
            if not isinstance(slot, h5py.Group):
                raise ValueError(f"{where}: expected a group")
            if target in known_primaries:
                raise ValueError(
                    f"{where}: duplicate primary mapping for UUID {target}"
                )
            length = _verify_slot(slot, where)
            primary_slots.append((name, target))
            primary_lengths.append((target, length))
            known_primaries.add(target)

    # /auxiliary/<uuid_a>__<uuid_b> — pairs that do not route through the reference.
    # Slot names are kept verbatim for the same reason as primary_slots.
    aux_slots: list[tuple[str, UUID, UUID]] = []
    known_aux_pairs: set[frozenset[UUID]] = set()
    if isinstance(aux_group := map_group.get("auxiliary"), h5py.Group):
        for name, pair in aux_group.items():
            where = f"{map_path}/auxiliary/{name}"
            # UUID hex never contains "__", so the separator is unambiguous.
            parts = name.split("__")
            if len(parts) != 2:
                raise ValueError(f"{where}: name is not '<uuid_a>__<uuid_b>'")
            uuid_a, uuid_b = (coerce_to_uuid(p) for p in parts)
            if uuid_a is None or uuid_b is None:
                raise ValueError(f"{where}: endpoint names are not UUIDs")
            if not isinstance(pair, h5py.Group):
                raise ValueError(f"{where}: expected a group")
            if uuid_a not in known_primaries or uuid_b not in known_primaries:
                raise ValueError(
                    "Found auxiliary matching groups without corresponding primaries!"
                )

            logical_pair = frozenset((uuid_a, uuid_b))
            if logical_pair in known_aux_pairs:
                raise ValueError(f"{where}: duplicate auxiliary mapping pair")
            known_aux_pairs.add(logical_pair)

            side_lengths = [
                _verify_map_array(pair.get(side), f"{where}/{side}")
                for side in ("source", "target")
            ]
            if side_lengths[0] != side_lengths[1]:
                raise ValueError(
                    f"{where}: auxiliary source and target must have the same length"
                )

            aux_slots.append((name, uuid_a, uuid_b))

    return MapLayout(
        path=map_path,
        reference=reference,
        # Sorted so every rank builds a byte-identical layout.
        primary_slots=tuple(sorted(primary_slots)),
        primary_lengths=tuple(sorted(primary_lengths)),
        aux_slots=tuple(sorted(aux_slots)),
    )


def discover_file(path: Path) -> FileLayout:
    """
    Walk a file once and produce a frozen, picklable layout with no live h5py handles.

    Reuses the walk logic from iopen's __make_group_map. Steps:
    1. Open with h5py.File(path, "r") in a with block.
    2. Build the group map.
    3. Find every /header group, every /data group, and every /map group.
    4. Read each header once, locally, via read_header.__wrapped__ (no world-comm
       broadcast), keyed by the group scope it governs.
    5. For each /data group, resolve its governing header as the nearest enclosing
       scope on the group's own ancestry.
    6. Extract column metadata (names, dtypes, row_count) for each group.
    7. Check for /index and /data_linked groups.
    8. Parse /map groups and extract mapping layouts.
    9. Sort groups and maps by path for determinism.

    Validity rule: a file is valid if it has (headers AND data) OR it has maps.
    A file with neither returns an error.

    On any structural failure, return FileLayout(path=path, groups=(), error=<message>) — never raise.
    """
    try:
        with h5py.File(path, "r") as f:
            file_map = _make_group_map(f)

            # Find all header and data groups.
            header_groups = sorted([k for k in file_map.keys() if k.endswith("header")])
            data_groups = sorted([k for k in file_map.keys() if k.endswith("/data")])
            map_groups = sorted([k for k in file_map.keys() if k.endswith("/map")])

            # Maps are parsed first: a mapping file has no header and no data, so
            # its validity cannot be decided until we know whether it carries maps.
            map_layouts: list[MapLayout] = []
            for map_path in map_groups:
                map_group = file_map[map_path]
                if not isinstance(map_group, h5py.Group):
                    continue
                try:
                    map_layouts.append(_read_map_layout(map_path, map_group))
                except ValueError as e:
                    return FileLayout(path=path, groups=(), error=str(e))

            maps = tuple(sorted(map_layouts, key=lambda m: m.path))

            # A mapping file carries a minimal, dataset-agnostic header
            # identifying it as an OpenCosmo file. Validate it here so a foreign
            # HDF5 file that merely happens to contain a "/map" group is rejected
            # with the same contract as every other malformed OpenCosmo file.
            # The header is only checked, never retained.
            if maps and len(data_groups) == 0:
                try:
                    read_map_header(f)
                except (KeyError, ValidationError) as e:
                    return FileLayout(
                        path=path,
                        groups=(),
                        error=f"Malformed mapping header: {e}",
                    )

            # A file is valid if it carries header+data or if it carries maps. A
            # mapping file has neither header nor data — it is not owned by any
            # simulation — so the old "must have both" rule is widened rather than
            # replaced: a file with neither still errors exactly as it did before.
            if not maps:
                if not header_groups:
                    return FileLayout(
                        path=path, groups=(), error="No header groups found in file"
                    )
                if not data_groups:
                    return FileLayout(
                        path=path, groups=(), error="No data groups found in file"
                    )

            # A pure mapping file stops here: there are no headers to read and no
            # data groups to describe.
            if not header_groups or not data_groups:
                return FileLayout(path=path, groups=(), maps=maps)

            # Otherwise, process headers and data groups as usual.
            read_header_local = read_header.__wrapped__  # type: ignore[attr-defined]
            scope_to_header: dict[str, tuple[str, OpenCosmoHeader]] = {}
            for header_path in header_groups:
                try:
                    header = read_header_local(file_map[header_path].parent)
                except (KeyError, ValueError, TypeError) as e:
                    return FileLayout(
                        path=path,
                        groups=(),
                        error=f"Malformed header at {header_path}: {e}",
                    )
                scope_to_header[_header_scope(header_path)] = (header_path, header)

            group_layouts = []
            for data_path in data_groups:
                # The owning group is the parent of /data ("/scidac1/data" -> "/scidac1").
                group_path = data_path.rsplit("/data", 1)[0] or "/"

                # Governing header = nearest enclosing scope on this group's ancestry.
                governing_scope = next(
                    (a for a in _iter_ancestors(group_path) if a in scope_to_header),
                    None,
                )
                if governing_scope is None:
                    return FileLayout(
                        path=path,
                        groups=(),
                        error=f"No header governs data group {data_path}",
                    )
                governing_header_path, header = scope_to_header[governing_scope]

                # Extract column metadata for this data group.
                data_prefix = data_path if data_path.endswith("/") else data_path + "/"
                columns_in_data = sorted(
                    [
                        k.split(data_prefix, 1)[1]
                        for k in file_map.keys()
                        if k.startswith(data_prefix)
                        and isinstance(file_map[k], h5py.Dataset)
                        and not k.endswith("/index")
                        and "header" not in k
                    ]
                )

                column_names = tuple(columns_in_data)
                column_handles = [
                    file_map[f"{data_path}/{col}"] for col in columns_in_data
                ]
                column_dtypes = tuple(str(h.dtype) for h in column_handles)
                column_units = tuple(
                    _normalize_attr(h.attrs.get("unit")) for h in column_handles
                )
                column_descriptions = tuple(
                    _normalize_attr(h.attrs.get("description")) for h in column_handles
                )

                # Every column in a group must agree on length: readers build a
                # single row index for the whole group. Shapes are already
                # fetched by the walk above, so checking here is free.
                row_count = 0
                if column_handles:
                    row_counts = {h.shape[0] for h in column_handles}
                    if len(row_counts) > 1:
                        return FileLayout(
                            path=path,
                            groups=(),
                            error=f"Not all columns in {data_path} are the same "
                            f"length: found lengths {sorted(row_counts)}",
                        )
                    row_count = column_handles[0].shape[0]

                # Check for index group.
                group_parent = (
                    group_path if group_path.endswith("/") else group_path + "/"
                )
                index_path = f"{group_parent}index"
                has_index = index_path in file_map

                # Check for data_linked group and extract its frozen slot layout.
                linked_target_names_set: set[str] = set()
                link_layout: LinkLayout | None = None
                data_linked_path = f"{group_parent}data_linked"
                if data_linked_path in file_map:
                    data_linked_group = file_map[data_linked_path]
                    if isinstance(data_linked_group, h5py.Group):
                        try:
                            link_layout = _read_link_layout(
                                data_linked_path, data_linked_group
                            )
                        except ValueError as e:
                            return FileLayout(path=path, groups=(), error=str(e))
                        linked_target_names_set.update(
                            slot.prefix for slot in link_layout.slots
                        )

                linked_target_names = tuple(sorted(linked_target_names_set))
                # Hash the /data group itself so parent groups' path prefixes do not
                # affect identity.
                data_group = file_map[data_path]
                if not isinstance(data_group, h5py.Group):
                    return FileLayout(
                        path=path,
                        groups=(),
                        error=f"Expected /data group at {data_path}",
                    )

                persistent_uuid = coerce_to_uuid(data_group.attrs.get("main_uuid"))
                group_uuid = get_dataset_uuid(data_group)

                group_layouts.append(
                    GroupLayout(
                        path=group_path,
                        header_path=governing_header_path,
                        header=header,
                        uuid=group_uuid,
                        has_persistent_uuid=persistent_uuid is not None,
                        column_names=column_names,
                        column_dtypes=column_dtypes,
                        column_units=column_units,
                        column_descriptions=column_descriptions,
                        row_count=row_count,
                        has_index=has_index,
                        linked_target_names=linked_target_names,
                        link_layout=link_layout,
                    )
                )

            # Sort groups and maps by path for determinism.
            sorted_groups = tuple(sorted(group_layouts, key=lambda g: g.path))
            return FileLayout(path=path, groups=sorted_groups, maps=maps)

    except Exception as e:
        return FileLayout(path=path, groups=(), error=str(e))


def discover_all(
    paths: list[Path],
    comm: Optional["MPI.Comm"] = None,
) -> tuple[FileLayout, ...]:
    """
    Discover metadata from all files, distributing the walk across MPI ranks.

    Rank ``i`` walks ``paths[i::nranks]`` (round-robin), then a single
    ``comm.allgather`` assembles the full list on every rank. The result is sorted
    by path, so every rank holds a byte-identical tuple. This one path covers every
    case uniformly: serial (no comm), a single file (only rank 0's slice is
    non-empty), and fewer files than ranks (surplus ranks contribute nothing).

    The only irreducible special case is the absence of a communicator: with no
    ``comm`` there is nothing to allgather, so the local walk is the final result.

    Layouts are memoized on disk via :mod:`opencosmo.io.cache`, keyed by path and
    mtime. Cache misses fall through to :func:`discover_file`, and all cache
    errors degrade to a full walk.
    """
    rank = comm.Get_rank() if comm is not None else 0
    nranks = comm.Get_size() if comm is not None else 1

    my_files = paths[rank::nranks]

    try:
        cached_layouts = get_cached_layouts(my_files)
    except Exception:
        logger.debug("cache read failed in discover_all", exc_info=True)
        cached_layouts = {}

    fresh_layouts = [discover_file(p) for p in my_files if p not in cached_layouts]

    try:
        # Safe to write per-rank: round-robin assignment (paths[rank::nranks])
        # guarantees that no two ranks ever discover or cache the same file.
        cache_layouts(fresh_layouts)
    except Exception:
        logger.debug("cache write failed in discover_all", exc_info=True)

    my_layouts = fresh_layouts + list(cached_layouts.values())

    if comm is not None:
        my_layouts = [
            layout
            for rank_layouts in comm.allgather(my_layouts)
            for layout in rank_layouts
        ]

    return tuple(sorted(my_layouts, key=lambda fl: str(fl.path)))


def group_data_type(group: GroupLayout) -> str:
    """Return the data_type from the group's header."""
    return str(group.header.file.data_type)


def is_lightcone_group(group: GroupLayout) -> bool:
    """Return True if the group's header marks it as a lightcone."""
    return bool(group.header.file.is_lightcone)


def has_linked_targets(group: GroupLayout) -> bool:
    """Return True if the group has any linked target names."""
    return len(group.linked_target_names) > 0


def is_healpix_map_group(group: GroupLayout) -> bool:
    """Return True if the group's data_type is healpix_map."""
    return group_data_type(group) == "healpix_map"


def is_particle_group(group: GroupLayout) -> bool:
    """Return True if the group's data_type contains 'particle'."""
    return "particle" in group_data_type(group)


def is_properties_group(group: GroupLayout) -> bool:
    """Return True if the group's data_type is halo_properties or galaxy_properties."""
    return group_data_type(group) in ("halo_properties", "galaxy_properties")


def has_maps(layout: FileLayout) -> bool:
    """Return True if the file layout contains any map groups."""
    return len(layout.maps) > 0


def encode_file_layout_blob(
    layout: FileLayout, *, sha256: bool = True
) -> dict[str, Any]:
    """Encode a :class:`FileLayout` into a JSON-safe SQLite blob dict.

    The returned dict is ready for :func:`json.dumps`.
    """
    groups_json = [group_to_dict(g) for g in layout.groups]
    maps_json = [map_to_dict(m) for m in layout.maps]

    blob: dict[str, Any] = {
        "path": str(layout.path),
        "error": layout.error,
        "groups": groups_json,
        "maps": maps_json,
    }
    if sha256:
        canon = json.dumps(
            blob, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        blob["sha256"] = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return blob


def decode_file_layout_blob(blob: dict[str, Any]) -> FileLayout | None:
    """Decode a SQLite blob dict back into :class:`FileLayout`.

    If ``sha256`` is present, it is validated against the canonical JSON encoding
    of the decoded blob fields (excluding ``sha256`` itself).
    """

    expected = blob.get("sha256")
    if expected is not None:
        test_blob = dict(blob)
        test_blob.pop("sha256", None)
        canon = json.dumps(
            test_blob, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        actual = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        if actual != expected:
            return None

    from pathlib import Path

    try:
        path = Path(blob["path"])
        error = blob.get("error")
        groups = tuple(group_from_dict(g) for g in blob.get("groups", []))
        maps = tuple(map_from_dict(m) for m in blob.get("maps", []))

        return FileLayout(path=path, groups=groups, error=error, maps=maps)
    except KeyError:
        # A cached blob written before these fields were introduced is missing
        # keys that group_from_dict now requires. Treat it as a cache miss
        # rather than raising.
        return None


def group_to_dict(group: GroupLayout) -> dict[str, Any]:
    return {
        "path": group.path,
        "header_path": group.header_path,
        "header": group.header.to_dict(),
        "column_names": list(group.column_names),
        "column_dtypes": list(group.column_dtypes),
        "column_units": list(group.column_units),
        "column_descriptions": list(group.column_descriptions),
        "row_count": group.row_count,
        "has_index": group.has_index,
        "linked_target_names": list(group.linked_target_names),
        "uuid": str(group.uuid),
        "has_persistent_uuid": group.has_persistent_uuid,
        "link_layout": link_to_dict(group.link_layout) if group.link_layout else None,
    }


def group_from_dict(data: dict[str, Any]) -> GroupLayout:
    from uuid import UUID

    from opencosmo.header import OpenCosmoHeader

    return GroupLayout(
        path=data["path"],
        header_path=data["header_path"],
        header=OpenCosmoHeader.from_dict(data["header"]),
        column_names=tuple(data["column_names"]),
        column_dtypes=tuple(data["column_dtypes"]),
        column_units=tuple(data["column_units"]),
        column_descriptions=tuple(data["column_descriptions"]),
        row_count=int(data["row_count"]),
        has_index=bool(data["has_index"]),
        linked_target_names=tuple(data["linked_target_names"]),
        uuid=UUID(data["uuid"]),
        has_persistent_uuid=bool(data["has_persistent_uuid"]),
        link_layout=link_from_dict(data["link_layout"])
        if data.get("link_layout")
        else None,
    )


def link_to_dict(layout: LinkLayout | None) -> dict[str, Any]:
    if layout is None:
        raise ValueError("link_to_dict called with None")
    return {
        "path": layout.path,
        "slots": [slot_to_dict(s) for s in layout.slots],
    }


def link_from_dict(data: dict[str, Any]) -> LinkLayout:
    return LinkLayout(
        path=data["path"],
        slots=tuple(slot_from_dict(s) for s in data["slots"]),
    )


def slot_to_dict(slot: LinkSlot) -> dict[str, Any]:
    return {
        "prefix": slot.prefix,
        "kind": slot.kind.value,
        "dataset_names": list(slot.dataset_names),
        "length": slot.length,
    }


def slot_from_dict(data: dict[str, Any]) -> LinkSlot:
    return LinkSlot(
        prefix=data["prefix"],
        kind=LinkSlotKind(data["kind"]),
        dataset_names=tuple(data["dataset_names"]),
        length=int(data["length"]),
    )


def map_to_dict(map_layout: MapLayout) -> dict[str, Any]:
    return {
        "path": map_layout.path,
        "reference": str(map_layout.reference),
        "primary_slots": [
            [name, str(uuid_)] for name, uuid_ in map_layout.primary_slots
        ],
        "primary_lengths": [
            [str(uuid_), length] for uuid_, length in map_layout.primary_lengths
        ],
        "aux_slots": [[name, str(a), str(b)] for name, a, b in map_layout.aux_slots],
    }


def map_from_dict(data: dict[str, Any]) -> MapLayout:
    from uuid import UUID

    primary_slots = tuple(
        (name, UUID(uuid_str)) for name, uuid_str in data["primary_slots"]
    )
    primary_lengths = tuple(
        (UUID(uuid_str), int(length)) for uuid_str, length in data["primary_lengths"]
    )
    aux_slots = tuple((name, UUID(a), UUID(b)) for name, a, b in data["aux_slots"])
    return MapLayout(
        path=data["path"],
        reference=UUID(data["reference"]),
        primary_slots=primary_slots,
        primary_lengths=primary_lengths,
        aux_slots=aux_slots,
    )
