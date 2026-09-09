from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from opencosmo.io.discover import LinkSlotKind
from opencosmo.mapping.mapping import ChunkedSlot, DatasetMatchSet

if TYPE_CHECKING:
    from uuid import UUID

    import h5py
    from opencosmo.io.discover import LinkLayout, MapLayout
    from opencosmo.mapping.mapping import MapSlot, SimpleH5pyIndex


def read_index_group(group: h5py.Group) -> "SimpleH5pyIndex":
    """Resolve a primary map slot into its live simple index dataset."""
    return group["index"]  # type: ignore[return-value]


def read_match_set(
    group: h5py.Group, layout: "MapLayout", available: "set[UUID]"
) -> DatasetMatchSet | None:
    """
    Resolve a live /map group into a DatasetMatchSet.

    ``layout`` comes from discovery, which has already validated the full /map
    structure and converted any malformation into ``FileLayout.error``. This function
    therefore assumes well-formedness and only (a) filters slots by availability and
    (b) resolves live h5py handles. Primary slot groups are resolved via
    ``read_index_group``; auxiliary sides are direct datasets.

    Slot groups are accessed by the verbatim on-disk names recorded in ``layout``
    (``layout.primary_slots`` and ``layout.aux_slots``). Names are never reconstructed
    from a parsed UUID, because the on-disk spelling is not guaranteed to match
    ``str(UUID)``.

    Primary maps are kept for every available target regardless of whether the
    reference dataset itself was opened. When the reference is absent, at least
    two primary targets must remain: target-to-target routing composes two primary
    index arrays through the reference's on-disk index without ever touching the
    reference dataset, but a lone target has nothing to compose with and is
    discarded. Auxiliary pairs whose own two endpoints are both present are always
    retained.

    ``reference_source`` always carries the reference recorded in the file, whether
    or not that dataset was opened. It names which dataset the primary maps are
    keyed against; substituting some other endpoint when the reference is absent
    would claim a routing that does not exist on disk.

    Returns None if nothing at all survives the availability check.

    Parameters
    ----------
    group : h5py.Group
        The live /map group. Only indexed into by slot name; its attrs are never
        read here, having already been consumed by discovery.
    layout : MapLayout
        The frozen layout produced by ``discover._read_map_layout`` for this group.
        Carries the reference UUID and the verbatim on-disk slot names with their
        parsed UUID identities.
    available : set[UUID]
        UUIDs of datasets that are present in the open set.

    Returns
    -------
    DatasetMatchSet | None
        The resolved match set with available endpoints only, or None if no maps
        remain after filtering.
    """

    primary_maps: dict[UUID, SimpleH5pyIndex] = {}
    for slot_name, target_uuid in layout.primary_slots:
        if target_uuid in available:
            primary_maps[target_uuid] = read_index_group(
                group[f"primary/{slot_name}"]  # type: ignore[arg-type]
            )

    if layout.reference not in available and len(primary_maps) < 2:
        primary_maps = {}

    aux_maps: dict[tuple[UUID, UUID], tuple[SimpleH5pyIndex, SimpleH5pyIndex]] = {}
    for slot_name, uuid_a, uuid_b in layout.aux_slots:
        if uuid_a in available and uuid_b in available:
            pair = group[f"auxiliary/{slot_name}"]  # type: ignore[index]
            source = pair["source"]  # type: ignore[index]
            target = pair["target"]  # type: ignore[index]
            for side, array in (("source", source), ("target", target)):
                values = array[:]
                if np.any(values < 0):
                    raise ValueError(
                        f"Malformed auxiliary map '{slot_name}': {side} indices "
                        "must be non-negative"
                    )
                if len(np.unique(values)) != len(values):
                    vals, count = np.unique(values, return_counts=True)

                    raise ValueError(
                        f"Malformed auxiliary map '{slot_name}': {side} indices "
                        "must be unique"
                    )
            aux_maps[(uuid_a, uuid_b)] = (
                source,
                target,
            )

    if not primary_maps and not aux_maps:
        return None

    return DatasetMatchSet(
        reference_source=layout.reference,
        primary_maps=primary_maps,
        aux_maps=aux_maps,
    )


def read_link_set(
    source_group: h5py.Group,
    layout: "LinkLayout",
    source_uuid: UUID,
    target_uuids: dict[str, UUID],
) -> DatasetMatchSet | None:
    """
    Resolve a live /data_linked group into a DatasetMatchSet.

    ``layout`` comes from discovery, which has already validated the full
    /data_linked structure. This function therefore only filters slots by opened
    target prefix and resolves their live h5py handles. Link mappings have a pure
    star topology: ``source_uuid`` is the reference source and every retained slot
    maps directly from it to one target. Links never invert or compose, so no
    auxiliary maps are created.

    Slot datasets are accessed by the verbatim on-disk names recorded in
    ``layout``. Names are never reconstructed from prefixes, because their on-disk
    spelling is the layout's authority. The returned datasets remain live and lazy:
    this reader does not read any link values, since discovery already validated
    their structure and link arrays may be structure-scale.

    Parameters
    ----------
    source_group : h5py.Group
        The live /data_linked group. It is indexed only by the dataset names
        recorded in ``layout``.
    layout : LinkLayout
        The frozen layout produced during discovery for ``source_group``.
    source_uuid : UUID
        UUID of the properties dataset owning the link group.
    target_uuids : dict[str, UUID]
        Mapping from on-disk link prefixes to UUIDs of target datasets that were
        opened by the caller. Slots whose prefixes are absent are omitted.

    Returns
    -------
    DatasetMatchSet | None
        A source-to-target-only match set with live slot datasets, or None if no
        target slots remain after filtering.
    """

    primary_maps: dict[UUID, MapSlot] = {}
    for slot in layout.slots:
        target_uuid = target_uuids.get(slot.prefix)
        if target_uuid is None:
            continue
        if slot.kind is LinkSlotKind.CHUNKED:
            start_name, size_name = slot.dataset_names
            primary_maps[target_uuid] = ChunkedSlot(
                source_group[start_name],  # type: ignore[arg-type]
                source_group[size_name],  # type: ignore[arg-type]
            )
        else:
            (idx_name,) = slot.dataset_names
            primary_maps[target_uuid] = source_group[idx_name]  # type: ignore[assignment]

    if not primary_maps:
        return None

    return DatasetMatchSet(
        reference_source=source_uuid,
        primary_maps=primary_maps,
        aux_maps={},
    )
