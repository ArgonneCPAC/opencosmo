from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

import numpy as np

from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.structure import structure as sc
from opencosmo.dataset import dataset as ocds
from opencosmo.index import coalesce_chunks, into_array, offset
from opencosmo.index.build import empty
from opencosmo.mapping.mapping import (
    get_mapping,
    get_slot_sizes,
    is_chunked_slot,
    rebuild_target_index,
)

if TYPE_CHECKING:
    from uuid import UUID

    import opencosmo as oc
    from opencosmo.index import DataIndex
    from opencosmo.mapping.mapping import DatasetMatchSet


"""
A tale in 3 acts:

Act 1: Patrick creates a LinkHandler which holds a pointer to the index that determines
which rows in halo/galaxy properties corresponds to rows in particle datasets.

Act 2: Patrick unifies metadata handling in Datasets, making the link handler
unncessary. Instead, ephemeral particle datasets are created when requested.

Act 3: Patrick realizes this solution makes it impossible to cache things, particularly
very expensive computations that the user has created with evaluate. Patrick re-introduces
the LinkHandler, but it's better this time or something.

"""

LINK_ALIASES = {  # Left: Name in file, right: Name in collection
    "sodbighaloparticles_star_particles": "star_particles",
    "sodbighaloparticles_dm_particles": "dm_particles",
    "sodbighaloparticles_gravity_particles": "gravity_particles",
    "sodbighaloparticles_agn_particles": "agn_particles",
    "sodbighaloparticles_gas_particles": "gas_particles",
    "fofbighaloparticles_star_particles": "star_particles",
    "fofbighaloparticles_dm_particles": "dm_particles",
    "fofbighaloparticles_gravity_particles": "gravity_particles",
    "fofbighaloparticles_agn_particles": "agn_particles",
    "fofbighaloparticles_gas_particles": "gas_particles",
    "sod_profile": "halo_profiles",
    "haloprofiles": "halo_profiles",
    "galaxyproperties": "galaxy_properties",
    "galaxyparticles_star_particles": "star_particles",
}


def _target_uuid(match_set: DatasetMatchSet, name: str) -> UUID:
    target_uuid = match_set.get_uuid(name)
    if target_uuid is None:
        raise ValueError(f"Unable to resolve link '{name}'")
    return target_uuid


def _slot_values(
    source: oc.Dataset, match_set: DatasetMatchSet, name: str, index: DataIndex
) -> tuple[np.ndarray, bool]:
    target_uuid = _target_uuid(match_set, name)
    if is_chunked_slot(match_set, target_uuid):
        return get_slot_sizes(match_set, target_uuid, index), True
    mapping = get_mapping(match_set, match_set.reference_source, target_uuid, index)
    assert mapping is not None
    return np.asarray(mapping), False


def link_slot_values(
    match_sets: dict[UUID, DatasetMatchSet],
    source: oc.Dataset | oc.Lightcone,
    name: str,
) -> tuple[np.ndarray, bool]:
    """Return per-source-row slot values for link ``name`` and whether it is chunked.

    For a chunked slot the values are the ``size`` column; for a simple slot they
    are the raw ``idx`` column including ``-1`` sentinels. Values are aligned with
    ``source``'s logical row order, matching ``get_metadata``.
    """
    if isinstance(source, ocds.Dataset):
        return _slot_values(source, match_sets[source.uuid], name, source.index)

    values: list[np.ndarray] = []
    chunked: Optional[bool] = None
    for _, step_source in source.items():
        step_values, step_chunked = _slot_values(
            step_source, match_sets[step_source.uuid], name, step_source.index
        )
        if chunked is not None and chunked != step_chunked:
            raise ValueError(f"Link '{name}' has inconsistent slot kinds across steps")
        chunked = step_chunked
        values.append(step_values)

    output = np.concatenate(values)
    sort_key = source._Lightcone__sort_key
    if sort_key is not None:
        order = np.argsort(source.select(sort_key[0]).get_data("numpy"))
        if sort_key[1]:
            order = order[::-1]
        output = output[order]
    assert chunked is not None
    return output, chunked


def compute_sort_index(source: oc.Dataset) -> np.ndarray:
    """Build an index from a source's file order to its current sorted order.

    Uses the source's own raw row numbers rather than any link's slot values:
    slot values (particularly chunked sizes) are not unique per row and cannot
    be used to recover a permutation.
    """
    unsorted_rows = into_array(source._state.raw_data_handler.index)
    sorted_rows = into_array(source.index)

    argsort_sorted_rows = np.argsort(sorted_rows)
    return argsort_sorted_rows[
        np.searchsorted(
            sorted_rows,
            unsorted_rows,
            sorter=argsort_sorted_rows,
        )
    ]


def compute_resort_index(
    source: oc.Dataset,
    match_set: DatasetMatchSet,
    name: str,
    sort_index: np.ndarray,
) -> DataIndex:
    """Build the take index restoring a linked dataset to source file order."""
    slot_values, chunked = _slot_values(source, match_set, name, source.index)
    if not chunked:
        valid_rows = slot_values >= 0
        return sort_index[valid_rows]

    chunk_boundaries = np.zeros(len(slot_values) + 1, dtype=np.int64)
    _ = np.cumsum(slot_values, out=chunk_boundaries[1:])
    starts = chunk_boundaries[sort_index]
    sizes = slot_values[sort_index]
    valid = sizes > 0
    return coalesce_chunks(starts[valid], sizes[valid])


def resort_datasets(
    source: oc.Dataset,
    datasets: Mapping[str, oc.Dataset | oc.Lightcone | oc.StructureCollection],
    match_sets: dict[UUID, DatasetMatchSet],
) -> dict[str, oc.Dataset | oc.Lightcone | oc.StructureCollection]:
    match_set = match_sets[source.uuid]
    sort_index = compute_sort_index(source)
    return {
        name: dataset.take_rows(
            compute_resort_index(source, match_set, name, sort_index)
        )
        for name, dataset in datasets.items()
    }


def apply_step_indices(
    target: oc.Lightcone | oc.StructureCollection,
    per_step_index: dict[Any, Optional[DataIndex]],
) -> oc.Lightcone | oc.StructureCollection:
    """Apply step-local indices to a Lightcone or nested StructureCollection."""

    if isinstance(target, lc.Lightcone):
        new_datasets = {
            step: target[step].take_rows(index if index is not None else empty())
            for step, index in per_step_index.items()
        }
        return lc.Lightcone.from_datasets(new_datasets)

    assert isinstance(target, sc.StructureCollection)
    source = target[str(target.header.file.data_type)]
    assert isinstance(source, lc.Lightcone)
    pieces: list[np.ndarray] = []
    running = 0
    for step, step_source in source.items():
        index = per_step_index.get(step)
        if index is not None:
            pieces.append(into_array(offset(index, running)))
        running += len(step_source)
    global_index = np.concatenate(pieces) if pieces else np.array([], dtype=np.int64)
    return target.take_rows(global_index)


def resolve_links_per_step(
    source: oc.Lightcone,
    datasets: Mapping[str, oc.Lightcone | sc.StructureCollection],
    match_sets: dict[UUID, DatasetMatchSet],
) -> dict[str, oc.Lightcone | oc.StructureCollection]:
    new_datasets: dict[str, oc.Lightcone | oc.StructureCollection] = {}
    for name, target in datasets.items():
        per_step_index: dict[Any, Optional[DataIndex]] = {}
        for step, step_source in source.items():
            match_set = match_sets[step_source.uuid]
            target_uuid = _target_uuid(match_set, name)
            index = get_mapping(
                match_set, match_set.reference_source, target_uuid, step_source.index
            )
            assert index is not None
            if not is_chunked_slot(match_set, target_uuid):
                index = np.asarray(index)
                index = index[index >= 0]
            per_step_index[step] = _none_if_empty(index)
        new_datasets[name] = apply_step_indices(target, per_step_index)
    return new_datasets


def _none_if_empty(index: DataIndex) -> Optional[DataIndex]:
    if isinstance(index, tuple):
        return index if len(index[0]) else None
    return index if len(index) else None


def rebuild_links_per_step(
    derived_from: oc.Lightcone,
    new_source: oc.Lightcone,
    datasets: Mapping[str, oc.Lightcone | sc.StructureCollection],
    match_sets: dict[UUID, DatasetMatchSet],
) -> dict[str, oc.Lightcone | oc.StructureCollection]:
    per_step_index: dict[str, dict[Any, Optional[DataIndex]]] = defaultdict(dict)
    for step, new_step_source in new_source.items():
        old_step_source = derived_from[step]
        match_set = match_sets[old_step_source.uuid]
        for name in datasets:
            target_uuid = _target_uuid(match_set, name)
            index = rebuild_target_index(
                match_set,
                target_uuid,
                old_step_source.index,
                new_step_source.index,
            )
            per_step_index[name][step] = _none_if_empty(index)
    return {
        name: apply_step_indices(target, per_step_index[name])
        for name, target in datasets.items()
    }


def resort_datasets_per_step(
    source: oc.Lightcone,
    datasets: Mapping[str, oc.Lightcone | sc.StructureCollection],
    match_sets: dict[UUID, DatasetMatchSet],
) -> dict[str, oc.Lightcone | oc.StructureCollection]:
    per_step_index: dict[str, dict[Any, Optional[DataIndex]]] = defaultdict(dict)
    for step, step_source in source.items():
        match_set = match_sets[step_source.uuid]
        sort_index = compute_sort_index(step_source)
        for name in datasets:
            index = compute_resort_index(step_source, match_set, name, sort_index)
            per_step_index[name][step] = _none_if_empty(index)
    return {
        name: apply_step_indices(target, per_step_index[name])
        for name, target in datasets.items()
    }


class LinkHandler:
    """Manage linked structure datasets and their deferred rebuilding."""

    def __init__(
        self,
        match_sets: dict[UUID, DatasetMatchSet],
        derived_from: Optional[oc.Dataset | oc.Lightcone],
    ) -> None:
        self.__derived_from = derived_from
        self.match_sets = match_sets

    def match_set_for(self, source: oc.Dataset) -> DatasetMatchSet:
        """Return the match set owning ``source``'s links."""
        return self.match_sets[source.uuid]

    @property
    def names(self) -> tuple[str, ...]:
        """Collection-facing link names, sorted for determinism."""
        return tuple(
            sorted(
                {
                    name
                    for match_set in self.match_sets.values()
                    for name in match_set.aliases
                }
            )
        )

    def prep_datasets(
        self,
        source: oc.Dataset | oc.Lightcone,
        datasets: dict[str, oc.Dataset | oc.Lightcone],
    ) -> dict[str, oc.Dataset | oc.Lightcone]:
        """Prepare linked datasets when their source is first opened."""
        if isinstance(source, lc.Lightcone):
            lightcone_datasets = cast(
                "Mapping[str, oc.Lightcone | sc.StructureCollection]", datasets
            )
            return cast(
                "dict[str, oc.Dataset | oc.Lightcone]",
                resolve_links_per_step(source, lightcone_datasets, self.match_sets),
            )

        match_set = self.match_set_for(source)
        new_datasets = dict(datasets)
        for name, dataset in datasets.items():
            target_uuid = match_set.get_uuid(name)
            if target_uuid is None:
                continue
            index = get_mapping(
                match_set, match_set.reference_source, target_uuid, source.index
            )
            assert index is not None
            if not is_chunked_slot(match_set, target_uuid):
                index = np.asarray(index)
                index = index[index >= 0]
            new_datasets[name] = dataset.take_rows(index)
        return new_datasets

    def make_derived(self, source: oc.Dataset) -> LinkHandler:
        """Record the source from which deferred linked rebuilding begins."""
        derived_from = self.__derived_from
        if self.__derived_from is None:
            derived_from = source
        return LinkHandler(self.match_sets, derived_from)

    def rebuild_datasets(
        self,
        new_source: oc.Dataset | oc.Lightcone,
        datasets: Mapping[str, oc.Dataset | oc.Lightcone | oc.StructureCollection],
    ) -> Mapping[str, oc.Dataset | oc.Lightcone | oc.StructureCollection]:
        """Rebuild linked datasets only after a derived source has changed rows."""
        if self.__derived_from is None:
            return datasets
        return self.__rebuild_datasets(self.__derived_from, new_source, datasets)

    def __rebuild_datasets(
        self,
        derived_from: oc.Dataset | oc.Lightcone,
        new_source: oc.Dataset | oc.Lightcone,
        datasets: Mapping[str, oc.Dataset | oc.Lightcone | oc.StructureCollection],
    ) -> Mapping[str, oc.Dataset | oc.Lightcone | oc.StructureCollection]:
        if isinstance(derived_from, lc.Lightcone):
            assert isinstance(new_source, lc.Lightcone)
            assert all(
                isinstance(dataset, (lc.Lightcone, sc.StructureCollection))
                for dataset in datasets.values()
            )
            lightcone_datasets = cast(
                "Mapping[str, oc.Lightcone | sc.StructureCollection]", datasets
            )
            return rebuild_links_per_step(
                derived_from, new_source, lightcone_datasets, self.match_sets
            )

        assert isinstance(new_source, ocds.Dataset)
        match_set = self.match_set_for(derived_from)
        new_datasets: dict[str, oc.Dataset | oc.Lightcone | oc.StructureCollection] = {}
        for name, dataset in datasets.items():
            target_uuid = _target_uuid(match_set, name)
            index = rebuild_target_index(
                match_set, target_uuid, derived_from.index, new_source.index
            )
            new_datasets[name] = dataset.take_rows(index)
        return new_datasets

    def resort(
        self,
        source: oc.Dataset | oc.Lightcone,
        datasets: dict[str, oc.Dataset | oc.Lightcone | oc.StructureCollection],
    ) -> dict[str, oc.Dataset | oc.Lightcone | oc.StructureCollection]:
        """Restore linked datasets to their source's original file order."""
        if source.sorted_by is None:
            return datasets
        if isinstance(source, lc.Lightcone):
            lightcone_datasets = cast(
                "Mapping[str, oc.Lightcone | sc.StructureCollection]", datasets
            )
            return cast(
                "dict[str, oc.Dataset | oc.Lightcone | oc.StructureCollection]",
                resort_datasets_per_step(source, lightcone_datasets, self.match_sets),
            )
        return resort_datasets(source, datasets, self.match_sets)
