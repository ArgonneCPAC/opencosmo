from __future__ import annotations

from collections import defaultdict
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, TypeGuard

import numpy as np

from opencosmo import dataset as d
from opencosmo import io
from opencosmo.collection.lightcone import lightcone as lc
from opencosmo.collection.structure import structure as sc
from opencosmo.collection.structure.handler import (
    LINK_ALIASES,
    LinkHandler,
    link_slot_values,
)
from opencosmo.io.index_spec import index_spec_for
from opencosmo.mapping.read import read_link_set

if TYPE_CHECKING:
    from uuid import UUID

    import h5py
    from mpi4py import MPI

    from opencosmo.io.discover import LinkLayout
    from opencosmo.io.iopen import DatasetTarget, FileTarget
    from opencosmo.mapping.mapping import DatasetMatchSet

ALLOWED_LINKS = {  # h5py.Files that can serve as a link holder and
    "halo_properties": ["halo_particles", "halo_profiles", "galaxy_properties"],
    "galaxy_properties": ["galaxy_particles"],
}


def resolve_link_targets(
    layout: LinkLayout,
    targets: Mapping[str, d.Dataset | sc.StructureCollection],
) -> tuple[dict[str, UUID], dict[str, UUID]]:
    """Resolve on-disk link prefixes against opened targets.

    Returns ``(prefix_to_uuid, name_to_uuid)``: the first keyed by on-disk link
    prefix for ``read_link_set``, the second by collection-facing name for
    ``DatasetMatchSet.with_aliases``.
    """
    prefix_to_uuid: dict[str, UUID] = {}
    name_to_uuid: dict[str, UUID] = {}
    for slot in layout.slots:
        name = LINK_ALIASES[slot.prefix]
        # ``targets`` is keyed the way StructureCollection will key it, so a
        # nested galaxy collection appears as "galaxies" rather than
        # "galaxy_properties". Follow that rename when it applies.
        if name == "galaxy_properties" and name not in targets:
            name = "galaxies"
        target = targets.get(name)
        if target is None:
            continue
        if isinstance(target, sc.StructureCollection):
            target = target[str(target.header.file.data_type)]
        assert isinstance(target, d.Dataset)
        prefix_to_uuid[slot.prefix] = target.uuid
        name_to_uuid[name] = target.uuid
    return prefix_to_uuid, name_to_uuid


def build_match_sets(
    source_target: DatasetTarget,
    source: d.Dataset,
    targets: Mapping[str, d.Dataset | sc.StructureCollection],
) -> dict[UUID, DatasetMatchSet]:
    """Build the resolved link match set for one properties source dataset."""
    layout = source_target["link_layout"]
    if layout is None or "data_linked" not in source_target["dataset_group"]:
        return {}
    prefix_to_uuid, name_to_uuid = resolve_link_targets(layout, targets)
    match_set = read_link_set(
        source_target["dataset_group"]["data_linked"],
        layout,
        source_target["uuid"],
        prefix_to_uuid,
    )
    if match_set is None:
        return {}
    return {source.uuid: match_set.with_aliases(name_to_uuid)}


def __with_galaxies_alias[T](targets: Mapping[str, T]) -> dict[str, T]:
    """Key nested galaxy collections as "galaxies", mirroring StructureCollection.

    ``StructureCollection.__init__`` renames ``galaxy_properties`` to ``galaxies``
    only when the target is a nested collection. Link aliases must agree, or the
    handler cannot resolve the link by name.
    """
    resolved = dict(targets)
    if isinstance(resolved.get("galaxy_properties"), sc.StructureCollection):
        resolved["galaxies"] = resolved.pop("galaxy_properties")
    return resolved


def __step_targets(
    targets: Mapping[str, d.Dataset | lc.Lightcone | sc.StructureCollection],
    step: int,
) -> dict[str, d.Dataset | sc.StructureCollection]:
    resolved: dict[str, d.Dataset | sc.StructureCollection] = {}
    for name, target in __with_galaxies_alias(targets).items():
        if isinstance(target, lc.Lightcone):
            resolved[name] = target[step]
        elif isinstance(target, sc.StructureCollection):
            source = target[str(target.header.file.data_type)]
            if isinstance(source, lc.Lightcone):
                resolved[name] = source[step]
            else:
                resolved[name] = target
        else:
            resolved[name] = target
    return resolved


def __build_lightcone_match_sets(
    source_targets: Iterable[DatasetTarget],
    source_datasets: Iterable[d.Dataset],
    targets: Mapping[str, d.Dataset | lc.Lightcone | sc.StructureCollection],
) -> dict[UUID, DatasetMatchSet]:
    sources = tuple(source_datasets)
    targets_by_step = {
        step: __step_targets(targets, step)
        for step in {
            ds.header.file.step for ds in sources if ds.header.file.step is not None
        }
    }
    match_sets: dict[UUID, DatasetMatchSet] = {}
    for source_target, source in zip(source_targets, sources, strict=True):
        step = source.header.file.step
        assert step is not None
        match_sets.update(
            build_match_sets(
                source_target,
                source,
                targets_by_step[step],
            )
        )
    return match_sets


def remove_empty(
    dataset: d.Dataset | lc.Lightcone,
    match_sets: dict[UUID, DatasetMatchSet],
    opened_datasets: Optional[Iterable[str]] = None,
) -> d.Dataset | lc.Lightcone:
    """
    Drop structures that are empty in the linked datasets that were actually
    opened. When a user opens, say, particles and profiles together, they should
    be able to assume every structure has both -- the source data keeps particles
    and profiles separately, and this reproduces that "all present" guarantee. The
    ignore_empty flag on open() exists to override it.

    Only the link columns belonging to opened datasets are considered. The source
    metadata always carries link columns for every data type in the file (e.g.
    galaxy or particle links), so restricting to opened datasets avoids dropping
    structures based on links the user never asked for.
    """
    names = {name for match_set in match_sets.values() for name in match_set.aliases}

    if opened_datasets is not None:
        # A nested galaxy collection is exposed as "galaxies" but keyed as
        # "galaxy_properties" in the link targets; treat them as the same link.
        opened = {
            "galaxies" if name == "galaxy_properties" else name
            for name in opened_datasets
        }
        names &= opened

    if not names:
        return dataset

    mask = np.ones(len(dataset), dtype=bool)
    for name in sorted(names):
        values, is_chunked = link_slot_values(match_sets, dataset, name)
        mask &= values != 0 if is_chunked else values != -1

    if not mask.all():
        dataset = dataset.take_rows(np.where(mask)[0])
    return dataset


def is_dataset(ds: Any) -> TypeGuard[d.Dataset]:
    return isinstance(ds, d.Dataset)


def validate_linked_groups(groups: dict[str, h5py.Group]):
    if "halo_properties" in groups:
        if "data_linked" not in groups["halo_properties"].keys():
            raise ValueError(
                "File appears to be a structure collection, but does not have links!"
            )
    elif "galaxy_properties" in groups:
        if "data_linked" not in groups["galaxy_properties"].keys():
            raise ValueError(
                "File appears to be a structure collection, but does not have links!"
            )
    if len(groups) == 1:
        raise ValueError("Structure collections must have more than one dataset")


def build_structure_collection(
    targets: list[FileTarget],
    ignore_empty: bool,
    index_kind: str = "none",
    is_empty_ref: bool = False,
):
    link_sources: dict[str, list[io.iopen.DatasetTarget]] = defaultdict(list)
    link_targets: dict[str, dict[str, list[d.Dataset | sc.StructureCollection]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    dataset_targets: list[io.iopen.DatasetTarget] = []
    for t in targets:
        dataset_targets.extend(t["dataset_targets"])
        for datasets in t["dataset_groups"].values():
            dataset_targets.extend(datasets)

    for target in dataset_targets:
        if target["header"].file.data_type == "halo_properties":
            link_sources["halo_properties"].append(target)
        elif target["header"].file.data_type == "galaxy_properties":
            link_sources["galaxy_properties"].append(target)
        elif str(target["header"].file.data_type).startswith("halo"):
            dataset = io.iopen.open_dataset(
                target, index_spec_for(index_kind, is_empty_ref, is_source=False)
            )
            name_source = target["dataset_group"]
            if (
                "particles" in name_source.parent.name
                or "profiles" in target["dataset_group"].parent.name
            ):
                name_source = target["dataset_group"].parent
            name = name_source.name.split("/")[-1]

            if not name:
                name = target["header"].file.data_type
            elif name.startswith("halo_properties"):
                name = name[16:]
            link_targets["halo_properties"][name].append(dataset)
        elif str(target["header"].file.data_type).startswith("galaxy"):
            dataset = io.iopen.open_dataset(
                target, index_spec_for(index_kind, is_empty_ref, is_source=False)
            )
            name_source = target["dataset_group"]
            if (
                "particles" in name_source.parent.name
                or "profiles" in target["dataset_group"].parent.name
            ):
                name_source = target["dataset_group"].parent
            name = name_source.name.split("/")[-1]

            if not name:
                name = target["header"].file.data_type
            elif name.startswith("galaxy_properties"):
                name = name[18:]
            link_targets["galaxy_properties"][name].append(dataset)
        else:
            raise ValueError(
                "Unknown data type for structure collection "
                f"{target['header'].file.data_type}"
            )

    if (
        index_kind == "redshift_step"
        or len(link_sources["halo_properties"]) > 1
        or len(link_sources["galaxy_properties"]) > 1
    ):
        # Lightcone structure collection. Under redshift-split we ALWAYS route here,
        # even when this rank holds a single step (one source per type), so every
        # rank builds the same collection kind and stays MPI-lockstep on the write
        # path.
        return build_lightcone_structure_collection(
            link_sources,
            link_targets,
            ignore_empty,
            index_kind=index_kind,
            is_empty_ref=is_empty_ref,
        )

    halo_properties_target = None
    galaxy_properties_target = None
    if link_sources["halo_properties"]:
        halo_properties_target = link_sources["halo_properties"][0]
    if link_sources["galaxy_properties"]:
        galaxy_properties_target = link_sources["galaxy_properties"][0]

    input_link_targets: dict[str, dict[str, d.Dataset | sc.StructureCollection]] = (
        defaultdict(dict)
    )
    for source_type, source_targets in link_targets.items():
        if any(len(ts) > 1 for ts in source_targets.values()):
            raise ValueError("Found more than one linked file of a given type!")
        input_link_targets[source_type] = {
            key: t[0] for key, t in source_targets.items()
        }

    return __build_structure_collection(
        halo_properties_target,
        galaxy_properties_target,
        input_link_targets,
        ignore_empty,
        index_kind=index_kind,
        is_empty_ref=is_empty_ref,
    )


def build_lightcone_structure_collection(
    link_sources: dict[str, list[io.iopen.DatasetTarget]],
    link_targets: dict[str, dict[str, list[d.Dataset | sc.StructureCollection]]],
    ignore_empty: bool = True,
    index_kind: str = "none",
    is_empty_ref: bool = False,
):
    found_redshift_steps: set[int] = set()
    for source_type, source_list in link_sources.items():
        if not all(t["header"].file.is_lightcone for t in source_list):
            raise ValueError("All sources must be lightcone datasets!")
        redshift_steps = set(t["header"].file.step for t in source_list)
        if found_redshift_steps and found_redshift_steps != redshift_steps:
            raise ValueError(
                "All source types must have the same set of redshift steps!"
            )
        if not all(
            t.header.file.is_lightcone
            for t in chain.from_iterable(link_targets[source_type].values())
        ):
            raise ValueError("All dataset must be lightcone datasets!")
        for targets in link_targets[source_type].values():
            target_redshift_steps = set(t.header.file.step for t in targets)
            if target_redshift_steps != redshift_steps:
                raise ValueError(
                    "All datasets must have the same set of redshift steps!"
                )
    # NOTE: link_targets is a defaultdict, so accessing link_targets[source_type]
    # in the validation loop above may have created an empty "galaxy_properties"
    # entry. Use a truthy check (non-empty dict) rather than `in` so that the
    # "galaxy properties but no galaxy particles" case does not fall into the
    # galaxy-particles branch below.
    if len(link_sources.get("galaxy_properties", [])) > 0 and link_targets.get(
        "galaxy_properties"
    ):
        # Galaxy properties and galaxy particles
        is_galaxy_source = len(link_sources.get("halo_properties", [])) == 0
        galaxy_datasets = [
            io.iopen.open_dataset(
                t,
                index_spec_for(index_kind, is_empty_ref, is_source=is_galaxy_source),
                metadata_group="data_linked",
            )
            for t in link_sources["galaxy_properties"]
        ]
        galaxy_source_by_step: dict[int, d.Dataset] = {}
        for ds in galaxy_datasets:
            assert ds.header.file.step is not None
            galaxy_source_by_step[ds.header.file.step] = ds
        galaxy_lightcone = lc.Lightcone.from_datasets(galaxy_source_by_step)
        galaxy_target_datasets = {}
        for target_type, targets in link_targets["galaxy_properties"].items():
            galaxy_target_datasets[target_type] = lc.Lightcone.from_datasets(
                {ds.header.file.step: ds for ds in targets}  # type: ignore
            )
        galaxy_match_sets = __build_lightcone_match_sets(
            link_sources["galaxy_properties"],
            galaxy_datasets,
            galaxy_target_datasets,
        )
        if len(link_sources.get("halo_properties", [])) > 0:
            collection = sc.StructureCollection(
                galaxy_lightcone,
                galaxy_target_datasets,
                False,
                LinkHandler(galaxy_match_sets, None),
                resolve_links=True,
            )
            link_targets["halo_properties"]["galaxy_properties"] = collection  # type: ignore[assignment]
        else:
            if ignore_empty:
                galaxy_lightcone = remove_empty(
                    galaxy_lightcone, galaxy_match_sets, galaxy_target_datasets.keys()
                )
            return sc.StructureCollection(
                galaxy_lightcone,
                galaxy_target_datasets,
                False,
                LinkHandler(galaxy_match_sets, None),
                resolve_links=True,
            )

    elif (
        len(link_sources.get("halo_properties", [])) > 0
        and len(link_sources.get("galaxy_properties", [])) > 0
    ):
        # Halo properties and galaxy properties, but no galaxy particles. Attach
        # the galaxy properties as a plain per-step linked dataset under the
        # halos, exactly like halo profiles.
        link_targets["halo_properties"]["galaxy_properties"] = [
            io.iopen.open_dataset(
                t, index_spec_for(index_kind, is_empty_ref, is_source=False)
            )
            for t in link_sources["galaxy_properties"]
        ]

    halo_source_list = link_sources["halo_properties"]
    halo_datasets = [
        io.iopen.open_dataset(
            t,
            index_spec_for(index_kind, is_empty_ref, is_source=True),
            metadata_group="data_linked",
        )
        for t in halo_source_list
    ]
    halo_source_by_step: dict[int, d.Dataset] = {}
    for ds in halo_datasets:
        assert ds.header.file.step is not None
        halo_source_by_step[ds.header.file.step] = ds
    source_lightcone = lc.Lightcone.from_datasets(halo_source_by_step)

    output_targets = {}
    # Iterate the linked type names in a deterministic (sorted) order so that
    # StructureCollection.make_schema's local, union-free iteration produces the
    # same child ordering on every rank under redshift-split -- defensive lockstep
    # for the mixed-type write path.
    for target_type in sorted(link_targets["halo_properties"].keys()):
        targets = link_targets["halo_properties"][target_type]
        if isinstance(targets, (d.Dataset, sc.StructureCollection)):
            output_targets[target_type] = targets
            continue
        output_targets_of_type: dict[int, d.Dataset] = {}
        for linked_ds in targets:
            assert isinstance(linked_ds, d.Dataset)
            assert linked_ds.header.file.step is not None
            output_targets_of_type[linked_ds.header.file.step] = linked_ds

        output_targets[target_type] = lc.Lightcone.from_datasets(output_targets_of_type)
    halo_match_sets = __build_lightcone_match_sets(
        halo_source_list,
        halo_datasets,
        __with_galaxies_alias(output_targets),
    )
    if ignore_empty:
        source_lightcone = remove_empty(
            source_lightcone, halo_match_sets, output_targets.keys()
        )
    return sc.StructureCollection(
        source_lightcone,
        output_targets,
        False,
        LinkHandler(halo_match_sets, None),
        resolve_links=True,
    )


def __build_structure_collection(
    halo_properties_target: Optional[io.iopen.DatasetTarget],
    galaxy_properties_target: Optional[io.iopen.DatasetTarget],
    link_targets: dict[str, dict[str, d.Dataset | sc.StructureCollection]],
    ignore_empty: bool,
    index_kind: str = "none",
    is_empty_ref: bool = False,
):
    if galaxy_properties_target is not None and "galaxy_properties" in link_targets:
        # Galaxy properties and galaxy particles
        source_dataset = io.iopen.open_dataset(
            galaxy_properties_target,
            index_spec_for(
                index_kind, is_empty_ref, is_source=halo_properties_target is None
            ),
            metadata_group="data_linked",
        )
        galaxy_match_sets = build_match_sets(
            galaxy_properties_target,
            source_dataset,
            link_targets["galaxy_properties"],
        )
        if ignore_empty and halo_properties_target is None:
            filtered_dataset = remove_empty(
                source_dataset,
                galaxy_match_sets,
                link_targets["galaxy_properties"].keys(),
            )
            assert isinstance(filtered_dataset, d.Dataset)
            source_dataset = filtered_dataset
        collection = sc.StructureCollection(
            source_dataset,
            link_targets["galaxy_properties"],
            False,
            LinkHandler(galaxy_match_sets, None),
            resolve_links=True,
        )
        if halo_properties_target is not None:
            link_targets["halo_properties"]["galaxy_properties"] = collection
        else:
            return collection

    if (
        halo_properties_target is not None
        and galaxy_properties_target is not None
        and "galaxy_properties" not in link_targets
    ):
        # Halo properties and galaxy properties, but no galaxy particles
        galaxy_properties = io.iopen.open_dataset(
            galaxy_properties_target,
            index_spec_for(index_kind, is_empty_ref, is_source=False),
        )
        link_targets["halo_properties"]["galaxy_properties"] = galaxy_properties

    if halo_properties_target is not None and link_targets["halo_properties"]:
        source_dataset = io.iopen.open_dataset(
            halo_properties_target,
            index_spec_for(index_kind, is_empty_ref, is_source=True),
            metadata_group="data_linked",
        )
        halo_match_sets = build_match_sets(
            halo_properties_target,
            source_dataset,
            __with_galaxies_alias(link_targets["halo_properties"]),
        )
        if ignore_empty:
            filtered_dataset = remove_empty(
                source_dataset,
                halo_match_sets,
                link_targets["halo_properties"].keys(),
            )
            assert isinstance(filtered_dataset, d.Dataset)
            source_dataset = filtered_dataset

        return sc.StructureCollection(
            source_dataset,
            link_targets["halo_properties"],
            False,
            LinkHandler(halo_match_sets, None),
            resolve_links=True,
        )


def do_idx_update(data: np.ndarray, comm: Optional[MPI.Comm] = None):
    # An idx metadata column links each structure to at most one row in a target
    # dataset, using -1 to mark structures with no linked row (e.g. halos without
    # a profile). The target dataset is written containing only the linked rows,
    # in structure order, so the rewritten idx must give each linked structure a
    # contiguous 0-based index while preserving the -1 sentinels. Under MPI the
    # target is concatenated across ranks, so each rank offsets its indices by the
    # number of linked rows on the ranks before it.
    valid = data >= 0
    n_valid = int(valid.sum())
    if comm is None:
        offset = 0
    else:
        counts = comm.allgather(n_valid)
        offset = int(np.sum(counts[: comm.Get_rank()]))
    result = np.full(len(data), -1, dtype=np.int64)
    result[valid] = np.arange(offset, offset + n_valid)
    return result


def do_start_update(data: np.ndarray, size: np.ndarray, comm: Optional[MPI.Comm]):
    psum = np.insert(np.cumsum(size), 0, 0)[:-1]
    if comm is None:
        return psum
    lengths = comm.allgather(np.sum(size))
    offsets = np.insert(np.cumsum(lengths), 0, 0)
    offset = offsets[comm.Get_rank()]
    return psum + offset


def rebuild_data_linked(source_schema):
    if (
        source_schema.type == io.schema.FileEntry.LIGHTCONE
        and "data" not in source_schema.children
    ):
        for key, value in source_schema.children.items():
            source_schema.children[key] = rebuild_data_linked(value)
        return source_schema

    for colname, column in source_schema.children["data_linked"].columns.items():
        if "idx" in colname:
            column.set_transformation(do_idx_update)
        elif "start" in colname:
            size_colname = colname.replace("start", "size")
            size_data = source_schema.children["data_linked"].columns[size_colname].data
            updater = partial(do_start_update, size=size_data)
            column.set_transformation(updater)
    return source_schema
