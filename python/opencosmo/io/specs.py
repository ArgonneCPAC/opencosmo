from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from opencosmo.io.discover import (
    FileLayout,
    group_data_type,
    has_linked_targets,
    is_healpix_map_group,
    is_lightcone_group,
    is_properties_group,
)

if TYPE_CHECKING:
    from pathlib import Path

    import opencosmo as oc
    from opencosmo.io.discover import GroupLayout
    from opencosmo.io.iopen import DatasetTarget

"""
Spec registry for opening OpenCosmo files.

Each file/collection *type* is described by one self-contained FileSpec that knows
only how to recognise itself (``matches``), sanity-check itself (``verify``) and turn
rehydrated targets into an object (``build_from_targets``). Opening a file is CORE
logic, not an extension point, so this registry deliberately does NOT use the plugin
hook system (opencosmo.plugins) — it is a plain module-level ordered tuple, and
dispatch is a plain ``match_spec(layouts)`` that returns the first spec whose
``matches()`` is true. Precedence is the order of the SPECS tuple, auditable in one
place.

A spec is stateless and does not orchestrate distribution or building — that is the
caller's job (``open_files``), which calls the free functions ``plan.distribute`` and
``plan.build_from_assignment`` directly, passing the matched spec as the builder.

Composition across simulations is not a spec either. A file (or file set) with more
than one governing header scope — one ``/header`` per simulation, e.g.
``/scidac1/header``, ``/scidac2/header`` — is a SimulationCollection of per-scope
children. ``group_by_scope`` splits the layouts by scope; the caller builds each
scope through the ordinary single-scope path and wraps the results. Every spec
therefore only ever sees a single-scope layout.

For plain Datasets and HealpixMaps, _build_single_dataset wraps the result of
open_dataset (which returns a raw Dataset) into a HealpixMap if needed after
open_dataset returns.
"""


def _all_groups(layouts: tuple[FileLayout, ...]) -> list[GroupLayout]:
    """Flatten all GroupLayouts from non-errored files (path-sorted order preserved)."""
    return [g for fl in layouts if fl.error is None for g in fl.groups]


def _strip_step_segment(path: str) -> str:
    """Drop a trailing redshift-step segment from a group path.

    A redshift-split lightcone is written with each step under a ``<step>_<name>``
    subgroup (``f"{step}_{name}"`` in the writer): the serial path yields
    ``/halo_properties/600_data``, the stacked/MPI path ``/halo_properties/600_600``.
    Either way the segment begins with ``<digits>_`` — and no dataset or simulation
    scope name begins with a digit — so that prefix unambiguously marks a step level.
    The logical dataset is one level up, so collapse it: all steps of a dataset then
    share a container and the collection is not mistaken for many independent ones.
    """
    parent, _, last = path.rpartition("/")
    if re.fullmatch(r"\d+_\w+", last):
        return parent or "/"
    return path


def _group_parent(path: str) -> str:
    """Parent container of a group's logical dataset.

    Collapses the ``<step>_data`` level first (see ``_strip_step_segment``) so a
    redshift-split lightcone dataset resolves to one container. Examples:
    ``/scidac1/halo_properties -> /scidac1``, ``/halo_properties -> /``,
    ``/halo_properties/600_data -> /`` (step collapsed), ``/ -> /``.
    """
    return _strip_step_segment(path).rsplit("/", 1)[0] or "/"


def _top_container(path: str) -> str:
    """First path segment — the simulation-collection bucket key.

    ``/scidac1/halo_properties -> scidac1``, ``/scidac1 -> scidac1``, ``/ -> /``.
    """
    return path.strip("/").split("/")[0] or "/"


def _scope_key(g: GroupLayout, file_path: Path) -> str:
    """Bucket key for a group in the multi-scope branch of group_by_scope.

    Nested groups key on their top-level container (``/scidac1/... -> scidac1``).
    Root groups (``/...``) all share the container ``"/"``, so they key on the
    simulation name from the header instead — otherwise two independent root
    datasets from two simulations would collapse into one unmatchable scope.

    Raises ``ValueError`` if the group is at the root and the header carries no
    ``simulation_info`` block (i.e. no ``header/simulation`` ``name`` attribute on
    disk), naming ``file_path`` in the message.
    """
    top = _top_container(g.path)
    if top != "/":
        return top
    try:
        return g.header.simulation["name"]
    except (AttributeError, KeyError):
        raise ValueError(
            f"{file_path}: opening datasets from different simulations together is "
            "only supported for HACC data carrying a header/simulation 'name' "
            "attribute."
        ) from None


def group_by_scope(
    layouts: tuple[FileLayout, ...],
) -> dict[str, tuple[FileLayout, ...]]:
    """Group discovered layouts into one scope per independent simulation.

    Returns an ordered mapping ``scope_name -> sub-layouts``. Most opens are a single
    scope (``"/"``): the caller returns that object directly. A SimulationCollection is
    the multi-scope case, and the caller wraps the per-scope children.

    The decision reproduces the old ``__get_collection_dataset_groups`` rule. A set of
    groups is ONE collection when it both matches a single spec (dataset / healpix /
    lightcone / structure collection) AND every dataset lives under the same parent
    container. Crucially, matching does NOT depend on header layout: a structure
    collection written to disk stores one header per dataset
    (``/halo_properties/header``, ``/dm_particles/header``, ...) yet is a single
    collection, so keying on ``header_path`` (as an earlier design did) would wrongly
    shatter it into a SimulationCollection.

    It is a SimulationCollection when either no single spec matches the whole layout
    (independent same-type datasets, e.g. ``/scidac1`` + ``/scidac2``, both
    ``halo_properties``) or the datasets are nested under more than one container
    (``/scidac1/*`` + ``/scidac2/*``, each an independent structure collection). Both
    are split by top-level container so each simulation is matched and built on its own.

    For root groups (path ``"/"``), the bucket key is the simulation name read from
    ``header.simulation["name"]`` rather than ``"/"`` — otherwise two independent
    root datasets from two different simulations would collapse into a single scope that
    no spec can match. Raises ``ValueError`` if a root group's header has no
    ``simulation_info`` block, or if two **root** groups from **different** file paths
    resolve to the same simulation name (same-simulation datasets must form a structure
    collection or be opened separately). Nested groups (keyed by container name) may be
    contributed by any number of files — that is how a multi-file nested collection is
    expressed — so the collision check does not apply to them.

    Errored files are skipped; callers raise on ``FileLayout.error`` before grouping.
    """
    non_errored = tuple(fl for fl in layouts if fl.error is None)
    all_groups = _all_groups(non_errored)
    if not all_groups:
        return {}

    parents = {_group_parent(g.path) for g in all_groups}
    if len(parents) == 1 and match_spec(non_errored) is not None:
        return {"/": non_errored}

    # Track which file path first claimed each root-derived simulation-name scope so
    # we can detect collisions from *different* files.  Nested groups (keyed by
    # container name) are intentionally excluded: multiple files sharing a container
    # name is how a multi-file nested collection is expressed.
    scope_first_path: dict[str, Path] = {}
    by_scope: dict[str, list[FileLayout]] = {}
    for fl in non_errored:
        scope_groups: dict[str, list[GroupLayout]] = {}
        scope_is_root: dict[str, bool] = {}
        for g in fl.groups:
            top = _top_container(g.path)
            is_root = top == "/"
            key = _scope_key(g, fl.path)
            scope_groups.setdefault(key, []).append(g)
            scope_is_root[key] = is_root
        for scope_name, gs in scope_groups.items():
            # Collision check: two *different* files claiming the same root-derived
            # simulation-name scope.  Only applies to root groups.
            if scope_is_root[scope_name]:
                if (
                    scope_name in scope_first_path
                    and scope_first_path[scope_name] != fl.path
                ):
                    first = scope_first_path[scope_name]
                    raise ValueError(
                        f"{first} and {fl.path} both come from simulation "
                        f"'{scope_name}'. Datasets from the same simulation must form "
                        "a structure collection or be opened separately."
                    )
                scope_first_path.setdefault(scope_name, fl.path)
            sub_fl = FileLayout(path=fl.path, groups=tuple(gs), error=None)
            by_scope.setdefault(scope_name, []).append(sub_fl)
    return {name: tuple(by_scope[name]) for name in sorted(by_scope)}


def _build_single_dataset(
    targets: list[DatasetTarget], open_kwargs: dict[str, Any]
) -> oc.Dataset | oc.collection.Collection:
    """Build one dataset from a single-target file list.

    Shared by DatasetSpec and HealpixMapSpec. open_dataset returns a raw Dataset;
    a healpix_map header is then wrapped into a HealpixMap here (DatasetSpec only
    ever matches plain non-lightcone, non-healpix groups, so its build passes the
    Dataset straight through). The single-dataset open is always spatially
    partitioned — serial opens (comm is None) fall through to no restriction inside
    the spatial spec.
    """
    from opencosmo.io.index_spec import spatial
    from opencosmo.io.iopen import _open_healpix_map, open_dataset

    ds = open_dataset(targets[0], spatial, open_kwargs=open_kwargs)
    if ds.header.file.data_type == "healpix_map":
        return _open_healpix_map(ds)
    return ds


def _verify_columns_consistent_per_dataset(layouts: tuple[FileLayout, ...]) -> None:
    """Verify each logical dataset has identical column names and dtypes across files.

    A logical dataset is identified by ``(data_type, group_path)``, not by ``data_type``
    alone: a single file's ``data_type`` (from its header) is shared by every group in
    it, so several distinct datasets can carry the same ``data_type``. For example a
    ``halo_particles`` file holds ``/dm_particles``, ``/star_particles``, ``/gas_particles``,
    ``/agn_particles`` — all ``data_type == "halo_particles"`` but with legitimately
    different columns. Conversely ``halo_properties`` and ``halo_profiles`` both sit at
    path ``/`` and are told apart by ``data_type``. Only the ``(data_type, path)`` pair is
    unique per file, so it is the right identity: the same dataset must match across
    redshift steps, while different datasets are free to differ. Column names/dtypes are
    already path-sorted in GroupLayout, so the comparison is order-stable.
    """
    by_dataset: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for g in _all_groups(layouts):
        key = (group_data_type(g), g.path)
        sig = (g.column_names, g.column_dtypes)
        if key not in by_dataset:
            by_dataset[key] = sig
        elif by_dataset[key] != sig:
            raise ValueError(
                f"Inconsistent columns for dataset {key}: {by_dataset[key]} vs "
                f"{sig}. Every file holding a given dataset must have identical "
                "column names and dtypes."
            )


def _verify_structure_collection_consistency(layouts: tuple[FileLayout, ...]) -> None:
    """Verify a structure collection is consistent across redshift steps.

    Ports the raises from the old __determine_collection_kind: every step must have
    an identical data_type set, and every step must contain a properties group with
    a /data_linked group. Single-step (snapshot) collections trivially pass. Raises
    ValueError otherwise.
    """
    by_step: dict[int | None, list[GroupLayout]] = {}
    for g in _all_groups(layouts):
        by_step.setdefault(g.header.file.step, []).append(g)

    type_sets = {frozenset(group_data_type(g) for g in gs) for gs in by_step.values()}
    if len(type_sets) > 1:
        raise ValueError(
            "Structure collection steps have inconsistent data_type sets: "
            f"{sorted(sorted(ts) for ts in type_sets)}. Every step must contain the "
            "same set of data types."
        )
    for step, gs in by_step.items():
        if not any(is_properties_group(g) and has_linked_targets(g) for g in gs):
            raise ValueError(
                f"Structure collection step {step} has no properties group with a "
                "/data_linked group."
            )


@runtime_checkable
class FileSpec(Protocol):
    """Contract for a single file/collection type in the open registry.

    Specs are stateless and describe only what varies per type. Distribution and
    building are handled by the caller via the plan.py free functions (the spec is
    passed to ``plan.build_from_assignment`` as the builder, which calls back into
    ``build_from_targets``).
    """

    name: str

    def matches(self, layouts: tuple[FileLayout, ...]) -> bool: ...

    def verify(self, layouts: tuple[FileLayout, ...]) -> None: ...

    def build_from_targets(
        self,
        targets: list[DatasetTarget],
        *,
        index_kind: str,
        is_empty_ref: bool,
        open_kwargs: dict[str, Any],
    ) -> oc.Dataset | oc.collection.Collection: ...


class StructureCollectionSpec:
    """Properties group + /data_linked + >=1 linked type present.

    Catches the lightcone structure collection too (LightconeSpec requires a single
    data_type, which a structure collection never has), so this spec is registered
    before LightconeSpec.
    """

    name = "structure_collection"

    def matches(self, layouts: tuple[FileLayout, ...]) -> bool:
        groups = _all_groups(layouts)
        # A properties file carrying /data_linked is necessary but not sufficient:
        # a lone properties file (or several properties files of one type across
        # redshift steps) references links whose children are not in the open set,
        # so it opens as a Dataset/Lightcone, not a structure collection. The
        # collection only exists once >=1 linked child type is actually present,
        # i.e. more than one distinct data_type is opened together (matching the
        # old properties + particles/profiles categorization).
        has_properties_link = any(
            is_properties_group(g) and has_linked_targets(g) for g in groups
        )
        n_data_types = len({group_data_type(g) for g in groups})
        return has_properties_link and n_data_types > 1

    def verify(self, layouts: tuple[FileLayout, ...]) -> None:
        _verify_columns_consistent_per_dataset(layouts)
        _verify_structure_collection_consistency(layouts)

    def build_from_targets(
        self,
        targets: list[DatasetTarget],
        *,
        index_kind: str,
        is_empty_ref: bool,
        open_kwargs: dict[str, Any],
    ) -> oc.Dataset | oc.collection.Collection:
        from opencosmo import collection as occ

        return occ.StructureCollection.open(
            targets, index_kind=index_kind, is_empty_ref=is_empty_ref, **open_kwargs
        )


class HealpixMapSpec:
    """A single healpix_map group. Shares the single-dataset build with DatasetSpec.

    _build_single_dataset wraps a healpix_map header to a HealpixMap after
    open_dataset returns a raw Dataset; this is a distinct spec purely for the
    match signal and documentation.
    """

    name = "healpix_map"

    def matches(self, layouts: tuple[FileLayout, ...]) -> bool:
        groups = _all_groups(layouts)
        return len(groups) == 1 and is_healpix_map_group(groups[0])

    def verify(self, layouts: tuple[FileLayout, ...]) -> None:
        return None

    def build_from_targets(
        self,
        targets: list[DatasetTarget],
        *,
        index_kind: str,
        is_empty_ref: bool,
        open_kwargs: dict[str, Any],
    ) -> oc.Dataset | oc.collection.Collection:
        return _build_single_dataset(targets, open_kwargs)


class LightconeSpec:
    """>=1 group, all is_lightcone, a single data_type."""

    name = "lightcone"

    def matches(self, layouts: tuple[FileLayout, ...]) -> bool:
        groups = _all_groups(layouts)
        if not groups or not all(is_lightcone_group(g) for g in groups):
            return False
        return len({group_data_type(g) for g in groups}) == 1

    def verify(self, layouts: tuple[FileLayout, ...]) -> None:
        _verify_columns_consistent_per_dataset(layouts)

    def build_from_targets(
        self,
        targets: list[DatasetTarget],
        *,
        index_kind: str,
        is_empty_ref: bool,
        open_kwargs: dict[str, Any],
    ) -> oc.Dataset | oc.collection.Collection:
        from opencosmo import collection as occ

        return occ.Lightcone.open(
            targets, index_kind=index_kind, is_empty_ref=is_empty_ref, **open_kwargs
        )


class DatasetSpec:
    """A single non-lightcone, non-healpix group -> plain Dataset."""

    name = "dataset"

    def matches(self, layouts: tuple[FileLayout, ...]) -> bool:
        groups = _all_groups(layouts)
        if len(groups) != 1:
            return False
        g = groups[0]
        return not is_lightcone_group(g) and not is_healpix_map_group(g)

    def verify(self, layouts: tuple[FileLayout, ...]) -> None:
        return None

    def build_from_targets(
        self,
        targets: list[DatasetTarget],
        *,
        index_kind: str,
        is_empty_ref: bool,
        open_kwargs: dict[str, Any],
    ) -> oc.Dataset | oc.collection.Collection:
        return _build_single_dataset(targets, open_kwargs)


SPECS: tuple[FileSpec, ...] = (
    StructureCollectionSpec(),
    HealpixMapSpec(),
    LightconeSpec(),
    DatasetSpec(),
)


def match_spec(layouts: tuple[FileLayout, ...]) -> FileSpec | None:
    """Return the first spec whose matches() is true for a single-scope layout, or None.

    Precedence is the order of SPECS (most-constrained first), reproducing the old
    query() 'first registration whose predicate is true wins' semantics. Callers pass
    a single-scope layout (see ``group_by_scope``); a multi-scope layout is a
    SimulationCollection and is decomposed before dispatch.
    """
    return next((s for s in SPECS if s.matches(layouts)), None)
