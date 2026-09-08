from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from opencosmo.collection.lightcone.distribute import partition_contiguous
from opencosmo.io.discover import (
    group_data_type,
    has_linked_targets,
    is_properties_group,
)

if TYPE_CHECKING:
    from uuid import UUID

    import opencosmo as oc
    from opencosmo.io.discover import FileLayout
    from opencosmo.io.io import MpiMode
    from opencosmo.io.iopen import DatasetTarget, FileTarget


@dataclass(frozen=True)
class Assignment:
    """Assignment of files to a single MPI rank."""

    rank: int
    """MPI rank number."""

    file_indices: tuple[int, ...]
    """Sorted tuple of indices into the layouts tuple."""

    index_kind: Literal["spatial", "redshift_step", "none"]
    """Type of indexing to apply: spatial, redshift_step (for contiguous step chunks), or none."""

    is_empty_ref: bool = False
    """True if this rank received the reference (lightest) step due to being empty."""


def distribute(
    layouts: tuple[FileLayout, ...],
    mpi_mode: MpiMode | None,
    nranks: int,
) -> tuple[Assignment, ...]:
    """
    Deterministic, broadcast-free distribution of files across MPI ranks.

    Every rank computes the identical Assignment tuple from identical inputs.
    No I/O, no comm, no header re-reads — everything comes from the passed-in layouts.

    Parameters
    ----------
    layouts : tuple[FileLayout, ...]
        Sorted tuple of discovered file layouts.
    mpi_mode : MpiMode | None
        Distribution mode (SPATIAL or REDSHIFT), or None (treat as SPATIAL).
    nranks : int
        Number of MPI ranks.

    Returns
    -------
    tuple[Assignment, ...]
        Tuple of length nranks, one Assignment per rank.
    """
    # Compute valid (non-errored) file indices, sorted.
    valid_indices = tuple(i for i, fl in enumerate(layouts) if fl.error is None)

    # Serial or single rank: return rank 0 with all valid files.
    if nranks <= 1:
        return (Assignment(rank=0, file_indices=valid_indices, index_kind="none"),)

    # Try redshift distribution if mode is REDSHIFT and we have > 1 valid file.
    if mpi_mode is not None and mpi_mode.value == "redshift" and len(valid_indices) > 1:
        redshift_assignments = _distribute_redshift(layouts, valid_indices, nranks)
        if redshift_assignments is not None:
            return redshift_assignments

    # Spatial (default / fallback): every rank gets every valid file.
    return tuple(
        Assignment(rank=r, file_indices=valid_indices, index_kind="spatial")
        for r in range(nranks)
    )


class FileMeta(NamedTuple):
    """Per-file metadata used when grouping redshift-step files by step."""

    file_index: int
    """Index into the layouts tuple."""

    weight: int
    """Row count of the file's top-level /data (its data-volume weight)."""

    is_properties_link: bool
    """True if this is a properties group carrying /data_linked targets."""


def _distribute_redshift(
    layouts: tuple[FileLayout, ...],
    valid_indices: tuple[int, ...],
    nranks: int,
) -> tuple[Assignment, ...] | None:
    """
    Distribute redshift-step lightcone files across ranks as contiguous chunks.

    Ports the step-grouping/weight logic from __compute_redshift_distribution_plan,
    operating on the in-memory layouts. Reads each file's step and top-level row
    count from its top-level group, groups files by step, and splits the
    redshift-ordered steps into ``nranks`` contiguous chunks of roughly-equal
    volume. Empty ranks get the lightest step (``is_empty_ref=True``).

    Distribution assumes verified input: cross-file compatibility checks
    (matching columns/dtypes across steps, structure-collection consistency)
    are the responsibility of the spec's ``verify()`` step, which runs before
    distribution. The only decision made here is whether the layout is
    distributable by step at all, or whether it is a nested diffsky-style
    lightcone that must fall back to spatial distribution.

    Parameters
    ----------
    layouts : tuple[FileLayout, ...]
        Full sorted layout tuple.
    valid_indices : tuple[int, ...]
        Tuple of valid (non-errored) file indices.
    nranks : int
        Number of MPI ranks.

    Returns
    -------
    tuple[Assignment, ...] or None
        Assignments for redshift-step distribution, or None to fall back to spatial.
    """
    # Per-file metadata, grouped by step. Each entry carries the file index, the
    # row count of its top-level /data, and whether it is a properties file with
    # /data_linked.
    file_info: dict[int | None, list[FileMeta]] = {}
    data_types: set[str] = set()

    for i in valid_indices:
        layout = layouts[i]
        # A valid (non-errored) file always has at least one data group —
        # discover_file sets FileLayout.error otherwise. If that invariant is
        # broken here the distribution is being fed garbage, so fail loudly
        # rather than silently dropping the file from the plan.
        if not layout.groups:
            raise ValueError(
                f"File {layout.path} has no data groups but was not marked as "
                "errored. This should be unreachable — discover_file sets "
                "FileLayout.error in that case."
            )
        # The file's step and data volume come from its top-level (/) group.
        top_group = next((g for g in layout.groups if g.path == "/"), layout.groups[0])

        step: int | None = top_group.header.file.step
        weight = top_group.row_count
        is_properties_link = is_properties_group(top_group) and has_linked_targets(
            top_group
        )
        data_types.add(group_data_type(top_group))

        file_info.setdefault(step, []).append(
            FileMeta(file_index=i, weight=weight, is_properties_link=is_properties_link)
        )

    # Decide whether this layout is distributable by redshift step. A plain
    # lightcone (one file per step, single data_type) or a lightcone structure
    # collection (some step carries a properties-with-/data_linked file) both
    # distribute by step. Anything else — multiple files/types per step with no
    # properties link, i.e. a nested diffsky-style lightcone — falls back to
    # spatial distribution (the None sentinel).
    is_plain = all(len(files) == 1 for files in file_info.values()) and (
        len(data_types) == 1
    )
    any_properties_link = any(
        fm.is_properties_link for files in file_info.values() for fm in files
    )
    if not is_plain and not any_properties_link:
        return None

    # Order steps by redshift (step is monotonic, None values sort last).
    ordered_steps = sorted(file_info.keys(), key=lambda x: (x is None, x))  # type: ignore
    step_file_indices: list[list[int]] = []
    step_weights: list[int] = []
    for step in ordered_steps:
        step_file_indices.append(sorted(fm.file_index for fm in file_info[step]))
        step_weights.append(sum(fm.weight for fm in file_info[step]))

    # Split the ordered steps into nranks CONTIGUOUS chunks of roughly-equal weight.
    step_groups = partition_contiguous(step_weights, nranks)

    # Reference = all file indices of the lightest step (for empty ranks).
    lightest = min(range(len(step_weights)), key=lambda i: step_weights[i])
    reference_indices = step_file_indices[lightest]

    # Build Assignments: gather file indices from all steps in this rank's group.
    # An empty rank instead gets the reference step's indices, flagged is_empty_ref.
    assignments = []
    for r in range(nranks):
        rank_step_group = step_groups[r]
        if rank_step_group:
            rank_indices = sorted(
                idx
                for step_idx in rank_step_group
                for idx in step_file_indices[step_idx]
            )
        else:
            rank_indices = list(reference_indices)
        assignments.append(
            Assignment(
                rank=r,
                file_indices=tuple(rank_indices),
                index_kind="redshift_step",
                is_empty_ref=not rank_step_group,
            )
        )

    return tuple(assignments)


@runtime_checkable
class SpecBuilder(Protocol):
    """
    Contract for file spec builders (T3 concrete specs implement this).

    Each FileSpec subclass knows how to build a Dataset or Collection from
    a list of FileTargets. The spec is responsible for routing to the appropriate
    builder (open_dataset, Lightcone.open, StructureCollection.open) with
    the correct set of targets for that collection type.
    """

    def build_from_targets(
        self,
        targets: list[FileTarget],
        *,
        index_kind: str,
        is_empty_ref: bool,
        open_kwargs: dict[str, Any],
    ) -> oc.Dataset | oc.collection.Collection: ...


def build_from_assignment(
    assignment: Assignment,
    layouts: tuple[FileLayout, ...],
    matched_spec: SpecBuilder,
    open_kwargs: dict[str, Any],
) -> tuple[oc.Dataset | oc.collection.Collection | None, frozenset[UUID]]:
    """
    Reopen only this rank's files and rehydrate FileTargets from live handles.

    Given an Assignment specifying which files this rank owns, and the
    in-memory FileLayout tuple from discovery, reopens those files and
    reconstructs the FileTarget TypedDict structures by navigating to the
    known group/data paths (no re-walk, no header re-read — reuses the
    already-discovered layout metadata). Re-runs evaluate_load_conditions
    live, then delegates to the matched spec's builder with kwargs derived
    from the Assignment.

    Parameters
    ----------
    assignment : Assignment
        This rank's file assignment from distribute().
    layouts : tuple[FileLayout, ...]
        The full sorted layouts tuple from discover_all().
    matched_spec : SpecBuilder
        The spec that matched these layouts. Must implement build_from_targets().
    open_kwargs : dict[str, Any]
        User kwargs for load condition evaluation and builder options.

    Returns
    -------
    tuple[oc.Dataset | oc.collection.Collection | None, frozenset[UUID]]
        A 2-tuple of: the built collection or dataset (or None if this scope
        was entirely filtered out by load/if conditions), and the frozenset of
        UUIDs of the datasets this rank actually opened *as map endpoints*.

        Every dataset now carries a runtime identity, but only datasets with a
        persistent on-disk ``main_uuid`` are eligible to be resolved as map
        endpoints and therefore appear in the returned frozenset.
    """
    import h5py

    from opencosmo.io.iopen import DatasetTarget, FileTarget, evaluate_load_conditions

    # Step A: Rehydrate this rank's files into FileTargets.
    file_targets: list[FileTarget] = []
    opened_uuids: set[UUID] = set()

    for file_idx in assignment.file_indices:
        # An Assignment only ever references valid files: distribute() builds
        # file_indices from these same layouts, skipping errored ones. A
        # violation of any of these invariants is a distribution bug, not a
        # recoverable condition — raise instead of silently dropping the file.
        layout = layouts[file_idx]
        if layout.error is not None:
            raise ValueError(
                f"Assignment references errored file {layout.path}: "
                f"{layout.error}. distribute() must not assign errored files."
            )
        if not layout.groups:
            raise ValueError(
                f"Assignment references file {layout.path} with no data groups. "
                "This should be unreachable — discover_file sets FileLayout.error "
                "in that case."
            )

        # Open the file (do not use a context manager; the live h5py handles in
        # the targets must outlive this function). The layout already records
        # every group/column/index path, so navigate straight to them — no need
        # to re-walk the file to rediscover what discovery already found.
        f = h5py.File(layout.path, "r")

        # Every group in the file becomes one DatasetTarget in dataset_targets;
        # dataset_groups stays empty. Both builders (build_structure_collection,
        # Lightcone.open) flatten dataset_targets and dataset_groups into a single
        # list, so there is nothing to gain from pre-bucketing — the group's own
        # path/data_type already carries the identity the builders key on.
        dataset_targets: list[DatasetTarget] = []

        for group in layout.groups:
            # rstrip("/") maps the root group "/" to "", so both the root and
            # named groups build their child paths the same way ("" -> "/data",
            # "/scidac1" -> "/scidac1/data").
            prefix = group.path.rstrip("/")
            data_path = f"{prefix}/data"
            index_path = f"{prefix}/index"
            data_linked_path = f"{prefix}/data_linked"

            # Columns are the /data datasets plus, when present, the /data_linked
            # datasets. The link columns (<target>_start/_size/_idx) live under
            # /data_linked and must reach the handler for a structure collection's
            # links to resolve — the old __find_datasets_under_group swept them in
            # the same way (everything under the group except header and index).
            columns_list = [f[f"{data_path}/{name}"] for name in group.column_names]
            if data_linked_path in f:
                data_linked_group = f[data_linked_path]
                for name in data_linked_group.keys():
                    if isinstance(data_linked_group[name], h5py.Dataset):
                        columns_list.append(data_linked_group[name])

            target: DatasetTarget = DatasetTarget(
                uuid=group.uuid,
                header=group.header,
                # dataset_group is the parent of /data, i.e. the group at group.path.
                dataset_group=f[group.path],
                columns=columns_list,
                spatial_index=f[index_path] if group.has_index else None,
                link_layout=group.link_layout,
            )

            # load/if conditions legitimately filter datasets out at open time,
            # so a dropped target here is expected behavior, not an error.
            if evaluate_load_conditions(target, open_kwargs):
                dataset_targets.append(target)
                if group.has_persistent_uuid:
                    opened_uuids.add(group.uuid)

        if dataset_targets:
            file_targets.append(
                FileTarget(
                    dataset_group_types={},
                    dataset_targets=dataset_targets,
                    dataset_groups={},
                )
            )

    # Every target in this scope was filtered out by load/if conditions (the whole
    # scope is gated behind an open flag the user did not pass). There is nothing to
    # build; the orchestrator drops this scope. This is deterministic across ranks
    # (same open_kwargs everywhere), so it stays collective-safe.
    if not file_targets:
        return None, frozenset()

    # Step B: Pass Assignment fields through to spec builder.
    return (
        matched_spec.build_from_targets(
            file_targets,
            index_kind=assignment.index_kind,
            is_empty_ref=assignment.is_empty_ref,
            open_kwargs=open_kwargs,
        ),
        frozenset(opened_uuids),
    )
