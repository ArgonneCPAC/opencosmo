from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

import opencosmo as oc
from opencosmo import collection as occ
from opencosmo.dataset import state as st
from opencosmo.header import OpenCosmoHeader
from opencosmo.io import plan
from opencosmo.io.discover import discover_all, has_maps, is_particle_group
from opencosmo.io.specs import group_by_scope, match_spec
from opencosmo.mpi import get_comm_world
from opencosmo.plugins.contexts import DatasetOpenCtx, HookPoint
from opencosmo.plugins.hook import fold
from opencosmo.spatial.builders import from_model
from opencosmo.spatial.region import FullSkyRegion, HealpixRegion
from opencosmo.spatial.tree import open_tree
from opencosmo.units import UnitConvention
from opencosmo.utils import normalize_kwarg_name
from opencosmo.uuid import get_column_uuid

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    import h5py

    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.discover import FileLayout, GroupLayout, LinkLayout
    from opencosmo.io.index_spec import IndexSpec
    from opencosmo.io.io import MpiMode
    from opencosmo.mapping.mapping import DatasetMatchSet
    from opencosmo.spatial.region import Region

"""
This file contains all the internal logic for opening a file or files.

There are a few file structures we have to be able to support.

1. header + data groups -> single dataset
2. header + several non-data groups structure collection or lightcone collection. If lightcone collection,
   all dataset will have the same data type and will have is_lightcone set to true in the header.
3. no header, serveral groups -> Structure Collection


When a user passes multiple files, there are basically two options (at present), structure collection
or lightcone collection.

The former will consist of halo properties or galaxy properties, particles, and/or profiles
The later will consist of several datasets, each with the same data type and is_lightcone set to true
"""


@dataclass(frozen=True)
class DatasetTarget:
    """One discovered dataset group, paired with the live file it lives in.

    Everything static comes from ``layout`` — discovery already captured the column
    names, dtypes, units, descriptions, row count and link structure, so nothing here
    re-reads them. The only live state is ``file``; the h5py accessors below resolve
    off it on first use and are cached, so a target that is opened but never read
    costs no HDF5 metadata access at all.
    """

    layout: GroupLayout
    """The group's discovered layout. Sole source of truth for static metadata."""

    file_path: Path
    """Resolved path of the file. Only used to mint column UUIDs."""

    file: h5py.File
    """Live handle. Held strongly: derived Hdf5Handlers acquire columns lazily and
    must not outlive it."""

    @property
    def uuid(self) -> UUID:
        return self.layout.uuid

    @property
    def header(self) -> OpenCosmoHeader:
        return self.layout.header

    @property
    def link_layout(self) -> LinkLayout | None:
        return self.layout.link_layout

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.layout.column_names

    @property
    def row_count(self) -> int:
        return self.layout.row_count

    @property
    def __prefix(self) -> str:
        # rstrip("/") maps the root group "/" to "", so the root and named groups
        # build child paths the same way ("" -> "/data", "/scidac1" -> "/scidac1/data").
        return self.layout.path.rstrip("/")

    @property
    def name(self) -> str:
        """Trailing segment of the group path. Empty for the root group, matching
        ``h5py.Group.name.split("/")[-1]``."""
        return self.layout.path.rsplit("/", 1)[-1]

    @property
    def parent_name(self) -> str:
        """Path of the group's parent, matching ``h5py.Group.parent.name``."""
        return self.layout.path.rsplit("/", 1)[0] or "/"

    @cached_property
    def column_units(self) -> dict[str, str | None]:
        return dict(zip(self.layout.column_names, self.layout.column_units))

    @cached_property
    def column_descriptions(self) -> dict[str, str | None]:
        return dict(zip(self.layout.column_names, self.layout.column_descriptions))

    @cached_property
    def column_uuids(self) -> dict[str, UUID]:
        return {
            name: get_column_uuid(self.file_path, f"{self.__prefix}/data/{name}")
            for name in self.layout.column_names
        }

    @cached_property
    def group(self) -> h5py.Group:
        return self.file[self.layout.path]

    @cached_property
    def data_group(self) -> h5py.Group:
        return self.file[f"{self.__prefix}/data"]

    @cached_property
    def spatial_index(self) -> h5py.Group | None:
        if not self.layout.has_index:
            return None
        return self.file[f"{self.__prefix}/index"]

    @cached_property
    def load_conditions(self) -> dict[str, bool] | None:
        """Conditions from the group's ``load/if`` attrs, or None when absent.

        Read once per target: both ``evaluate_load_conditions`` and
        ``state_from_target`` need these, and each open used to probe the group
        separately.
        """
        try:
            return dict(self.group["load/if"].attrs)
        except KeyError:
            return None


def open_files(
    paths: list[Path],
    open_kwargs: dict[str, Any],
    mpi_mode: "MpiMode | None" = None,
) -> oc.Dataset | oc.collection.Collection:
    """
    Main back-end entry point for opening files.

    Pipeline: discover metadata for every file (one collective allgather), group the
    discovered layouts by governing header scope, then for each scope match a spec,
    verify it, distribute files across ranks, and build this rank's object. A single
    scope returns its object directly; multiple scopes wrap in a SimulationCollection.
    Every rank computes identical scopes/specs/assignments from identical, path-sorted
    layouts, so the only collective in the whole path is the discovery allgather.
    """
    if mpi_mode is None:
        from opencosmo.io.io import MpiMode

        mpi_mode = MpiMode.SPATIAL

    comm = get_comm_world()
    rank = comm.Get_rank() if comm is not None else 0
    nranks = comm.Get_size() if comm is not None else 1

    layouts = discover_all(paths, comm)

    # Discovery captures structural failures into FileLayout.error rather than raising
    # mid-collective. Every rank holds the identical errored set (allgather), so raising
    # here is collective. The message includes each bad path (test_open_bad_data asserts
    # the offending path appears in the ValueError).
    errored = [fl for fl in layouts if fl.error is not None]
    if errored:
        details = "\n".join(f"  {fl.path}: {fl.error}" for fl in errored)
        raise ValueError(f"Failed to open one or more files:\n{details}")

    # Partition layouts into map-carrying and ordinary. A layout may have both
    # maps and groups; route it to the map side if has_maps, to the ordinary side
    # if it has groups, so a hybrid file participates in both pipelines.
    map_layouts = [fl for fl in layouts if has_maps(fl)]
    ordinary_layouts = [fl for fl in layouts if fl.groups]

    if len(map_layouts) > 1:
        map_paths = ", ".join(str(layout.path) for layout in map_layouts)
        raise ValueError(
            "Opening multiple dataset mapping files is not supported. "
            f"Mapping files: {map_paths}"
        )

    # If there are maps but no ordinary layouts, raise: a mapping file has no
    # meaning without endpoints to connect.
    if map_layouts and not ordinary_layouts:
        raise ValueError(
            "Cannot open a dataset mapping on its own. A mapping file defines "
            "row-level correspondences between datasets in different simulations "
            "and requires both endpoint datasets to be present."
        )

    # Particle files carry a "*_particles" data_type in their header. They only make
    # sense when opened alongside the properties dataset that links to them (a
    # StructureCollection). Opening particles on their own is unsupported: if every
    # discovered group is a particle group, there is nothing to anchor them, so fail
    # loudly rather than fabricate a SimulationCollection of orphaned particle types.
    groups = [g for fl in ordinary_layouts for g in fl.groups]

    groups_by_uuid = {group.uuid: group for group in groups if group.uuid is not None}
    for file_layout in map_layouts:
        for map_layout in file_layout.maps:
            reference = groups_by_uuid.get(map_layout.reference)
            if reference is None:
                continue
            invalid_lengths = [
                (target, length)
                for target, length in map_layout.primary_lengths
                if length != reference.row_count
            ]
            if invalid_lengths:
                details = ", ".join(
                    f"{target} has length {length}"
                    for target, length in invalid_lengths
                )
                raise ValueError(
                    f"Primary mappings must have the reference dataset length "
                    f"{reference.row_count}; {details}."
                )

    if groups and all(is_particle_group(g) for g in groups):
        raise ValueError(
            "Cannot open particle data on its own. Particle datasets must be opened "
            "together with their properties dataset (e.g. halo or galaxy properties), "
            "which links to them as a StructureCollection."
        )

    scopes = group_by_scope(tuple(ordinary_layouts))
    if not scopes:
        raise ValueError("No valid datasets found!")

    children: dict[str, oc.Dataset | oc.collection.Collection] = {}
    # UUIDs of every dataset this rank actually opened. Only the identities are
    # needed (map endpoint resolution is pure set membership), so nothing here
    # requires the Dataset objects themselves.
    children_by_uuid: dict[str, frozenset[UUID]] = {}
    available_uuids: set[UUID] = set()
    for scope_name in sorted(scopes):
        sub = scopes[scope_name]
        spec = match_spec(sub)
        if spec is None:
            raise ValueError(
                "Failed to open file. This is likely a bug. Please report it on github"
            )
        spec.verify(sub)
        assignments = plan.distribute(sub, mpi_mode, nranks)
        child, child_uuids = plan.build_from_assignment(
            assignments[rank], sub, spec, open_kwargs
        )
        # A scope whose every dataset was gated out by load/if conditions builds to
        # None (see plan.build_from_assignment); drop it.
        if child is not None:
            children[scope_name] = child
            children_by_uuid[scope_name] = child_uuids
            available_uuids |= child_uuids

    if not children:
        raise ValueError("No valid datasets found!")

    # Resolve mapping files against the UUIDs actually opened on this rank.

    match_set = _resolve_match_set(map_layouts, available_uuids)

    # --- Cross-simulation connectivity check ---
    #
    # Detect root-origin scopes structurally: a scope is root-origin when every one
    # of its GroupLayouts sits at the root path ("/" or "").  Nested multi-simulation
    # scopes (e.g. /scidac1, /scidac2) have non-root paths and are excluded.  We do
    # NOT sniff the scope-key string to decide this — g.path is the real signal.
    #
    # Only scopes that actually produced a child count: a scope whose every dataset
    # was gated out by load/if conditions builds to None and was dropped from children
    # (see the `if child is not None:` guard above).
    root_scope_names = [
        scope_name
        for scope_name, sub in scopes.items()
        if scope_name in children
        and all(g.path in ("/", "") for fl in sub for g in fl.groups)
    ]

    if len(root_scope_names) > 1:
        # The set of UUIDs the mapping file connects (post-availability-filter).
        # See DatasetMatchSet.endpoints for why reference_source is included even
        # when that dataset was not opened.
        endpoints: frozenset[UUID] = (
            frozenset() if match_set is None else match_set.endpoints
        )

        # Per-scope UUIDs come from the discovered layouts, not from the built
        # children: layouts are identical on every rank, whereas a rank's actual
        # assignment is not under MpiMode.REDSHIFT.  Deriving the check from
        # layouts keeps it collective-safe.  Datasets without a persistent
        # on-disk ``main_uuid`` have a synthesized identity that can never be a
        # map endpoint.
        unconnected = [
            name
            for name in root_scope_names
            if not frozenset(
                g.uuid
                for fl in scopes[name]
                for g in fl.groups
                if g.has_persistent_uuid
            )
            & endpoints
        ]
        if unconnected:
            raise ValueError(
                "Cannot open datasets from different simulations without a connecting "
                "mapping file. Pass a mapping file that defines the correspondence "
                "between them. Unconnected datasets: "
                + ", ".join(f"'{n}'" for n in sorted(unconnected))
                + "."
            )
    if match_set is not None:
        if any(len(uuids) > 1 for uuids in children_by_uuid.values()):
            raise ValueError("Mapping is currently only supported for single catalogs")
        match_set = match_set.with_aliases(
            {
                normalize_kwarg_name(name): next(iter(uuids))
                for name, uuids in children_by_uuid.items()
            }
        )

    # A mapping resolves to nothing usable whenever fewer than two of its endpoints
    # were opened — the "ignore unresolvable endpoints" rule applied consistently.
    # The result is then exactly what it would have been without the mapping file,
    # including the bare single-child object.
    if len(children) == 1 and match_set is None:
        return next(iter(children.values()))

    return occ.SimulationCollection(children, match_set=match_set)


def _resolve_match_set(
    map_layouts: list[FileLayout], available: set[UUID]
) -> DatasetMatchSet | None:
    """
    Resolve discovered mapping files against the datasets actually opened.

    ``available`` is the set of UUIDs the caller actually opened; this function does
    not rebuild it.

    Only slots whose endpoints are all present survive, so one suite-wide mapping
    file can be opened against any subset of its simulations. Files are opened
    without a context manager and their handles deliberately outlive this call: the
    match set slices them lazily, exactly as build_from_assignment does for data.

    Pre-filtering with ``MapLayout.endpoints`` avoids opening files that cannot
    possibly resolve: a file is skipped entirely when none of its maps mention any
    available UUID. Files that are opened but yield no resolving map have their
    handle closed before moving on.

    First resolvable map wins. Multiple connecting map files are not merged — that
    is a real design decision, not a cleanup omission.

    This is pure computation over the layouts every rank already holds plus local
    file opens, so it introduces no collective and cannot diverge across ranks.
    """
    if not map_layouts:
        return None

    import h5py

    from opencosmo.mapping.read import read_match_set

    for layout in map_layouts:
        # Pre-filter: skip this file entirely if no map it carries mentions any
        # available UUID.  endpoints is a necessary-but-not-sufficient condition —
        # read_match_set may still return None for a file that passes (e.g. exactly
        # one primary target available with the reference absent).
        if not any(map_layout.endpoints & available for map_layout in layout.maps):
            continue

        f = h5py.File(layout.path, "r")
        winner = None
        for map_layout in layout.maps:
            group = f[map_layout.path]
            assert isinstance(group, h5py.Group)
            match_set = read_match_set(group, map_layout, available)
            if match_set is not None:
                winner = match_set
                break

        if winner is not None:
            # Leave this file's handle open: the match set slices it lazily.
            # First resolvable map wins; multiple connecting map files are not merged.
            return winner

        # No map in this file resolved — close the handle rather than leaking it.
        f.close()

    return None


def open_dataset(
    target: DatasetTarget,
    index: "IndexSpec",
    *,
    open_kwargs: dict[str, Any] = {},
) -> oc.Dataset:
    header = target.header

    assert header is not None

    try:
        box_size = header.with_units("scalefree").simulation["box_size"].value
    except AttributeError:
        box_size = None

    sim_region = None
    if header.file.region is not None:
        sim_region = from_model(header.file.region)
    elif header.file.data_type == "healpix_map":
        assert header.healpix_map["full_sky"]
        sim_region = FullSkyRegion()
    elif not header.file.is_lightcone:
        p1 = (0, 0, 0)
        p2 = tuple(header.simulation["box_size"].value for _ in range(3))
        sim_region = oc.make_box(p1, p2)

    if (spatial_index := target.spatial_index) is not None:
        tree = open_tree(
            spatial_index, box_size, header.file.is_lightcone, region=sim_region
        )
    else:
        tree = None

    comm = get_comm_world()
    data_index, sim_region = index(
        comm, header, target, tree, target.row_count, sim_region
    )
    if tree is not None:
        tree = tree.with_region(sim_region)

    state = st.state_from_target(
        target,
        UnitConvention.COMOVING,
        sim_region,
        open_kwargs,
        data_index,
        tree=tree,
    )

    dataset = oc.Dataset(
        state,
    )
    dataset = fold(HookPoint.DatasetOpen, DatasetOpenCtx(dataset, open_kwargs)).dataset
    return dataset


def _open_healpix_map(dataset: oc.Dataset):
    header = dataset.header
    sim_region: Region = FullSkyRegion()
    if header.file.region is not None:
        sim_region = from_model(header.file.region)

    if (comm := get_comm_world()) is not None and isinstance(
        sim_region, HealpixRegion
    ):  # partitioning has to be done manually since we don't store a spatial index
        pixels = sim_region.pixels
        splits = np.array(comm.allgather(len(dataset)))
        splits = np.insert(np.cumsum(splits), 0, 0)
        rank = comm.Get_rank()
        sim_region = HealpixRegion(
            pixels[splits[rank] : splits[rank + 1]],
            nside=header.healpix_map["nside"],
        )
    elif isinstance(sim_region, FullSkyRegion) or header.healpix_map["full_sky"]:
        sim_region = HealpixRegion(dataset.index, nside=header.healpix_map["nside"])

    assert isinstance(sim_region, HealpixRegion)
    return occ.HealpixMap(
        {"data": dataset},
        header.healpix_map["nside"],
        header.healpix_map["nside_lr"],
        header.healpix_map["ordering"],
        header.healpix_map["full_sky"],
        header.healpix_map["z_range"],
        region=sim_region,
    )


def _expand_lightcone_region(region, tree):
    region = region or tree.get_region()

    pixels = region.pixels
    npix_ratio = hp.nside2npix(2**tree.max_level) // hp.nside2npix(region.nside)
    pixels = pixels[:, None] * npix_ratio + np.arange(npix_ratio)
    pixels = pixels.flatten()

    full_pixels = tree.get_partitions_with_data(tree.max_level)
    full_pixels = np.intersect1d(pixels, full_pixels)
    return HealpixRegion(full_pixels, 2**tree.max_level)


def evaluate_load_conditions(
    target: DatasetTarget, open_kwargs: dict[str, bool]
) -> bool:
    """
    Datasets can define conditional loading via an addition group called "load/if".
    the "if" group can define parameters which must either be true or false for the
    given group to be loaded. These parameters can then be provided by the user to the
    "open" function. Parameters not specified by the user default to False.

    Note that some open kwargs may be used in other places in the opening process,
    and will just be ignored here.
    """
    conditions = target.load_conditions
    if conditions is None:
        return True
    return all(
        open_kwargs.get(key, False) == condition
        for key, condition in conditions.items()
    )
