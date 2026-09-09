from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid5

import pytest
from opencosmo.io.discover import FileLayout, GroupLayout
from opencosmo.io.specs import (
    DatasetSpec,
    HealpixMapSpec,
    LightconeSpec,
    StructureCollectionSpec,
    group_by_scope,
    match_spec,
)
from opencosmo.uuid import NAMESPACE


# Mock header structure to build GroupLayouts without touching HDF5. group_data_type
# does str(header.file.data_type), so a plain str data_type behaves like the real
# DatasetType enum's str().
@dataclass(frozen=True)
class _FakeFile:
    step: int | None
    data_type: str
    is_lightcone: bool


class _FakeHeader:
    """Fake header that mimics OpenCosmoHeader's simulation_info access pattern.

    ``header.simulation["name"]`` works when a simulation name is set;
    accessing ``.simulation_info`` raises ``AttributeError`` when the name is
    ``None``, matching the real header's behaviour for files that lack the
    ``header/simulation`` ``name`` attribute.
    """

    def __init__(self, file: _FakeFile, simulation_name: str | None = None) -> None:
        self.file = file
        self._simulation_name = simulation_name

    @property
    def simulation(self) -> dict[str, str]:
        if self._simulation_name is None:
            raise AttributeError("simulation")
        return {"name": self._simulation_name}


def _group(
    *,
    path: str = "/",
    header_path: str = "/header",
    data_type: str = "halo_properties",
    is_lightcone: bool = False,
    step: int | None = None,
    linked: tuple[str, ...] = (),
    columns: tuple[str, ...] = ("x", "y"),
    dtypes: tuple[str, ...] = ("float64", "float64"),
    has_index: bool = True,
    simulation_name: str | None = "sim_a",
    uuid_=None,
    has_persistent_uuid: bool = False,
) -> GroupLayout:
    if uuid_ is None:
        uuid_ = uuid5(NAMESPACE, f"{Path(path).resolve()}::{path}/data")
    return GroupLayout(
        path=path,
        header_path=header_path,
        header=_FakeHeader(_FakeFile(step, data_type, is_lightcone), simulation_name),  # type: ignore[arg-type]
        column_names=columns,
        column_dtypes=dtypes,
        row_count=100,
        has_index=has_index,
        linked_target_names=linked,
        uuid=uuid_,
        has_persistent_uuid=has_persistent_uuid,
    )


def _file(path: str, *groups: GroupLayout, error: str | None = None) -> FileLayout:
    return FileLayout(path=Path(path), groups=tuple(groups), error=error)


class TestMatchSpec:
    """match_spec returns the correct spec for each single-scope file type."""

    def test_dataset(self) -> None:
        layouts = (_file("/a.hdf5", _group(data_type="halo_particles")),)
        assert isinstance(match_spec(layouts), DatasetSpec)

    def test_healpix_map(self) -> None:
        layouts = (_file("/m.hdf5", _group(data_type="healpix_map")),)
        assert isinstance(match_spec(layouts), HealpixMapSpec)

    def test_lightcone_single_file(self) -> None:
        layouts = (
            _file(
                "/lc.hdf5",
                _group(data_type="halo_particles", is_lightcone=True, step=0),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)

    def test_lightcone_multi_file(self) -> None:
        layouts = (
            _file(
                "/lc0.hdf5",
                _group(data_type="halo_particles", is_lightcone=True, step=0),
            ),
            _file(
                "/lc1.hdf5",
                _group(data_type="halo_particles", is_lightcone=True, step=1),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)

    def test_structure_collection(self) -> None:
        layouts = (
            _file(
                "/sc.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(path="/halo_particles", data_type="halo_particles"),
            ),
        )
        assert isinstance(match_spec(layouts), StructureCollectionSpec)

    def test_lone_properties_with_links_is_lightcone_not_sc(self) -> None:
        # Regression: a single properties file carrying /data_linked, opened without
        # its linked children, opens as a Lightcone (old oc.open behavior) — NOT a
        # structure collection. The collection requires >=1 linked child type to be
        # present (i.e. more than one distinct data_type opened together).
        layouts = (
            _file(
                "/haloproperties.hdf5",
                _group(
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)

    def test_multiple_properties_of_one_type_is_lightcone_not_sc(self) -> None:
        # Regression: several properties files of a single data_type across redshift
        # steps (each with /data_linked) open as a Lightcone, not a structure
        # collection — there is still only one distinct data_type.
        layouts = (
            _file(
                "/hp0.hdf5",
                _group(
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
            ),
            _file(
                "/hp1.hdf5",
                _group(
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=1,
                ),
            ),
        )
        assert isinstance(match_spec(layouts), LightconeSpec)


class TestPrecedence:
    """Spec ordering resolves the ambiguous cases correctly."""

    def test_lightcone_structure_collection_matches_sc_not_lightcone(self) -> None:
        # A lightcone structure collection has several data_types, so LightconeSpec
        # (single data_type) must not fire; StructureCollectionSpec catches it.
        layouts = (
            _file(
                "/lcsc.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                ),
            ),
        )
        assert isinstance(match_spec(layouts), StructureCollectionSpec)


class TestGroupByScope:
    """group_by_scope decomposes a multi-scope layout into per-simulation scopes.

    A simulation collection is no longer a spec: the caller splits by header scope
    and builds each scope through the ordinary single-scope path, so each spec only
    ever sees a single scope.
    """

    def test_single_scope_is_one_group(self) -> None:
        layouts = (
            _file(
                "/sc.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(path="/halo_particles", data_type="halo_particles"),
            ),
        )
        scopes = group_by_scope(layouts)
        assert list(scopes) == ["/"]
        # The single scope round-trips to its leaf spec.
        assert isinstance(match_spec(scopes["/"]), StructureCollectionSpec)

    def test_nested_scopes_split_and_each_matches_a_leaf(self) -> None:
        layouts = (
            _file(
                "/sim.hdf5",
                _group(
                    path="/scidac1/halo_properties",
                    header_path="/scidac1/header",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(
                    path="/scidac1/halo_particles",
                    header_path="/scidac1/header",
                    data_type="halo_particles",
                ),
                _group(
                    path="/scidac2/halo_properties",
                    header_path="/scidac2/header",
                    data_type="halo_properties",
                    linked=("haloparticles",),
                ),
                _group(
                    path="/scidac2/halo_particles",
                    header_path="/scidac2/header",
                    data_type="halo_particles",
                ),
            ),
        )
        scopes = group_by_scope(layouts)
        # Sorted per-simulation scopes, each a single-scope sub-layout.
        assert list(scopes) == ["scidac1", "scidac2"]
        for name, sub in scopes.items():
            assert {g.header_path for fl in sub for g in fl.groups} == {
                f"/{name}/header"
            }
            assert isinstance(match_spec(sub), StructureCollectionSpec)

    def test_errored_files_skipped(self) -> None:
        layouts = (
            _file("/good.hdf5", _group(data_type="halo_particles")),
            _file("/bad.hdf5", error="malformed"),
        )
        scopes = group_by_scope(layouts)
        assert list(scopes) == ["/"]
        assert isinstance(match_spec(scopes["/"]), DatasetSpec)

    def test_two_root_groups_different_simulations_split_by_name(self) -> None:
        # Two separate files, each with one root halo_properties group from a
        # different simulation. They do not match any spec together (DatasetSpec
        # requires exactly one group), so they fall through to the split branch and
        # must be keyed by simulation name.
        layouts = (
            _file(
                "/sim_a.hdf5",
                _group(data_type="halo_properties", simulation_name="alpha"),
            ),
            _file(
                "/sim_b.hdf5",
                _group(data_type="halo_properties", simulation_name="beta"),
            ),
        )
        scopes = group_by_scope(layouts)
        assert list(scopes) == ["alpha", "beta"]

    def test_root_group_missing_simulation_name_raises(self) -> None:
        # A root group whose header has no simulation_info must raise ValueError
        # naming the offending file path.
        layouts = (
            _file(
                "/sim_a.hdf5",
                _group(data_type="halo_properties", simulation_name="alpha"),
            ),
            _file(
                "/no_name.hdf5",
                _group(data_type="halo_properties", simulation_name=None),
            ),
        )
        with pytest.raises(ValueError, match="/no_name.hdf5"):
            group_by_scope(layouts)

    def test_two_root_groups_same_simulation_name_different_files_raises(self) -> None:
        # Two root groups from different files that resolve to the same simulation
        # name must raise ValueError naming both paths and the shared name.
        layouts = (
            _file(
                "/step310.hdf5",
                _group(data_type="halo_properties", simulation_name="LastJourney"),
            ),
            _file(
                "/step624.hdf5",
                _group(data_type="halo_properties", simulation_name="LastJourney"),
            ),
        )
        with pytest.raises(ValueError, match="LastJourney") as exc_info:
            group_by_scope(layouts)
        msg = str(exc_info.value)
        assert "/step310.hdf5" in msg
        assert "/step624.hdf5" in msg

    def test_nested_multi_sim_split_across_files_no_collision(self) -> None:
        # Regression: a nested multi-simulation structure collection split across two
        # files — file A holds /scidac1/halo_properties and /scidac2/halo_properties,
        # file B holds /scidac1/halo_particles and /scidac2/halo_particles.  Both files
        # legitimately contribute to scopes "scidac1" and "scidac2"; this must NOT
        # raise even though two different file paths share the same scope keys.
        file_a = _file(
            "/props.hdf5",
            _group(
                path="/scidac1/halo_properties",
                header_path="/scidac1/header",
                data_type="halo_properties",
                linked=("haloparticles",),
            ),
            _group(
                path="/scidac2/halo_properties",
                header_path="/scidac2/header",
                data_type="halo_properties",
                linked=("haloparticles",),
            ),
        )
        file_b = _file(
            "/parts.hdf5",
            _group(
                path="/scidac1/halo_particles",
                header_path="/scidac1/header",
                data_type="halo_particles",
            ),
            _group(
                path="/scidac2/halo_particles",
                header_path="/scidac2/header",
                data_type="halo_particles",
            ),
        )
        layouts = (file_a, file_b)
        scopes = group_by_scope(layouts)
        assert list(scopes) == ["scidac1", "scidac2"]
        # Each scope must contain groups from both files (two FileLayouts per scope).
        for name, sub in scopes.items():
            assert len(sub) == 2
            assert isinstance(match_spec(sub), StructureCollectionSpec)


class TestVerify:
    """verify() raises on cross-file structural inconsistency."""

    def test_lightcone_column_mismatch_raises(self) -> None:
        layouts = (
            _file(
                "/lc0.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                    columns=("x", "y"),
                ),
            ),
            _file(
                "/lc1.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=1,
                    columns=("x", "z"),
                ),
            ),
        )
        spec = match_spec(layouts)
        assert isinstance(spec, LightconeSpec)
        with pytest.raises(ValueError):
            spec.verify(layouts)

    def test_lightcone_dtype_mismatch_raises(self) -> None:
        layouts = (
            _file(
                "/lc0.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                    dtypes=("float64", "float64"),
                ),
            ),
            _file(
                "/lc1.hdf5",
                _group(
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=1,
                    dtypes=("float64", "int64"),
                ),
            ),
        )
        with pytest.raises(ValueError):
            LightconeSpec().verify(layouts)

    def test_structure_collection_inconsistent_type_sets_raises(self) -> None:
        # step0 has {halo_properties, halo_particles}, step1 only {halo_properties}.
        layouts = (
            _file(
                "/step0.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                ),
            ),
            _file(
                "/step1.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=1,
                ),
            ),
        )
        spec = match_spec(layouts)
        assert isinstance(spec, StructureCollectionSpec)
        with pytest.raises(ValueError):
            spec.verify(layouts)

    def test_consistent_structure_collection_passes(self) -> None:
        layouts = (
            _file(
                "/step0.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=0,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=0,
                ),
            ),
            _file(
                "/step1.hdf5",
                _group(
                    path="/halo_properties",
                    data_type="halo_properties",
                    is_lightcone=True,
                    linked=("haloparticles",),
                    step=1,
                ),
                _group(
                    path="/halo_particles",
                    data_type="halo_particles",
                    is_lightcone=True,
                    step=1,
                ),
            ),
        )
        spec = match_spec(layouts)
        assert isinstance(spec, StructureCollectionSpec)
        # Consistent columns + type sets across steps -> no raise.
        spec.verify(layouts)
