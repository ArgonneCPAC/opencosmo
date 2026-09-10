from __future__ import annotations

import dataclasses
import json
import pickle
import random
import shutil
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from uuid import UUID

import h5py
import numpy as np
import pytest
from opencosmo.header import OpenCosmoHeader
from opencosmo.io.discover import (
    FileLayout,
    LinkSlotKind,
    decode_file_layout_blob,
    discover_all,
    discover_file,
    encode_file_layout_blob,
    group_data_type,
    has_linked_targets,
    is_healpix_map_group,
    is_lightcone_group,
    is_particle_group,
    is_properties_group,
)
from opencosmo.units.get import parse_unit_string
from opencosmo.uuid import get_column_uuid

import opencosmo as oc

if TYPE_CHECKING:
    from conftest import TestDataPaths


def _normalize_attr_for_test(value) -> str | None:
    """Independently normalize an HDF5 attribute value for comparison.

    Mirrors what discovery does (bytes -> utf-8 str) without importing
    discovery's own normalization helper, so the expectation stays an
    independent read. ``None`` (attribute absent) is preserved as-is and
    stays distinct from an empty string (attribute present but empty).
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_file_expected_layout(path: Path) -> dict:
    """
    Read a file with h5py and compute expected layout values.
    Returns dict with keys: column_names, column_dtypes, column_units,
    column_descriptions, row_count, has_index, linked_target_names.
    """
    with h5py.File(path, "r") as f:
        # Find data group (handle both "/" and nested cases)
        data_groups = []
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                if "data" in f[key]:
                    data_groups.append("/" + key if key != "/" else "/")

        if not data_groups:
            if "data" in f:
                data_groups = ["/"]

        result = {}
        for group_key in data_groups:
            group = f[group_key] if group_key != "/" else f

            # Get data group
            if "data" in group:
                data_grp = group["data"]
                columns_in_data = sorted(
                    [
                        k
                        for k in data_grp.keys()
                        if isinstance(data_grp[k], h5py.Dataset)
                    ]
                )
                column_names = tuple(columns_in_data)
                column_dtypes = tuple(
                    str(data_grp[col].dtype) for col in columns_in_data
                )
                column_units = tuple(
                    _normalize_attr_for_test(data_grp[col].attrs.get("unit"))
                    for col in columns_in_data
                )
                column_descriptions = tuple(
                    _normalize_attr_for_test(data_grp[col].attrs.get("description"))
                    for col in columns_in_data
                )
                row_count = (
                    data_grp[columns_in_data[0]].shape[0] if columns_in_data else 0
                )

                # Check for index
                has_index = "index" in group

                # Check for linked targets
                linked_target_names_set = set()
                if "data_linked" in group:
                    data_linked = group["data_linked"]
                    for key in data_linked.keys():
                        for suffix in ("_start", "_size", "_idx"):
                            if key.endswith(suffix):
                                linked_target_names_set.add(key.rsplit(suffix, 1)[0])
                                break

                linked_target_names = tuple(sorted(linked_target_names_set))

                result[group_key if group_key != "" else "/"] = {
                    "column_names": column_names,
                    "column_dtypes": column_dtypes,
                    "column_units": column_units,
                    "column_descriptions": column_descriptions,
                    "row_count": row_count,
                    "has_index": has_index,
                    "linked_target_names": linked_target_names,
                }

        return result


class TestDiscoverSingleFiles:
    """Test single-file discovery for various dataset types."""

    @pytest.mark.parametrize(
        "filename,is_lc",
        [
            ("haloproperties.hdf5", False),  # snapshot
            ("test_map.hdf5", False),  # healpix map
            ("lj_487.hdf5", False),  # diffsky
        ],
    )
    def test_discover_single_files(self, filename: str, is_lc: bool, test_data):
        """Test discovery of single-file datasets."""
        if filename == "haloproperties.hdf5":
            path = test_data.snapshot.primary.halo_properties
        elif filename == "test_map.hdf5":
            path = test_data.healpix_map
        else:
            path = test_data.diffsky.core(487)

        layout = discover_file(path)

        # No errors
        assert layout.error is None, f"Unexpected error: {layout.error}"

        # At least one group
        assert len(layout.groups) > 0

        # Each group matches what we read from the file
        expected_by_group = _read_file_expected_layout(path)

        for group in layout.groups:
            # Group path should be in expected data
            assert group.path in expected_by_group or "/" in expected_by_group

            group_key = group.path if group.path in expected_by_group else "/"
            expected = expected_by_group[group_key]

            # Check column names and dtypes
            assert group.column_names == expected["column_names"], (
                f"Column names mismatch for {group.path}"
            )
            assert group.column_dtypes == expected["column_dtypes"], (
                f"Column dtypes mismatch for {group.path}"
            )
            assert group.column_units == expected["column_units"], (
                f"Column units mismatch for {group.path}"
            )
            assert group.column_descriptions == expected["column_descriptions"], (
                f"Column descriptions mismatch for {group.path}"
            )

            # Check row count
            assert group.row_count == expected["row_count"], (
                f"Row count mismatch for {group.path}"
            )

            # Check index
            assert group.has_index == expected["has_index"], (
                f"has_index mismatch for {group.path}"
            )

            # Check linked targets
            assert group.linked_target_names == expected["linked_target_names"], (
                f"linked_target_names mismatch for {group.path}"
            )

            # Header should be an OpenCosmoHeader
            assert isinstance(group.header, OpenCosmoHeader)

    def test_discover_lightcone_file(self, test_data):
        """Test discovery of a lightcone file."""
        path = test_data.lightcone.step(600).halo_properties
        layout = discover_file(path)

        assert layout.error is None
        assert len(layout.groups) > 0

        group = layout.groups[0]
        assert is_lightcone_group(group)
        assert has_linked_targets(group)

        # Verify expected structure
        expected = _read_file_expected_layout(path)["/"]
        assert group.column_names == expected["column_names"]
        assert group.column_units == expected["column_units"]
        assert group.column_descriptions == expected["column_descriptions"]
        assert group.row_count == expected["row_count"]
        assert group.has_index == expected["has_index"]

    def test_discover_healpix_map(self, test_data):
        """Test discovery of HEALPix map file."""
        path = test_data.healpix_map
        layout = discover_file(path)

        assert layout.error is None
        assert len(layout.groups) > 0

        group = layout.groups[0]
        assert is_healpix_map_group(group)
        assert not group.has_index  # HEALPix maps don't have index

    def test_discover_no_h5py_leaks(self, test_data):
        """Test that no h5py handles leak from FileLayout."""
        path = test_data.snapshot.primary.halo_properties
        layout = discover_file(path)

        # Should be picklable (no live h5py handles)
        pickled = pickle.dumps(layout)
        unpickled = pickle.loads(pickled)

        assert unpickled.error is None
        assert len(unpickled.groups) == len(layout.groups)

    def test_dataset_uuid_persistence_and_stability(self, test_data):
        """Test persistent and synthesized dataset identities."""
        legacy_path = test_data.snapshot.primary.galaxy_properties
        first_legacy = discover_file(legacy_path).groups[0]
        second_legacy = discover_file(legacy_path).groups[0]

        assert first_legacy.uuid == second_legacy.uuid
        assert not first_legacy.has_persistent_uuid
        assert first_legacy.uuid is not None

        persistent_path = test_data.snapshot.primary.halo_properties
        persistent_group = discover_file(persistent_path).groups[0]
        with h5py.File(persistent_path, "r") as file:
            on_disk_uuid = UUID(str(file["data"].attrs["main_uuid"]))

        assert persistent_group.has_persistent_uuid
        assert persistent_group.uuid == on_disk_uuid

    def test_synthesized_dataset_uuid_resolves_file_path(self, test_data):
        """Test that synthesized identities do not depend on path spelling."""
        path = test_data.snapshot.primary.galaxy_properties
        alternate_spelling = path.parent / ".." / path.parent.name / path.name

        absolute_group = discover_file(path.absolute()).groups[0]
        alternate_group = discover_file(alternate_spelling).groups[0]

        assert absolute_group.uuid == alternate_group.uuid

    def test_synthesized_dataset_uuids_are_distinct(self, test_data):
        """Test that distinct legacy data groups have distinct identities."""
        galaxy_group = discover_file(
            test_data.snapshot.primary.galaxy_properties
        ).groups[0]
        profile_group = discover_file(test_data.snapshot.primary.halo_profiles).groups[
            0
        ]
        multi_groups = discover_file(test_data.snapshot.multi_simulation).groups

        assert galaxy_group.uuid != profile_group.uuid
        assert len({group.uuid for group in multi_groups}) == len(multi_groups)


class TestDiscoverMultiFile:
    """Test serial multi-file discovery."""

    def test_discover_all_serial(self, test_data):
        """Test discover_all on multiple files (serial mode)."""
        paths = [
            test_data.lightcone.step(600).halo_properties,
            test_data.lightcone.step(601).halo_properties,
        ]

        layouts = discover_all(paths, comm=None)

        assert len(layouts) == 2
        assert all(fl.error is None for fl in layouts)

        # Should be sorted by path
        path_strs = [str(fl.path) for fl in layouts]
        assert path_strs == sorted(path_strs)

    def test_discover_all_determinism(self, test_data):
        """Test that discover_all returns same result regardless of input order."""
        paths = [
            test_data.lightcone.step(600).halo_properties,
            test_data.lightcone.step(601).halo_properties,
        ]

        layouts1 = discover_all(paths, comm=None)

        # Shuffle and re-discover
        paths_shuffled = paths.copy()
        random.shuffle(paths_shuffled)
        layouts2 = discover_all(paths_shuffled, comm=None)

        # Should be identical (sorted by path)
        assert len(layouts1) == len(layouts2)
        for fl1, fl2 in zip(layouts1, layouts2):
            assert str(fl1.path) == str(fl2.path)
            assert len(fl1.groups) == len(fl2.groups)
            for g1, g2 in zip(fl1.groups, fl2.groups):
                assert g1.path == g2.path
                assert g1.header_path == g2.header_path


class TestDiscoverMalformed:
    """Test error handling for malformed files."""

    def test_discover_empty_hdf5(self, tmp_path):
        """Test discovery of an empty HDF5 file with no /header."""
        import h5py

        file_path = tmp_path / "empty.hdf5"
        with h5py.File(file_path, "w"):
            # Create an empty file with no header or data
            pass

        layout = discover_file(file_path)

        # Should return error, not raise
        assert layout.error is not None
        assert layout.groups == ()

    def test_discover_non_hdf5(self, tmp_path):
        """Test discovery of a non-HDF5 file."""
        file_path = tmp_path / "not_hdf5.txt"
        file_path.write_text("This is not an HDF5 file")

        layout = discover_file(file_path)

        # Should return error, not raise
        assert layout.error is not None
        assert layout.groups == ()

    def test_discover_header_but_no_data(self, tmp_path):
        """Test discovery of a file with /header but no /data."""
        import h5py

        file_path = tmp_path / "header_only.hdf5"
        with h5py.File(file_path, "w") as f:
            # Create header but no data
            header_grp = f.create_group("header")
            file_grp = header_grp.create_group("file")
            file_grp.attrs["data_type"] = "halo_properties"

        layout = discover_file(file_path)

        # Should return error (no data groups)
        assert layout.error is not None
        assert layout.groups == ()


class TestDiscoverLinkLayouts:
    """Test frozen /data_linked layouts and their validation."""

    @staticmethod
    def _copy_properties_file(destination: Path, test_data: TestDataPaths) -> None:
        """Copy a valid file whose header permits focused link-layout mutations."""
        shutil.copyfile(test_data.snapshot.primary.halo_properties, destination)

    @staticmethod
    def _replace_link_group(
        path: Path,
        slots: dict[str, tuple[np.dtype, list[int]]],
    ) -> None:
        """Replace /data_linked with the supplied flat datasets."""
        with h5py.File(path, "r+") as file:
            del file["data_linked"]
            link_group = file.create_group("data_linked")
            for name, (dtype, values) in slots.items():
                link_group.create_dataset(name, data=np.asarray(values, dtype=dtype))

    def test_chunked_slots_record_link_layout(
        self, tmp_path: Path, test_data: TestDataPaths
    ) -> None:
        """Chunked start/size slots retain their names, kind, and row length."""
        path = tmp_path / "chunked.hdf5"
        self._copy_properties_file(path, test_data)
        self._replace_link_group(
            path,
            {
                "halo_start": (np.dtype("int64"), [0, 4]),
                "halo_size": (np.dtype("uint32"), [4, 2]),
            },
        )

        layout = discover_file(path)

        assert layout.error is None
        assert layout.groups[0].link_layout is not None
        assert layout.groups[0].link_layout.path == "/data_linked"
        assert layout.groups[0].link_layout.slots[0].prefix == "halo"
        assert layout.groups[0].link_layout.slots[0].kind is LinkSlotKind.CHUNKED
        assert layout.groups[0].link_layout.slots[0].dataset_names == (
            "halo_start",
            "halo_size",
        )
        assert layout.groups[0].link_layout.slots[0].length == 2
        assert layout.groups[0].linked_target_names == ("halo",)

    def test_idx_slot_records_simple_kind(
        self, tmp_path: Path, test_data: TestDataPaths
    ) -> None:
        """An idx slot is retained as an explicitly simple link slot."""
        path = tmp_path / "idx.hdf5"
        self._copy_properties_file(path, test_data)
        self._replace_link_group(path, {"galaxy_idx": (np.dtype("int64"), [4, 2, 0])})

        link_layout = discover_file(path).groups[0].link_layout

        assert link_layout is not None
        assert len(link_layout.slots) == 1
        assert link_layout.slots[0].prefix == "galaxy"
        assert link_layout.slots[0].kind is LinkSlotKind.SIMPLE
        assert link_layout.slots[0].dataset_names == ("galaxy_idx",)
        assert link_layout.slots[0].length == 3

    def test_mixed_link_slots_are_sorted_and_picklable(
        self, tmp_path: Path, test_data: TestDataPaths
    ) -> None:
        """Mixed flat slot representations are sorted and retain no live handles."""
        path = tmp_path / "mixed.hdf5"
        self._copy_properties_file(path, test_data)
        self._replace_link_group(
            path,
            {
                "zeta_idx": (np.dtype("int64"), [1]),
                "alpha_size": (np.dtype("uint32"), [2]),
                "alpha_start": (np.dtype("int64"), [0]),
            },
        )

        layout = discover_file(path)

        assert layout.error is None
        assert layout.groups[0].link_layout is not None
        assert tuple(slot.prefix for slot in layout.groups[0].link_layout.slots) == (
            "alpha",
            "zeta",
        )
        assert pickle.loads(pickle.dumps(layout)) == layout

    @pytest.mark.parametrize(
        "slots",
        [
            {"halo_start": (np.dtype("int64"), [0])},
            {"halo_size": (np.dtype("uint32"), [1])},
            {
                "halo_start": (np.dtype("int64"), [0]),
                "halo_size": (np.dtype("uint32"), [1]),
                "halo_idx": (np.dtype("int64"), [0]),
            },
            {"halo_idx": (np.dtype("float64"), [0])},
            {"halo_idx": (np.dtype("int64"), [[0]])},
            {
                "halo_start": (np.dtype("int64"), [0]),
                "halo_size": (np.dtype("uint32"), [1, 2]),
            },
        ],
    )
    def test_malformed_link_slots_return_file_error(
        self,
        tmp_path: Path,
        test_data: TestDataPaths,
        slots: dict[str, tuple[np.dtype, list[int]]],
    ) -> None:
        """Malformed link slots fail discovery without raising."""
        path = tmp_path / "malformed-links.hdf5"
        self._copy_properties_file(path, test_data)
        self._replace_link_group(path, slots)

        layout = discover_file(path)

        assert layout.error is not None
        assert layout.groups == ()

    def test_map_chunked_slots_remain_rejected(
        self, tmp_path: Path, test_data: TestDataPaths
    ) -> None:
        """The /map validator must continue to reject chunked map slots."""
        path = tmp_path / "chunked-map.hdf5"
        self._copy_properties_file(path, test_data)
        with h5py.File(path, "r+") as file:
            map_group = file.create_group("map")
            map_group.attrs["reference"] = "12345678-1234-5678-1234-567812345678"
            map_group.attrs["format_version"] = 1
            slot = map_group.create_group(
                "primary/87654321-4321-8765-4321-876543218765"
            )
            slot.create_dataset("start", data=np.asarray([0], dtype=np.int64))
            slot.create_dataset("size", data=np.asarray([1], dtype=np.uint32))

        layout = discover_file(path)

        assert layout.error is not None
        assert "chunked" in layout.error

    def test_real_link_layout_prefixes(self, test_data: TestDataPaths) -> None:
        """The snapshot halo-properties file exposes its flat link prefixes."""
        layout = discover_file(test_data.snapshot.primary.halo_properties)

        assert layout.error is None
        assert layout.groups[0].link_layout is not None
        assert {
            "sodbighaloparticles_star_particles",
            "sod_profile",
            "galaxyproperties",
        }.issubset({slot.prefix for slot in layout.groups[0].link_layout.slots})


class TestHelperFunctions:
    """Test helper functions on discovered groups."""

    def test_group_data_type(self, test_data):
        """Test group_data_type helper."""
        path = test_data.snapshot.primary.halo_properties
        layout = discover_file(path)
        group = layout.groups[0]

        dtype = group_data_type(group)
        assert dtype == "halo_properties"

    def test_is_particle_group(self, test_data):
        """Test is_particle_group helper."""
        # haloproperties is not a particle group
        path = test_data.snapshot.primary.halo_properties
        layout = discover_file(path)
        group = layout.groups[0]

        assert not is_particle_group(group)

        # haloparticles should be a particle group
        path = test_data.snapshot.primary.halo_particles
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_particle_group(group)

    def test_is_properties_group(self, test_data):
        """Test is_properties_group helper."""
        # haloproperties should be a properties group
        path = test_data.snapshot.primary.halo_properties
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_properties_group(group)

        # galaxyproperties should be a properties group
        path = test_data.snapshot.primary.galaxy_properties
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_properties_group(group)

    def test_is_lightcone_group(self, test_data):
        """Test is_lightcone_group helper."""
        path = test_data.lightcone.step(600).halo_properties
        layout = discover_file(path)
        group = layout.groups[0]

        assert is_lightcone_group(group)

    def test_is_healpix_map_group(self, test_data):
        """Test is_healpix_map_group helper."""
        path = test_data.healpix_map
        layout = discover_file(path)
        group = layout.groups[0]
        assert is_healpix_map_group(group)


class TestFileLayoutBlobRoundTrip:
    """column_units/column_descriptions must survive the JSON cache round trip
    with None (absent), "" (present but empty), and a real unit string all
    still distinct from one another."""

    def test_units_and_descriptions_survive_json_round_trip(self, test_data):
        """encode -> json.dumps -> json.loads -> decode preserves the mix."""
        # Cheapest way to get a real OpenCosmoHeader without constructing one
        # by hand: discover a real file and reuse its group's header.
        path = test_data.snapshot.primary.halo_properties
        discovered = discover_file(path)
        base_group = discovered.groups[0]

        n = len(base_group.column_names)
        assert n >= 3, "test file needs at least 3 columns to exercise the mix"

        column_units = tuple(
            None if i % 3 == 0 else ("" if i % 3 == 1 else "comoving Mpc/h")
            for i in range(n)
        )
        column_descriptions = tuple(
            None if i % 3 == 0 else ("" if i % 3 == 1 else "a real description")
            for i in range(n)
        )

        group = dataclasses.replace(
            base_group,
            column_units=column_units,
            column_descriptions=column_descriptions,
        )
        layout = FileLayout(path=discovered.path, groups=(group,))

        blob = encode_file_layout_blob(layout)
        json_blob = json.loads(json.dumps(blob))
        decoded = decode_file_layout_blob(json_blob)

        assert decoded is not None, "sha256 check should have passed"

        decoded_group = decoded.groups[0]
        assert decoded_group.column_units[0] is None
        assert decoded_group.column_units[1] == ""
        assert decoded_group.column_units[2] == "comoving Mpc/h"

        assert decoded_group.column_descriptions[0] is None
        assert decoded_group.column_descriptions[1] == ""
        assert decoded_group.column_descriptions[2] == "a real description"


class TestWarmCacheZeroAttributeReads:
    """The whole point of caching column_units/column_descriptions/uuids on
    GroupLayout: opening a dataset with a warm layout cache must perform
    zero per-column HDF5 attribute reads. Without this test, a future change
    could silently reintroduce those reads with no visible symptom."""

    def test_warm_open_reads_no_unit_or_description_attrs(self, test_data, monkeypatch):
        paths = [
            test_data.lightcone.step(600).halo_properties,
            test_data.lightcone.step(601).halo_properties,
        ]

        # First open warms the on-disk layout cache. Not measured: this open
        # is expected to read attributes since the cache starts cold.
        oc.open(*paths)

        calls: list[object] = []
        orig_get = h5py.AttributeManager.get
        orig_getitem = h5py.AttributeManager.__getitem__

        def _get(self, k, *a, **kw):
            calls.append(k)
            return orig_get(self, k, *a, **kw)

        def _getitem(self, k):
            calls.append(k)
            return orig_getitem(self, k)

        monkeypatch.setattr(h5py.AttributeManager, "get", _get)
        monkeypatch.setattr(h5py.AttributeManager, "__getitem__", _getitem)

        # Same path list, so the warm cache actually hits.
        oc.open(*paths)

        unit_or_description_calls = [k for k in calls if k in ("unit", "description")]
        assert unit_or_description_calls == [], (
            f"warm open should not read column attrs, got: {calls}"
        )


class TestLayoutEquivalenceWithH5py:
    """Units, descriptions, and column UUIDs obtained via the cached
    layout/open path must match what a direct h5py attribute read produces."""

    def test_units_descriptions_and_uuids_match_direct_h5py_reads(self, test_data):
        path = test_data.snapshot.primary.halo_properties
        resolved_path = path.resolve()

        ds = oc.open(path)

        # Reaching into Dataset internals: there is no public accessor for
        # the per-column UnitApplicator or the raw base unit.
        applicators = ds._state.unit_handler._UnitHandler__applicators
        column_map = ds._state.column_map

        with h5py.File(resolved_path, "r") as f:
            data_grp = f["data"]
            for name in ds.columns:
                col = data_grp[name]

                expected_unit = parse_unit_string(col.attrs.get("unit"))
                actual_unit = applicators[name].base_unit
                if expected_unit is None:
                    assert actual_unit is None, f"unit mismatch for {name}"
                else:
                    assert actual_unit is not None, f"unit mismatch for {name}"
                    assert actual_unit == expected_unit, f"unit mismatch for {name}"

                expected_description = col.attrs.get("description")
                assert ds.descriptions[name] == expected_description, (
                    f"description mismatch for {name}"
                )

                expected_uuid = get_column_uuid(resolved_path, col.name)
                assert column_map[name] == expected_uuid, f"uuid mismatch for {name}"
