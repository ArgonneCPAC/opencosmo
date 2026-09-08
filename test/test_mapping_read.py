from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import h5py
import numpy as np
import pytest
from opencosmo.io.discover import (
    LinkSlotKind,
    _read_link_layout,
    _read_map_layout,
    discover_file,
)
from opencosmo.mapping.mapping import (
    ChunkedSlot,
    DatasetMatchSet,
    get_auxillary_mapping,
    get_mapping,
    get_primary_mapping,
    get_slot_sizes,
    rebuild_single_with_new_source,
    rebuild_single_with_source,
    rebuild_target_index,
)
from opencosmo.mapping.read import read_link_set, read_match_set

from opencosmo.index import coalesce_chunks

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import TestDataPaths
    from opencosmo.io.discover import LinkLayout, MapLayout

    from opencosmo.index import DataIndex

# Fixed UUIDs used across tests.
_REF = UUID("00000000-0000-0000-0000-000000000001")
_A = UUID("00000000-0000-0000-0000-000000000002")
_B = UUID("00000000-0000-0000-0000-000000000003")
_C = UUID("00000000-0000-0000-0000-000000000004")


# Frozen copies of the pre-migration index rebuilders that used to live in
# collection/structure/handler.py. They are the oracle that rebuild_target_index
# must keep agreeing with, so they are pinned here rather than imported.
def rebuild_row_index(
    original_metadata_column: np.ndarray,
    index_into_original: np.ndarray,
) -> np.ndarray:
    valid_rows = original_metadata_column >= 0
    index = np.full(len(original_metadata_column), -1, dtype=np.int64)
    index[valid_rows] = np.arange(0, sum(valid_rows))
    index_to_take = index[index_into_original]
    return index_to_take[index_to_take >= 0]


def rebuild_chunk_index(
    original_size_column: np.ndarray,
    index_into_original: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_boundaries = np.zeros(len(original_size_column) + 1, dtype=np.int64)
    _ = np.cumsum(original_size_column, out=chunk_boundaries[1:])
    valid_rows = original_size_column[index_into_original] > 0

    starts = chunk_boundaries[index_into_original[valid_rows]]
    sizes = original_size_column[index_into_original[valid_rows]]

    return coalesce_chunks(starts, sizes)


def _make_chunked_match_set(file: h5py.File, sizes: np.ndarray) -> DatasetMatchSet:
    start = file.create_dataset("start", data=np.array([0, 2, 2, 5, 7], dtype=np.int64))
    size = file.create_dataset("size", data=sizes)
    return DatasetMatchSet(_REF, {_A: ChunkedSlot(start, size)}, {})


def _make_link_group(
    file: h5py.File, slots: dict[str, np.ndarray]
) -> tuple[h5py.Group, LinkLayout]:
    """Write flat link datasets and parse their layout through discovery."""
    link_group = file.create_group("data_linked")
    for name, values in slots.items():
        link_group.create_dataset(name, data=values)
    return link_group, _read_link_layout("/data_linked", link_group)


def test_read_link_set_chunked_slot_maps_source_rows(tmp_path: Path) -> None:
    starts = np.array([0, 2, 2, 5], dtype=np.int64)
    sizes = np.array([2, 0, 3, 1], dtype=np.uint32)
    source_index = np.array([3, 0, 2, 1], dtype=np.int64)
    with h5py.File(tmp_path / "links.hdf5", "w") as file:
        link_group, layout = _make_link_group(
            file,
            {
                "particles_start": starts,
                "particles_size": sizes,
            },
        )
        match_set = read_link_set(link_group, layout, _REF, {"particles": _A})

        assert match_set is not None
        result = get_primary_mapping(match_set, _REF, _A, source_index)

    assert isinstance(match_set.primary_maps[_A], ChunkedSlot)
    assert isinstance(result, tuple)
    expected = coalesce_chunks(
        starts[source_index][sizes[source_index] > 0],
        sizes[source_index][sizes[source_index] > 0],
    )
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_read_link_set_idx_slot_resolves_simple_slot(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "links.hdf5", "w") as file:
        link_group, layout = _make_link_group(
            file, {"galaxies_idx": np.array([4, -1, 2], dtype=np.int64)}
        )
        match_set = read_link_set(link_group, layout, _REF, {"galaxies": _A})

        assert match_set is not None
        slot = match_set.primary_maps[_A]
        assert not isinstance(slot, ChunkedSlot)
        result = get_primary_mapping(match_set, _REF, _A, np.array([2, 0]))
        assert slot.name == "/data_linked/galaxies_idx"

    np.testing.assert_array_equal(result, [2, 4])


def test_read_link_set_mixed_slots_and_source_topology(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "links.hdf5", "w") as file:
        link_group, layout = _make_link_group(
            file,
            {
                "profiles_start": np.array([0, 2], dtype=np.int64),
                "profiles_size": np.array([2, 1], dtype=np.uint32),
                "particles_idx": np.array([3, 1], dtype=np.int64),
            },
        )
        match_set = read_link_set(
            link_group, layout, _REF, {"profiles": _A, "particles": _B}
        )

        assert match_set is not None
        assert isinstance(match_set.primary_maps[_A], ChunkedSlot)
        assert not isinstance(match_set.primary_maps[_B], ChunkedSlot)
        assert match_set.reference_source == _REF
        assert match_set.aux_maps == {}


def test_read_link_set_skips_unopened_targets_and_returns_none(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "links.hdf5", "w") as file:
        link_group, layout = _make_link_group(
            file,
            {
                "profiles_start": np.array([0], dtype=np.int64),
                "profiles_size": np.array([1], dtype=np.uint32),
                "particles_idx": np.array([0], dtype=np.int64),
            },
        )
        match_set = read_link_set(link_group, layout, _REF, {"profiles": _A})
        missing = read_link_set(link_group, layout, _REF, {})

    assert match_set is not None
    assert set(match_set.primary_maps) == {_A}
    assert missing is None


def test_read_link_set_real_hacc_chunked_slot(test_data: TestDataPaths) -> None:
    path = test_data.snapshot.primary.halo_properties
    if not path.is_file():
        pytest.skip("repository test data is not available")

    layout = discover_file(path)
    assert layout.error is None
    link_layout = layout.groups[0].link_layout
    assert link_layout is not None
    chunked_slot = next(
        slot for slot in link_layout.slots if slot.kind is LinkSlotKind.CHUNKED
    )
    target_uuid = _A

    with h5py.File(path, "r") as file:
        link_group = file[link_layout.path]
        match_set = read_link_set(
            link_group, link_layout, _REF, {chunked_slot.prefix: target_uuid}
        )
        assert match_set is not None
        start_name, size_name = chunked_slot.dataset_names
        source_index = np.arange(min(8, chunked_slot.length), dtype=np.int64)
        starts = link_group[start_name][source_index]
        sizes = link_group[size_name][source_index].astype(np.int64)
        result = get_primary_mapping(match_set, _REF, target_uuid, source_index)

    assert isinstance(result, tuple)
    expected = coalesce_chunks(starts[sizes > 0], sizes[sizes > 0])
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_chunked_primary_mapping_round_trips_and_coalesces(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        match_set = _make_chunked_match_set(
            file, np.array([2, 0, 3, 2, 1], dtype=np.uint32)
        )
        index = np.arange(5, dtype=np.int64)

        primary = get_primary_mapping(match_set, _REF, _A, index)
        mapped = get_mapping(match_set, _REF, _A, index)

    assert isinstance(primary, tuple)
    assert mapped is not None
    assert isinstance(mapped, tuple)
    np.testing.assert_array_equal(primary[0], [0])
    np.testing.assert_array_equal(primary[1], [8])
    np.testing.assert_array_equal(mapped[0], primary[0])
    np.testing.assert_array_equal(mapped[1], primary[1])
    assert len(primary[0]) != len(index)


def test_chunked_mappings_reject_unsupported_routes(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        match_set = _make_chunked_match_set(
            file, np.array([1, 1, 1, 1, 1], dtype=np.int64)
        )
        simple = file.create_dataset("simple", data=np.arange(5, dtype=np.int64))
        aux_source = file.create_dataset("aux_source", data=[0])
        aux_target = file.create_dataset("aux_target", data=[0])
        match_set = DatasetMatchSet(
            _REF,
            {_A: match_set.primary_maps[_A], _B: simple},
            {(_REF, _A): (aux_source, aux_target)},
        )
        index = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError, match="reference-to-target"):
            get_primary_mapping(match_set, _A, _REF, index)
        with pytest.raises(ValueError, match="reference-to-target"):
            get_primary_mapping(match_set, _A, _B, index)
        with pytest.raises(ValueError, match="reference-to-target"):
            get_auxillary_mapping(match_set, _REF, _A, index)


def test_rebuild_target_index_matches_structure_rebuilders(tmp_path: Path) -> None:
    old_source_index = np.array([4, 0, 3, 1, 2], dtype=np.int64)
    new_source_index = np.array([1, 4, 2], dtype=np.int64)
    index_into_original = np.array([3, 0, 4], dtype=np.int64)
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        chunked_match_set = _make_chunked_match_set(
            file, np.array([2, 1, 3, 0, 4], dtype=np.uint32)
        )
        simple = file.create_dataset("simple", data=[8, -1, 12, 9, 10])
        simple_match_set = DatasetMatchSet(_REF, {_A: simple}, {})

        chunked = rebuild_target_index(
            chunked_match_set, _A, old_source_index, new_source_index
        )
        simple_result = rebuild_target_index(
            simple_match_set, _A, old_source_index, new_source_index
        )

    expected_chunked = rebuild_chunk_index(
        np.array([4, 2, 0, 1, 3], dtype=np.int64), index_into_original
    )
    expected_simple = rebuild_row_index(
        np.array([10, 8, 9, -1, 12], dtype=np.int64), index_into_original
    )
    assert isinstance(chunked, tuple)
    np.testing.assert_array_equal(chunked[0], expected_chunked[0])
    np.testing.assert_array_equal(chunked[1], expected_chunked[1])
    np.testing.assert_array_equal(simple_result, expected_simple)


def test_chunked_slot_sizes_and_rebuilt_indices_are_signed_int64(
    tmp_path: Path,
) -> None:
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        match_set = _make_chunked_match_set(
            file, np.array([2, 0, 3, 2, 1], dtype=np.uint32)
        )
        simple = file.create_dataset("simple", data=np.arange(5, dtype=np.int64))
        simple_match_set = DatasetMatchSet(_REF, {_A: simple}, {})
        index = np.arange(5, dtype=np.int64)

        sizes = get_slot_sizes(match_set, _A, index)
        rebuilt = rebuild_target_index(match_set, _A, index, index)
        with pytest.raises(ValueError, match="do not have a size column"):
            get_slot_sizes(simple_match_set, _A, index)

    assert sizes.dtype == np.int64
    assert isinstance(rebuilt, tuple)
    assert rebuilt[0].dtype == np.int64
    assert rebuilt[1].dtype == np.int64


@pytest.mark.parametrize("writer", ("make_schema", "with_source", "with_new_source"))
def test_writing_chunked_slots_is_rejected(tmp_path: Path, writer: str) -> None:
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        match_set = _make_chunked_match_set(
            file, np.array([1, 1, 1, 1, 1], dtype=np.int64)
        ).with_aliases({"reference": _REF, "target": _A})
        indices: dict[str, DataIndex] = {
            "reference": np.arange(5, dtype=np.int64),
            "target": np.arange(5, dtype=np.int64),
        }

        with pytest.raises(
            ValueError, match="Writing chunked mappings is not yet supported"
        ):
            if writer == "make_schema":
                match_set.make_schema({"reference": _REF, "target": _A}, indices)
            elif writer == "with_source":
                rebuild_single_with_source(
                    match_set, {"reference": _REF, "target": _A}, indices, "reference"
                )
            else:
                rebuild_single_with_new_source(
                    match_set, {"reference": _REF, "target": _A}, indices, "target"
                )


def test_missing_primary_route_has_clear_error() -> None:
    match_set = DatasetMatchSet(
        reference_source=_REF,
        primary_maps={},
        aux_maps={},
        aliases={"source": _A, "target": _B},
    )

    with pytest.raises(
        ValueError,
        match="Unable to map from 'source' to 'target'.*no primary mapping route",
    ):
        get_mapping(match_set, "source", "target", np.array([0], dtype=np.int64))


@pytest.mark.parametrize(
    ("source", "target", "index", "expected"),
    (
        (_REF, _A, [3, 0, 1, 2], [1, 2, -1, 0]),
        (_A, _REF, [3, 1, 2, 0], [-1, 3, 0, 2]),
        (_REF, _B, [3, 0, 1, 2], [0, 1, 2, -1]),
        (_B, _REF, [4, 0, 1, 2], [-1, 3, 0, 1]),
        (_A, _B, [3, 1, 2, 0], [4, 0, 1, -1]),
        (_B, _A, [4, 0, 1, 2], [3, 1, 2, -1]),
    ),
)
def test_get_mapping_directions(tmp_path, source, target, index, expected) -> None:
    with h5py.File(tmp_path / "mapping.hdf5", "w") as file:
        primary_a = file.create_dataset("primary_a", data=[2, -1, 0, 1])
        primary_b = file.create_dataset("primary_b", data=[1, 2, -1, 0])
        auxiliary_a = file.create_dataset("auxiliary_a", data=[3])
        auxiliary_b = file.create_dataset("auxiliary_b", data=[4])
        match_set = DatasetMatchSet(
            reference_source=_REF,
            primary_maps={_A: primary_a, _B: primary_b},
            aux_maps={(_A, _B): (auxiliary_a, auxiliary_b)},
        )

        result = get_mapping(
            match_set, source, target, np.asarray(index, dtype=np.int64)
        )

    # Every route in this parametrization uses simple slots, so the result is a
    # plain index array rather than a chunked (start, size) pair.
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)


def _write_simple_slot(group: h5py.Group, name: str, data: list[int]) -> None:
    """Write a simple (index-only) slot under *group*."""
    slot = group.require_group(name)
    slot.create_dataset("index", data=np.array(data, dtype=np.int64))


def _write_aux_pair(
    aux_group: h5py.Group,
    uuid_a: UUID,
    uuid_b: UUID,
    source_data: list[int],
    target_data: list[int],
) -> None:
    """Write an auxiliary pair slot under *aux_group*."""
    pair_name = f"{uuid_a}__{uuid_b}"
    pair = aux_group.require_group(pair_name)
    pair.create_dataset("source", data=np.array(source_data, dtype=np.int64))
    pair.create_dataset("target", data=np.array(target_data, dtype=np.int64))


def _make_map_group(
    f: h5py.File,
    reference: UUID,
    primary_targets: dict[UUID, list[int]] | None = None,
    aux_pairs: dict[tuple[UUID, UUID], tuple[list[int], list[int]]] | None = None,
) -> tuple[h5py.Group, MapLayout]:
    """
    Build a /map group in *f* with the given reference and optional primary/aux slots.
    Returns the /map group and its parsed MapLayout (via the real discovery parser),
    so that tests exercise the actual name<->UUID pairing rather than hand-constructing
    a layout that might diverge from what the reader sees.
    """
    map_group = f.require_group("map")
    map_group.attrs["reference"] = str(reference)
    map_group.attrs["format_version"] = 1

    if primary_targets:
        primary = map_group.require_group("primary")
        for target_uuid, data in primary_targets.items():
            _write_simple_slot(primary, str(target_uuid), data)

    if aux_pairs:
        auxiliary = map_group.require_group("auxiliary")
        for (uuid_a, uuid_b), (src, tgt) in aux_pairs.items():
            _write_aux_pair(auxiliary, uuid_a, uuid_b, src, tgt)

    layout = _read_map_layout("/map", map_group)
    return map_group, layout


class TestReferenceAbsentTwoPrimaryTargets:
    """Bug fix: reference absent + two primary targets -> match set returned."""

    def test_returns_dataset_match_set(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                },
            )
            # Reference (_REF) is NOT in available.
            available = {_A, _B}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert result.reference_source == _REF
            assert set(result.primary_maps.keys()) == {_A, _B}
            assert result.aux_maps == {}


class TestReferenceAbsentOnePrimaryNoAux:
    """Reference absent, exactly one primary target, no auxiliary -> None."""

    def test_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1, 2]},
            )
            available = {_A}
            result = read_match_set(map_group, layout, available)

            assert result is None


class TestAuxiliaryRequiresPrimaries:
    """Every auxiliary endpoint must have a corresponding primary map."""

    def test_missing_primary_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            with pytest.raises(ValueError, match="without corresponding primaries"):
                _make_map_group(
                    f,
                    reference=_REF,
                    primary_targets={_A: [0, 1, 2]},
                    aux_pairs={(_A, _B): ([0, 1], [2, 0])},
                )


class TestMapArrayValidation:
    @pytest.mark.parametrize(
        ("data", "message"),
        (
            (np.array([[0, 1]], dtype=np.int64), "one-dimensional"),
            (np.array([0.0, 1.0]), "integer dtype"),
        ),
    )
    def test_primary_array_shape_and_dtype(self, tmp_path: Path, data, message) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            slot = map_group.require_group(f"primary/{_A}")
            slot.create_dataset("index", data=data)

            with pytest.raises(ValueError, match=message):
                _read_map_layout("/map", map_group)

    def test_chunked_mapping_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            slot = map_group.require_group(f"primary/{_A}")
            slot.create_dataset("start", data=np.array([0, 1], dtype=np.int64))
            slot.create_dataset("size", data=np.array([1, 1], dtype=np.int64))

            with pytest.raises(ValueError, match="chunked.*not supported"):
                _read_map_layout("/map", map_group)

        layout = discover_file(path)
        assert layout.error is not None
        assert "chunked" in layout.error

    def test_slot_cannot_mix_simple_and_chunked_arrays(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            slot = map_group.require_group(f"primary/{_A}")
            slot.create_dataset("index", data=np.array([0, 1], dtype=np.int64))
            slot.create_dataset("start", data=np.array([0, 1], dtype=np.int64))
            slot.create_dataset("size", data=np.array([1, 1], dtype=np.int64))

            with pytest.raises(ValueError, match="chunked.*not supported"):
                _read_map_layout("/map", map_group)

    @pytest.mark.parametrize("version", (2, "1"))
    def test_unsupported_format_version(self, tmp_path: Path, version) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = version

            with pytest.raises(ValueError, match="only version 1 is supported"):
                _read_map_layout("/map", map_group)

    def test_auxiliary_sides_require_equal_lengths(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            with pytest.raises(ValueError, match="source and target.*same length"):
                _make_map_group(
                    f,
                    reference=_REF,
                    primary_targets={_A: [0, 1], _B: [1, 0]},
                    aux_pairs={(_A, _B): ([0, 1], [1])},
                )

    @pytest.mark.parametrize(
        ("side", "data", "message"),
        (
            ("source", np.array([[0, 1]], dtype=np.int64), "one-dimensional"),
            ("target", np.array([0.0, 1.0]), "integer dtype"),
        ),
    )
    def test_auxiliary_array_shape_and_dtype(
        self, tmp_path: Path, side, data, message
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, _ = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): ([0, 1], [1, 0])},
            )
            pair = map_group[f"auxiliary/{_A}__{_B}"]
            del pair[side]
            pair.create_dataset(side, data=data)

            with pytest.raises(ValueError, match=message):
                _read_map_layout("/map", map_group)

    @pytest.mark.parametrize("side", ("source", "target"))
    def test_auxiliary_side_must_be_direct_dataset(self, tmp_path: Path, side) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, _ = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): ([0, 1], [1, 0])},
            )
            pair = map_group[f"auxiliary/{_A}__{_B}"]
            del pair[side]
            _write_simple_slot(pair, side, [0, 1])

            with pytest.raises(ValueError, match="expected a dataset"):
                _read_map_layout("/map", map_group)

    @pytest.mark.parametrize(
        ("source", "target", "message"),
        (
            ([-1, 1], [0, 1], "source indices must be non-negative"),
            ([0, 1], [-1, 1], "target indices must be non-negative"),
            ([0, 0], [0, 1], "source indices must be unique"),
            ([0, 1], [0, 0], "target indices must be unique"),
        ),
    )
    def test_auxiliary_values_are_one_to_one(
        self, tmp_path: Path, source, target, message
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): (source, target)},
            )

            with pytest.raises(ValueError, match=message):
                read_match_set(map_group, layout, {_REF, _A, _B})


class TestMapUUIDCollisions:
    def test_primary_spellings_cannot_resolve_to_same_uuid(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1])
            _write_simple_slot(primary, _A.hex, [0, 1])

            with pytest.raises(ValueError, match="duplicate primary mapping"):
                _read_map_layout("/map", map_group)

    def test_auxiliary_spellings_cannot_resolve_to_same_pair(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1])
            _write_simple_slot(primary, str(_B), [1, 0])
            auxiliary = map_group.require_group("auxiliary")
            _write_aux_pair(auxiliary, _A, _B, [0], [1])
            reversed_pair = auxiliary.require_group(f"{_B.hex}__{_A.hex}")
            reversed_pair.create_dataset("source", data=np.array([1], dtype=np.int64))
            reversed_pair.create_dataset("target", data=np.array([0], dtype=np.int64))

            with pytest.raises(ValueError, match="duplicate auxiliary mapping pair"):
                _read_map_layout("/map", map_group)


def test_get_auxiliary_mapping_override_alignment_one_to_one(tmp_path) -> None:
    """Regression test for correct alignment of auxiliary overrides.

    In particular, the override indices returned by `get_auxillary_mapping`
    must be aligned to the positions in the caller-provided `index_arr`.
    """

    path = tmp_path / "mapping.hdf5"
    with h5py.File(path, "w") as file:
        primary_a = file.create_dataset("primary_a", data=[0, 2, -1, 3])
        primary_b = file.create_dataset("primary_b", data=[0, 3, 1, -1])

        # Auxiliary overrides for source->target when source index is 1.
        # This should override the mapping for the *correct position* in the
        # provided index array.
        auxiliary_source = file.create_dataset("aux_source", data=[1])
        auxiliary_target = file.create_dataset("aux_target", data=[2])

        match_set = DatasetMatchSet(
            reference_source=_REF,
            primary_maps={_A: primary_a, _B: primary_b},
            aux_maps={(_A, _B): (auxiliary_source, auxiliary_target)},
            aliases={"source": _A, "target": _B},
        )

        # index positions: only the entry equal to aux_source[0] should be overridden.
        index_arr = np.asarray([3, 1, 0], dtype=np.int64)
        result = get_mapping(match_set, "source", "target", index_arr)
        assert result is not None

        # Ensure injective mapping for the source rows we selected.
        # -1 entries are unmatched and ignored for one-to-one.
        mapped = result[result != -1]
        assert mapped.size == np.unique(mapped).size


class TestReferencePresent:
    """Reference present -> primary maps populated (pre-existing behaviour)."""

    def test_primary_maps_populated(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                },
            )
            # Reference is present this time.
            available = {_REF, _A, _B}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert result.reference_source == _REF
            assert set(result.primary_maps.keys()) == {_A, _B}

    def test_auxiliary_maps_resolve_direct_datasets(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={_A: [0, 1], _B: [1, 0]},
                aux_pairs={(_A, _B): ([2, 3], [4, 5])},
            )

            result = read_match_set(map_group, layout, {_REF, _A, _B})

            assert result is not None
            source, target = result.aux_maps[(_A, _B)]
            assert source.name.endswith("/source")
            assert target.name.endswith("/target")
            np.testing.assert_array_equal(source[:], [2, 3])
            np.testing.assert_array_equal(target[:], [4, 5])


class TestTargetsNotInAvailableFiltered:
    """Targets not in available are filtered out."""

    def test_unavailable_target_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _B: [2, 0, 1],
                    _C: [1, 2, 0],
                },
            )
            # _C is not available.
            available = {_REF, _A, _B}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert _C not in result.primary_maps
            assert set(result.primary_maps.keys()) == {_A, _B}

    def test_aux_pair_with_unavailable_endpoint_excluded(self, tmp_path: Path) -> None:
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group, layout = _make_map_group(
                f,
                reference=_REF,
                primary_targets={
                    _A: [0, 1, 2],
                    _C: [1, 2, 0],
                },
                # _C is not available, so this pair must be dropped.
                aux_pairs={(_A, _C): ([0], [1])},
            )
            available = {_REF, _A}
            result = read_match_set(map_group, layout, available)

            # Only one primary target with reference present -> primary kept,
            # but aux dropped because _C is absent.
            assert result is not None
            assert (_A, _C) not in result.aux_maps


class TestMissingReferenceAttr:
    """Missing reference attr is a discovery-time error, not a read_match_set None return."""

    def test_read_map_layout_raises_on_missing_reference(self, tmp_path: Path) -> None:
        """_read_map_layout raises ValueError mentioning 'reference' when the attr is absent."""
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["format_version"] = 1
            # Deliberately omit the "reference" attr.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1, 2])

            with pytest.raises(ValueError, match="reference"):
                _read_map_layout("/map", map_group)

    def test_discover_file_yields_error_on_missing_reference(
        self, tmp_path: Path
    ) -> None:
        """discover_file returns a FileLayout with a non-None error for a missing reference."""
        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["format_version"] = 1
            # Deliberately omit the "reference" attr.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, str(_A), [0, 1, 2])

        layout = discover_file(path)
        assert layout.error is not None
        assert "reference" in layout.error


class TestNonCanonicalSlotName:
    """Regression guard: a /primary slot whose on-disk name is a non-canonical UUID
    spelling (e.g. uppercase or 32-char hex without hyphens) must still resolve
    correctly through the layout.  This guards against anyone later 'simplifying'
    the reader to rebuild names with f'{uuid}' instead of using the recorded name."""

    def test_uppercase_uuid_slot_resolves(self, tmp_path: Path) -> None:
        # Verify that uuid.UUID accepts the uppercase spelling we use on disk.
        uppercase_name = str(_A).upper()
        assert UUID(uppercase_name) == _A, "precondition: UUID() accepts uppercase"

        path = tmp_path / "map.hdf5"
        with h5py.File(path, "w") as f:
            map_group = f.require_group("map")
            map_group.attrs["reference"] = str(_REF)
            map_group.attrs["format_version"] = 1

            # Write the primary slot with an uppercase (non-canonical) name.
            primary = map_group.require_group("primary")
            _write_simple_slot(primary, uppercase_name, [0, 1, 2])

            layout = _read_map_layout("/map", map_group)

            # The layout must record the verbatim on-disk name, not str(_A).
            assert len(layout.primary_slots) == 1
            slot_name, target_uuid = layout.primary_slots[0]
            assert slot_name == uppercase_name
            assert target_uuid == _A

            # read_match_set must resolve the live handle correctly via the recorded name.
            available = {_REF, _A}
            result = read_match_set(map_group, layout, available)

            assert result is not None
            assert _A in result.primary_maps
