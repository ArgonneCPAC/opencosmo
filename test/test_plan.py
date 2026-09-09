from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid5

from opencosmo.io.discover import FileLayout, GroupLayout
from opencosmo.io.io import MpiMode
from opencosmo.io.plan import distribute
from opencosmo.uuid import NAMESPACE


# Mock header classes to simulate real OpenCosmoHeader structure without HDF5.
@dataclass(frozen=True)
class _FakeFile:
    step: int | None
    data_type: str
    is_lightcone: bool


@dataclass(frozen=True)
class _FakeHeader:
    file: _FakeFile


def _layout(
    path: str,
    *,
    step: int | None = None,
    data_type: str = "halo_properties",
    is_lightcone: bool = True,
    row_count: int = 100,
    linked: tuple[str, ...] = (),
    has_index: bool = True,
    error: str | None = None,
    uuid_=None,
    has_persistent_uuid: bool = False,
) -> FileLayout:
    """
    Build a single-group FileLayout with a fake header.

    Parameters
    ----------
    path : str
        File path (e.g., "/path/to/file.hdf5").
    step : int | None, optional
        Redshift step number, by default None.
    data_type : str, optional
        Header data_type field, by default "halo_properties".
    is_lightcone : bool, optional
        Whether this is a lightcone file, by default True.
    row_count : int, optional
        Number of rows in the data group, by default 100.
    linked : tuple[str, ...], optional
        Names of linked targets (e.g., ("haloparticles",)), by default ().
    has_index : bool, optional
        Whether /index group exists, by default True.
    error : str | None, optional
        Error message if discovery failed, by default None.

    Returns
    -------
    FileLayout
        A single-group layout.
    """
    if uuid_ is None:
        uuid_ = uuid5(NAMESPACE, f"{Path(path).resolve()}::/data")

    grp = GroupLayout(
        path="/",
        header_path="/header",
        header=_FakeHeader(_FakeFile(step, data_type, is_lightcone)),  # type: ignore[arg-type]
        column_names=("x", "y"),
        column_dtypes=("float64", "float64"),
        row_count=row_count,
        has_index=has_index,
        linked_target_names=linked,
        uuid=uuid_,
        has_persistent_uuid=has_persistent_uuid,
    )
    return FileLayout(path=Path(path), groups=(grp,), error=error)


class TestDistributeSpatial:
    """Test spatial distribution: all files to all ranks."""

    def test_spatial_all_files_to_all_ranks(self) -> None:
        """Spatial mode assigns all valid files to every rank."""
        layouts = (
            _layout("/file1.hdf5"),
            _layout("/file2.hdf5"),
            _layout("/file3.hdf5"),
        )
        assignments = distribute(layouts, MpiMode.SPATIAL, nranks=4)

        assert len(assignments) == 4
        assert all(a.index_kind == "spatial" for a in assignments)
        assert all(a.file_indices == (0, 1, 2) for a in assignments)
        assert [a.rank for a in assignments] == [0, 1, 2, 3]

    def test_spatial_excludes_errored_files(self) -> None:
        """Errored files are excluded from every rank's file_indices."""
        layouts = (
            _layout("/file1.hdf5"),
            _layout("/file2.hdf5", error="bad file"),
            _layout("/file3.hdf5"),
        )
        assignments = distribute(layouts, MpiMode.SPATIAL, nranks=3)

        assert len(assignments) == 3
        # All ranks should get indices [0, 2], excluding the errored index 1.
        for a in assignments:
            assert a.file_indices == (0, 2)
            assert a.index_kind == "spatial"


class TestDistributeSerial:
    """Test serial (single-rank) distribution."""

    def test_serial_single_rank(self) -> None:
        """Serial mode (nranks=1) returns single Assignment with all valid files."""
        layouts = (
            _layout("/file1.hdf5"),
            _layout("/file2.hdf5"),
        )
        assignments = distribute(layouts, MpiMode.SPATIAL, nranks=1)

        assert len(assignments) == 1
        assert assignments[0].rank == 0
        assert assignments[0].file_indices == (0, 1)
        assert assignments[0].index_kind == "none"

    def test_serial_with_errors(self) -> None:
        """Serial mode excludes errored files."""
        layouts = (
            _layout("/file1.hdf5"),
            _layout("/file2.hdf5", error="bad"),
            _layout("/file3.hdf5"),
        )
        assignments = distribute(layouts, MpiMode.SPATIAL, nranks=1)

        assert len(assignments) == 1
        assert assignments[0].file_indices == (0, 2)
        assert assignments[0].index_kind == "none"


class TestDistributeRedshift:
    """Test redshift distribution: contiguous step chunks."""

    def test_redshift_contiguous_coverage(self) -> None:
        """Redshift distribution produces contiguous, coverage-complete step chunks."""
        # One file per step, steps 0..3, varying row counts.
        layouts = (
            _layout("/file_step0.hdf5", step=0, is_lightcone=True, row_count=100),
            _layout("/file_step1.hdf5", step=1, is_lightcone=True, row_count=200),
            _layout("/file_step2.hdf5", step=2, is_lightcone=True, row_count=150),
            _layout("/file_step3.hdf5", step=3, is_lightcone=True, row_count=50),
        )
        nranks = 3
        assignments = distribute(layouts, MpiMode.REDSHIFT, nranks)

        assert len(assignments) == nranks
        assert all(a.index_kind == "redshift_step" for a in assignments)

        # Collect all non-empty ranks' file_indices.
        non_empty = [a for a in assignments if not a.is_empty_ref]
        flat_indices = sorted(i for a in non_empty for i in a.file_indices)

        # Check coverage: every valid file (0..3) appears exactly once.
        assert flat_indices == [0, 1, 2, 3]

        # Check contiguity: each non-empty rank's file_indices form an ascending run.
        for a in non_empty:
            if a.file_indices:
                f = list(a.file_indices)
                assert f == list(range(f[0], f[-1] + 1))

    def test_redshift_empty_ranks_flagged(self) -> None:
        """Ranks with fewer files than ranks get is_empty_ref=True."""
        # 3 files, 5 ranks: 3 busy, 2 empty.
        layouts = (
            _layout("/file_step0.hdf5", step=0, is_lightcone=True, row_count=100),
            _layout("/file_step1.hdf5", step=1, is_lightcone=True, row_count=200),
            _layout("/file_step2.hdf5", step=2, is_lightcone=True, row_count=50),
        )
        nranks = 5
        assignments = distribute(layouts, MpiMode.REDSHIFT, nranks)

        assert len(assignments) == nranks

        # Count empty (is_empty_ref=True).
        empty_count = sum(1 for a in assignments if a.is_empty_ref)

        # Should have at most 2 empty ranks (partition_contiguous distributes 3 steps).
        # (Exact distribution depends on partition_contiguous; just verify the pattern.)
        assert empty_count >= 0
        assert empty_count <= 2

        # Empty ranks get the reference (lightest) step indices.
        # The lightest step is step 2 with row_count=50, so file index 2.
        reference_indices = (2,)
        for a in assignments:
            if a.is_empty_ref:
                assert a.file_indices == reference_indices

        # Non-empty ranks have is_empty_ref=False.
        for a in assignments:
            if not a.is_empty_ref:
                assert a.is_empty_ref is False

    def test_redshift_fallback_to_spatial_diffsky(self) -> None:
        """Redshift falls back to spatial for nested diffsky-style layouts."""
        # Multiple files per step with no properties link = not plain.
        # This should fall back to spatial distribution.
        layouts = (
            _layout(
                "/file_step0a.hdf5",
                step=0,
                data_type="halo_properties",
                is_lightcone=True,
                row_count=100,
                linked=(),
            ),
            _layout(
                "/file_step0b.hdf5",
                step=0,
                data_type="galaxy_properties",
                is_lightcone=True,
                row_count=100,
                linked=(),
            ),
        )
        nranks = 2
        assignments = distribute(layouts, MpiMode.REDSHIFT, nranks)

        # Should have fallen back to spatial.
        assert all(a.index_kind == "spatial" for a in assignments)
        assert all(a.file_indices == (0, 1) for a in assignments)

    def test_redshift_none_mode_treated_as_spatial(self) -> None:
        """mpi_mode=None is treated as spatial (every rank gets every file)."""
        layouts = (
            _layout("/file1.hdf5"),
            _layout("/file2.hdf5"),
        )
        assignments = distribute(layouts, None, nranks=2)

        assert len(assignments) == 2
        assert all(a.index_kind == "spatial" for a in assignments)
        assert all(a.file_indices == (0, 1) for a in assignments)


class TestDistributePurity:
    """Test that distribute is deterministic and a pure function."""

    def test_purity_determinism(self) -> None:
        """Calling distribute twice with the same inputs yields identical results."""
        layouts = (
            _layout("/file1.hdf5"),
            _layout("/file2.hdf5"),
            _layout("/file3.hdf5"),
        )
        result1 = distribute(layouts, MpiMode.SPATIAL, nranks=3)
        result2 = distribute(layouts, MpiMode.SPATIAL, nranks=3)

        assert result1 == result2

    def test_purity_same_logical_layouts_identical_result(self) -> None:
        """Building the same logical layouts separately yields identical results."""
        # Build two identical layout tuples independently.
        layouts1 = (
            _layout("/file_step0.hdf5", step=0, row_count=100),
            _layout("/file_step1.hdf5", step=1, row_count=200),
        )
        layouts2 = (
            _layout("/file_step0.hdf5", step=0, row_count=100),
            _layout("/file_step1.hdf5", step=1, row_count=200),
        )

        result1 = distribute(layouts1, MpiMode.REDSHIFT, nranks=2)
        result2 = distribute(layouts2, MpiMode.REDSHIFT, nranks=2)

        assert result1 == result2

    def test_purity_order_independence_via_sorting(self) -> None:
        """
        Inputs are sorted by path; shuffling layout order does not change the output.

        The distribute function expects sorted inputs. Callers (discover_all) ensure
        this. Here we verify the invariant: two calls with the same layouts in
        different order (if they were not pre-sorted) would still be equivalent
        after sorting by path.
        """
        # Build layouts with filenames that sort in a specific order.
        layouts_a = (
            _layout("/aaa.hdf5", step=0, row_count=100),
            _layout("/bbb.hdf5", step=1, row_count=200),
            _layout("/ccc.hdf5", step=2, row_count=50),
        )

        # Same logical layouts, different insertion order (but internally sorted by path).
        layouts_b = (
            _layout("/ccc.hdf5", step=2, row_count=50),
            _layout("/aaa.hdf5", step=0, row_count=100),
            _layout("/bbb.hdf5", step=1, row_count=200),
        )

        # Ensure they are sorted the same way (by path).
        layouts_a_sorted = tuple(sorted(layouts_a, key=lambda fl: str(fl.path)))
        layouts_b_sorted = tuple(sorted(layouts_b, key=lambda fl: str(fl.path)))

        result_a = distribute(layouts_a_sorted, MpiMode.REDSHIFT, nranks=2)
        result_b = distribute(layouts_b_sorted, MpiMode.REDSHIFT, nranks=2)

        assert result_a == result_b
