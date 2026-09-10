from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _uuid_blob_path(cache_dir: Path, path: Path) -> Path:
    from opencosmo.uuid import get_path_uuid

    return cache_dir / f"{get_path_uuid(path)}.json"


def _decode_layout_blob_for_cache_roundtrip(blob_path: Path) -> object:
    from opencosmo.io.discover import decode_file_layout_blob

    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    return decode_file_layout_blob(blob)


def _cache_dir_for_reads(path: Path) -> Path:
    from opencosmo.io.cache import get_directory_write_cache_dir

    return get_directory_write_cache_dir(path.parent)


def _layout_entry_from_db(cache_dir: Path, path: Path) -> sqlite3.Row | None:
    from opencosmo.io.cache import open_cache_db_for_read

    with open_cache_db_for_read(cache_dir) as conn:
        if conn is None:
            return None
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM files WHERE path = ?", (str(path),))
        row = cur.fetchone()
        return row


def _discover_all_sorted(paths: list[Path]) -> tuple:
    from opencosmo.io.discover import discover_all

    return discover_all(paths, comm=None)


def test_stale_mtime_is_miss(tmp_path: Path, test_data) -> None:
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_file

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    layout = discover_file(path)
    assert layout.error is None
    _ = _discover_all_sorted([path])

    hit = get_cached_layouts([path])
    assert path in hit

    old_mtime = path.stat().st_mtime
    os.utime(path, (time.time(), old_mtime + 5.0))
    miss = get_cached_layouts([path])
    assert miss == {}


def test_corrupt_blob_json_is_miss(tmp_path: Path, test_data) -> None:
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    _ = discover_all([path], comm=None)
    cache_dir = _cache_dir_for_reads(path)
    blob_path = _uuid_blob_path(cache_dir, path)
    assert blob_path.exists()

    blob_path.write_text("{not-json", encoding="utf-8")

    miss = get_cached_layouts([path])
    assert miss == {}


def test_sha256_mismatch_is_miss(tmp_path: Path, test_data) -> None:
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    _ = discover_all([path], comm=None)
    cache_dir = _cache_dir_for_reads(path)
    blob_path = _uuid_blob_path(cache_dir, path)

    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    assert "sha256" in blob
    # Mutate a field while leaving sha256 intact.
    blob["path"] = str(path) + "::mismatch"
    blob_path.write_text(json.dumps(blob), encoding="utf-8")

    miss = get_cached_layouts([path])
    assert miss == {}


def test_missing_blob_file_is_miss(tmp_path: Path, test_data) -> None:
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    _ = discover_all([path], comm=None)
    cache_dir = _cache_dir_for_reads(path)
    blob_path = _uuid_blob_path(cache_dir, path)
    assert blob_path.exists()
    blob_path.unlink()

    miss = get_cached_layouts([path])
    assert miss == {}


@pytest.mark.parametrize("mutator", ["corrupt", "sha256", "missing"])
def test_none_free_discover_all_on_cache_failures(
    tmp_path: Path, test_data, mutator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())
    _ = discover_all([path], comm=None)
    cache_dir = _cache_dir_for_reads(path)
    blob_path = _uuid_blob_path(cache_dir, path)

    if mutator == "corrupt":
        blob_path.write_text("not-json", encoding="utf-8")
    elif mutator == "sha256":
        blob = json.loads(blob_path.read_text(encoding="utf-8"))
        blob["path"] = str(path) + "::mismatch"
        blob_path.write_text(json.dumps(blob), encoding="utf-8")
    elif mutator == "missing":
        blob_path.unlink()
    else:
        raise AssertionError("unreachable")

    assert get_cached_layouts([path]) == {}
    layouts = discover_all([path], comm=None)
    assert isinstance(layouts, tuple)
    assert len(layouts) == 1
    assert layouts[0].path == path
    assert layouts[0].error is None
    assert layouts[0].groups is not None


def test_layout_version_bump_invalidates(tmp_path: Path, test_data) -> None:
    import opencosmo.io.cache as cache
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())
    _ = discover_all([path], comm=None)

    assert path in get_cached_layouts([path])

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(cache, "LAYOUT_VERSION", cache.LAYOUT_VERSION + 1)
        miss = get_cached_layouts([path])
        assert miss == {}
    finally:
        monkeypatch.undo()


def test_errored_layouts_not_persisted(tmp_path: Path, test_data) -> None:
    from opencosmo.io.cache import cache_layouts
    from opencosmo.io.discover import FileLayout

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    cache_layouts([FileLayout(path=path, groups=(), error="boom")])
    cache_dir = _cache_dir_for_reads(path)
    blob_path = _uuid_blob_path(cache_dir, path)
    assert not blob_path.exists()


def test_warm_cold_round_trip_and_no_walk_on_warm(
    tmp_path: Path, test_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opencosmo.io.cache import read_layouts_from_cache
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    p1 = tmp_path / src.name
    p2 = tmp_path / (src.name + ".2")
    p1.write_bytes(src.read_bytes())
    p2.write_bytes(src.read_bytes())

    cold = discover_all([p1, p2], comm=None)
    warm = discover_all([p1, p2], comm=None)
    assert [str(fl.path) for fl in warm] == [str(fl.path) for fl in cold]
    assert [fl.error for fl in warm] == [fl.error for fl in cold]
    assert [len(fl.groups) for fl in warm] == [len(fl.groups) for fl in cold]

    def _boom(_: Path) -> object:
        raise AssertionError("should not walk")

    monkeypatch.setattr("opencosmo.io.discover.discover_file", _boom)
    warm2 = discover_all([p1, p2], comm=None)
    assert [str(fl.path) for fl in warm2] == [str(fl.path) for fl in cold]
    assert [fl.error for fl in warm2] == [fl.error for fl in cold]
    assert [len(fl.groups) for fl in warm2] == [len(fl.groups) for fl in cold]

    for p in (p1, p2):
        cache_dir = _cache_dir_for_reads(p)
        cached = read_layouts_from_cache(cache_dir, [p])
        assert p in cached


def test_cache_hit_does_not_write(tmp_path: Path, test_data) -> None:
    """A warm read must not update the row; an UPDATE per open costs an fsync."""
    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    _ = discover_all([path], comm=None)
    cache_dir = _cache_dir_for_reads(path)
    blob = _uuid_blob_path(cache_dir, path)

    before_row = _layout_entry_from_db(cache_dir, path)
    before_blob = blob.read_bytes()
    assert before_row is not None
    time.sleep(0.01)

    assert path in get_cached_layouts([path])

    assert dict(_layout_entry_from_db(cache_dir, path)) == dict(before_row)  # type: ignore
    assert blob.read_bytes() == before_blob


def test_disable_cache_escape_hatch(
    tmp_path: Path, test_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    import opencosmo.io.cache as cache
    from opencosmo.io.cache import cache_layouts, get_cached_layouts
    from opencosmo.io.discover import FileLayout

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())

    monkeypatch.setattr(cache, "CACHE_DISABLED", True)
    assert get_cached_layouts([path]) == {}
    cache_layouts([FileLayout(path=path, groups=(), error=None)])
    cache_dir = _cache_dir_for_reads(path)
    blob_path = _uuid_blob_path(cache_dir, path)
    assert not blob_path.exists()


def test_connections_are_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_data
) -> None:
    # sqlite3.Connection.close is not patchable (immutable type) on some
    # Python builds. Instead, patch the cache module's context manager to
    # observe close behavior.
    close_calls: list[int] = []

    import opencosmo.io.cache as cache

    original_sqlite_connection = cache._sqlite_connection

    def _tracking_sqlite_connection(cache_db_path: Path, *, readonly: bool) -> object:
        cm = original_sqlite_connection(cache_db_path, readonly=readonly)

        import contextlib

        @contextlib.contextmanager
        def _wrap():
            with cm as conn:
                try:
                    yield conn
                finally:
                    close_calls.append(1)

        return _wrap()

    monkeypatch.setattr(cache, "_sqlite_connection", _tracking_sqlite_connection)

    from opencosmo.io.cache import get_cached_layouts
    from opencosmo.io.discover import discover_all

    src = test_data.snapshot.primary.halo_properties
    path = tmp_path / src.name
    path.write_bytes(src.read_bytes())
    _ = discover_all([path], comm=None)

    for _ in range(20):
        _ = get_cached_layouts([path])

    assert len(close_calls) >= 20


def test_populate_directory_cache_writes_shared_cache(
    test_data, tmp_path: "Path"
) -> None:
    """populate_directory_cache indexes a directory into <dir>/.opencosmo."""
    from opencosmo.io.cache import populate_directory_cache

    src = test_data.snapshot.primary.halo_properties
    for name in ("a.hdf5", "b.hdf5"):
        (tmp_path / name).write_bytes(src.read_bytes())

    result = populate_directory_cache(tmp_path)

    assert result.cache_dir == tmp_path.resolve() / ".opencosmo"
    assert len(result.cached) == 2
    assert result.failed == {}
    assert (result.cache_dir / "files.db").is_file()


def test_populate_directory_cache_skips_and_reports_bad_files(
    test_data, tmp_path: "Path"
) -> None:
    """Files that fail discovery are reported, not cached."""
    from opencosmo.io.cache import populate_directory_cache

    src = test_data.snapshot.primary.halo_properties
    (tmp_path / "good.hdf5").write_bytes(src.read_bytes())
    bad = tmp_path / "bad.hdf5"
    bad.write_text("not an hdf5 file")

    result = populate_directory_cache(tmp_path)

    assert result.cached == (tmp_path.resolve() / "good.hdf5",)
    assert bad.resolve() in result.failed


def test_populate_directory_cache_rejects_non_directory(tmp_path: "Path") -> None:
    from opencosmo.io.cache import populate_directory_cache

    missing = tmp_path / "nope"
    with pytest.raises(NotADirectoryError):
        populate_directory_cache(missing)


def test_shared_cache_is_readable_from_readonly_directory(
    test_data, tmp_path: "Path"
) -> None:
    """
    A populated shared cache must be usable once the data dir is read-only.

    Regression test: write_layouts puts the DB in WAL mode, and SQLite cannot
    open a WAL database even read-only without creating -wal/-shm sidecars, so
    a published cache silently missed on every lookup.
    """
    import opencosmo.io.discover as discover_module
    from opencosmo.io.cache import (
        get_directory_read_cache_dir,
        populate_directory_cache,
    )

    src = test_data.snapshot.primary.halo_properties
    data_dir = tmp_path / "shared"
    data_dir.mkdir()
    path = data_dir / "haloproperties.hdf5"
    path.write_bytes(src.read_bytes())

    populate_directory_cache(data_dir)

    cache_dir = data_dir.resolve() / ".opencosmo"
    assert not list(cache_dir.glob("*-wal")), "WAL sidecar breaks read-only reads"

    original_modes = {p: p.stat().st_mode for p in (data_dir, *data_dir.rglob("*"))}
    for p in original_modes:
        os.chmod(p, 0o555)
    try:
        assert get_directory_read_cache_dir(data_dir.resolve()) == cache_dir

        # The warm read must not touch HDF5 at all.
        def _fail(p: "Path"):
            raise AssertionError(f"walked {p}")

        original_discover = discover_module.discover_file
        discover_module.discover_file = _fail  # type: ignore[assignment]
        try:
            layouts = discover_module.discover_all([path.resolve()], comm=None)
        finally:
            discover_module.discover_file = original_discover  # type: ignore[assignment]

        assert len(layouts) == 1
        assert layouts[0].error is None
    finally:
        for p, mode in original_modes.items():
            os.chmod(p, mode)


class TestCacheDirSelection:
    """The read path may use a directory-shared cache; the write path never may."""

    def test_read_falls_back_when_shared_db_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opencosmo.io.cache import get_directory_read_cache_dir

        data_dir = tmp_path / "data"
        (data_dir / ".opencosmo").mkdir(parents=True)
        monkeypatch.setattr("os.access", lambda p, mode: False)

        assert get_directory_read_cache_dir(data_dir) != data_dir / ".opencosmo"

    def test_read_uses_shared_when_shared_db_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opencosmo.io.cache import get_directory_read_cache_dir

        data_dir = tmp_path / "data"
        shared = data_dir / ".opencosmo"
        shared.mkdir(parents=True)
        (shared / "files.db").write_bytes(b"")
        monkeypatch.setattr("os.access", lambda p, mode: False)

        assert get_directory_read_cache_dir(data_dir) == shared

    def test_write_never_uses_shared_cache_even_when_writable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opencosmo.io.cache import get_directory_write_cache_dir

        data_dir = tmp_path / "data"
        shared = data_dir / ".opencosmo"
        shared.mkdir(parents=True)
        (shared / "files.db").write_bytes(b"")
        monkeypatch.setattr("os.access", lambda p, mode: True)

        chosen = get_directory_write_cache_dir(data_dir)
        assert chosen is not None
        assert chosen != shared
