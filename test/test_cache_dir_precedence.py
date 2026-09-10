from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_directory_cache_read_falls_back_when_shared_db_missing(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from opencosmo.io.cache import get_directory_read_cache_dir

    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    shared_cache_dir = readonly_dir / ".opencosmo"
    shared_cache_dir.mkdir()
    # No files.db created.

    # Simulate non-writable directory (so we should never take the writable path).
    monkeypatch.setattr("os.access", lambda p, mode: False)

    chosen = get_directory_read_cache_dir(readonly_dir)
    assert chosen != shared_cache_dir


def test_directory_cache_read_uses_shared_when_shared_db_exists(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from opencosmo.io.cache import get_directory_read_cache_dir

    readonly_dir = tmp_path / "readonly2"
    readonly_dir.mkdir()
    shared_cache_dir = readonly_dir / ".opencosmo"
    shared_cache_dir.mkdir()
    (shared_cache_dir / "files.db").write_bytes(b"")

    monkeypatch.setattr("os.access", lambda p, mode: False)

    chosen = get_directory_read_cache_dir(readonly_dir)
    assert chosen == shared_cache_dir


def test_directory_cache_write_never_uses_shared_cache(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from opencosmo.io.cache import get_directory_write_cache_dir

    writable_dir = tmp_path / "writable3"
    writable_dir.mkdir()
    shared_cache_dir = writable_dir / ".opencosmo"
    shared_cache_dir.mkdir()
    (shared_cache_dir / "files.db").write_bytes(b"")

    # Simulate writable directory; writes must still go to the user cache.
    monkeypatch.setattr("os.access", lambda p, mode: True)

    chosen = get_directory_write_cache_dir(writable_dir)
    assert chosen != shared_cache_dir
