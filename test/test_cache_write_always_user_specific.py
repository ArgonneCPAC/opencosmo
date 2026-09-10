from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_directory_cache_write_uses_user_specific_cache_when_writable(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    writable_dir = tmp_path / "writable"
    writable_dir.mkdir()

    # Simulate writable directory.
    monkeypatch.setattr("os.access", lambda p, mode: True)

    from opencosmo.io.cache import get_directory_write_cache_dir

    chosen = get_directory_write_cache_dir(writable_dir)
    assert chosen is not None
    assert chosen != writable_dir / ".opencosmo"
