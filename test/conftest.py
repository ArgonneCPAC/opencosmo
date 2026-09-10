import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class StructurePaths:
    halo_properties: Path
    halo_particles: Path | None = None
    halo_profiles: Path | None = None
    galaxy_properties: Path | None = None
    galaxy_particles: Path | None = None

    @property
    def halos(self) -> list[Path]:
        return [
            path
            for path in (
                self.halo_properties,
                self.halo_particles,
                self.halo_profiles,
            )
            if path is not None
        ]

    @property
    def galaxies(self) -> list[Path]:
        return [
            path
            for path in (self.galaxy_properties, self.galaxy_particles)
            if path is not None
        ]

    @property
    def all(self) -> list[Path]:
        return self.halos + self.galaxies


@dataclass(frozen=True)
class SnapshotPaths:
    root: Path

    @property
    def primary(self) -> StructurePaths:
        return StructurePaths(
            halo_properties=self.root / "haloproperties.hdf5",
            halo_particles=self.root / "haloparticles.hdf5",
            halo_profiles=self.root / "sodproperties.hdf5",
            galaxy_properties=self.root / "galaxyproperties.hdf5",
            galaxy_particles=self.root / "galaxyparticles.hdf5",
        )

    def scidac(self, simulation: int) -> StructurePaths:
        root = self.root / f"scidac_{simulation:03d}"
        return StructurePaths(
            halo_properties=root / "haloproperties.hdf5",
            galaxy_properties=root / "galaxyproperties.hdf5",
        )

    @property
    def header(self) -> Path:
        return self.root / "header.hdf5"

    @property
    def multi_simulation(self) -> Path:
        return self.root / "haloproperties_multi.hdf5"

    @property
    def alternate_step(self) -> Path:
        return self.root / "haloproperties_step310.hdf5"

    @property
    def mapping_reference(self) -> Path:
        return self.root / "haloproperties_go.hdf5"

    @property
    def halo_mapping(self) -> Path:
        return self.root / "halo_mapping.hdf5"


@dataclass(frozen=True)
class LightconePaths:
    root: Path

    def step(self, step: int) -> StructurePaths:
        root = self.root / f"step_{step}"
        return StructurePaths(
            halo_properties=root / "haloproperties.hdf5",
            halo_particles=root / "haloparticles.hdf5",
            halo_profiles=root / "haloprofiles.hdf5",
            galaxy_properties=root / "galaxyproperties.hdf5",
            galaxy_particles=root / "galaxyparticles.hdf5",
        )


@dataclass(frozen=True)
class DiffskyPaths:
    root: Path

    def core(self, pixel: int) -> Path:
        return self.root / f"lj_{pixel}.hdf5"

    @property
    def invalid(self) -> Path:
        return self.root / "random_data.hdf5"


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path

    @property
    def mass_function(self) -> Path:
        return self.root / "mass_fn.npy"

    @property
    def stacked_profile(self) -> Path:
        return self.root / "stacked_profile.npy"


@dataclass(frozen=True)
class TestDataPaths:
    root: Path

    @property
    def snapshot(self) -> SnapshotPaths:
        return SnapshotPaths(self.root / "snapshot")

    @property
    def lightcone(self) -> LightconePaths:
        return LightconePaths(self.root / "lightcone")

    @property
    def diffsky(self) -> DiffskyPaths:
        return DiffskyPaths(self.root / "diffsky")

    @property
    def healpix_map(self) -> Path:
        return self.root / "healpix_map" / "test_map.hdf5"

    @property
    def analysis(self) -> AnalysisPaths:
        return AnalysisPaths(self.root / "analysis")


@pytest.fixture(scope="session")
def test_data() -> TestDataPaths:
    root = os.environ.get("OPENCOSMO_DATA_PATH")
    return TestDataPaths(
        Path(root).expanduser().resolve()
        if root
        else Path(__file__).parents[1] / "test_data"
    )


@pytest.fixture(scope="session", autouse=True)
def _redirect_user_cache_dir(
    tmp_path_factory: "pytest.TempPathFactory",
) -> "Iterator[None]":
    """
    Point the discovery layout cache at a temp dir for the whole session.

    Without this, every test that opens a file writes to the developer's real
    user cache and leaks state between runs.
    """
    import opencosmo.io.cache as cache

    temp_cache_dir = tmp_path_factory.mktemp("opencosmo-user-cache") / "opencosmo"
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    # Module-level names are not name-mangled, so this patches the real
    # resolver that both the read and write cache-dir paths call.
    original = cache.__user_cache_dir  # type: ignore[attr-defined]
    cache.__user_cache_dir = lambda: temp_cache_dir  # type: ignore[attr-defined]
    try:
        yield
    finally:
        cache.__user_cache_dir = original  # type: ignore[attr-defined]
