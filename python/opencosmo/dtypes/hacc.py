# ruff: noqa: TC001 TC003
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional

import astropy.cosmology.units as cu  # type: ignore
import astropy.units as u  # type: ignore
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from opencosmo.units import UnitConvention

from .cosmology import CosmologyParameters
from .diffsky import DiffskyCatalogInfo, DiffskyVersionInfo
from .supermock import SuperMockParams
from .units import register_units
from .utils import empty_string_to_none


class HaccSimulationParameters(BaseModel):
    ACCESS_PATH: ClassVar[str] = "simulation"
    model_config = ConfigDict(frozen=True)

    box_size: float = Field(ge=0, description="Size of the simulation box (Mpc/h)")
    z_ini: float = Field(ge=0.01, description="Initial redshift of the simulation")
    z_end: float = Field(ge=0.0, description="Final redshift of the simulation")
    n_gravity: Optional[int] = Field(
        ge=2,
        default=None,
        description="Number of gravity-only particles (per dimension). "
        "In hydrodynamic simulations, this parameter will not be set.",
    )
    n_steps: int = Field(ge=1, description="Number of time steps")
    pm_grid: int = Field(ge=2, description="Number of grid points (per dimension)")
    offset_gravity_ini: Optional[float] = Field(
        description="Lagrangian offset for gravity-only particles", default=None
    )

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            return {k: empty_string_to_none(v) for k, v in data.items()}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def step_zs(self) -> list[float]:
        """
        Get the redshift of the steps in this simulation. Outputs such that
        redshift[step_number] returns the redshift for that step. Keep in
        mind that steps go from high z -> low z.
        """
        a_ini = 1 / (1 + self.z_ini)
        a_end = 1 / (1 + self.z_end)
        # Steps are evenly spaced in a with step zero corresponding to the first step
        # after the initial conditions
        step_as = np.linspace(a_ini, a_end, self.n_steps + 1)[1:]
        return np.round(1 / step_as - 1, 3).tolist()  # type: ignore


HaccGravityOnlySimulationParameters = HaccSimulationParameters


def subgrid_alias_generator(name: str) -> str:
    return f"subgrid_{name}"


class HaccHydroSimulationParameters(HaccSimulationParameters):
    model_config = ConfigDict(frozen=True)
    n_gas: int = Field(
        description="Number of gas particles (per dimension)", alias="n_bar"
    )
    n_dm: int = Field(
        ge=2, description="Number of dark matter particles (per dimension)"
    )
    offset_gas_ini: float = Field(
        description="Lagrangian offset for gas particles", alias="offset_bar_ini"
    )
    offset_dm_ini: float = Field(
        description="Lagrangian offset for dark matter particles"
    )
    agn_kinetic_eps: float = Field(
        description="AGN feedback efficiency", alias="subgrid_agn_kinetic_eps"
    )
    agn_kinetic_jet_vel: float = Field(
        description="AGN feedback velocity", alias="subgrid_agn_kinetic_jet_vel"
    )
    agn_nperh: float = Field(
        description="AGN sphere of influence", alias="subgrid_agn_nperh"
    )
    agn_seed_mass: float = Field(
        description="AGN seed mass (Msun / h)", alias="subgrid_agn_seed_mass"
    )
    wind_egy_w: float = Field(
        description="Wind mass loading factor", alias="subgrid_wind_egy_w"
    )
    wind_kappa_w: float = Field(
        description="Wind velocity", alias="subgrid_wind_kappa_w"
    )


class CosmoToolsParameters(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "cosmotools"
    cosmotools_steps: frozenset[int]
    fof_linking_length: float
    fof_pmin: int
    sod_pmin: int
    sod_delta_crit: float
    sod_concentration_pmin: int
    sodbighaloparticles_pmin: int
    profiles_nbins: int
    galaxy_dbscan_neighbors: Optional[int] = None
    galaxy_aperture_radius: Optional[int] = None
    galaxy_pmin: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            data = {k: empty_string_to_none(v) for k, v in data.items()}
        return data

    @field_serializer("cosmotools_steps")
    def serialize_steps(self, steps) -> list[int]:
        return sorted(steps)


class ReformatParameters(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "reformat"
    cosmotools_lc_path: Optional[Path] = None
    cosmotools_path: Path
    indat_path: Path
    is_hydro: bool
    lightcone_analysis_path_pattern: Optional[str] = None
    machine: str
    mass_threshold_sodbighaloparticles: Optional[float] = None
    mass_threshold_sodpropertybins: Optional[float] = None
    max_level: int = 0
    max_level_lc: Optional[frozenset[tuple[int, int]]] = None
    npart_threshold_galaxyproperties: Optional[int] = None
    output_lc_path_pattern: Optional[str] = None
    rearrange_output_path_pattern: str
    rearrange_output_lc_path_pattern: Optional[str] = None
    simulation_date: date
    simulation_name: str
    snapshot_analysis_path_pattern: Optional[str] = None
    temporary_path: Optional[Path] = None

    @field_serializer(
        "cosmotools_lc_path",
        "cosmotools_path",
        "indat_path",
        "temporary_path",
        "lightcone_analysis_path_pattern",
        "output_lc_path_pattern",
        "rearrange_output_lc_path_pattern",
        "snapshot_analysis_path_pattern",
    )
    def handle_path(self, v):
        if v is None:
            return ""
        return str(v)

    @field_serializer("max_level_lc")
    def serialize_max_level(self, value):
        if value is None:
            return
        return list(value)

    @field_serializer("simulation_date")
    def handle_date(self, v):
        return v.isoformat()

    @field_validator("is_hydro", mode="before")
    @classmethod
    def numpy_bool_to_base(cls, value):
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            data = {k: empty_string_to_none(v) for k, v in data.items()}
        return data


class HaccSimulationInfo(BaseModel):
    """
    Models the attributes on the ``header/simulation`` group itself (as opposed
    to ``header/simulation/parameters``).

    Only ``name`` is modelled deliberately: ``date`` would shadow the
    module-level ``from datetime import date`` import, and the raw config blobs
    (``raw_indat``, ``raw_cosmotools_config``) are not worth carrying.
    """

    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "simulation"
    name: str


class MapParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    ACCESS_PATH: ClassVar[str] = "healpix_map"
    z_range: Optional[tuple[float, float]] = None
    nside: Optional[int] = None
    nside_lr: Optional[int] = None
    map_type: Optional[str] = None
    ordering: Optional[str] = None
    full_sky: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def empty_string_to_none(cls, data):
        if isinstance(data, dict):
            return {k: empty_string_to_none(v) for k, v in data.items()}
        return data

    @field_validator("full_sky", mode="before")
    @classmethod
    def numpy_bool_to_base(cls, value):
        if isinstance(value, np.bool_):
            return bool(value)
        return value


ORIGIN_PARAMETERS = {
    "required": {
        "simulation/parameters": HaccHydroSimulationParameters
        | HaccGravityOnlySimulationParameters,
        "simulation/cosmotools": CosmoToolsParameters,
        "simulation/cosmology": CosmologyParameters,
    },
    "optional": {
        "reformat_hacc/config": ReformatParameters,
        "simulation": HaccSimulationInfo,
    },
}

DATATYPE_PARAMETERS: dict[str, dict[str, dict[str, type[BaseModel]]]] = {
    "halo_properties": {},
    "galaxy_properties": {},
    "halo_particles": {},
    "galaxy_particles": {},
    "halo_profiles": {},
    "synthetic_galaxies": {
        "optional": {
            "diffsky_versions": DiffskyVersionInfo,
            "catalog_info": DiffskyCatalogInfo,
            "supermock": SuperMockParams,
        }
    },
    "healpix_map": {"required": {"map_params": MapParams}},
}

register_units(
    HaccSimulationParameters, "box_size", u.Mpc / cu.littleh, UnitConvention.SCALEFREE
)
register_units(
    HaccHydroSimulationParameters,
    "box_size",
    u.Mpc / cu.littleh,
    UnitConvention.SCALEFREE,
)
register_units(
    HaccHydroSimulationParameters,
    "agn_seed_mass",
    u.Msun / cu.littleh,
    UnitConvention.SCALEFREE,
)
