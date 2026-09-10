from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np

import opencosmo as oc
from opencosmo.column import norm_cols
from opencosmo.plugins.contexts import HookPoint
from opencosmo.plugins.hook import hook

if TYPE_CHECKING:
    from opencosmo import Lightcone
    from opencosmo.plugins.contexts import LightconeOpenCtx


@hook(
    HookPoint.LightconeOpen,
    when=lambda ctx: ctx.lightcone.dtype in ["halo_properties", "galaxy_properties"],
)
def _ensure_coordinates(ctx: LightconeOpenCtx) -> LightconeOpenCtx:
    known_columns = set(ctx.lightcone.columns)
    if known_columns.issuperset(("phi", "theta")) or known_columns.issuperset(
        ("ra", "dec")
    ):
        return ctx

    prefix = "fof_halo_center"
    if ctx.lightcone.dtype == "galaxy_properties":
        prefix = "gal_center"

    coord_columns = {coord: f"{prefix}_{coord}" for coord in ["x", "y", "z"]}
    if not set(ctx.lightcone.columns).issuperset(coord_columns.values()):
        raise ValueError("Unable to find coordinate columns for this lightcone dataset")
    chi = norm_cols(*list(coord_columns.values()))
    phi = oc.col(coord_columns["y"]).arctan2(oc.col(coord_columns["x"]))
    theta = (oc.col(coord_columns["z"]) / oc.col("chi")).arccos()
    new_lightcone = ctx.lightcone.with_new_columns(chi=chi, phi=phi, theta=theta)
    return dataclasses.replace(ctx, lightcone=new_lightcone)


@hook(HookPoint.LightconeOpen)
def _ensure_redshift_column(ctx: LightconeOpenCtx) -> LightconeOpenCtx:
    """Ensures a column called 'redshift' exists on every lightcone."""
    lightcone: Lightcone = ctx.lightcone
    if (
        "particles" in lightcone.dtype or "profiles" in lightcone.dtype
    ):  # Particles or profiles, redshift handled at structure collection level
        return ctx
    if "redshift" in lightcone.columns:
        return ctx
    elif "fof_halo_center_a" in lightcone.columns:
        z_col = 1 / oc.col("fof_halo_center_a") - 1
    elif "redshift_true" in lightcone.columns:
        z_col = oc.col("redshift_true")
    elif "zp" in lightcone.columns:
        z_col = oc.col("zp")
    elif "chi" in lightcone.columns:
        lightcone = lightcone.evaluate(
            redshift_from_chi,
            cosmology=lightcone.cosmology,
            vectorize=True,
        )
        return dataclasses.replace(ctx, lightcone=lightcone)

    else:
        raise ValueError(
            "Unable to find a redshift or scale factor column for this lightcone dataset"
        )
    return dataclasses.replace(
        ctx, lightcone=lightcone.with_new_columns(redshift=z_col)
    )


@hook(HookPoint.LightconeOpen)
def _make_radec_columns(ctx: LightconeOpenCtx):
    lightcone: Lightcone = ctx.lightcone
    if "ra" in lightcone.columns and "dec" in lightcone.columns:
        pass
    elif "theta" in lightcone.columns and "phi" in lightcone.columns:
        lightcone = lightcone.evaluate(
            radec_from_thetaphi, vectorize=True, insert=True, format="numpy"
        )
    elif "properties" in lightcone.dtype:
        warnings.warn(
            "Could not find coordinates in this catalog. Spatial queries will not be available"
        )

    return dataclasses.replace(ctx, lightcone=lightcone)


def radec_from_thetaphi(theta, phi):
    theta_deg = theta * 180 / np.pi
    phi_deg = phi * 180 / np.pi
    return {"ra": phi_deg * u.deg, "dec": (90.0 - theta_deg) * u.deg}


def redshift_from_chi(chi: u.Quantity, cosmology):
    distance = chi.to(u.Mpc, cu.with_H0(cosmology.H0))

    if len(distance) == 0:
        # astropy's z_at_value (used by the redshift-distance equivalency below)
        # opens an np.nditer over the input, which raises "Iteration of
        # zero-sized operands is not enabled" on empty input. Surplus MPI ranks
        # hold zero-length datasets, so short-circuit with an empty result of the
        # right unit rather than invoking the solver.
        return {"redshift": distance.value * cu.redshift}

    z_max = distance.max().to(
        cu.redshift, cu.redshift_distance(cosmology, kind="comoving")
    )
    z_min = distance.min().to(
        cu.redshift, cu.redshift_distance(cosmology, kind="comoving")
    )
    z_grid = np.linspace(z_min.value, z_max.value, 4096)
    chi_grid = cosmology.comoving_distance(z_grid).to(u.Mpc)
    z = np.interp(distance.to_value(u.Mpc), chi_grid.to_value(u.Mpc), z_grid)
    return {"redshift": z * cu.redshift}
