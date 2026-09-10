from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, TypeVar

import astropy.units as u  # type: ignore
import numpy as np
from healpy import (  # type: ignore
    ang2vec,
    max_pixrad,
    pix2ang,
    query_disc,
    query_strip,
)

from opencosmo.index import get_length, into_array
from opencosmo.spatial.models import (
    BoxRegionModel,
    ConeRegionModel,
    HealpixRegionModel,
    SkyboxRegionModel,
)
from opencosmo.spatial.relations import (
    contains_2d,
    contains_3d,
    intersects_2d,
    intersects_3d,
)

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord  # type: ignore
    from astropy.cosmology import FLRW

    from opencosmo.index import DataIndex
    from opencosmo.spatial.protocols import Region
    from opencosmo.units import UnitConvention
    from opencosmo.units.handler import UnitHandler

T = TypeVar("T", float, u.Quantity)

Point3d = tuple[float, float, float]
Point2d = tuple[T, T]
BoxSize = tuple[float, float, float]


def physical_to_scalefree(value: float, cosmology: FLRW, z: float):
    a = cosmology.scale_factor(z)
    comoving_value = value / a
    return comoving_to_scalefree(comoving_value, cosmology)


def comoving_to_scalefree(value: float, cosmology: FLRW):
    h = cosmology.h
    scalefree_value = value * h
    return scalefree_value


class ConeRegion:
    """
    Cone region for querying lightcones. Defined by RA/Dec coordinate and an angular
    size. Should always be constructed with :py:meth:`opencosmo.make_cone`

    Note that the underlying coordinate representation of the data may or may not be
    in RA and Dec. Spatial queries handle all the necessary conversions, but this may
    mean that the coordinates in the query output do not appear to match the coordinates
    of the region used to perform the query.
    """

    def __init__(self, center: SkyCoord, radius: u.Quantity):
        self.__center = center
        self.__radius = radius

    def __hash__(self):
        return hash(
            (self.__center.ra.deg, self.center.dec, self.radius.to(u.deg).value)
        )

    def __repr__(self):
        ra = self.__center.ra.deg
        dec = self.__center.dec.deg
        radius = self.__radius.to(u.deg).value
        return (
            f"Cone Region (center: RA={ra:.4f}°, Dec={dec:.4f}°, radius={radius:.4f}°)"
        )

    def into_base_convention(self, *args, **kwargs):
        return self

    def into_model(self) -> ConeRegionModel:
        return ConeRegionModel(
            center=(self.center.ra.deg, self.center.dec.deg),
            radius=self.__radius.value,
        )

    @property
    def center(self):
        """
        The center of this region.

        Returns
        -------
        coordinate: astropy.coordinates.SkyCoord
        """
        return self.__center

    @property
    def radius(self):
        """
        The angular radius of the region.

        Returns
        -------
        radius: astropy.units.Quantity
        """
        return self.__radius

    def contains(self, other: Any):
        """
        Check if this ConeRegion contains another ConeRegion or a sky coordinate.
        A region does not contain itself — the test object must be strictly interior.

        Parameters
        ----------
        other: ConeRegion | astropy.coordinates.SkyCoord
        """
        return contains_2d(self, other)

    def intersects(self, other: Any):
        """
        Check if this ConeRegion intersects with another cone region. Regions which
        touch at a single point but do not cross will return False.

        Parameters
        ----------
        other: :py:class:`opencosmo.spatial.ConeRegion`
            The other region to query.

        Return
        ------
        intersects: bool
            Whether the two regions intersect
        """
        return intersects_2d(self, other)

    def get_healpix_intersections(self, nside: int, nest: bool = True):
        vec = ang2vec(self.center.ra.value, self.center.dec.value, lonlat=True)
        radius = self.radius.to(u.rad).value
        return query_disc(nside, vec, radius, inclusive=True, nest=nest)


class SkyboxRegion:
    """A declination-bounded sky box with an eastward circular RA interval.

    The RA interval starts at ``p1.ra`` and extends eastward to ``p2.ra``. Its
    width is ``(p2.ra - p1.ra) % 360``, so boxes that cross RA=0 retain their
    intended narrow extent.
    """

    def __init__(self, p1: SkyCoord, p2: SkyCoord):
        self.__p1 = p1
        self.__p2 = p2
        self.__ra_start = p1.ra.deg % 360.0
        self.__ra_width = (p2.ra.deg - p1.ra.deg) % 360.0
        self.__dec_bounds = (
            min(p1.dec.deg, p2.dec.deg),
            max(p1.dec.deg, p2.dec.deg),
        )

    def __repr__(self):
        ra0, ra1 = self.ra_bounds
        dec0, dec1 = self.__dec_bounds
        return (
            f"Sky Box Region (RA: {ra0:.4f}°–{ra1:.4f}°, Dec: {dec0:.4f}°–{dec1:.4f}°)"
        )

    @property
    def ra_bounds(self) -> tuple[float, float]:
        """Return the monotonic eastward RA interval, whose upper bound may exceed 360."""
        return self.__ra_start, self.__ra_start + self.__ra_width

    @property
    def ra_start(self) -> float:
        """Starting RA of the eastward interval, normalized to [0, 360)."""
        return self.__ra_start

    @property
    def ra_width(self) -> float:
        """Eastward RA interval width in degrees, in [0, 360)."""
        return self.__ra_width

    @property
    def dec_bounds(self) -> tuple[float, float]:
        return self.__dec_bounds

    def get_healpix_intersections(self, nside: int, nest: bool = True):
        # A skybox is bounded by lines of constant declination, not the great
        # circles that query_polygon draws between corners. A great circle
        # between two equal-declination corners bows toward the nearer pole, so
        # query_polygon would omit pixels along the box's lower-declination edge.
        # Instead, select the full declination band with query_strip and then
        # restrict to the RA bounds.
        dec0, dec1 = self.__dec_bounds
        theta1 = np.radians(90.0 - dec1)
        theta2 = np.radians(90.0 - dec0)
        strip = query_strip(nside, theta1, theta2, inclusive=True, nest=nest)
        strip_ra, _ = pix2ang(nside, strip, lonlat=True, nest=nest)
        # Widen the RA window by one pixel radius (corrected for declination) so
        # pixels straddling the RA edges are treated as intersecting.
        cos_dec = min(np.cos(np.radians(dec0)), np.cos(np.radians(dec1)))
        margin = np.degrees(max_pixrad(nside)) / max(cos_dec, 1e-6)
        if self.__ra_width >= 360.0:
            keep = np.ones(strip.shape, dtype=bool)
        else:
            offset = (strip_ra - self.__ra_start) % 360.0
            keep = (offset <= self.__ra_width + margin) | (offset >= 360.0 - margin)
        # query_strip does not return pixels in ascending order; downstream index
        # projection requires the intersection pixels to be sorted.
        return np.sort(strip[keep])

    def into_base_convention(self, *args, **kwargs):
        return self

    def into_model(self) -> SkyboxRegionModel:
        p1 = (self.__p1.ra.deg, self.__p1.dec.deg)
        p2 = (self.__p2.ra.deg, self.__p2.dec.deg)
        return SkyboxRegionModel(p1=p1, p2=p2)

    def contains(self, other: Any):
        return contains_2d(self, other)

    def intersects(self, other: Any):
        return intersects_2d(self, other)


class HealpixRegion:
    def __init__(self, idxs: DataIndex, nside: int, ordering: str = "nested"):
        self.__idxs = idxs
        self.__nside = nside
        self.__ordering = ordering

    def __repr__(self):
        if get_length(self.__idxs) == 0:
            return "Empty region"
        res = (
            f"Healpix Region (nside = {self.nside}, ordering = {self.ordering})\n"
            f"{get_length(self.__idxs)} pixels in range: {self.pixels.min()} -> {self.pixels.max()}"
        )
        return res

    def into_base_convention(self, *args, **kwargs):
        return self

    def into_model(self):
        return HealpixRegionModel(pixels=into_array(self.__idxs), nside=self.nside)

    def combine(self, *others: HealpixRegion) -> HealpixRegion:
        if any(o.nside != self.nside for o in others):
            raise ValueError("Cannot combine healpix regions with different nsides!")
        if any(o.ordering != self.ordering for o in others):
            raise ValueError("Cannot combine healpix regions with different orderings!")

        output = reduce(
            lambda left, right: np.union1d(left, right),
            (o.pixels for o in others),
            self.pixels,
        )
        return HealpixRegion(output, self.nside, self.ordering)

    @property
    def pixels(self):
        """
        The Healpix pixels contained in this region.
        """
        return into_array(self.__idxs)

    @property
    def nside(self):
        """
        The nside of the map the pixels are drawn from
        """
        return self.__nside

    @property
    def ordering(self):
        """
        The pixel ordering
        """
        return self.__ordering

    def get_healpix_intersections(self, nside: int):
        if nside == self.nside:
            return self.pixels
        else:
            raise ValueError(
                "Healpix regions can only be compared to each other if they have the same nside"
            )

    def contains(self, other: Any):
        return contains_2d(self, other)

    def intersects(self, other: Any):
        return intersects_2d(self, other)


class FullSkyRegion:
    def __init__(self):
        pass

    def __repr__(self):
        return "Full Sky Region"

    def into_base_convention(self, *args, **kwargs):
        return self

    def into_model(self):
        return None

    def contains(self, other: Any):
        return contains_2d(self, other)

    def intersects(self, other: Any):
        return intersects_2d(self, other)


class BoxRegion:
    """
    A region representing a 3-dimensional box of arbitrary length, width, and depth. A
    BoxRegion can be used to query snapshot data. BoxRegions can be constructed with
    :py:meth:`opencosmo.make_box`

    When used in a spatial query, the values defining a box region will always be
    interpreted in the same unit convention as the dataset it is being used to query.
    This means two queries on the same dataset with the same box region can return
    different results if the unit convention changes between them.
    """

    def __init__(self, center: Point3d, halfwidths: BoxSize):
        self.__center = center
        self.__halfwidths = halfwidths
        self.__bounds: list[tuple[float, float]] = [
            (c - hw, c + hw) for c, hw in zip(self.__center, self.__halfwidths)
        ]

    def __repr__(self):
        return f"Box with bounds {self.bounds}"

    def into_model(self) -> BoxRegionModel:
        p1 = tuple(b[0] for b in self.bounds)
        p2 = tuple(b[1] for b in self.bounds)
        return BoxRegionModel(p1=p1, p2=p2)

    def bounding_box(self) -> BoxRegion:
        return self

    def into_base_convention(
        self,
        unit_handler: UnitHandler,
        columns: Iterable[str],
        from_: UnitConvention,
        unit_kwargs: dict[str, Any] = {},
    ):
        center = {col: dim for col, dim in zip(columns, self.__center)}
        halfwidth = {col: dim for col, dim in zip(columns, self.__halfwidths)}

        new_center = tuple(
            v.value
            for v in unit_handler.into_base_convention(center, unit_kwargs).values()
        )
        new_halfwidth = tuple(
            v.value
            for v in unit_handler.into_base_convention(halfwidth, unit_kwargs).values()
        )

        return BoxRegion(new_center, new_halfwidth)

    @property
    def bounds(self):
        """
        The bounds of this region in the form [(min,max) ...]

        Returns
        -------
        bounds: list[tuple(float,float), ....]
        """
        return self.__bounds

    def contains(self, other: Any) -> bool | np.ndarray:
        """
        Check if this box region contains another region or set of points.
        Points should be passed in as a numpy array with shape (3, n_points). Note
        that this method requires that the bounds of the test region be fully inside the
        other region. Functionally this means that a region does not contain itself.

        Parameters
        ----------
        other: BoxRegion | np.ndarray
            The region or points to check

        Returns
        -------
        contained: bool or np.ndarray[bool]
            Whether the region or points are contained in the region

        Raises
        ------
        ValueError
            If the input not a region or points
        """
        return contains_3d(self, other)

    def intersects(self, other: Region) -> bool:
        """
        Check if this BoxRegion intersects another BoxRegion. This function
        returns true of the regions intersect in any way, including if one
        contains the other. Regions that share a bound but do not cross
        will return False.

        Parameters
        ----------
        other: BoxRegion
            The region to check

        Returns
        -------
        intersects: bool
            Whether the regions intersect

        Raises
        ------
        ValueError:
            If the input is not a BoxRegion

        """
        return intersects_3d(self, other)
