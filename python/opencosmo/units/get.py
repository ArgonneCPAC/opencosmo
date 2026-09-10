from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Callable, Optional

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.constants import m_p

if TYPE_CHECKING:
    import h5py
    import numpy as np
    from astropy.cosmology import Cosmology
    from numpy.typing import ArrayLike

from opencosmo.units.convention import UnitConvention
from opencosmo.units.converters import get_unit_transitions

KNOWN_UNITS = {
    "comoving Mpc/h": u.Mpc / cu.littleh,
    "comoving (Mpc/h)^2": (u.Mpc / cu.littleh) ** 2,
    "comoving km/s": u.km / u.s,
    "comoving (km/s)^2": (u.km / u.s) ** 2,
    "Msun/h": u.Msun / cu.littleh,
    "Msun/yr": u.Msun / u.yr,
    "K": u.K,
    "comoving (Msun/h * (km/s) * Mpc/h)": (u.Msun / cu.littleh)
    * (u.km / u.s)
    * (u.Mpc / cu.littleh),
    "log10(erg/s)": u.DexUnit("erg/s"),
    "h^2 keV / (comoving cm)^3": (cu.littleh**2) * u.keV / (u.cm**3),
    "keV * cm^2": u.keV * u.cm**2,
    "cm^-3": u.cm**-3,
    "Gyr": u.Gyr,
    "Msun/h / (comoving Mpc/h)^3": (u.Msun / cu.littleh) / (u.Mpc / cu.littleh) ** 3,
    "Msun/h * km/s": (u.Msun / cu.littleh) * (u.km / u.s),
    "H0^-1": (u.s * (1 * u.Mpc).to(u.km).value).to(u.Gyr) / (100 * cu.littleh),
    "m_hydrogen": m_p,
    "Msun * (km/s)^2": (u.Msun) * (u.km / u.s) ** 2,
}


class UnitApplicator:
    def __init__(
        self,
        units: dict[UnitConvention, u.Unit],
        base_convention: UnitConvention,
        converters: dict[UnitConvention, Callable],
        invserse_converters: dict[UnitConvention, Callable],
        prefactor: Optional[float] = None,
    ):
        self.__units = units
        self.__base_convention = base_convention
        self.__converters = converters
        self.__inv_converters = invserse_converters
        self.__prefactor = prefactor

    @classmethod
    def static(cls, base_unit: u.Unit, base_convention: UnitConvention):
        """
        Unit applicator for a column that does not transform under changes in unit convention.
        This column CAN still be transformed explicitly to an equivalent unit.
        """
        units = {base_convention: base_unit}
        if base_convention in [UnitConvention.COMOVING, UnitConvention.SCALEFREE]:
            units[UnitConvention.PHYSICAL] = base_unit
        if base_convention == UnitConvention.SCALEFREE:
            units[UnitConvention.COMOVING] = base_unit
        return UnitApplicator(units, base_convention, {}, {})

    @classmethod
    def from_unit(
        cls,
        base_unit: Optional[u.Unit],
        base_convention: UnitConvention,
        cosmology: Cosmology,
        is_comoving: bool = True,
    ):
        if base_unit is None:
            # Certain "units" are not actually units (e.g. m_p)
            return UnitApplicator({}, base_convention, {}, {})

        if isinstance(base_unit, u.Quantity):
            trans, inv_trans, units = get_unit_transitions(
                base_unit.unit, base_convention, cosmology, is_comoving
            )
            prefactor = base_unit.value
        else:
            trans, inv_trans, units = get_unit_transitions(
                base_unit, base_convention, cosmology, is_comoving
            )
            prefactor = None

        return UnitApplicator(
            units, base_convention, trans, inv_trans, prefactor=prefactor
        )

    @property
    def base_unit(self):
        return self.unit_in_convention(self.__base_convention)

    def unit_in_convention(self, convention: UnitConvention):
        return self.__units.get(convention)

    def apply(
        self,
        value: ArrayLike,
        convention: UnitConvention,
        convert_to: Optional[u.Unit] = None,
        unit_kwargs: dict[str, Any] = {},
    ) -> ArrayLike | u.Quantity:
        if not self.__units or convention == UnitConvention.UNITLESS:
            return value
        if hasattr(value, "unit"):
            raise ValueError(
                "Units can only be applied to unitless scalars and numpy arrays"
            )
        new_value = value * self.__units[self.__base_convention]
        if convention != self.__base_convention:
            new_value = self.__convert(new_value, convention, unit_kwargs)

        if convert_to is not None:
            new_value = new_value.to(convert_to)

        if self.__prefactor is not None:
            new_value *= self.__prefactor
        return new_value

    def can_convert(self, to_: u.Unit, convention: UnitConvention):
        unit_to_convert = self.__units.get(convention)
        if unit_to_convert is None:
            return False
        return to_.is_equivalent(unit_to_convert)

    def convert_to_base(
        self,
        value: np.ndarray | float,
        convention: UnitConvention,
        unit_kwargs: dict[str, Any] = {},
    ):
        if not self.__units:
            return value

        if not isinstance(value, u.Quantity):
            value = value * self.__units[convention]

        converter = self.__inv_converters.get(convention)
        if converter is not None:
            return converter(value, **unit_kwargs)
        return value

    def __convert(
        self, value: u.Quantity, to_: UnitConvention, unit_kwargs: dict[str, Any]
    ) -> u.Quantity:
        converter = self.__converters.get(to_)
        if converter is not None:
            return converter(value, **unit_kwargs)
        return value


def get_unit_applicators_dict(
    units: dict[str, u.Unit | None],
    base_convention: UnitConvention | str,
    cosmology: Cosmology,
    is_comoving: bool = True,
):
    # FileParameters sets use_enum_values=True, so header.file.unit_convention is
    # a plain str at runtime despite its annotation. UnitConvention is a bare Enum
    # with no str mixin, so the value patterns in get_unit_transitions never match
    # a raw string and it raises.
    base_convention = UnitConvention(base_convention)
    applicators = {}
    for name, base_unit in units.items():
        applicators[name] = UnitApplicator.from_unit(
            base_unit, base_convention, cosmology, is_comoving
        )
    return applicators


@cache
def parse_unit_string(unit_string: str | None) -> Optional[u.Unit]:
    """
    Parse a unit string captured from an HDF5 column's "unit" attribute.

    `None` (the attribute was absent) and the literal strings "None" and ""
    (the attribute was present but empty) all mean "no unit" here; the
    distinction between an absent and an empty/"None" attribute is preserved
    by the caller, not by this function.

    Memoized: anything not in `KNOWN_UNITS` falls through to astropy's grammar
    parser, which is expensive, and a catalog re-presents the same handful of
    unit strings for every column of every file it is opened alongside. `u.Unit`
    values are immutable, so sharing one instance between callers is safe.
    """
    if unit_string is None or unit_string == "None" or unit_string == "":
        return None
    if (unit := KNOWN_UNITS.get(unit_string)) is not None:
        return unit
    try:
        return u.Unit(unit_string)
    except ValueError:
        return None


def get_raw_units(column: h5py.Dataset) -> Optional[u.Unit]:
    return parse_unit_string(column.attrs.get("unit"))
