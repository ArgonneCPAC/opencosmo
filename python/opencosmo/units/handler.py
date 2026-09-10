from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Mapping, Optional

import astropy.units as u

from opencosmo.units import UnitConvention
from opencosmo.units.get import (
    UnitApplicator,
    get_unit_applicators_dict,
    get_unit_applicators_hdf5,
    parse_unit_string,
)

if TYPE_CHECKING:
    from uuid import UUID

    import h5py
    import numpy as np
    from astropy.cosmology import Cosmology

    from opencosmo.header import OpenCosmoHeader


def make_unit_handler_from_hdf5(
    columns: list[h5py.Dataset],
    header: "OpenCosmoHeader",
    target_convention: Optional[UnitConvention] = None,
):
    applicators = get_unit_applicators_hdf5(columns, header)
    if target_convention is None:
        target_convention = header.file.unit_convention
    if not isinstance(target_convention, UnitConvention):
        target_convention = UnitConvention(target_convention)

    return UnitHandler(
        UnitConvention(header.file.unit_convention),
        target_convention,
        header.cosmology,
        applicators,
    )


def make_unit_handler_from_units(
    columns: dict[str, u.Unit | None],
    header: "OpenCosmoHeader",
    target_convention: Optional[UnitConvention] = None,
):
    applicators = get_unit_applicators_dict(
        columns, header.file.unit_convention, header.cosmology
    )
    if target_convention is None:
        target_convention = header.file.unit_convention
    if not isinstance(target_convention, UnitConvention):
        target_convention = UnitConvention(target_convention)

    return UnitHandler(
        UnitConvention(header.file.unit_convention),
        target_convention,
        header.cosmology,
        applicators,
    )


def make_unit_handler_from_unit_strings(
    unit_strings: dict[str, str | None],
    header: "OpenCosmoHeader",
    target_convention: Optional[UnitConvention] = None,
):
    """
    Cached-metadata counterpart to `make_unit_handler_from_hdf5`. Builds a
    `UnitHandler` from unit strings captured during discovery (e.g.
    `DatasetTarget.column_units`) rather than reading `unit` attributes
    from live h5py datasets.
    """
    columns = {name: parse_unit_string(us) for name, us in unit_strings.items()}
    return make_unit_handler_from_units(columns, header, target_convention)


class UnitHandler:
    def __init__(
        self,
        base_convention: UnitConvention,
        current_convention: UnitConvention,
        cosmology: Cosmology,
        applicators: dict[str, UnitApplicator],
        conversions: dict[str, u.Unit] = {},
        column_conversions: dict[str, u.Unit] = {},
    ):
        if not isinstance(base_convention, UnitConvention):
            raise ValueError
        self.__base_convention = base_convention
        self.__current_convention = current_convention
        self.__cosmology = cosmology
        self.__applicators = applicators
        self.__conversions = conversions
        self.__column_conversions = column_conversions

    @property
    def current_convention(self):
        return self.__current_convention

    @property
    def base_convention(self):
        return self.__base_convention

    @property
    def columns_with_conversions(self):
        return {
            name
            for name, applicator in self.__applicators.items()
            if name in self.__column_conversions
            or str(applicator.unit_in_convention(self.current_convention))
            in self.__conversions
        }

    @cached_property
    def base_units(self):
        return {key: app.base_unit for key, app in self.__applicators.items()}

    @property
    def current_units(self):
        return {
            key: self.__conversions.get(
                str(app.unit_in_convention(self.__current_convention)),
                app.unit_in_convention(self.__current_convention),
            )
            for key, app in self.__applicators.items()
        } | self.__column_conversions

    @cached_property
    def unitless_columns(self):
        return set(
            key for key, app in self.__applicators.items() if app.base_unit is None
        )

    def with_static_columns(self, **columns: u.Unit):
        new_applicators = {}
        for colname, unit in columns.items():
            new_applicators[colname] = UnitApplicator.static(
                unit, self.__base_convention
            )
        return UnitHandler(
            self.__base_convention,
            self.__current_convention,
            self.__cosmology,
            self.__applicators | new_applicators,
            self.__conversions,
            self.__column_conversions,
        )

    def with_new_columns(self, **columns: u.Unit):
        new_applicators = {}
        for colname, unit in columns.items():
            new_applicators[colname] = UnitApplicator.from_unit(
                unit, self.__base_convention, self.__cosmology
            )
        return UnitHandler(
            self.__base_convention,
            self.__current_convention,
            self.__cosmology,
            self.__applicators | new_applicators,
            self.__conversions,
            self.__column_conversions,
        )

    def verify_conversions(
        self, conversions: dict[str, u.Unit], convention: UnitConvention
    ):
        conversion_keys = set(conversions.keys())
        unitless_columns = self.unitless_columns.intersection(conversion_keys)
        if unitless_columns:
            raise ValueError(f"Columns {unitless_columns} do not carry units!")

        for name, unit in conversions.items():
            if not self.__applicators[name].can_convert(unit, convention):
                raise ValueError(
                    f"Cannot convert values with units {self.__applicators[name].unit_in_convention(convention)} to value with units {unit}"
                )

    def with_convention(self, convention: str | UnitConvention):
        convention = UnitConvention(convention)
        return UnitHandler(
            self.__base_convention, convention, self.__cosmology, self.__applicators
        )

    def with_conversions(
        self, conversions: dict[u.Unit, u.Unit], columns: dict[str, u.Unit]
    ):
        if not conversions and not columns:
            return self

        self.verify_conversions(columns, self.__current_convention)
        new_column_conversions = self.__column_conversions | columns
        new_conversions = self.__conversions | {
            str(key): value for key, value in conversions.items()
        }
        return UnitHandler(
            self.__base_convention,
            self.__current_convention,
            self.__cosmology,
            self.__applicators,
            new_conversions,
            new_column_conversions,
        )

    def into_base_convention(
        self,
        data: Mapping[str, float | np.ndarray | u.Quantity],
        unit_kwargs: dict = {},
    ):
        return {
            name: self.__applicators[name].convert_to_base(
                val, self.__current_convention, unit_kwargs=unit_kwargs
            )
            if name in self.__applicators
            else val
            for name, val in data.items()
        }

    def apply_raw_units(self, data: dict[str, np.ndarray], unit_kwargs):
        if self.__current_convention == UnitConvention.UNITLESS:
            return data
        columns: dict[str, np.ndarray] = {}
        for colname, value in data.items():
            columns[colname] = value
            applicator = self.__applicators.get(colname)
            if applicator is not None and applicator.base_unit is not None:
                unitful = applicator.apply(
                    columns[colname], self.__current_convention, unit_kwargs=unit_kwargs
                )
                assert isinstance(unitful, u.Quantity)

                columns[colname] = unitful
        return columns

    def apply_unit_conversions(
        self,
        data: dict[UUID, dict[str, u.Quantity | np.ndarray]],
        unit_kwargs,
    ) -> dict[UUID, dict[str, u.Quantity | np.ndarray]]:
        # Only apply the unit CONVERSIONS. Useful for cached data.
        # Does not return data that was not updated.
        if not self.__conversions and not self.__column_conversions:
            return {}
        output: dict[UUID, dict[str, u.Quantity | np.ndarray]] = {}
        for uuid, uuid_data in data.items():
            for colname, column in uuid_data.items():
                if not isinstance(column, u.Quantity):
                    continue
                column_conversion = self.__column_conversions.get(colname)
                unit_conversion = self.__conversions.get(str(column.unit))
                if unit_conversion is not None and column_conversion is None:
                    output.setdefault(uuid, {})[colname] = column.to(unit_conversion)
                elif column_conversion is not None:
                    output.setdefault(uuid, {})[colname] = column.to(column_conversion)
        return output

    def apply_units(self, data: dict[str, np.ndarray], unit_kwargs):
        if self.__current_convention == UnitConvention.UNITLESS:
            return data
        columns = {}

        for colname, value in data.items():
            applicator = self.__applicators.get(colname)
            column_conversion = self.__column_conversions.get(colname)
            if applicator is not None and applicator.base_unit is not None:
                unitful_value = applicator.apply(
                    value, self.__current_convention, unit_kwargs=unit_kwargs
                )
                assert isinstance(unitful_value, u.Quantity)
                unit_conversion = self.__conversions.get(str(unitful_value.unit))
                if unit_conversion is not None and column_conversion is None:
                    unitful_value = unitful_value.to(unit_conversion)
                elif column_conversion is not None:
                    unitful_value = unitful_value.to(column_conversion)
                columns[colname] = unitful_value
            else:
                columns[colname] = value

        return columns
