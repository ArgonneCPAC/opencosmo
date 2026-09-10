from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from warnings import warn

import astropy.units as u
import numpy as np

from opencosmo.column.reducer import default_reducer
from opencosmo.dataset import state as st
from opencosmo.dataset.evaluate import build_evaluated_column, visit_dataset
from opencosmo.dataset.formats import convert_data, verify_format
from opencosmo.dataset.take import (
    get_end_take_index,
    get_random_take_index,
    get_range_take_index,
)
from opencosmo.index import empty, get_range, into_array, mask, project
from opencosmo.spatial import check
from opencosmo.units.convention import UnitConvention
from opencosmo.units.converters import get_scale_factor

if TYPE_CHECKING:
    from typing import Callable, Iterable, Literal

    from opencosmo.column.column import (
        ColumnMask,
        ConstructedColumn,
        DerivedScalarValue,
    )
    from opencosmo.dataset.state import DatasetState
    from opencosmo.index import DataIndex


def get_data(
    state: DatasetState,
    format: str,
    unpack: bool = False,
    wrap_single: bool = False,
    **kwargs,
):
    verify_format(format)

    if state.convention.value == "physical":
        scale_factor = get_scale_factor(
            state, state.header.cosmology, state.header.file.redshift
        )
        unit_kwargs = {"scale_factor": scale_factor}
    else:
        unit_kwargs = {}

    data = st.get_data(
        state,
        unit_kwargs=unit_kwargs,
        **kwargs,
    )  # dict
    if unpack:
        data = {
            key: value[0]
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == 1
            else value
            for key, value in data.items()
        }

    return convert_data(data, format, wrap_single=wrap_single)


def filter(state: DatasetState, *masks: ColumnMask, mode: str = "global"):
    reducer = default_reducer(mode)
    masks = tuple(m.with_reducer(reducer) for m in masks)
    bool_mask = np.ones(len(state), dtype=bool)
    required_columns = set()
    for m in masks:
        required_columns |= m.requires_names

    selected_state = st.select(state, required_columns)
    data = get_data(selected_state, "astropy", wrap_single=True)

    for m in masks:
        bool_mask &= m.apply(data)

    new_state = st.take_rows(state, np.where(bool_mask)[0])
    return new_state


def select(
    state: DatasetState,
    *columns: str | Iterable[str],
    mode: str = "global",
    **derived_columns: ConstructedColumn | DerivedScalarValue,
):
    from opencosmo.column.column import Column, DerivedScalarValue
    from opencosmo.column.reducer import default_reducer

    all_columns: set[str] = set()
    for col_group in columns:
        if isinstance(col_group, str):
            col_group = {col_group}
        all_columns.update(col_group)

    scalars: dict[str, DerivedScalarValue] = {}
    non_scalars: dict[str, ConstructedColumn] = {}
    for name, col in derived_columns.items():
        if isinstance(col, DerivedScalarValue):
            scalars[name] = col
        else:
            non_scalars[name] = col

    if scalars and (all_columns or non_scalars):
        raise ValueError(
            "Scalar selections cannot be mixed with column selections. "
            "Call select() with only scalar kwargs, or only column selections."
        )

    reducer = default_reducer(mode)
    derived_columns = {
        k: v.with_reducer(reducer) if isinstance(v, (Column, DerivedScalarValue)) else v
        for k, v in derived_columns.items()
    }

    new_state = state
    if derived_columns:
        new_state = st.with_new_columns(new_state, {}, False, **derived_columns)
        all_columns.update(derived_columns.keys())

    if all_columns:
        new_state = st.select(new_state, all_columns)
    return new_state


def drop(state: DatasetState, *columns: str | Iterable[str]):
    all_columns: set[str] = set()
    for col_group in columns:
        if isinstance(col_group, str):
            col_group = {col_group}
        all_columns.update(col_group)
    return st.select(state, all_columns, drop=True)


def sort_by(state: DatasetState, column: str | None, invert: bool) -> DatasetState:
    if column is None:
        sort_key = None
    elif column not in state.columns:
        raise ValueError(f"This dataset has no column {column}")
    else:
        sort_key = (column, invert, False)

    return dataclasses.replace(state, sort_key=sort_key)


def with_new_columns(
    state: DatasetState,
    descriptions: str | dict[str, str] = {},
    allow_overwrite: bool = False,
    mode: str = "global",
    **new_columns: ConstructedColumn | np.ndarray | u.Quantity,
):
    from opencosmo.column.column import Column, DerivedScalarValue
    from opencosmo.column.reducer import default_reducer

    if any(isinstance(col, DerivedScalarValue) for col in new_columns.values()):
        raise ValueError(
            "Scalar values cannot be added to an existing dataset, but can be retrieved with Dataset.select()"
        )
    reducer = default_reducer(mode)
    new_columns = {
        k: v.with_reducer(reducer) if isinstance(v, Column) else v
        for k, v in new_columns.items()
    }
    if isinstance(descriptions, str):
        descriptions = {key: descriptions for key in new_columns.keys()}
    return st.with_new_columns(state, descriptions, allow_overwrite, **new_columns)


def evaluate(
    state: DatasetState,
    func: Callable,
    vectorize=False,
    insert=True,
    format="astropy",
    batch_size: int = -1,
    allow_overwrite: bool = False,
    _verify: bool = True,
    **evaluate_kwargs,
):
    verify_format(format)
    evaluated_column = build_evaluated_column(
        state, func, vectorize, insert, format, batch_size, evaluate_kwargs
    )

    if not insert:
        output = visit_dataset(evaluated_column, state, batch_size)
        return output
    return with_new_columns(
        state, allow_overwrite=allow_overwrite, **{func.__name__: evaluated_column}
    )


def take_range(
    state: DatasetState,
    start: int,
    end: int,
    mode: Literal["local", "global"],
) -> DatasetState:
    if start < 0 or end < 0:
        raise ValueError("start and end must be positive.")
    if end < start:
        raise ValueError("end must be greater than start.")

    take_index = get_range_take_index(state, state.sort_key, start, end - start, mode)
    return st.take_rows(state, take_index)


def take_rows(state: DatasetState, rows: DataIndex) -> DatasetState:
    row_range = get_range(rows)
    if row_range[0] < 0 or row_range[1] > len(state):
        raise ValueError(
            "Row indices must be between 0 and the length of this dataset - 1!"
        )

    return st.take_rows(state, rows)


def take(
    state: DatasetState,
    n: int,
    at: str,
    mode: Literal["local", "global"],
) -> DatasetState:
    if at == "start":
        return take_range(state, 0, n, mode)
    elif at == "end":
        take_index = get_end_take_index(n, state, state.sort_key, mode)
        return take_rows(state, take_index)
    elif at != "random":
        raise ValueError(f"Unknown take type {at}")

    row_indices = get_random_take_index(n, len(state), mode)
    return take_rows(state, row_indices)


def rows(
    state: DatasetState,
    include_units: bool = True,
):
    if state.convention.value == "physical":
        scale_factor = get_scale_factor(
            state, state.header.cosmology, state.header.file.redshift
        )
        unit_kwargs = {"scale_factor": scale_factor}
    else:
        unit_kwargs = {}

    for row in st.iter_rows(state, unit_kwargs):
        output_data = row
        if not isinstance(output_data, dict):
            output_data = {state.columns[0]: row}

        if not include_units:
            output_data = {
                name: val.value if isinstance(val, u.Quantity) else val
                for name, val in output_data.items()
            }
        yield output_data


def bound(state: DatasetState, region, select_by):
    if state.tree is None:
        raise AttributeError(
            "Your dataset does not contain a spatial index, "
            "so spatial querying is not available"
        )

    if not state.header.file.is_lightcone:
        columns = check.find_coordinates_3d(
            state, str(state.header.file.data_type), select_by
        )

        check_region = region.into_base_convention(
            state.unit_handler,  # type: ignore[arg-type]
            columns,
            state.convention,
            {
                "scale_factor": state.header.cosmology.scale_factor(
                    state.header.file.redshift
                ).value
            },
        )
    else:
        check_region = region

    if not state.region.intersects(check_region):
        return st.take_rows(state, empty())

    if not state.region.contains(check_region):
        warn(
            "You're querying with a region that is not fully contained by the "
            "region this dataset is in. This may result in unexpected behavior"
        )

    contained_index: DataIndex
    intersects_index: DataIndex
    contained_index, intersects_index = state.tree.query(check_region)

    contained_index = project(state.raw_index, contained_index)
    intersects_index = project(state.raw_index, intersects_index)

    check_state = st.take_rows(state, intersects_index)
    if not state.header.file.is_lightcone:
        check_state = with_units(check_state, "scalefree", {})

    if len(check_state) > 0:
        index_mask = check.check_containment(
            check_state, check_region, state.header.file, select_by
        )
        new_intersects_index = mask(intersects_index, index_mask)
    else:
        new_intersects_index = np.array([], dtype=np.int64)

    new_index = np.sort(
        np.concatenate([into_array(contained_index), into_array(new_intersects_index)])
    )

    new_tree = state.tree.with_region(check_region)

    new_state = st.take_rows(state, new_index)
    return dataclasses.replace(new_state, tree=new_tree)


def with_units(
    state: DatasetState,
    convention: str | None,
    conversions: dict[u.Unit, u.Unit],
    **columns: u.Unit,
) -> DatasetState:
    if convention is None:
        convention_ = state.unit_handler.current_convention
    else:
        convention_ = UnitConvention(convention)

    if (
        convention_ == UnitConvention.SCALEFREE
        and UnitConvention(state.header.file.unit_convention)
        != UnitConvention.SCALEFREE
    ):
        raise ValueError(
            f"Cannot convert units with convention {state.header.file.unit_convention} to convention scalefree"
        )
    column_keys = set(columns.keys())
    missing_columns = column_keys - set(state.columns)
    if missing_columns:
        raise ValueError(f"Dataset does not have columns {missing_columns}")
    return st.with_units(
        state,
        convention_,
        conversions,
        columns,
        state.header.cosmology,
        state.header.file.redshift,
    )
