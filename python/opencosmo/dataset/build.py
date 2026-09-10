from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, TypeVar

import astropy.units as u
import h5py
import numpy as np

import opencosmo.dataset.state as state
from opencosmo.dataset import Dataset
from opencosmo.spatial.healpix import HealPixIndex
from opencosmo.spatial.tree import Tree
from opencosmo.spatial.utils import combine_upwards

if TYPE_CHECKING:
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.spatial import Region

T = TypeVar("T")
GroupedColumnData = dict[str, dict[str, T]]
SpatialIndexData = dict[int, tuple[np.ndarray, int]]


def build_dataset_from_data(
    data: GroupedColumnData[np.ndarray],
    header: OpenCosmoHeader,
    region: Region,
    spatial_index_data: Optional[SpatialIndexData],
    descriptions: GroupedColumnData[str] = {},
) -> Dataset:
    data_keys = set(data.keys())
    if data_keys != {"data"}:
        raise ValueError("Data must have exactly one `data` group")
    if descriptions and not set(descriptions.keys()).issubset(data.keys()):
        raise ValueError(
            "Descriptions should be organized into the same groups as the data!"
        )

    tree = None
    if isinstance(spatial_index_data, dict):
        spatial_index_columns = make_spatial_index(spatial_index_data)
        tree = Tree(HealPixIndex(), spatial_index_columns)
    data_group = data.pop("data")

    data_descriptions = descriptions.get("data", {})
    new_state = state.state_in_memory(
        data_group,
        header,
        header.file.unit_convention,
        region,
        {},
        data_descriptions,
        tree=tree,
    )
    return Dataset(new_state)


def make_spatial_index(data: SpatialIndexData):
    """
    allowed input (for now)

    a single level > 0
    """
    if len(data) != 1:
        raise ValueError("Spatial index creation routines should have a single level")
    level = next(iter(data.keys()))
    size, fold_factor = data[level]
    if level <= 0:
        raise ValueError("Data for creating spatial index should include one level > 0")
    name = uuid.uuid1()
    file = h5py.File(f"{name}.hdf5", "w", driver="core", backing_store=False)
    data = combine_upwards(size, fold_factor, level, file)
    output = {}
    for group in data.values():
        assert isinstance(group, h5py.Group)
        output.update({ds.name[1:]: ds for ds in group.values()})
    return output


def split_data_and_metadata(data: dict[str, np.ndarray], descriptions: dict[str, str]):
    output_data = {}
    output_metadata = {}
    for colname, coldata in data.items():
        output_data[colname] = coldata
        column_metadata = {}
        if isinstance(coldata, u.Quantity):
            column_metadata["unit"] = str(coldata.unit)
            output_data[colname] = coldata.value
        if colname in descriptions:
            column_metadata["description"] = descriptions[colname]
        output_metadata[colname] = column_metadata
    return output_data, output_metadata
