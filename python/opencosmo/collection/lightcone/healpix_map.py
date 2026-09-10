from __future__ import annotations

from functools import cached_property, reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Optional, Self
from warnings import warn

import astropy.units as u  # type: ignore
import healpy as hp
import healsparse as hsp
import numpy as np
from astropy.table import Column as AstroColumn  # type: ignore
from astropy.table import vstack

import opencosmo as oc
from opencosmo.column.column import Column
from opencosmo.dataset.build import build_dataset_from_data
from opencosmo.index import from_size, into_array
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import get_comm_world
from opencosmo.spatial.region import ConeRegion, FullSkyRegion, HealpixRegion

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.dataset import Dataset
    from opencosmo.dataset.build import GroupedColumnData
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.iopen import DatasetTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial import Region


def make_healsparse_maps(
    table,
    nside: int,
    nside_lr: int,
) -> dict[str, hsp.HealSparseMap]:
    sentinel = np.float32(hp.UNSEEN)
    pixels = table["pixel"].value

    # Build coverage map once, shared across all columns.
    cov_map = hsp.HealSparseCoverage.make_empty(nside_lr, nside)
    cov_pix = cov_map.cov_pixels(pixels)
    unique_cov_pix = np.unique(cov_pix)
    cov_map.initialize_pixels(unique_cov_pix)
    sparse_indices = pixels + cov_map[cov_pix]
    sparse_map_size = (len(unique_cov_pix) + 1) * cov_map.nfine_per_cov

    result = {}
    for name, col in table.items():
        if name != "pixel":
            sparse_map = np.full(sparse_map_size, sentinel, dtype=np.float32)
            sparse_map[sparse_indices] = col.value.astype(np.float32)
            result[name] = hsp.HealSparseMap(
                cov_map=cov_map,
                sparse_map=sparse_map,
                nside_sparse=nside,
                sentinel=sentinel,
            )
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def take_from_sorted(
    healpix_map: "HealpixMap", sort_by: str, invert: bool, n: int, at: str | int
):
    column = np.concatenate(
        [ds.select(sort_by).get_data("numpy") for ds in healpix_map.values()]
    )
    if invert:
        column = -column
    sort_index = np.argsort(column)
    if at == "start":
        sort_index = sort_index[:n]
    elif at == "end":
        sort_index = sort_index[-n:]
    elif isinstance(at, int):
        if at + n > len(sort_index) or at < 0:
            raise ValueError(
                "Requested a range that is outside the size of this dataset!"
            )
        sort_index = sort_index[at : at + n]

    sorted_indices = np.sort(sort_index)
    return sorted_indices


class HealpixMap(dict):
    """
    A HealpixMap contains one or more datasets of map format. Each dataset will
    typically contain a different type of data over a specified integrated
    redshift range. The HealpixMap object provides an API identical to the standard
    Dataset API, however the data that is provided is returned in healpix or healsparse
    format, which are different than other opencosmo datasets. This also contains some
    convenience functions for standard operations.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        nside: int,
        nside_lr: int,
        ordering: str,
        full_sky: bool,
        z_range: tuple[float, float],
        region: HealpixRegion,
        hidden: Optional[set[str]] = None,
        ordered_by: Optional[tuple[str, bool]] = None,
    ):
        if len(datasets) > 1:
            raise NotImplementedError(
                "Healpix map is not currently implemented correctly for multiple layers"
            )

        assert isinstance(region, HealpixRegion)
        self.update(datasets)
        self.__nside = nside
        self.__nside_lr = nside_lr
        self.__full_sky = full_sky
        self.__z_range = z_range
        self.__ordering = ordering

        columns: set[str] = reduce(
            lambda left, right: left.union(set(right.columns)), self.values(), set()
        )
        if len(columns) != len(next(iter(self.values())).columns):
            raise ValueError("Not all map datasets have the same columns!")

        header = next(iter(self.values())).header
        self.__header = header
        if hidden is None:
            hidden = set()

        self.__hidden = hidden
        self.__ordered_by = ordered_by
        self.__region = region

    def __getattr__(self, key: str):
        try:
            return self.header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def __dir__(self):
        return list(self.header.parameters.keys()) + super().__dir__()

    @property
    def nside(self):
        """
        The healpix nside resolution parameter for this map

        Returns
        -------
        dtype: int
        """
        return self.__nside

    @property
    def pixels(self):
        """
        The healpix pixels that are included in this map
        """
        return self.__region.pixels

    @property
    def nside_lr(self):
        """
        The low resolution nside resolution parameter used to
        access this map with healsparse.
        Returns
        -------
        dtype: int
        """
        return self.__header.healpix_map["nside_lr"]

    @property
    def ordering(self):
        """
        The order of pixelization for the map. Either
        NESTED or RING. Maps are currently always saved
        in NESTED format.

        Returns
        -------
        dtype: str
        """
        return self.__header.healpix_map["ordering"]

    @property
    def full_sky(self):
        """
        Whether the map has full-sky coverage or not
        (note if not you must ask for the data in
        healsparse format and not full healpix format)
        Returns
        -------
        dtype: bool
        """
        return self.__full_sky

    def __repr__(self):
        """
        A basic string representation of the dataset
        """
        length = len(self)

        if len(self) < 10:
            repr_ds = self
            table_head = ""
        else:
            repr_ds = self.take(10, at="start")
            table_head = "First 10 rows:\n"

        table_repr = repr_ds.data.__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        head = f"OpenCosmo Healpix Map Dataset (length={length}, "
        cosmo_repr = f"Cosmology: {self.cosmology.__repr__()}" + "\n"
        return head + cosmo_repr + table_head + table_repr

    def __len__(self):
        return sum(len(ds) for ds in self.values())

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    @property
    def header(self) -> OpenCosmoHeader:
        """
        The header associated with this dataset.

        OpenCosmo headers generally contain information about the original data this
        dataset was produced from, as well as any analysis that was done along
        the way.

        Returns
        -------
        header: opencosmo.header.OpenCosmoHeader

        """
        return self.__header

    @property
    def columns(self) -> list[str]:
        """
        The names of the columns in this map.

        Returns
        -------
        columns: list[str]
        """
        cols = next(iter(self.values())).columns
        cols = list(filter(lambda col: col not in self.__hidden, cols))
        return cols

    @cached_property
    def descriptions(self) -> dict[str, Optional[str]]:
        """
        Return the descriptions (if any) of the columns in this map as a dictonary.
        Columns without a description will be included in the dictionary with a value
        of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        descriptions = next(iter(self.values())).descriptions
        descriptions = dict(
            filter(lambda kv: kv[0] not in self.__hidden, descriptions.items())
        )
        return descriptions

    @property
    def region(self) -> Region:
        """
        The region this dataset is contained in. If no spatial
        queries have been performed, this will be the full sky for
        lightcone maps.

        Returns
        -------
        region: opencosmo.spatial.Region

        """
        return self.__region

    @property
    def z_range(self):
        """
        The redshift range of the data which created this map.

        Returns
        -------
        z_range: tuple[float, float]
        """

        return self.__header.healpix_map["z_range"]

    def get_data(self, format="healsparse", nside_out: Optional[int] = None, **kwargs):
        """
        Get the data in this dataset as healsparse map or as healpix maps
        (nest-ordered numpy array). Note that a dataset does not load data from
        disk into memory until this function is called. As a result, you should
        not call this function until you have performed any transformations you
        plan to on the data.

        You can get the data in two formats, "healsparse" (the default) and "healpix".
        "healsparse" format will return the data as a healsparse sparse map.
        "healpix" will return the data as a dictionary of numpy arrays. If the map does not
        cover the full sky, this wll be a masked numpy array. For map data,
        due to format requirements, no units will be attached to the data itself,
        although these will match the units from the data attributes.


        Parameters
        ----------
        output: str, default="healsparse"
            The format to output the data in

        Returns
        -------
        data: HealsparseMap | Column | dict[str, ndarray] | ndarray
            The data in this dataset.
        """

        if "output" in kwargs:
            warn(
                "The `output` argument of the `get_data` function has been renamed to `format`. Passing the `output` argument will cause a failure in a future version"
            )
            format = kwargs["output"]
        if format not in {"healsparse", "healpix"}:
            raise ValueError(f"Unknown format type {format}")

        if nside_out is not None:
            return self.with_resolution(nside_out).get_data(format)

        data = [ds.get_data(unpack=False) for ds in self.values()]
        pixels = self.pixels
        table = vstack(data, join_type="exact")

        colnames = table.colnames
        if len(colnames) == 1:
            table.rename_column(colnames[0], self.columns[0])

        table["pixel"] = pixels
        table.sort("pixel", reverse=False)

        if len(table.colnames) == 1:
            table = next(table.itercols())

        if format == "healpix":
            npix = hp.nside2npix(self.nside)
            if isinstance(table, (u.Quantity, AstroColumn)):
                vals = np.zeros(npix, dtype=np.float32)
                vals[pixels] = table.value
                storage = {"vals": vals}

            else:
                table.remove_columns(self.__hidden)
                storage = {
                    name: np.zeros(npix, dtype=np.float32)
                    for name in table.columns
                    if name != "pixel"
                }
                for name, arr in storage.items():
                    arr[pixels] = table[name].value

            if len(pixels) != hp.nside2npix(self.nside):
                mask = np.zeros(hp.nside2npix(self.nside), dtype=bool)
                mask[pixels] = True
                storage = {
                    name: np.ma.masked_array(arr, mask) for name, arr in storage.items()
                }
            if len(storage) == 1:
                return next(iter(storage.values()))
            return storage

        elif format == "healsparse":
            return make_healsparse_maps(table, self.nside, self.nside_lr)

    @property
    def data(self):
        """
        Return the data in the dataset in healsparse format. The value of this
        attribute is equivalent to the return value of
        :code:`Dataset.get_data("healsparse")`.


        Returns
        -------
        data : HealsparseMap
            The data in the dataset.

        """
        return self.get_data("healsparse")

    def with_resolution(self, nside) -> HealpixMap:
        """
        Return a copy of the map with a new nside resolution.

        The new resolution must be strictly less than the current
        resolution.

        """
        nside_out = nside
        if nside_out == self.nside:
            return self
        elif nside_out > self.nside:
            raise ValueError(
                "You cannot change the resolution of a map to be higher than its original resolution!"
            )

        pixels = self.pixels
        data = [ds.get_data(unpack=False) for ds in self.values()]
        table = vstack(data, join_type="exact")
        if len(table.colnames) == 1:
            table.rename_column(table.colnames[0], self.columns[0])

        table["pixel"] = pixels

        table.sort("pixel", reverse=False)

        if nside_out >= self.__nside:
            raise ValueError(
                f"New nside {nside_out} is greater than or equal to input value: {self.__nside}."
            )
        if not hp.isnsideok(nside_out):
            raise ValueError(f"New nside {nside_out} is invalid.")

        nside_ratio = self.__nside // nside_out
        pixel_lores = table["pixel"].value // (nside_ratio * nside_ratio)

        new_pixels, boundaries = np.unique(pixel_lores, return_index=True)
        counts = np.add.reduceat(np.ones_like(table["pixel"].value), boundaries).astype(
            float
        )

        new_data: GroupedColumnData = {"data": {}}
        for name in self.columns:
            new_data["data"][name] = (
                np.add.reduceat(table[name].value, boundaries) / counts
            ).astype(np.float32)

        new_header = self.header.with_parameters({"map_params/nside": nside_out})

        out_npix = hp.nside2npix(nside_out)

        new_dataset = build_dataset_from_data(
            new_data,
            new_header,
            self.region,
            None,
            descriptions={
                "data": {
                    key: desc
                    for key, desc in self.descriptions.items()
                    if desc is not None
                }
            },
        )

        is_full_sky = self.full_sky and get_comm_world() is None

        if len(new_pixels) != out_npix:
            region = HealpixRegion(new_pixels, nside_out, self.__ordering)
        else:
            region = HealpixRegion(from_size(out_npix), nside_out, self.__ordering)

        return HealpixMap(
            {"data": new_dataset},
            nside_out,
            self.nside_lr,
            self.ordering,
            is_full_sky,
            self.z_range,
            region,
            self.__hidden,
            self.__ordered_by,
        )

    @classmethod
    def open(cls, targets: list[DatasetTarget], **kwargs):
        raise NotImplementedError()

    def __map(
        self,
        method,
        *args,
        hidden: Optional[set[str]] = None,
        mapped_arguments: dict[str, dict[str, Any]] = {},
        construct: bool = True,
        **kwargs,
    ):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {}
        hidden = hidden if hidden is not None else self.__hidden
        for ds_name, dataset in self.items():
            dataset_mapped_arguments = {
                arg_name: args[ds_name] for arg_name, args in mapped_arguments.items()
            }
            output[ds_name] = getattr(dataset, method)(
                *args, **kwargs, **dataset_mapped_arguments
            )

        if construct:
            return HealpixMap(
                output,
                self.nside,
                self.nside_lr,
                self.ordering,
                self.full_sky,
                self.z_range,
                self.__region,
                self.__hidden,
                self.__ordered_by,
            )
        return output

    def make_schema(self) -> Schema:
        children = {}
        for name, dataset in self.items():
            ds_schema = dataset.make_schema()
            children[name] = ds_schema
        if len(children) == 1:
            schema = next(iter(children.values()))
            schema = make_schema(
                "/",
                FileEntry.HEALPIX_MAP,
                ds_schema.children,
                ds_schema.columns,
                ds_schema.attributes,
            )

        else:
            schema = make_schema("/", FileEntry.HEALPIX_MAP, children=children)

        comm_world = get_comm_world()
        is_full_sky = comm_world is None and self.full_sky
        is_full_sky |= comm_world is not None and comm_world.allreduce(
            len(self.pixels)
        ) == hp.nside2npix(self.nside)

        if not is_full_sky:
            new_header = self.header.with_region(self.region).with_parameter(
                "map_params/full_sky", False
            )
        else:
            new_header = self.header.with_region(FullSkyRegion())

        header_schema = new_header.dump()
        schema.children["header"] = header_schema

        return schema

    def bound(self, region: Region, inclusive: bool = False):
        """
        Restrict this map to some subregion. Be default this will
        include all pixels whose centers fall within the subregion. You can additionally
        include pixels that overalp without there centers being within the
        specified region by passing :code:`inclusive=True`

        If trying to query in a circular region, consider using
        :py:meth:`cone_search <opencosmo.HealpixMap.cone_search>` for simplicity.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query.

        incluive: bool, default = Flase
            Whether to include pixels that overlap but whose centers are not in the region1

        Returns
        -------
        new_map: opencosmo.HealpixMap
            The map including the pixels within the region.

        Raises
        ------
        ValueError
            If the query region does not overlap with the coverage of this map
            in
        """
        # The best we can do here is turn
        if not isinstance(region, ConeRegion):
            raise TypeError(
                "Currently only cone regions are supported when performing spatial queries on HealpixMaps"
            )

        vec = hp.ang2vec(region.center.ra.value, region.center.dec.value, lonlat=True)
        pixels = hp.query_disc(
            self.nside,
            vec,
            region.radius.to(u.radian).value,
            inclusive=inclusive,
            nest=self.__ordering == "NESTED",
        )
        new_datasets = {}
        current_pixels = self.pixels

        for name, dataset in self.items():
            rows_to_take = np.where(
                np.isin(current_pixels, pixels, assume_unique=True)
            )[0]
            new_datasets[name] = dataset.take_rows(rows_to_take)
            assert len(new_datasets[name]) == len(rows_to_take)

        return HealpixMap(
            new_datasets,
            self.nside,
            self.nside_lr,
            self.ordering,
            False,
            self.z_range,
            HealpixRegion(current_pixels[rows_to_take], self.nside),
            self.__hidden,
            self.__ordered_by,
        )

    def cone_search(self, center: tuple | SkyCoord, radius: float | u.Quantity):
        """
        Perform a search for pixels within some angular distance of some
        given point on the sky. This is a convinience function around
        :py:meth:`bound <opencosmo.HealpixMap.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        This search is inclusive, meaning this will include all pixels which overlap
        with the region by any amount.

        Parameters
        ----------
        center: tuple | SkyCoord
            The center of the region to search. If a tuple and no units are provided
            assumed to be RA and Dec in degrees.

        radius: float | astropy.units.Quantity
            The angular radius of the region to query. If no units are provided,
            assumed to be degrees.

        Returns
        -------
        new_map: opencosmo.HealpixMap
            The pixels in these maps that fall within the given region.

        """
        region = oc.make_cone(center, radius)
        return self.bound(region)

    def box_search(self, p1: tuple | SkyCoord, p2: tuple | SkyCoord):
        """
        Perform a search for objects within some range in RA and Dec.
        This is a convinience function around
        :py:meth:`bound <opencosmo.HealpixMap.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        This search is inclusive, meaning this will include all pixels which overlap
        with the region by any amount.

        Parameters
        ----------
        p1: tuple | SkyCoord
            One corner of the box. If a tuple and no units are provided
            assumed to be RA and Dec in degrees.

        p2: tuple | SkyCoord
            The opposite corner of the box. If a tuple and no units are provided,
            assumed to be degrees.

        Returns
        -------
        new_map: opencosmo.HealpixMap
            The pixels in these maps that fall within the given region.

        """
        region = oc.make_box(p1, p2)
        return self.bound(region)

    def evaluate(
        self,
        func: Callable,
        format: str = "numpy",
        vectorize=False,
        insert=True,
        **evaluate_kwargs,
    ):
        """
        Iterate over the rows in this collection, apply `func` to each, and collect
        the result as new columns in the dataset. You may also choose to simply return thevalues
        instead of inserting them as a column

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.HealpixMap.with_new_columns>`
        for cases where the new column is not a simple algebraic combination of existing columns. Unlike
        :code:`with_new_columns`, this method will evaluate the results immediately and the resulting
        columns will not change under unit transformations.

        The function should take in arguments with the same name as the columns in this dataset that
        are needed for the computation, and should return a dictionary of output values.
        The dataset will automatically select the needed columns to avoid unnecessarily reading
        data from disk. The new columns will have the same names as the keys of the output dictionary
        See :ref:`Evaluating On Datasets` for more details.

        If vectorize is set to True, the full columns will be pased to the dataset. Otherwise,
        rows will be passed to the function one at a time.

        This function behaves identically to :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`

        Parameters
        ----------
        func: Callable
            The function to evaluate on the rows in the dataset.

        format: str, default = "numpy"
            The format of the data that is provided to your function. If "astropy", will be a dictionary of
            astropy quantities. If "numpy", will be a dictionary of numpy arrays.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False)

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. Otherwise the data will be returned.

        Returns
        -------
        dataset : HealpixMap
            The new lightcone dataset with the evaluated column(s)
        """

        result = self.__map(
            "evaluate",
            func=func,
            format=format,
            vectorize=vectorize,
            insert=insert,
            mapped_arguments={},
            construct=insert,
            **evaluate_kwargs,
        )
        if next(iter(result.values())) is None:
            return

        if insert:
            assert isinstance(result, HealpixMap)
            return result

        keys = next(iter(result.values())).keys()
        output = {}
        for key in keys:
            output[key] = np.concatenate([r[key] for r in result.values()])
        return output

    def filter(self, *masks: ColumnMask, **kwargs) -> HealpixMap:
        """
        Filter the map based on some criteria. See :ref:`Querying Based on Column
        Values` for more information.

        Parameters
        ----------
        *masks : Mask
            The masks to apply to dataset, constructed with :func:`opencosmo.col`

        Returns
        -------
        dataset : Dataset
            The new dataset with the masks applied.

        Raises
        ------
        ValueError
            If the given  refers to columns that are
            not in the dataset, or the  would return zero rows.

        """
        # This one in particular will need to change if and when we deal with multiple maps at once
        # For the moment we have to assume there is only one dataset to get this right
        ds = next(iter(self.values()))
        init_index = ds.index
        ds = ds.filter(*masks)

        init_index_arr = into_array(init_index)
        final_index_arr = into_array(ds.index)

        is_saved = np.where(
            np.isin(init_index_arr, final_index_arr, assume_unique=True)
        )
        new_pixels = self.pixels[is_saved]
        output = {next(iter(self.keys())): ds}
        return HealpixMap(
            output,
            self.nside,
            self.nside_lr,
            self.ordering,
            False,
            self.z_range,
            HealpixRegion(new_pixels, self.nside),
            self.__hidden,
            self.__ordered_by,
        )

    def rows(self) -> Generator[dict[str, float | u.Quantity], None, None]:
        """
        Iterate over the pixels in the map, returning their individual values.
        Rows are returned as a dictionary. For performance, it is recommended
        to first select the columns you need to work with.

        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset with units.
        """
        yield from chain.from_iterable(v.rows() for v in self.values())

    def select(
        self, *columns: str | Iterable[str], **derived_columns: ConstructedColumn
    ) -> Self:
        """
        Create a new map from a subset of columns in this map. This function
        behaves the same as :py:meth:`Dataset.select <opencosmo.Dataset.select>`.

        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to select.
        **derived_columns: Column
            Any new derived columns that will be instantiated as part of the select

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected columns.

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        all_columns: set[str] = set()
        for col_group in columns:
            if isinstance(col_group, str):
                col_group = {col_group}
            all_columns.update(col_group)

        hidden = self.__hidden | set()

        if self.__ordered_by is not None and self.__ordered_by[0] not in all_columns:
            all_columns.add(self.__ordered_by[0])
            hidden.add(self.__ordered_by[0])

        return self.__map(
            "select",
            all_columns,
            hidden=hidden,
            mapped_arguments={},
            construct=True,
            **derived_columns,
        )

    def drop(self, columns: str | Iterable[str]) -> Self:
        """
        Produce a new dataset by dropping columns from this map.

        Parameters
        ----------
        columns : str or list[str]
            The column or columns to drop.

        Returns
        -------
        dataset : Dataset
            The new dataset without the dropped columns

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        if isinstance(columns, str):
            columns = [columns]

        dropped_columns = set(columns)
        current_columns = set(self.columns)
        if missing := dropped_columns.difference(current_columns):
            raise ValueError(
                f"Tried to drop columns that are not in this dataset: {missing}"
            )
        kept_columns = current_columns - dropped_columns
        return self.select(kept_columns)

    def take(self, n: int, at: str = "random") -> "HealpixMap":
        """
        Create a new dataset from some number of rows from this map.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "random".

        Returns
        -------
        dataset : Dataset
            The new dataset with only the selected rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        if n > len(self):
            raise ValueError(
                "Number of rows to take must be less than number of rows in dataset"
            )
        if at == "random":
            indices = np.random.choice(len(self), n, replace=False)
            indices = np.sort(indices)
            return self.__take_rows(indices)

        elif self.__ordered_by is not None:
            index = take_from_sorted(self, *self.__ordered_by, n=n, at=at)
            return self.__take_rows(index)
        elif at == "start":
            return self.take_range(0, n)
        elif at == "end":
            return self.take_range(len(self) - n, len(self))
        else:
            raise ValueError(
                f'"at" should be one of ("start", "end", "random", got {at}'
            )

    def take_range(self, start: int, end: int):
        if start < 0 or end > len(self):
            raise ValueError("Got row indices that are out of range!")

        if self.__ordered_by is not None:
            indices = take_from_sorted(self, *self.__ordered_by, end - start, at=start)
            return self.__take_rows(indices)

        ends = np.cumsum(np.fromiter((len(ds) for ds in self.values()), dtype=int))
        starts = np.insert(ends, 0, 0)[:-1]
        clipped_starts = np.clip(starts, a_min=start, a_max=None)
        clipped_ends = np.clip(ends, a_min=None, a_max=end)

        output = {}
        for i, (name, dataset) in enumerate(self.items()):
            if starts[i] == clipped_starts[i] and ends[i] == clipped_ends[i]:
                output[name] = dataset
            elif clipped_starts[i] >= clipped_ends[i]:
                continue
            else:
                output[name] = dataset.take_range(
                    clipped_starts[i] - starts[i], clipped_ends[i] - starts[i]
                )

        pixels = self.pixels[start:end]
        return HealpixMap(
            output,
            self.nside,
            self.nside_lr,
            self.ordering,
            False,
            self.z_range,
            HealpixRegion(pixels, self.nside),
            self.__hidden,
            self.__ordered_by,
        )

    def take_rows(self, rows: np.ndarray):
        """
        Take the rows of a map specified by the :code:`rows` argument.
        :code:`rows` should be an array of integers. Note that for healpix
        maps the rows refers to the pixel indices.

        Parameters
        ----------
        rows : np.ndarray[int]
            The indices of the rows to take.

        Returns
        -------
        dataset: The dataset with only the specified rows included

        Raises:
        -------
        ValueError:
            If any of the indices is less than 0 or greater than the length of the
            map.

        """
        rows = np.sort(rows)
        if rows[-1] >= len(self) or rows[0] < 0:
            raise ValueError(
                "Rows must be between 0 and the length of this dataset - 1"
            )
        if self.__ordered_by is not None:
            data = np.concatenate(
                [
                    ds.select(self.__ordered_by[0]).get_data("numpy")
                    for ds in self.values()
                ]
            )
            if self.__ordered_by[1]:
                data = -data
            sort_index = np.argsort(data)
            rows = sort_index[rows]
            rows.sort()

        return self.__take_rows(rows)

    def __take_rows(self, rows: np.ndarray):
        """
        Takes rows from this map while ignoring sort. "rows" is assumed to be sorted.
        For internal use only.
        """
        ds_ends = np.cumsum(np.fromiter((len(ds) for ds in self.values()), dtype=int))
        partitions = np.searchsorted(rows, ds_ends)
        splits = np.split(rows, partitions)
        output = {**self}
        rs = 0
        for split, (name, dataset) in zip(splits, self.items()):
            if len(split) > 0:
                output[name] = dataset.take_rows(split - rs)
            rs += len(dataset)
        return HealpixMap(
            output,
            self.nside,
            self.nside_lr,
            self.ordering,
            False,
            self.z_range,
            HealpixRegion(self.pixels[rows], self.nside),
            self.__hidden,
            self.__ordered_by,
        )

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] = {},
        **columns: Column | np.ndarray | u.Quantity,
    ):
        """
        Create a new map with additional columns. These new columns can be derived
        from columns already in the dataset, or a numpy array.  See :ref:`Adding Custom Columns`
        and :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`
        for examples.

        Parameters
        ----------
        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`HealpixMap.descriptions <opencosmo.HealpixMap.descriptions>`. If a dictionary,
            should have keys matching the column names.

        ** columns : opencosmo.Column | np.ndarray | u.quantity
            The new columns

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """

        derived = {}
        raw = {}
        for name, column in columns.items():
            if isinstance(column, Column):
                derived[name] = column
            elif len(column) != len(self):
                raise ValueError(
                    f"New column {name} has length {len(column)} but this dataset "
                    f"has length {len(self)}"
                )
            else:
                raw[name] = column

        split_points = np.cumsum([len(ds) for ds in self.values()])
        split_points = np.insert(0, 0, split_points)[:-1]
        raw_split = {name: np.split(arr, split_points) for name, arr in raw.items()}
        new_datasets = {}
        for i, (ds_name, ds) in enumerate(self.items()):
            raw_columns = {name: arrs[i] for name, arrs in raw_split.items()}
            columns_input = raw_columns | derived
            new_dataset = ds.with_new_columns(descriptions, **columns_input)
            new_datasets[ds_name] = new_dataset
        return HealpixMap(
            new_datasets,
            self.nside,
            self.nside_lr,
            self.ordering,
            self.full_sky,
            self.z_range,
            self.__region,
            self.__hidden,
            self.__ordered_by,
        )

    def sort_by(self, column: str, invert: bool = False):
        """
        Sort this map by the values in a given column. By default sorting is in
        ascending order (least to greatest). Pass invert = True to sort in descending
        order (greatest to least).

        This is not generally particular useful in map queries, but can be used to
        enforce ordering schemes or find outlier pixels.

        Parameters
        ----------
        column : str
            The column in the map dataset to
            order the collection by.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """

        if column not in self.columns:
            raise ValueError(f"Column {column} does not exist in this dataset!")
        return HealpixMap(
            dict(self),
            self.nside,
            self.nside_lr,
            self.ordering,
            self.full_sky,
            self.z_range,
            self.__region,
            self.__hidden,
            (column, invert),
        )

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> Self:
        r"""
        Unit conversion is usually supported for OpenCosmo datasets, however maps tend to be integrated
        quantities over a range of redshifts which correspond to observed units so applying unit conversions
        is not generally easy or appropriate.
        """

        raise NotImplementedError(
            "Unit conversions not supported on maps, these are integrated over redshift so conversions are non-trivial!"
        )
