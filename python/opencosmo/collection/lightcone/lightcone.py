from __future__ import annotations

from collections import defaultdict
from functools import cached_property, reduce
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Self,
)
from warnings import warn

import healpy as hp
import numpy as np
from astropy.table import vstack  # type: ignore

import opencosmo as oc
from opencosmo.collection.lightcone import io as lcio
from opencosmo.collection.lightcone import utils as lcutils
from opencosmo.collection.lightcone.instantiate import evaluate_scope
from opencosmo.collection.lightcone.stack import stack_lightcone_datasets_in_schema
from opencosmo.column.column import (
    Column,
    DerivedScalarValue,
    EvaluatedColumn,
    mask_contains_scalar,
    resolve_mask_scalars,
)
from opencosmo.column.reducer import default_reducer
from opencosmo.dataset import Dataset
from opencosmo.dataset.evaluate import build_evaluated_column
from opencosmo.dataset.formats import concat_chunks, convert_data, verify_format
from opencosmo.dataset.take import (
    get_end_take_index,
    get_random_take_index,
    get_range_take_index,
    get_rows_take_index,
)
from opencosmo.deprecated import deprecated
from opencosmo.index import get_range, into_array, rebuild_by_ranges
from opencosmo.io import iopen
from opencosmo.io.index_spec import index_spec_for
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import get_comm_world
from opencosmo.plugins.contexts import (
    HookPoint,
    LightconeInstantiateCtx,
    LightconeOpenCtx,
    PostSortCtx,
)
from opencosmo.plugins.hook import fold

if TYPE_CHECKING:
    from uuid import UUID

    import astropy.units as u  # type: ignore
    import numpy.typing as npt
    from astropy.coordinates import SkyCoord

    from opencosmo.column.column import (
        ColumnMask,
        ConstructedColumn,
    )
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.index import DataIndex
    from opencosmo.io.iopen import FileTarget
    from opencosmo.io.schema import Schema
    from opencosmo.spatial import Region


class Lightcone(dict):
    """
    A lightcone contains two or more datasets that are part of a lightcone. Typically
    each dataset will cover a specific redshift range. The Lightcone object
    hides these details, providing an API that is identical to the standard
    Dataset API. Additionally, the lightcone contains some convinience functions
    for standard operations.

    Lightcones can be nested. In this case, the top level will split the datasets
    up by step, while the second level will split the datasets up by type. This nested
    scheme (at present) is used for Diffsky catalogs, which may contain both cores
    and synthetic cores that need to be adddressed (and more importantly, written)
    seperately from one another.
    """

    def __init__(
        self,
        datasets: Mapping[Any, Dataset | Lightcone],
        z_range: Optional[tuple[float, float]] = None,
        hidden: Optional[set[str]] = None,
        sort_key: Optional[tuple[str, bool]] = None,
        scope: Optional[Any] = None,
    ):
        from opencosmo.collection.lightcone.scope import LightconeScope

        self.update(datasets)
        z_range = (
            z_range
            if z_range is not None
            else lcutils.get_redshift_range(list(datasets.values()))
        )

        columns: set[str] = reduce(
            lambda left, right: left.union(set(right.columns)), self.values(), set()
        )
        if len(columns) != len(next(iter(self.values())).columns):
            raise ValueError("Not all lightcone datasets have the same columns!")
        header = next(iter(self.values())).header
        self.__header = header.with_parameter("lightcone/z_range", z_range)

        if hidden is None:
            hidden = set()
        if scope is None:
            scope = LightconeScope()

        self.__hidden = hidden
        self.__sort_key = sort_key
        self.__scope = scope

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

        table_repr = repr_ds.get_data().__repr__()
        # remove the first line
        table_repr = table_repr[table_repr.find("\n") + 1 :]
        z_range = self.z_range
        head = (
            f"OpenCosmo Lightcone Dataset (length={length}, "
            f"{z_range[0]} < z < {z_range[1]})\n"
        )
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

    def __getattr__(self, key: str):
        try:
            return self.__header.parameters[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def __dir__(self):
        return list(self.header.parameters.keys()) + super().__dir__()

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
    def scope(self):
        """The lightcone-level scope for scoped columns."""
        return self.__scope

    @property
    def columns(self) -> list[str]:
        """
        The names of the columns in this dataset.

        Returns
        -------
        columns: list[str]
        """
        cols = next(iter(self.values())).columns
        cols = list(filter(lambda col: col not in self.__hidden, cols))
        cols.extend(name for name in self.__scope.names() if name not in cols)
        return cols

    # Internal identity used by link/mapping resolution.
    @property
    def uuid(self) -> UUID:
        return next(iter(self.values())).uuid

    @cached_property
    def descriptions(self) -> dict[str, Optional[str]]:
        """
        Return the descriptions (if any) of the columns in this lightcone as a dictonary.
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

    @cached_property
    def units(self) -> dict[str, Optional[u.Unit]]:
        """
        Return the units of the columns in this lightcone. Columns without a unit will
        return a value of None

        Returns
        -------

        descriptions : dict[str, str | None]
            The column descriptions
        """
        units = next(iter(self.values())).units
        units = dict(filter(lambda kv: kv[0] not in self.__hidden, units.items()))
        return units

    @property
    def region(self) -> Region:
        """
        The region this dataset is contained in. If no spatial
        queries have been performed, this will be the entire
        simulation box for snapshots or the full sky for lightcones

        Returns
        -------
        region: opencosmo.spatial.Region

        """
        regions = [v.region for v in self.values()]
        if len(regions) == 1:
            return regions[0]
        return regions[0].combine(*regions[1:])

    @property
    def sorted_by(self) -> Optional[str]:
        """
        The column this dataset is sorted by. If not sorted, returns None.

        Returns
        -------
        column: Optional[str]
        """
        return self.__sort_key[0] if self.__sort_key is not None else None

    def get_pixels(self, nside: int = 64):
        """
        Return the HEALPix pixels occupied by this lightcone at a given resolution.

        Pixel indices are returned in nested ordering. The ``nside`` parameter
        controls angular resolution: larger values produce finer pixels. The
        requested resolution may not exceed the resolution of the spatial index
        stored in the file.

        Parameters
        ----------
        nside : int, default = 64
            The HEALPix resolution parameter. Must be a positive power of two.

        Returns
        -------
        pixels : numpy.ndarray[int]
            HEALPix pixel indices (nested ordering) occupied by this lightcone
            at the given resolution.

        Raises
        ------
        ValueError
            If ``nside`` is not a positive power of two, if ``nside`` exceeds
            the maximum resolution of the spatial index, or if the lightcone
            does not have a spatial index.
        """

        level = np.log2(nside)
        if not level.is_integer() or level < 0:
            raise ValueError("nside must be a positive power of two!")

        return lcutils.get_pixels(self, int(level))

    def get_data(
        self,
        format="astropy",
        unpack: bool = True,
        wrap_single: bool = False,
        **kwargs,
    ):
        """
        Get the data in this dataset as an astropy table/column or as
        numpy array(s). Note that a dataset does not load data from disk into
        memory until this function is called. As a result, you should not call
        this function until you have performed any transformations you plan to
        on the data.

        You can get the data in two formats, "astropy" (the default) and "numpy".
        "astropy" format will return the data as an astropy table with associated
        units. "numpy" will return the data as a dictionary of numpy arrays. The
        numpy values will be in the associated unit convention, but no actual
        units will be attached.

        If the dataset only contains a single column, it will be returned as an
        astropy.table.Column or a single numpy array. Pass :code:`wrap_single=True`
        to always return the format's multi-column container (QTable, DataFrame,
        dict, ...) regardless of column count.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in. Currently supported are "astropy", "numpy",
            "pandas", "polars", and "arrow"

        wrap_single: bool, default=False
            If True, always return the format's natural multi-column container even
            when only one column is present.

        Returns
        -------
        data: Table | Column | dict[str, ndarray] | ndarray
            The data in this dataset.
        """
        if "output" in kwargs:
            warn(
                "The `output` argument of the `get_data` function has been renamed to `format`. Passing the `output` argument will cause a failure in a future version"
            )
            format = kwargs["output"]
        verify_format(format)
        lightcone = fold(
            HookPoint.LightconeInstantiate, LightconeInstantiateCtx(self)
        ).lightcone

        data = [
            ds.get_data(unpack=False, wrap_single=True) for ds in lightcone.values()
        ]
        data_with_length = [d for d in data if len(d) > 0]
        if len(data_with_length) == 0:
            vstacked = data[0]
        else:
            vstacked = vstack(data_with_length, join_type="exact")

        vstacked, scope_scalars = evaluate_scope(self.__scope, vstacked, unpack=unpack)
        if scope_scalars is not None:
            return convert_data(scope_scalars, format, wrap_single=wrap_single)

        if self.__sort_key is not None and not kwargs.get("ignore_sort", False):
            order = vstacked.argsort(self.__sort_key[0], reverse=self.__sort_key[1])
            vstacked = vstacked[order]
            vstacked = fold(
                HookPoint.PostSort, PostSortCtx(self, vstacked, np.argsort(order))
            ).data

        to_remove = self.__hidden.intersection(vstacked.colnames)
        vstacked.remove_columns(to_remove)
        if len(vstacked) == 1 and unpack:
            output_data = {
                key: value[0] if len(value) == 1 else value
                for key, value in vstacked.items()
            }
            return convert_data(output_data, format, wrap_single=wrap_single)

        if format != "astropy":
            return convert_data(dict(vstacked), format, wrap_single=wrap_single)
        elif len(vstacked.columns) == 1 and not wrap_single:
            return next(iter(dict(vstacked).values()))

        return vstacked

    @property
    @deprecated(
        msg="Accessing data through the .data attribute is deprecated and will be removed in a future version. Use get_data()",
    )
    def data(self):
        """
        Return the data in the dataset in astropy format. The value of this
        attribute is equivalent to the return value of
        :code:`Dataset.get_data("astropy")`.

        Returns
        -------
        data : astropy.table.Table or astropy.table.Column
            The data in the dataset.

        """
        return self.get_data("astropy")

    @classmethod
    def open(
        cls,
        targets: list[FileTarget],
        index_kind: str = "none",
        is_empty_ref: bool = False,
        **kwargs,
    ):
        datasets: dict[int, dict[str, Dataset]] = defaultdict(dict)
        dataset_targets = []
        for target in targets:
            dataset_targets.extend(target["dataset_targets"])
            for group in target["dataset_groups"].values():
                dataset_targets += group
        for i, ds_target in enumerate(dataset_targets):
            group_name = ds_target["dataset_group"].name.split("/")[-1]
            group_name = group_name.lstrip(f"{ds_target['header'].file.step}_")

            open_kwargs = dict(kwargs)
            ds = iopen.open_dataset(
                ds_target,
                index_spec_for(index_kind, is_empty_ref, is_source=True),
                open_kwargs=open_kwargs,
            )
            step = ds_target["header"].file.step
            if step is None:
                step = i
            datasets[step][group_name] = ds

        output: dict[int, Dataset | Lightcone] = {}
        for key, ds_group in datasets.items():
            if len(ds_group) == 1:
                output[key] = next(iter(ds_group.values()))
            else:
                output[key] = Lightcone(ds_group)

        if not all(type(ds) is oc.Dataset for ds in output.values()) and not all(
            type(ds) is Lightcone for ds in output.values()
        ):
            raise ValueError()

        result = cls(output)
        return fold(HookPoint.LightconeOpen, LightconeOpenCtx(result, kwargs)).lightcone

    @classmethod
    def from_datasets(
        cls,
        datasets: Mapping[int, oc.Dataset],
        z_range: Optional[tuple[float, float]] = None,
        scope: Optional[Any] = None,
        **open_kwargs,
    ):
        result = cls(datasets, z_range, scope=scope)
        return fold(
            HookPoint.LightconeOpen, LightconeOpenCtx(result, open_kwargs)
        ).lightcone

    def with_redshift_range(self, z_low: float, z_high: float):
        """
        Restrict this lightcone to a specific redshift range. Lightcone datasets will
        always contain a column titled "redshift." This function is always operates on
        this column.

        This function also updates the value in
        :py:meth:`Lightcone.z_range <opencosmo.collection.Lightcone.z_range>`,
        so you should always use it rather than filteringo n the column directly.
        """
        z_range = self.__header.lightcone["z_range"]
        if z_high < z_low:
            z_high, z_low = z_low, z_high

        if z_high < z_range[0] or z_low > z_range[1]:
            raise ValueError(
                f"This lightcone only ranges from z = {z_range[0]} to z = {z_range[1]}"
            )

        elif z_low == z_high:
            raise ValueError("Low and high values of the redshift range are the same!")
        new_datasets = {}
        for key, dataset in self.items():
            if not lcutils.is_in_range(dataset, z_low, z_high):
                continue
            new_dataset = dataset.filter(
                oc.col("redshift") > z_low, oc.col("redshift") < z_high
            )
            if len(new_dataset) > 0:
                new_datasets[key] = new_dataset
        return Lightcone(
            new_datasets, (z_low, z_high), self.__hidden, self.__sort_key, self.__scope
        )

    def __map(
        self,
        method,
        *args,
        hidden: Optional[set[str]] = None,
        mapped_arguments: dict[str, dict[str, Any]] = {},
        construct: bool = True,
        scope: Optional[Any] = None,
        **kwargs,
    ):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type, so it is always safe to map operations
        across all of them.
        """
        output = {}
        hidden = hidden if hidden is not None else self.__hidden
        scope = scope if scope is not None else self.__scope
        zero_length_output = {}

        for ds_name, dataset in self.items():
            dataset_mapped_arguments = {
                arg_name: args[ds_name] for arg_name, args in mapped_arguments.items()
            }
            new_ds = getattr(dataset, method)(
                *args, **kwargs, **dataset_mapped_arguments
            )
            if len(new_ds) == 0:
                zero_length_output[ds_name] = new_ds
                continue
            output[ds_name] = new_ds

        if not output:
            output = zero_length_output
        if construct:
            return Lightcone(output, self.z_range, hidden, self.__sort_key, scope)
        return output

    def __map_attribute(self, attribute):
        return {k: getattr(v, attribute) for k, v in self.items()}

    def make_schema(
        self, name: str = "", _min_size=100_000, no_stack: bool = False
    ) -> Schema:
        from opencosmo.mpi import get_all_keys

        datasets = lcio.order_by_redshift_range(self)
        for key in datasets:
            if isinstance(datasets[key], Lightcone):
                datasets[key] = dict(datasets[key])
        output_datasets = lcio.combine_adjacent_datasets(
            datasets, min_dataset_size=_min_size, no_stack=no_stack
        )
        children = {}

        # Use get_all_keys to iterate over the union of steps across all ranks
        # This ensures every rank calls stack_lightcone_datasets_in_schema the same
        # number of times in the same order, preventing MPI deadlock
        all_steps = get_all_keys(output_datasets, get_comm_world())

        for step in all_steps:
            datasets = output_datasets.get(step, {})
            if len(datasets) == 0:
                stack_lightcone_datasets_in_schema(datasets, None, None, no_stack)
                continue

            all_datasets = list(chain(*tuple(lst for lst in datasets.values())))
            header_zrange = lcutils.get_redshift_range(all_datasets)
            my_zrange = self.z_range
            zrange = (
                max(header_zrange[0], my_zrange[0]),
                min(header_zrange[1], my_zrange[1]),
            )

            child_schemas = stack_lightcone_datasets_in_schema(
                datasets, step, zrange, no_stack
            )
            child_schemas = {
                f"{step}_{name}": schema for name, schema in child_schemas.items()
            }
            children.update(child_schemas)

        return make_schema(name, FileEntry.LIGHTCONE, children=children)

    def bound(self, region: Region, select_by: Optional[str] = None):
        """
        Restrict the dataset to some subregion. The subregion will always be evaluated
        in the same units as the current dataset. For example, if the dataset is
        in the default "comoving" unit convention, positions are always in units of
        comoving Mpc. However Region objects themselves do not carry units.
        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query.

        Returns
        -------
        dataset: opencosmo.Dataset
            The portion of the dataset inside the selected region

        Raises
        ------
        ValueError
            If the query region does not overlap with the region this dataset resides
            in
        AttributeError:
            If the dataset does not contain a spatial index
        """
        return self.__map("bound", region, select_by)

    def cone_search(self, center: tuple | SkyCoord, radius: float | u.Quantity):
        """
        Perform a search for objects within some angular distance of some
        given point on the sky. This is a convinience function around
        :py:meth:`bound <opencosmo.Lightcone.bound>` and is exactly
        equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

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
        new_lightcone: opencosmo.Lightcone
            The rows in this lightcone that fall within the given region.

        """
        region = oc.make_cone(center, radius)
        return self.bound(region)

    def box_search(self, p1: tuple | SkyCoord, p2: tuple | SkyCoord):
        """
        Perform a box search in a given RA and Dec range. Of course this is not
        really a "box" on the surface of a sphere, but it's the closet thing we got.
        This metbhod is exactly equivalent to

        .. code-block:: python

            region = oc.make_cone(center, radius)
            ds = ds.bound(region)

        Parameters
        ----------

        p1: tuple | astropy.coordinates.SkyCoord
            A point defining one corner of the box. If a tuple with no units, values are assumed to
            be RA and Dec in degrees.
        p2: tuple | astropy.coordinates.SkyCoord
            A point defining the opposite corner of the box. If a tuple with no units, values are assumed to
            be RA and Dec in degrees.

        Returns
        -------

        new_lightcone: opencosmo.Lightcone
            The new dataset, only including data within the specified region.
        """
        region = oc.make_skybox(p1, p2)
        return self.bound(region)

    def pixel_search(self, pixels: npt.NDArray[np.int_], nside: int = 64):
        """
        Return the subset of this lightcone that falls within a set of HEALPix pixels.

        Pixels must be specified in nested ordering and must be valid indices at
        the given ``nside``. Duplicate pixel indices are ignored. Use
        :py:meth:`get_pixels <opencosmo.Lightcone.get_pixels>` to discover
        which pixels this lightcone covers.

        Parameters
        ----------
        pixels : array_like[int]
            HEALPix pixel indices to query, in nested ordering. Must be a 1-D
            array of non-negative integers. Values must be less than
            ``healpy.nside2npix(nside)``.
        nside : int, default = 64
            The HEALPix resolution parameter. Must be a positive power of two
            and must not exceed the resolution of the spatial index stored in
            the file.

        Returns
        -------
        lightcone : opencosmo.Lightcone
            A new lightcone containing only the objects that fall within the
            specified pixels.

        Raises
        ------
        ValueError
            If ``nside`` is not a positive power of two, or if ``pixels``
            contains values that are out of range for the given ``nside``.
        """
        level = np.log2(nside)
        if not level.is_integer() or level < 0:
            raise ValueError("nside must be a positive power of two!")
        level = int(level)
        pixels = np.atleast_1d(pixels)
        pixels = np.unique(pixels)
        if not np.isdtype(pixels.dtype, "integral") or len(pixels) == 0:
            raise ValueError("Pixels must be a 1d array of positive integers")
        if pixels[0] < 0 or pixels[-1] >= hp.nside2npix(nside):
            raise ValueError("Pixels must be a 1d array of positive integers")
        output = {}
        for name, ds in self.items():
            if isinstance(ds, Lightcone):
                output[name] = ds.pixel_search(pixels, nside)
                continue
            rows = ds.tree.project_on_index(level, ds.index, pixels)
            output[name] = ds.take_rows(rows)
        return Lightcone(
            output, self.z_range, self.__hidden, self.__sort_key, self.__scope
        )

    def evaluate(
        self,
        func: Callable,
        vectorize=False,
        insert=True,
        format: str = "astropy",
        batch_size: int = -1,
        allow_overwrite: bool = False,
        **evaluate_kwargs,
    ):
        """
        Iterate over the rows in this collection, apply `func` to each, and collect
        the result as new columns in the dataset. You may also choose to simply return thevalues
        instead of inserting them as a column

        This function is the equivalent of :py:meth:`with_new_columns <opencosmo.Lightcone.with_new_columns>`
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

        If a :code:`batch_size` is set, opencosmo will pass data to your function in batches of rows. In a lightcone,
        batches may be smaller than the given chunk size but will never be larger. Exact batch sizes
        will depend on the layout of the lightcone. Setting a batch size overrides the :code:`vectorize`
        flag.

        This function behaves (mostly) identically to :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`

        Parameters
        ----------
        func: Callable
            The function to evaluate on the rows in the dataset.

        format: str, default = "astropy"
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Lightcone.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.

        vectorize: bool, default = False
            Whether to provide the values as full columns (True) or one row at a time (False)

        insert: bool, default = True
            If true, the data will be inserted as a column in this dataset. Otherwise the data will be returned.

        Returns
        -------
        dataset : Lightcone
            The new lightcone dataset with the evaluated column(s)
        """

        mapped_kwargs = {}
        kwargs_names = list(evaluate_kwargs.keys())
        for name in kwargs_names:
            if isinstance(evaluate_kwargs[name], dict) and set(
                evaluate_kwargs[name].keys()
            ) == set(self.keys()):
                mapped_kwargs[name] = evaluate_kwargs.pop(name)

        if insert:
            name, dataset_to_verify = next(iter(self.items()))
            ds_kwargs = evaluate_kwargs | {
                argname: vals[name] for argname, vals in mapped_kwargs.items()
            }

            # Workaround until Lightcone holds Dataset states. Future PR.
            if isinstance(dataset_to_verify, Dataset):
                dataset_state_to_verify = dataset_to_verify._state
            elif isinstance(dataset_to_verify, Lightcone):
                dataset_to_verify = next(iter(dataset_to_verify.values()))
                dataset_state_to_verify = dataset_to_verify._state

            evaluated_column = build_evaluated_column(
                dataset_state_to_verify,
                func,
                vectorize,
                insert,
                format,
                batch_size,
                ds_kwargs,
            )
            mapped_evaluated_columns = {
                func.__name__: {
                    name: evaluated_column.with_kwargs(
                        **{
                            argname: vals[name]
                            for argname, vals in mapped_kwargs.items()
                        }
                    )
                    for name in self.keys()
                }
            }
            return self.__map(
                "with_new_columns",
                mapped_arguments=mapped_evaluated_columns,
                construct=True,
                allow_overwrite=allow_overwrite,
            )

        result = self.__map(
            "evaluate",
            func=func,
            format=format,
            vectorize=vectorize,
            insert=insert,
            mapped_arguments=mapped_kwargs,
            batch_size=batch_size,
            allow_overwrite=allow_overwrite,
            construct=insert,
            **evaluate_kwargs,
        )
        if next(iter(result.values())) is None:
            return

        assert isinstance(result, dict)
        keys = next(iter(result.values())).keys()
        output = {}
        for key in keys:
            output[key] = concat_chunks([r[key] for r in result.values()], format)
        return output

    def filter(self, *masks: ColumnMask, mode: str = "global", **kwargs) -> Self:
        """
        Filter the dataset based on some criteria. See :ref:`Querying Based on Column
        Values` for more information.

        Parameters
        ----------
        *masks : Mask
            The masks to apply to dataset, constructed with :func:`opencosmo.col`

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions (e.g. ``oc.col("x").mean()``) used
            inside the masks are computed when running under MPI. Defaults to
            ``"global"`` so every rank filters against the same cross-rank
            scalar; pass ``"local"`` to use each rank's per-rank scalar.

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
        new_masks = [
            resolve_mask_scalars(mask, self.__scalar_evaluator(mode))
            if mask_contains_scalar(mask)
            else mask
            for mask in masks
        ]
        return self.__map("filter", *new_masks, mode=mode, **kwargs)

    def __scalar_evaluator(self, mode: str) -> Callable[[DerivedScalarValue], Any]:
        """
        Build a function that evaluates a single DerivedScalarValue against the
        current lightcone state. Used for eager resolution in filter masks.
        """
        reducer = default_reducer(mode)

        def evaluate(scalar: DerivedScalarValue) -> Any:
            scalar = scalar.with_reducer(reducer) if scalar.reducer is None else scalar
            required = scalar._traverse_names()
            if not required:
                return scalar.evaluate({})
            # `required` is the set of raw child names this scalar reads, so
            # self.select(*required) routes through the positional (child)
            # path — partition_columns treats no derived_columns as
            # all-child-scoped — and never re-enters the scope evaluator.
            # If a future caller passes a scope-name in `required`, that
            # would recurse through evaluate_scope; bypass via take_rows or
            # a private _get_raw if that becomes possible.
            table = self.select(*required).get_data(
                "astropy", unpack=False, wrap_single=True
            )
            data = {name: table[name] for name in required}
            return scalar.evaluate(data)

        return evaluate

    def rows(self) -> Generator[dict[str, float | u.Quantity], None, None]:
        """
        Iterate over the rows in the dataset. Rows are returned as a dictionary
        For performance, it is recommended to first select the columns you need to
        work with.

        Yields
        -------
        row : dict
            A dictionary of values for each row in the dataset with units.
        """
        yield from chain.from_iterable(v.rows() for v in self.values())

    def select(
        self,
        *columns: str | Iterable[str],
        mode: str = "global",
        **derived_columns: ConstructedColumn,
    ) -> Self:
        """

        Create a new lightcone dataset from a subset of columns in this lightcone dataset.
        This function accepts wildcards. For exampe, "lsst*" will select all columns
        that start with "lsst", while "*host*" will select all columns that have
        "host" somewhere in the middle.

        You can also create new columns as part of this call, as long as they are
        derived from other columns in the dataset. For example:

        .. code-block:: python

            import opencosmo as oc
            from opencosmo.column import add_mag_cols

            dataset = oc.open("galaxy_catalog.hdf5")
            total_mag = add_mag_cols("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
            # Note, you can also use oc.col to do this manually


            dataset = dataset.select("ra", "dec", "*host*", "lsst*", lsst_total = total_mag)

        This new dataset will contain the :code:`ra` and :code:`dec` columns, all the columns
        with :code:`host` somewhere in the name, all the columns that start with :code:`lsst`
        and a newly-constructed :code:`lsst_total` column.



        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to select.

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions inside derived column expressions
            are computed when running under MPI. Defaults to ``"global"``
            (cross-rank); pass ``"local"`` for per-rank scalars.

        **derived_columns : Column
            Additional columns to create as part of the selection.

        Returns
        -------
        dataset : Lightcone
            The new lightcone with only the selected columns.

        Raises
        ------
        ValueError
            If any of the required columns are not in the dataset.
        """
        from opencosmo.column.column import Column

        all_columns: set[str] = set()
        for col_group in columns:
            if isinstance(col_group, str):
                col_group = {col_group}
            all_columns.update(col_group)

        scalars = {
            name: col
            for name, col in derived_columns.items()
            if isinstance(col, DerivedScalarValue)
        }
        non_scalars = {
            name: col
            for name, col in derived_columns.items()
            if not isinstance(col, DerivedScalarValue)
        }
        if scalars and (all_columns or non_scalars):
            raise ValueError(
                "Scalar selections cannot be mixed with column selections. "
                "Call select() with only scalar kwargs, or only column selections."
            )

        reducer = default_reducer(mode)
        derived_columns = {
            k: v.with_reducer(reducer)
            if isinstance(v, (Column, DerivedScalarValue))
            else v
            for k, v in derived_columns.items()
        }

        raw_child_columns = set(next(iter(self.values())).columns)
        plan = self.__scope.plan_select(all_columns, derived_columns, raw_child_columns)

        hidden = set(self.__hidden) | plan.hidden_additions

        # Some columns must be retained even when the user does not ask for them,
        # otherwise the lightcone cannot be written. They are kept but hidden.
        underlying_columns = set(next(iter(self.values())).columns)
        required = lcutils.get_required_columns(underlying_columns, self.dtype)
        if self.__sort_key is not None:
            required.add(self.__sort_key[0])

        required_extras = required.difference(all_columns)
        hidden = self.__hidden.union(required_extras)
        additional_columns = set(plan.child_additional).union(required_extras)

        return self.__map(
            "select",
            plan.child_positional,
            additional_columns,
            mapped_arguments={},
            hidden=hidden,
            construct=True,
            scope=plan.new_scope,
            mode=mode,
            **plan.child_scoped,
        )

    def drop(self, *columns: str | Iterable[str]) -> Self:
        """
        Produce a new dataset by dropping columns from this dataset.

        Parameters
        ----------
        *columns : str or list[str]
            The column or columns to drop.

        Returns
        -------
        dataset : Lightcone
            The new lightcone without the dropped columns

        Raises
        ------
        ValueError
            If any of the given columns are not in the dataset.
        """
        dropped_columns: set[str] = set()
        for col_group in columns:
            if isinstance(col_group, str):
                col_group = {col_group}
            dropped_columns.update(col_group)

        current_columns = set(self.columns)
        if missing := dropped_columns.difference(current_columns):
            raise ValueError(
                f"Tried to drop columns that are not in this dataset: {missing}"
            )
        kept_columns = current_columns - dropped_columns
        return self.select(kept_columns)

    def take(
        self, n: int, at: str = "random", mode: Literal["local", "global"] = "local"
    ) -> "Lightcone":
        """
        Create a new dataset from some number of rows from this dataset.

        Can take the first n rows, the last n rows, or n random rows
        depending on the value of 'at'.

        Parameters
        ----------
        n : int
            The number of rows to take.
        at : str
            Where to take the rows from. One of "start", "end", or "random".
            The default is "random".
        mode : str, "local" or "global", default = "local"
            Controls how ``n`` is interpreted when running under MPI. Has no
            effect if you are not using MPI.

            * ``"local"`` (default): ``n`` rows are taken independently on
              each rank.
            * ``"global"``: ``n`` is the total number of rows to select across
              all ranks combined. Each rank receives the portion of those rows
              that it owns. If the dataset is sorted, ranks will coordinate
              to take from the globally-sorted dataset.

        Returns
        -------
        lightcone : Lightcone
            The new lightcone with only the selected rows.

        Raises
        ------
        ValueError
            If n is negative or greater than the number of rows in the dataset,
            or if 'at' is invalid.

        """
        if at == "random":
            index = get_random_take_index(n, len(self), mode)
        elif at == "start":
            index = get_range_take_index(self, self.__sort_key, 0, n, mode)
            if self.__sort_key is not None:
                sort_index = self.__make_sort_index()
                index = np.sort(sort_index[into_array(index)])
        elif at == "end":
            index = get_end_take_index(n, self, self.__sort_key, mode)
            if self.__sort_key is not None:
                sort_index = self.__make_sort_index()
                index = np.sort(sort_index[into_array(index)])
        else:
            raise ValueError(
                f'"at" should be one of ("start", "end", "random", got {at}'
            )
        return self.__take_rows(index)

    def take_range(
        self, start: int, end: int, mode: Literal["local", "global"] = "local"
    ):
        """
        Create a new lightcone from a row range in this lightcone. We use standard
        indexing conventions, so the rows included will be start -> end - 1. Because
        lightcones are stacked by redshift, this operation effectively takes a
        redshift range. If you know the exact redshift range you want, use
        :py:meth:`with_redshift_range <opencosmo.Lightcone.with_redshift_range>`.

        Parameters
        ----------
        start : int
            The beginning of the range.
        end : int
            The end of the range (exclusive).
        mode : str, "local" or "global", default = "local"
            Controls how ``start`` and ``end`` are interpreted when running
            under MPI. Has no effect if you are not using MPI.

            * ``"local"`` (default): the range is applied independently on
              each rank.
            * ``"global"``: ``start`` and ``end`` index into the global row
              space across all ranks combined. Each rank receives the portion
              of that range it owns. If the lightcone is sorted, ranks will
              coordinate to take from the globally-sorted lightcone.

        Returns
        -------
        lightcone : opencosmo.Lightcone
            The lightcone with only the specified range of rows.

        Raises
        ------
        ValueError
            If start or end are negative or greater than the length of the dataset
            or if end is greater than start.

        """
        if start < 0:
            raise ValueError("Tried to take negative rows!")

        index = get_range_take_index(self, self.__sort_key, start, end - start, mode)
        if self.__sort_key is not None:
            sort_index = self.__make_sort_index()
            index = np.sort(sort_index[into_array(index)])
        return self.__take_rows(index)

    def take_rows(self, rows: DataIndex):
        """
        Take the rows of a lightcone specified by the :code:`rows` argument.
        :code:`rows` should be an array of integers.

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
            lightcone.

        """
        index_range = get_range(rows)

        if index_range[0] < 0 or index_range[1] > len(self):
            raise ValueError(
                "Rows must be between 0 and the length of this dataset - 1"
            )
        rows = get_rows_take_index(self, rows, self.__sort_key)
        return self.__take_rows(rows)

    def __make_sort_index(self):
        if self.__sort_key is None:
            return None
        data = np.concatenate(
            [ds.select(self.__sort_key[0]).get_data("numpy") for ds in self.values()]
        )
        if self.__sort_key[1]:
            data = -data
        return np.argsort(data)

    def __take_rows(self, rows: DataIndex):
        """
        Takes rows from this lightcone while ignoring sort. "rows" is assumed to be sorted.
        For internal use only.
        """
        sizes = np.fromiter((len(ds) for ds in self.values()), dtype=np.int64)
        starts = np.zeros_like(sizes)
        starts[1:] = np.cumsum(sizes)[:-1]
        projected = rebuild_by_ranges(rows, (starts, sizes))
        output = {}
        for (name, ds), index in zip(self.items(), projected):
            output[name] = ds.take_rows(index)
        if all(len(ds) == 0 for ds in output.values()):
            key = next(iter(output.keys()))
            output = {key: output[key]}
        else:
            output = {key: ds for key, ds in output.items() if len(ds) > 0}

        return Lightcone(
            output, self.z_range, self.__hidden, self.__sort_key, self.__scope
        )

    def with_new_columns(
        self,
        descriptions: str | dict[str, str] = {},
        allow_overwrite: bool = False,
        mode: str = "global",
        **columns: ConstructedColumn | np.ndarray | u.Quantity,
    ):
        """
        Create a new dataset with additional columns. These new columns can be derived
        from columns already in the dataset, a numpy array, or an Astropy quantity
        array. When a column is derived from other columns, it will behave
        appropriately under unit transformations. See :ref:`Adding Custom Columns`
        and :py:meth:`Dataset.with_new_columns <opencosmo.Dataset.with_new_columns>`
        for examples.

        Parameters
        ----------
        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`Lightcone.descriptions <opencosmo.Lighcone.descriptions>`. If a dictionary,
            should have keys matching the column names.

        mode : str, "local" or "global", default = "global"
            Controls how scalar reductions nested inside derived column
            expressions are computed when running under MPI. Defaults to
            ``"global"`` (cross-rank); pass ``"local"`` for per-rank scalars.

        ** columns : opencosmo.Column | np.ndarray | u.quantity
            The new columns

        Returns
        -------
        dataset : opencosmo.Dataset
            This dataset with the columns added

        """
        if any(isinstance(col, DerivedScalarValue) for col in columns.values()):
            raise ValueError(
                "Scalar values cannot be added to an existing dataset, but can be retrieved with Dataset.select()"
            )
        reducer = default_reducer(mode)
        columns = {
            k: v.with_reducer(reducer) if isinstance(v, Column) else v
            for k, v in columns.items()
        }
        derived = {}
        raw = {}
        for name, column in columns.items():
            if isinstance(column, (Column, EvaluatedColumn)):
                derived[name] = column

            elif not isinstance(column, np.ndarray):
                raise ValueError(f"Invalid column type: {type(columns)}")
            elif len(column) != len(self):
                raise ValueError(
                    f"New column {name} has length {len(column)} but this dataset "
                    f"has length {len(self)}"
                )
            else:
                raw[name] = column

        if self.__sort_key is not None:
            sort_index = self.__make_sort_index()
            sort_index = np.argsort(sort_index)
            raw = {name: raw_data[sort_index] for name, raw_data in raw.items()}

        from opencosmo.collection.lightcone.scope import partition_columns

        raw_child_columns = set(next(iter(self.values())).columns) | set(raw.keys())
        scoped, child_scoped = partition_columns(
            derived,  # type: ignore
            raw_child_columns,
            self.__scope,
        )
        new_scope = self.__scope.add(derived, raw_child_columns)  # type: ignore

        split_points = np.cumsum([len(ds) for ds in self.values()])
        split_points = np.insert(0, 0, split_points)[:-1]
        raw_split = {name: np.split(arr, split_points) for name, arr in raw.items()}
        new_datasets = {}
        for i, (ds_name, ds) in enumerate(self.items()):
            raw_columns = {name: arrs[i] for name, arrs in raw_split.items()}
            columns_input = raw_columns | child_scoped
            new_dataset = ds.with_new_columns(
                descriptions,
                allow_overwrite=allow_overwrite,
                mode=mode,
                **columns_input,
            )
            new_datasets[ds_name] = new_dataset
        return Lightcone(
            new_datasets, self.z_range, self.__hidden, self.__sort_key, new_scope
        )

    def sort_by(self, column: Optional[str], invert: bool = False):
        """
        Sort this dataset by the values in a given column. By default sorting is in
        ascending order (least to greatest). Pass invert = True to sort in descending
        order (greatest to least).

        This can be used to, for example, select largest halos in a given
        dataset:

        .. code-block:: python

            dataset = oc.open("haloproperties.hdf5")
            dataset = dataset
                        .sort_by("fof_halo_mass")
                        .take(100, at="start")

        Parameters
        ----------
        column : Optional[str]
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by. Pass None to remove sorting.

        invert : bool, default = False
            If False (the default), ordering will be from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : Dataset
            A new Dataset ordered by the given column.


        """

        if column is None:
            sort_key = None
        elif column not in self.columns:
            raise ValueError(f"Column {column} does not exist in this dataset!")
        else:
            sort_key = (column, invert)

        return Lightcone(
            dict(self), self.z_range, self.__hidden, sort_key, self.__scope
        )

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> Self:
        r"""
        Create a new lightcone from this one with a different unit convention or
        with certain columns converted to a different compatible unit.

        Unit conversions are always performed after a change of convention, and
        changing conventions clears any existing unit conversions.

        For more, see :doc:`units`.

        .. code-block:: python

            import astropy.units as u

            # this works
            lc = lc.with_units(fof_halo_mass=u.kg)

            # this clears the previous conversion
            lc = lc.with_units("scalefree")

            # This now fails, because the units of masses
            # are Msun / h, which cannot be converted to kg
            lc = lc.with_units(fof_halo_mass=u.kg)

            # this will now work, wince the units of halo mass in the "physical"
            # convention are Msun (no h).
            lc = lc.with_units("physical", fof_halo_mass=u.kg, fof_halo_center_x=u.lyr)

            # Suppose you want your distances in lightyears, but the x coordinate of your
            # halo center in kilometers, for some reason ¯\_(ツ)_/¯
            blanket_conversions = {u.Mpc: u.lyr}
            lc = lc.with_units(conversions = blanket_conversions, fof_halo_center_x = u.km)

        Parameters
        ----------
        convention : str, optional
            The unit convention to use. One of "physical", "comoving",
            "scalefree", or "unitless".

        conversions: dict[astropy.units.Unit, astropy.units.Unit]
            Conversions that apply to all columns in the lightcone with the
            unit given by the key.

        **column_conversions: astropy.units.Unit
            Custom unit conversions for specific columns
            in this dataset.

        Returns
        -------
        lightcone : Lightcone
            The new lightcone with the requested unit convention and/or conversions.
        """
        return self.__map(
            "with_units",
            convention=convention,
            conversions=conversions,
            **columns,
        )
