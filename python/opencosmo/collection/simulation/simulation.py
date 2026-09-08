from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Mapping, Optional, cast

import numpy as np

from opencosmo.collection import structure as sc
from opencosmo.collection.simulation.io import resort_simulation_collection
from opencosmo.column.select import do_multi_dataset_drops, do_multi_dataset_selections
from opencosmo.dataset import Dataset
from opencosmo.dataset import operations as dsops
from opencosmo.dataset import state as st
from opencosmo.dataset.state import DatasetState
from opencosmo.index import into_array
from opencosmo.index.mpi import is_in_global_index
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mapping.mapping import get_mapping
from opencosmo.mpi import get_comm_world, has_mpi
from opencosmo.utils import normalize_kwarg_name

if TYPE_CHECKING:
    import astropy.units as u
    import h5py

    from opencosmo.collection.protocols import Collection
    from opencosmo.column.column import ColumnMask, ConstructedColumn
    from opencosmo.dataset.dataset import OpenCosmoData
    from opencosmo.header import OpenCosmoHeader
    from opencosmo.io.iopen import FileTarget
    from opencosmo.io.schema import Schema
    from opencosmo.mapping.mapping import DatasetMatchSet
    from opencosmo.spatial.protocols import Region


def verify_datasets_exist(file: h5py.File, datasets: Iterable[str]):
    """
    Verify a set of datasets exist in a given file.
    """
    if not set(datasets).issubset(set(file.keys())):
        raise ValueError(f"Some of {', '.join(datasets)} not found in file.")


def prepare_matched_datasets(
    match_set: DatasetMatchSet,
    datasets: Mapping[str, DatasetState],
    source: str,
):
    # Guard: active sort keys would cause dataset/state.py to re-sort the
    # carefully ordered rows produced below, silently misaligning everything.
    for name, ds in datasets.items():
        if ds.sort_key is not None and name != source:
            raise ValueError(
                f"Dataset '{name}' has an active sort key. Call match() before sort_by()."
            )

    reference_dataset = datasets[source]
    rows_to_keep = np.ones(len(reference_dataset), dtype=bool)
    index = reference_dataset.raw_index

    mappings = {}
    for name, dataset in datasets.items():
        if name == source:
            continue
        mapping = get_mapping(match_set, source, name, index)
        if mapping is None:
            raise ValueError(f"Unable to find mapping for dataset {dataset}")
        if not isinstance(mapping, np.ndarray):
            raise ValueError("Dataset matching does not support chunked mappings")
        rows_to_keep = rows_to_keep & (mapping >= 0)
        mappings[name] = mapping

    if has_mpi():
        return prepare_matched_datasets_mpi(datasets, mappings, rows_to_keep, source)

    # The np.isin pass below is a required precondition for the unchecked
    # searchsorted in the final loop: it guarantees every value in `wanted`
    # is present in `target_index`, making the lookup safe without bounds checks.

    for name, mapping in mappings.items():
        mappings_to_keep = mapping[rows_to_keep]
        mappings_in_index = np.isin(
            mappings_to_keep, into_array(datasets[name].raw_index)
        )
        rows_to_keep[rows_to_keep] &= mappings_in_index

    # Must be taken only after the refinement above, so the source and every
    # target end up with the same number of rows.
    new_datasets = {
        source: dsops.take_rows(datasets[source], np.where(rows_to_keep)[0])
    }
    for name, mapping in mappings.items():
        target_index = into_array(datasets[name].raw_index)
        wanted = mapping[rows_to_keep]
        order = np.argsort(target_index, kind="stable")
        rows_to_take = order[np.searchsorted(target_index, wanted, sorter=order)]
        new_datasets[name] = dsops.take_rows(datasets[name], rows_to_take)

    return new_datasets


def prepare_matched_datasets_mpi(
    datasets: Mapping[str, DatasetState],
    mappings: dict[str, np.ndarray],
    rows_to_keep: np.ndarray,
    source: str,
):
    """
    Match under MPI, where a rank pulls the GLOBAL target rows that its local
    source rows map to. Target rows therefore need not live on this rank, so the
    serial rank-local membership test does not apply.

    A row is still only matchable if it is present somewhere in the current
    global state. Rows dropped by an earlier match are gone from every rank and
    must not reappear, so membership is tested against the union of every rank's
    index rather than the local index alone.
    """

    comm = get_comm_world()
    assert comm is not None

    for name, mapping in mappings.items():
        rows_to_keep[rows_to_keep] &= is_in_global_index(
            mapping[rows_to_keep],
            into_array(datasets[name].raw_index),
            comm,
        )

    new_datasets = {
        source: dsops.take_rows(datasets[source], np.where(rows_to_keep)[0])
    }
    for name, mapping in mappings.items():
        dataset = datasets[name]
        new_datasets[name] = st.redistribute(dataset, mapping[rows_to_keep])
    return new_datasets


class SimulationCollection:
    """
    A collection of datasets of the same type from different
    simulations. In general this exposes the exact same API
    as the individual datasets, but maps the results across
    all of them.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset | DatasetState | Collection],
        match_set: DatasetMatchSet | None = None,
        match_source: str | None = None,
        rebuilt: dict[str, bool] | None = None,
    ):
        def normalize(
            value: Dataset | DatasetState | Collection,
        ) -> DatasetState | sc.StructureCollection:
            if isinstance(value, Dataset):
                return value._state
            elif isinstance(value, (DatasetState, sc.StructureCollection)):
                return value
            raise ValueError(
                "Simulation collection only accepts datasets and structure collections"
            )

        if match_set is not None and not all(
            isinstance(v, (Dataset, DatasetState)) for v in datasets.values()
        ):
            raise ValueError(
                "Dataset matching is only supported for simple datasets (no collections)"
            )

        self.__datasets = {
            normalize_kwarg_name(k): normalize(v) for k, v in dict(datasets).items()
        }
        self.__match_set = match_set
        self.__match_source = match_source
        self.__rebuilt = rebuilt

    def __getattr__(self, key: str):
        output = {}
        try:
            for name, ds in self.__datasets.items():
                output[name] = getattr(ds.header, key)
        except AttributeError:
            return object.__getattribute__(self, key)
        return output

    def __dir__(self):
        keys: set[str] = set()
        for ds in self.__datasets.values():
            keys.update(ds.header.parameters.keys())
        return list(keys) + list(super().__dir__())

    def keys(self):
        return self.__datasets.keys()

    def values(self):
        self.__rebuild_all()

        values = []
        for v in self.__datasets.values():
            if isinstance(v, DatasetState):
                values.append(Dataset(v))
            else:
                values.append(v)
        return values

    def items(self):
        self.__rebuild_all()
        out = {}
        for k, v in self.__datasets.items():
            if isinstance(v, DatasetState):
                out[k] = Dataset(v)
            else:
                out[k] = v
        return out.items()

    def __len__(self):
        return len(self.__datasets)

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key):
        if (
            self.__match_source is not None
            and key != self.__match_source
            and not self.__rebuilt[key]
        ):
            datasets = {
                self.__match_source: self.__datasets[self.__match_source],
                key: self.__datasets[key],
            }

            new_datasets = prepare_matched_datasets(
                self.__match_set, datasets, self.__match_source
            )
            self.__datasets |= new_datasets
            self.__rebuilt[key] = True
        value = self.__datasets[key]
        if isinstance(value, DatasetState):
            return Dataset(value)
        return value

    def __rebuild_all(self):
        if self.__match_source is None:
            return

        datasets = {
            key: ds
            for key, ds in self.__datasets.items()
            if key != self.__match_source and not self.__rebuilt[key]
        }
        if not datasets:
            return
        self.__datasets |= prepare_matched_datasets(
            self.__match_set,
            datasets | {self.__match_source: self.__datasets[self.__match_source]},
            self.__match_source,
        )
        self.__rebuilt = {key: True for key in self.__datasets.keys()}

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        for dataset in self.values():
            try:
                dataset.close()
            except ValueError:
                continue

    def __repr__(self):
        n_collections = sum(
            1
            for v in self.values()
            if isinstance(v, (SimulationCollection, sc.StructureCollection))
        )
        n_datasets = sum(1 for v in self.values() if isinstance(v, Dataset))
        return (
            f"SimulationCollection({n_collections} collections, {n_datasets} datasets)"
        )

    @classmethod
    def open(cls, targets: list[FileTarget], **kwargs) -> Collection | Dataset:
        raise NotImplementedError()

    def make_schema(self) -> Schema:
        children = {}

        new_uuids = {}
        indices = {}
        self.__rebuild_all()

        for name, dataset in self.__datasets.items():
            if isinstance(dataset, DatasetState):
                children[name] = st.make_schema(dataset)
                new_uuids[name] = (
                    children[name].children["data"].attributes[""]["main_uuid"]
                )
                indices[name] = dataset.raw_index
                continue

            children[name] = dataset.make_schema()
            if isinstance(dataset, sc.StructureCollection):
                continue
            new_uuids[name] = (
                children[name].children["data"].attributes[""]["main_uuid"]
            )
            indices[name] = dataset.index

        if self.__match_set is not None and len(self) > 1:
            match_set_schema = self.__match_set.make_schema(
                new_uuids, indices, self.__match_source
            )
            children["map"] = match_set_schema

        schema = make_schema("/", FileEntry.SIMULATION_COLLECTION, children=children)
        return resort_simulation_collection(schema)

    def __map(
        self,
        method,
        *args,
        construct=True,
        datasets: Optional[str | Iterable[str]] = None,
        **kwargs,
    ):
        """
        This type of collection will only ever be constructed if all the underlying
        datasets have the same data type.
        """
        if isinstance(datasets, str):
            datasets = [datasets]
        if self.__match_source is not None and method in (
            "take",
            "take_range",
            "take_rows",
            "filter",
            "bound",
            "sort_by",
        ):
            # datasets=None means "all datasets", which for a matched collection
            # resolves to the active source. Any explicit request must name that
            # source and nothing else.
            if datasets is not None and (
                not isinstance(datasets, Iterable)
                or tuple(datasets) != (self.__match_source,)
            ):
                raise ValueError(
                    f"When working with a matched collection, {method} can only be called on the active source. Got datasets = {datasets}"
                )

            method_impl = getattr(dsops, method)
            new_source = method_impl(
                self.__datasets[self.__match_source], *args, **kwargs
            )
            new_datasets = self.__datasets | {self.__match_source: new_source}
            return SimulationCollection(
                new_datasets,
                self.__match_set,
                self.__match_source,
                {
                    ds_name: False
                    for ds_name in self.__datasets.keys()
                    if ds_name != self.__match_source
                },
            )

        regular_kwargs = {}
        mapped_kwargs = {}
        if isinstance(datasets, str):
            datasets = [datasets]
        elif datasets is None:
            datasets = self.keys()
        requested_datasets = set(datasets)
        if not requested_datasets.issubset(self.keys()):
            raise ValueError(
                f"Unknown datasets {requested_datasets.difference(self.keys())}"
            )

        for name, value in kwargs.items():
            if isinstance(value, dict) and set(value.keys()) == requested_datasets:
                mapped_kwargs[name] = value
            else:
                regular_kwargs[name] = value

        output = dict(self.__datasets) if construct else {}
        for name in requested_datasets:
            output[name] = self.__dispatch_dataset_operation(
                name,
                method,
                *args,
                dataset_mapped_kwargs={
                    key: kw[name] for key, kw in mapped_kwargs.items()
                },
                regular_kwargs=regular_kwargs,
            )
        if construct:
            return SimulationCollection(
                output,
                self.__match_set,
                self.__match_source,
                copy(self.__rebuilt) if self.__rebuilt is not None else None,
            )
        return output

    def __dispatch_dataset_operation(
        self,
        dataset_name: str,
        method: str,
        *args,
        dataset_mapped_kwargs: dict[str, object],
        regular_kwargs: dict[str, object],
    ) -> DatasetState | sc.StructureCollection:
        """Private dispatcher for operations on DatasetState vs nested Collections."""
        target = self.__datasets[dataset_name]
        if isinstance(target, DatasetState):
            # Route DatasetState operations through the shared state operation layer.
            fn = getattr(dsops, method)
            return fn(target, *args, **regular_kwargs, **dataset_mapped_kwargs)
        # Higher-level collections already implement the operation.
        assert isinstance(target, sc.StructureCollection)
        return getattr(target, method)(*args, **regular_kwargs, **dataset_mapped_kwargs)

    @property
    def dtype(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, v in self.__datasets.items():
            if isinstance(v, DatasetState):
                out[key] = str(v.header.file.data_type)
            else:
                out[key] = v.dtype
        return out

    @property
    def header(self) -> dict[str, OpenCosmoHeader]:
        return {k: v.header for k, v in self.__datasets.items()}

    def get_data(
        self, format: str = "astropy", wrap_single: bool = False, unpack: bool = True
    ) -> dict[str, OpenCosmoData]:
        """
        Retrieve the data from all datasets in this collection, and return it as a dictionary of
        type (dataset_name, data). This method requires that the underlying entries into this
        SimulationCollection are :py:class:`datasets <opencosmo.Dataset>` or :py:class:`lightcones <opencosmo.Lightcone>`,
        other entries will raise an error.

        Parameters
        ----------
        output: str, default="astropy"
            The format to output the data in.
            Currently supported are "astropy", "numpy", "pandas", "polars", "arrow", "jax"

        wrap_single: bool, default=False
            If True, always return the format's natural multi-column container even
            when only one column is present.

        Returns
        -------
        data: dict
            The data the datasets. Keys will be the same as the keys of this
            collection.

        """
        if not all(isinstance(ds, DatasetState) for ds in self.__datasets.values()):
            raise TypeError(
                "get_data is only supported when this collection contains datasets!"
            )
        return {
            name: ds.get_data(format=format, wrap_single=wrap_single, unpack=unpack)
            for name, ds in self.items()
        }

    def match(self, dataset: str) -> SimulationCollection:
        """
        Create a new simulation collection where the datasets are ordered so that matched
        objects appear in the same row across every dataset. All datasets are matched
        to `source`, and only rows that are available in every simulation are included.
        This feature requires that the underlying data has a pre-built matching index,
        and will raise an error if the index is not present.

        For example, suppose you have one gravity-only sim and one hydro sim.

        .. code-block:: python

            collection = collection.match("gravity_only")

        Further operations such as :py:meth:`take <opencosmo.SimulationCollection.take>` and
        :py:meth:`filter <opencosmo.SimulationCollection.filter>` will now operate exclusively
        on the gravity_only sim, and retain matching objects in the other sim. For a matched
        collection, calling :py:meth:`get_data <opencosmo.SimulationCollection.get_data>` will
        always return datasets with equal length

        To clear a match, use :py:meth:`clear_match <opencosmo.SimulationCollection.clear_match>`.
        Clearning a match does not undo any previous operations, it simple causes later
        operations to be applied to all datasets regardless of matching between rows.

        Parameters
        ----------
        dataset: str
            The dataset to match the collection against. You can get the available
            names with SimulationCollection.keys()

        Returns
        -------
        dataset: opencosmo.SimulationCollection
            The collection with the matching performed.

        Raises
        -------
        ValueError: If the data does not contain a matching index
        """

        if self.__match_set is None:
            raise ValueError(
                "This SimulationCollection does not contain matching information!"
            )
        elif dataset not in self.keys():
            raise ValueError(
                f"This SimulationCollection does not have a simulation named {dataset}"
            )
        assert all(isinstance(ds, DatasetState) for ds in self.__datasets.values())

        new_datasets = prepare_matched_datasets(
            self.__match_set,
            cast("dict[str, DatasetState]", self.__datasets),
            dataset,
        )

        return SimulationCollection(
            new_datasets,
            self.__match_set,
            dataset,
            {name: True for name in self.__datasets.keys()},
        )

    def clear_match(self):
        """
        Clear a previously set :py:meth`match <opencosmo.SimulationCollection.match>`. This does not
        alter any datasets. Instead, it ensures that any future operations will be applied uniformly
        across all datasets, rather than only to the match source.

        Take the following example:

        .. code-block:: python

            collection = collection.match("gravity_only")
            collection = collection.filter(oc.col("fof_halo_mass") > 1e14)
            collection = collection.clear_match()

        This collection will contain all objects in the gravity-only simulations with
        a mass above 10**14 and their matched objects in the other simulations. The
        clear_match call has no effect until a further operation is performed. If you
        retrieve data, the rows in each dataset will be matched up.

        Another example:

        .. code-block:: python

            collection = collection.match("gravity_only")
            collection = collection.clear_match()
            collection = collection.filter(oc.col("fof_halo_mass") > 1e14)

        This collection will contain all objects which have any a match to an object in the
        gravity-only simulation, and is above a mass 10**14. However since the filter
        was performed after clear_match, the order of the datasets are no longer
        guaranteed and the datasets may be of different lengths.

        If the collection does not contain a matching index, or no matching has been performed,
        this operations has no effect.

        Returns
        -------
        dataset: opencosmo.SimulationCollection
            The collection with the matching cleared.
        """

        self.__rebuild_all()
        return SimulationCollection(self.__datasets, self.__match_set)

    def bound(
        self, region: Region, select_by: Optional[str] = None
    ) -> SimulationCollection:
        """
        Restrict the datasets to some region. Note that the SimulationCollection does
        not do any checking to ensure its members have identical boxes. As a result
        this method can in principle fail for some of the simulations in the
        collection and not others. This should never happen when working with official
        OpenCosmo data products.

        See :doc:`spatial_ref` for details of how to construct regions.

        Parameters
        ----------
        region: opencosmo.spatial.Region
            The region to query

        Returns
        -------
        dataset: opencosmo.SimulationCollection
            The portion of each dataset inside the selected region

        """
        return self.__map("bound", region, select_by)

    def filter(self, *masks: ColumnMask, **kwargs) -> SimulationCollection:
        """
        Filter the datasets in the collection. This method behaves
        exactly like :meth:`opencosmo.Dataset.filter` or
        :meth:`opencosmo.StructureCollection.filter`, but
        it applies the filter to all the datasets or collections
        within this collection. The result is a new collection.

        If this dataset has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulatinCollection.match>`,
        the filter will only run on the dataset matched against, with all matching rows retained
        in the remaining datasets.

        Parameters
        ----------
        filters:
            The filters constructed with :func:`opencosmo.col`

        Returns
        -------
        SimulationCollection
            A new collection with the same datasets, but only the
            particles that pass the filter.
        """
        return self.__map("filter", *masks, **kwargs)

    def select(self, *args, **kwargs) -> SimulationCollection:
        """
        Select a set of columns in the datasets in this collection. This method
        calls the underlying method in :class:`opencosmo.Dataset`, or
        :class:`opencosmo.StructureCollection` depending on the context. As such
        its behavior and arguments can vary depending on what this collection
        contains. See the documentation for those objects to determine
        the expected arguments.

        If the collection holds datasets with different column sets (e.g a matched
        gravity-only and hydro sim) it will make a best-effort attempt to distribute
        the selections to the relevant dataset. Datasets with no matches will
        be retained unchanged.


        Parameters
        ----------
        args:
            The arguments to pass to the select method. This is
            usually a list of column names to select.
        kwargs:
            The keyword arguments to pass to the select method.
            This is usually a dictionary of column names to select.

        Returns
        -------
        SimulationCollection
            A new collection with only the specified columns

        """
        if not all(isinstance(dataset, Dataset) for dataset in self.values()):
            return self.__map("select", *args, **kwargs)

        datasets = cast("dict[str, Dataset]", self.__datasets)
        output = do_multi_dataset_selections(datasets, args, kwargs)
        return SimulationCollection(
            output, self.__match_set, self.__match_source, copy(self.__rebuilt)
        )

    def drop(self, *args, **kwargs) -> SimulationCollection:
        """
        Drop columns by automatically matching their names to datasets in this
        collection. Wildcards are applied to every dataset, while datasets
        without a match are unchanged. For nested collections, matching follows
        their :meth:`drop` behavior.

        To target datasets explicitly, pass dataset names as keyword arguments.
        This form is forwarded to the underlying datasets or collections.

        If the collection holds datasets with different column sets (e.g a matched
        gravity-only and hydro sim) it will make a best-effort attempt to distribute
        the selections to the relevant dataset. Datasets with no matches will
        be retained unchanged.


        Parameters
        ----------
        args : str or Iterable[str]
            Column names or wildcard patterns to match automatically across all
            datasets in the collection.
        kwargs
            Explicit dataset-keyed drop selections.

        """
        if not all(
            isinstance(dataset, DatasetState) for dataset in self.__datasets.values()
        ):
            return self.__map("drop", *args, **kwargs)

        datasets = cast("dict[str, DatasetState]", self.__datasets)
        output = do_multi_dataset_drops(datasets, args)
        for dataset_name, columns in kwargs.items():
            if dataset_name not in self:
                raise ValueError(f"Dataset {dataset_name} not found in collection.")
            output[dataset_name] = dsops.drop(output[dataset_name], columns)
        return SimulationCollection(
            output, self.__match_set, self.__match_source, copy(self.__rebuilt)
        )

    def take(
        self, n: int, at: str = "random", mode: Literal["local", "global"] = "local"
    ) -> SimulationCollection:
        """
        Take a subest of rows from all datasets or collections in this collection.
        This method will delegate to the underlying method in
        :class:`opencosmo.Dataset`, or :class:`opencosmo.StructureCollection` depending
        on  the context. As such, behavior may vary depending on what this collection
        contains. See their documentation for more info.

        If this dataset has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulatinCollection.match>`,
        the take operation will only run on the dataset matched against, with all matching rows retained
        in the remaining datasets.


        Parameters
        ----------
        n: int
            The number of rows to take
        at: str, default = "random"
            The method to use to take rows. Must be one of "start", "end", "random".

        """
        return self.__map("take", n, at, mode=mode)

    def take_range(
        self, start: int, end: int, mode: Literal["local", "global"] = "local"
    ):
        """
        Take a range of rows from all datasets or collections in this collection.
        This method will fail if :code:`start` < 0, or any of the datasets are not at least
        :code:`end` long.

        If this dataset has been matched with :py:meth:`SimulationCollection.match <opencosmo.SimulatinCollection.match>`,
        the take operation will only run on the dataset matched against, with all matching rows retained
        in the remaining datasets.

        Parameters
        ----------
        n: int
            The number of rows to take
        at: str, default = "random"
            The method to use to take rows. Must be one of "start", "end", "random".

        Returns
        -------
        SimulationCollection
            The new simulation collection with only the specified rows.

        """
        return self.__map("take_range", start, end, mode=mode)

    def with_new_columns(
        self,
        *args,
        datasets: Optional[str | Iterable[str]] = None,
        descriptions: str | dict[str, str] = {},
        allow_overwrite: bool = False,
        **new_columns: ConstructedColumn | np.ndarray,
    ):
        """
        Update the datasets within this collection with a set of new columns.
        This method simply calls :py:meth:`opencosmo.Dataset.with_new_columns` or
        :py:meth:`opencosmo.StructureCollection.with_new_columns`, as appropriate.

        You can also optionally pass the "datasets" keyword argument to specify that the
        operation should only be performed on a subset of the datasets.

        If passing in numpy arrays or astropy quantities, they should be provided
        as a dictionary where the keys are the same as the keys in this dataset.

        Parameters
        ----------
        datasets: str | list[str], optional
            The datasets to add the columns to.

        descriptions : str | dict[str, str], optional
            A description for the new columns. These descriptions will be accessible through
            :py:attr:`SimulationCollection(datasets).descriptions <opencosmo.SimulationCollection.descriptions>`.
            If a dictionary, should have keys matching the column names.

        allow_overwrite: bool, default = False


        ** columns : opencosmo.Column | np.ndarray | units.Quantity
            The new columns
        """
        return self.__map(
            "with_new_columns",
            *args,
            descriptions=descriptions,
            datasets=datasets,
            allow_overwrite=allow_overwrite,
            **new_columns,
        )

    def evaluate(
        self,
        func: Callable,
        datasets: Optional[str | Iterable[str]] = None,
        format: str = "astropy",
        vectorize: bool = False,
        insert: bool = False,
        allow_overwrite: bool = False,
        **evaluate_kwargs,
    ):
        """
        Evaluate the function :code:`func` on each of the datasets or collections
        held by this SimulationCollection. This function simply delegates to the
        either :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.Evaluate>`
        or :py:meth:`Dataset.evaluate <opencosmo.Dataset.Evaluate>` as appropriate. Refer
        to :ref:`Evaluating Complex Expressions on Datasets and Collections` for more details.

        If "datasets" is provided, the evaluation will only be performed on the provided
        datasets.

        Parameters
        ----------

        func: Callable
            The function to evaluate
        datasets: str | list[str], optional
            The datasets to evaluate on. If not provided, will be evaluated on all datasets
        format: str, default = "astropy"
            The format in which to provide column data to your function. Supports the same formats
            as :py:meth:`get_data <opencosmo.Dataset.get_data>` ("astropy", "numpy", "pandas",
            "polars", "arrow", "jax"). When :code:`insert=True`, the function's output is converted
            back to numpy before being stored.

        vectorize: bool, default = False
            Whether to vectorize the computation. See :py:meth:`StructureCollection.evaluate <opencosmo.StructureCollection.Evaluate>`
            and/or :py:meth:`Dataset.evaluate <opencosmo.Dataset.Evaluate>` for more details.
        insert: bool, default = True
            Whether or not to insert the results as columns in the datasets. If false, the results will
            be returned directly. If true, this method will return a new Simulation Collection.

        Returns
        -------
        results: SimulationCollection | dict[str, np.ndarray] | dict[str, astropy.units.Quantity]
            The results of the computation, or a new simulation collection with the results inserted.
        """
        if self.__match_source is not None:
            assert self.__match_set is not None
            self.__rebuild_all()

        results = self.__map(
            "evaluate",
            func,
            vectorize=vectorize,
            insert=insert,
            format=format,
            allow_overwrite=allow_overwrite,
            construct=insert,
            datasets=datasets,
            **evaluate_kwargs,
        )
        if next(iter(results.values())) is None:
            return
        return results

    def sort_by(self, column: str, invert: bool = False):
        """
        Re-order the individual datasets in the collection based on a column. See
        :py:meth:`Dataset.sort_by <opencosmo.Dataset.sort_by>` for usage details.

        Parameters
        ----------
        column : str
            The column in the halo_properties or galaxy_properties dataset to
            order the collection by.

        invert : bool, default = False
            If False (the default) ordering will be done from least to greatest.
            Otherwise greatest to least.

        Returns
        -------
        result : SimulationCollection
            A new SimulationCollection with the datasets ordered by the given column.

        """
        return self.__map("sort_by", column=column, invert=invert)

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns: u.Unit,
    ) -> SimulationCollection:
        """
        Transform all datasets or collections to use the given unit convention, convert
        all columns with a given unit into a different unit, and/or convert specific column(s)
        to a compatible unit. This method behaves exactly like :meth:`opencosmo.Dataset.with_units`.

        Parameters
        ----------
        convention: str
            The unit convention to use. One of "unitless",
            "scalefree", "comoving", or "physical".

        conversions: dict[astropy.units.Unit, astropy.units.Unit]
            Conversions that apply to all columns in the collection with the
            unit given by the key.

        **column_conversions: astropy.units.Unit
            Custom unit conversions for any column with a specific
            name in the datasets in this collection.

        Returns
        -------
        collection
            A new simulation collection with the requested unit conventions and conversions.


        """
        return self.__map("with_units", convention, conversions=conversions, **columns)
