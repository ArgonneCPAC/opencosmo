from __future__ import annotations

import json
from collections import defaultdict
from copy import copy
from functools import cache
from itertools import chain
from types import UnionType
from typing import TYPE_CHECKING, Any, Optional

import h5py
import numpy as np
from pydantic import ValidationError

from opencosmo.dtypes import (
    FileParameters,
    dtype,
    origin,
    read_header_attributes,
    write_header_attributes,
)
from opencosmo.dtypes.units import apply_units
from opencosmo.file import broadcast_read, file_reader, file_writer
from opencosmo.io.schema import FileEntry, add_metadata, empty_schema
from opencosmo.io.writer import ColumnCombineStrategy
from opencosmo.units import UnitConvention

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from opencosmo.io.schema import Schema
    from opencosmo.spatial.protocols import Region

HEADER_WRITE_OVERRIDES = {"region_pixels": ColumnCombineStrategy.CONCAT}


def _json_default_serializer(obj: Any) -> Any:
    """Best-effort JSON serializer for header transport.

    The goal is to preserve list ordering by only converting array-like types
    into plain Python lists (instead of reordering or permuting).
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


class OpenCosmoHeader:
    """
    A class to represent the header of an OpenCosmo file. The header contains
    information about the simulation the data is a part of, as well as other
    meatadata that are useful to the library in various contexts. Most files
    will have a single unique header, but it is possible to have multiple
    headers in a SimulationCollection.
    """

    def __init__(
        self,
        file_pars: FileParameters,
        required_origin_parameters: dict[str, BaseModel],
        optional_origin_parameters: dict[str, BaseModel],
        dtype_parameters: dict[str, BaseModel],
        unit_convention: UnitConvention = UnitConvention.SCALEFREE,
    ):
        self.__file_pars = file_pars
        self.__required_origin_parameters = required_origin_parameters
        self.__optional_origin_parameters = optional_origin_parameters
        self.__dtype_parameters = dtype_parameters
        self.unit_convention = unit_convention

    def __eq__(self, other):
        return (
            self.__file_pars == other.__file_pars
            and self.__required_origin_parameters == other.__required_origin_parameters
            and self.__optional_origin_parameters == other.__optional_origin_parameters
            and self.__dtype_parameters == other.__dtype_parameters
        )

    def __dir__(self):
        return list(self.parameters.keys()) + list(super().__dir__())

    def __hash__(self):
        # Create a frozenset of the items in the dictionary
        # Each item is a tuple of (key, hash of the model)
        iter_ = chain(
            {"file": self.__file_pars}.items(),
            {"unit_convention": self.unit_convention}.items(),
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
            self.__dtype_parameters.items(),
        )
        s = frozenset((key, hash(model)) for key, model in iter_)
        return hash(s)

    def with_units(self, convention: UnitConvention | str):
        convention = UnitConvention(convention)
        if convention == self.unit_convention:
            return self
        return OpenCosmoHeader(
            self.__file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
            self.__dtype_parameters,
            convention,
        )

    @cache
    def __get_access_table(self):
        all_models = chain(
            {"file": self.__file_pars}.values(),
            self.__required_origin_parameters.values(),
            self.__optional_origin_parameters.values(),
            self.__dtype_parameters.values(),
        )
        table = get_access_table(all_models, self.unit_convention, self.file.redshift)

        return dict(table)

    @property
    def parameters(self):
        """
        Return the parametrs stored in this header as
        key-value pairs. The values will be Pydantic models.

        Any block of parameters that is returned from this method can
        also be accessed with standard dot notation. For example, HACC
        data contains a "simulation" block that contains the parameters that
        were used to run the original simulation. The following calls are
        equivalent:

        .. code-block:: python

            header.simulation
            header.parameters["simulation"]

        Returns
        -------
        parameters: dict[str, pydantic.BaseModel]
            The parameter blocks associated with this header

        """
        return self.__get_access_table()

    def __getattr__(self, key: str):
        if key.startswith("__"):  # avoid infinite recursion when serailizing for MPI
            raise AttributeError(key)

        table = self.__get_access_table()
        try:
            return table[key]
        except KeyError:
            return object.__getattribute__(self, key)

    def with_region(self, region: Region):
        if region is not None:
            region_model = region.into_model()
        else:
            region_model = None
        new_file_pars = self.__file_pars.model_copy(update={"region": region_model})
        new_header = OpenCosmoHeader(
            new_file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
            self.__dtype_parameters,
        )
        return new_header

    def with_parameter(self, key: str, value: Any):
        """
        Update a dtype parameter with a new value. This in general should never
        be called by the user. Returns a copy.
        """
        path = key.split("/")
        if len(path) != 2:
            raise ValueError("Can only update top-level dtype parameters")
        new_dtype_parameters = copy(self.__dtype_parameters)
        model = new_dtype_parameters[path[0]]
        new_model = model.model_copy(update={path[1]: value})
        new_dtype_parameters[path[0]] = new_model
        return OpenCosmoHeader(
            self.__file_pars,
            self.__required_origin_parameters,
            self.__optional_origin_parameters,
            new_dtype_parameters,
        )

    def with_parameters(self, updates: dict[str, Any]):
        if not updates:
            return self
        new_header = self
        for key, val in updates.items():
            new_header = new_header.with_parameter(key, val)
        return new_header

    def dump(self) -> Schema:
        to_write = chain(
            [("file", self.__file_pars)],
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
            self.__dtype_parameters.items(),
        )
        schema = empty_schema("header", FileEntry.METADATA)

        for path, model in to_write:
            data = model.model_dump(by_alias=True, exclude_none=True)
            schema = add_metadata(path, schema, data, HEADER_WRITE_OVERRIDES)

        return schema

    def write(self, file: h5py.File | h5py.Group) -> None:
        write_header_attributes(file, "file", self.__file_pars)
        to_write = chain(
            self.__required_origin_parameters.items(),
            self.__optional_origin_parameters.items(),
            self.__dtype_parameters.items(),
        )
        for path, data in to_write:
            write_header_attributes(file, path, data)

    @property
    def file(self) -> FileParameters:
        """
        All files must at minimum have a "file" block in their header. This block
        contains basic information like the original source of the data and
        its data type.
        """
        return self.__file_pars

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this header.

        Notes
        -----
        This representation is intended for transport/storage (e.g. SQLite
        caches). Values are JSON-safe primitives and nested dict/list
        structures.
        """

        def _model_block(models: dict[str, BaseModel]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, model in models.items():
                # Round-tripping through json coerces numpy scalars/arrays that
                # some models emit into JSON-safe primitives.
                data = model.model_dump(by_alias=True, exclude_none=True)
                out[key] = json.loads(
                    json.dumps(
                        data, default=_json_default_serializer, separators=(",", ":")
                    )
                )
            return out

        return {
            "file": json.loads(
                json.dumps(
                    self.__file_pars.model_dump(by_alias=True, exclude_none=True),
                    default=_json_default_serializer,
                )
            ),
            "unit_convention": self.unit_convention.value,
            "required_origin_parameters": _model_block(
                self.__required_origin_parameters
            ),
            "optional_origin_parameters": _model_block(
                self.__optional_origin_parameters
            ),
            "dtype_parameters": _model_block(self.__dtype_parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenCosmoHeader:
        """Reconstruct an :class:`~opencosmo.header.OpenCosmoHeader` from ``to_dict``."""

        unit_convention = UnitConvention(data["unit_convention"])

        file_pars = FileParameters.model_validate(data["file"])
        origin_parameter_models = origin.get_origin_parameters(file_pars.origin)
        required_origin_models = origin_parameter_models.get("required", {})
        optional_origin_models = origin_parameter_models.get("optional", {})

        def _postprocess_payload(payload: dict[str, Any]) -> dict[str, Any]:
            for k, v in list(payload.items()):
                if k.endswith("cosmotools_steps") and isinstance(v, np.ndarray):
                    payload[k] = v.tolist()
            return payload

        required_origin_parameters: dict[str, BaseModel] = {}
        for key, payload in data["required_origin_parameters"].items():
            payload = _postprocess_payload(payload)
            model_type = required_origin_models.get(key)
            if model_type is None:
                raise ValueError(f"Unknown required origin parameter: {key}")
            if isinstance(model_type, UnionType):
                for inner_model in model_type.__args__:
                    try:
                        required_origin_parameters[key] = inner_model.model_validate(
                            payload
                        )
                        break
                    except ValidationError as ve:
                        if any(
                            e["type"] == "missing" or e["input"] is None
                            for e in ve.errors()
                        ):
                            continue
                        raise ValueError(
                            "Parsing header paramter model raised a validation error: "
                            f"\n {ve}"
                        )
                else:
                    raise ValueError(
                        "Input attributes do not match any of the models in the union"
                    )
            else:
                required_origin_parameters[key] = model_type.model_validate(payload)

        optional_origin_parameters: dict[str, BaseModel] = {}
        for key, payload in data["optional_origin_parameters"].items():
            payload = _postprocess_payload(payload)
            model_type = optional_origin_models.get(key)
            if model_type is None:
                raise ValueError(f"Unknown optional origin parameter: {key}")
            if isinstance(model_type, UnionType):
                for inner_model in model_type.__args__:
                    try:
                        optional_origin_parameters[key] = inner_model.model_validate(
                            payload
                        )
                        break
                    except ValidationError as ve:
                        if any(
                            e["type"] == "missing" or e["input"] is None
                            for e in ve.errors()
                        ):
                            continue
                        raise ValueError(
                            "Parsing header paramter model raised a validation error: "
                            f"\n {ve}"
                        )
                else:
                    raise ValueError(
                        "Input attributes do not match any of the models in the union"
                    )
            else:
                optional_origin_parameters[key] = model_type.model_validate(payload)

        dtype_parameter_models = dtype.get_dtype_parameters(file_pars)
        required_dtype_models = dtype_parameter_models.get("required", {})
        optional_dtype_models = dtype_parameter_models.get("optional", {})

        dtype_parameters: dict[str, BaseModel] = {}
        for key, payload in data["dtype_parameters"].items():
            payload = _postprocess_payload(payload)
            model_type = required_dtype_models.get(key) or optional_dtype_models.get(
                key
            )
            if model_type is None:
                raise ValueError(f"Unknown dtype parameter: {key}")
            if isinstance(model_type, UnionType):
                for inner_model in model_type.__args__:
                    try:
                        dtype_parameters[key] = inner_model.model_validate(payload)
                        break
                    except ValidationError as ve:
                        if any(
                            e["type"] == "missing" or e["input"] is None
                            for e in ve.errors()
                        ):
                            continue
                        raise ValueError(
                            "Parsing header paramter model raised a validation error: "
                            f"\n {ve}"
                        )
                else:
                    raise ValueError(
                        "Input attributes do not match any of the models in the union"
                    )
            else:
                dtype_parameters[key] = model_type.model_validate(payload)

        return cls(
            file_pars,
            required_origin_parameters,
            optional_origin_parameters,
            dtype_parameters,
            unit_convention,
        )


@file_writer
def write_header(
    path: Path, header: OpenCosmoHeader, dataset_name: Optional[str] = None
) -> None:
    """
    Write the header of an OpenCosmo file

    Parameters
    ----------
    file : h5py.File
        The file to write to
    header : OpenCosmoHeader
        The header information to write

    """
    with h5py.File(path, "w") as f:
        if dataset_name is not None:
            group = f.require_group(dataset_name)
        else:
            group = f
        header.write(group)


def get_access_table(all_models, unit_convention, redshift):
    table = defaultdict(dict)
    known_paramater_exports = set()
    all_models = list(all_models)
    cosmology_pars = [
        i
        for i, m in enumerate(all_models)
        if getattr(m, "ACCESS_PATH", None) == "cosmology"
    ]
    if len(cosmology_pars) == 1:
        table["cosmology"] = all_models[cosmology_pars[0]].ACCESS_TRANSFORMATION()

    del all_models[cosmology_pars[0]]

    cosmology = table.get("cosmology")
    scale_factor = None
    if redshift is not None:
        scale_factor = cosmology.scale_factor(redshift)

    for model in all_models:
        if hasattr(model, "PARAMETER_ACCESS_PATHS"):
            for name, path in model.PARAMETER_ACCESS_PATHS.items():
                if path in known_paramater_exports:
                    raise ValueError(
                        f"Duplicate access path detected in header: {name}"
                    )
                table[path] = getattr(model, name)
                known_paramater_exports.add(path)

        if not hasattr(model, "ACCESS_PATH"):
            continue

        if model.ACCESS_PATH in known_paramater_exports:
            raise ValueError(
                f"Duplicate access path detected in header: {model.ACCESS_PATH}"
            )
        if hasattr(model, "ACCESS_TRANSFORMATION"):
            data = model.ACCESS_TRANSFORMATION()
        else:
            data = model

        table[model.ACCESS_PATH] |= apply_units(
            data,
            type(model),
            cosmology,
            unit_convention,
            unit_kwargs={"scale_factor": scale_factor},
        )

    return dict(table)


@broadcast_read
@file_reader
def read_header(
    file: h5py.File | h5py.Group,
    unit_convention: UnitConvention = UnitConvention.COMOVING,
) -> OpenCosmoHeader:
    """
    Read the header of an OpenCosmo file

    This function may be useful if you just want to access some basic
    information about the simulation but you don't plan to actually
    read any data.

    Parameters
    ----------
    file : str | Path
        The path to the file

    Returns
    -------
    header : OpenCosmoHeader
        The header information from the file


    """
    try:
        file_parameters = read_header_attributes(file, "file", FileParameters)
    except KeyError as e:
        raise KeyError(
            "File header is malformed. Are you sure it is an OpenCosmo file?\n "
            f"Error: {e}"
        )

    origin_parameter_models = origin.get_origin_parameters(file_parameters.origin)
    required_origin_params, optional_origin_params = read_parameter_groups(
        file,
        origin_parameter_models,
        additional_required=file_parameters.require_header_groups,
    )
    dtype_parameter_models = dtype.get_dtype_parameters(file_parameters)
    required_dtype_params, optional_dtype_params = read_parameter_groups(
        file,
        dtype_parameter_models,
        additional_required=file_parameters.require_header_groups,
    )

    h = OpenCosmoHeader(
        file_parameters,
        required_origin_params,
        optional_origin_params,
        required_dtype_params | optional_dtype_params,
        unit_convention,
    )
    return h


def read_parameter_groups(
    file: h5py.File | h5py.Group,
    parameter_groups: dict[str, dict[str, type[BaseModel]]],
    additional_required: Optional[tuple] = None,
):
    """
    An "origin" describes the original source of a given dataset. Currently the only
    origin we support in the OpenCosmo toolkit is HACC.

    Origins can define a set of required and optional parameters.
    """
    if not parameter_groups:
        return {}, {}

    if additional_required is None:
        additional_required = ()
    required = parameter_groups.get("required", {})
    required_output = {}
    for path, model in required.items():
        if isinstance(model, UnionType):
            required_output[path] = load_union_model(file, path, model)
        else:
            required_output[path] = read_header_attributes(file, path, model)

    optional_output = {}
    optional = parameter_groups.get("optional", {})
    for path, model in optional.items():
        if isinstance(origin, UnionType):
            read_fn = load_union_model
        else:
            read_fn = read_header_attributes
        try:
            optional_output[path] = read_fn(file, path, model)
        except (ValidationError, KeyError):
            if path not in additional_required:
                continue
            raise

    return required_output, optional_output


def load_union_model(
    file: h5py.File | h5py.Group, path: str, allowed_models: UnionType, **kwargs
):
    for model in allowed_models.__args__:
        try:
            return read_header_attributes(file, path, model)
        except ValidationError as ve:
            if any(e["type"] == "missing" or e["input"] is None for e in ve.errors()):
                continue
            else:
                raise ValueError(
                    f"Parsing header paramter model raised a validation error: \n {ve}"
                )
    raise ValueError("Input attributes do not match any of the models in the union")
