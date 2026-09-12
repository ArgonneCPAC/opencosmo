from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import FileEntry

if TYPE_CHECKING:
    from .schema import Schema


"""
Verification is split into two independent tiers, each with its own return channel:

1. ``verify_structure`` checks that a schema is *well-formed*. It raises
   ``ValueError`` on genuine corruption and never inspects row counts, so a
   zero-length dataset (or an empty lightcone partition) is structurally valid.
2. ``schema_data_length`` reports how many data rows *this rank* actually
   contributes. It is a pure, local computation that never raises.

Keeping the two concerns separate is what lets an MPI write distinguish a rank
that holds no data (excluded from the write) from a rank whose schema is broken
(a hard error), without one empty sub-group poisoning the whole rank.

Verification is intentionally centralized, such that if you create a new data
type and implement "make_schema" it will fail at this step until you add
verification.
"""


def verify_structure(
    schema: Schema,
):
    match schema.type:
        case FileEntry.DATASET:
            return verify_dataset_data(schema)
        case FileEntry.STRUCTURE_COLLECTION:
            return verify_structure_collection_data(schema)
        case FileEntry.LIGHTCONE:
            verify_lightcone_collection_schema(schema)
        case FileEntry.SIMULATION_COLLECTION:
            for name, ds_schema in schema.children.items():
                match ds_schema.type:
                    case FileEntry.DATASET:
                        verify_dataset_data(ds_schema)
                    case FileEntry.STRUCTURE_COLLECTION:
                        verify_structure_collection_data(ds_schema)
        case FileEntry.HEALPIX_MAP:
            verify_dataset_data(schema, has_index=False)
        case _:
            raise ValueError("Unknown file structure!")


def data_group_length(schema: Schema) -> int:
    """
    Number of rows in a node's direct "data" group, or 0 if it has none.
    """
    if "data" not in schema.children:
        return 0
    column = next(iter(schema.children["data"].columns.values()), None)
    return 0 if column is None else column.shape[0]


def schema_data_length(schema: Schema) -> int:
    """
    Total number of data rows this rank will contribute for a schema. Purely
    local: MPI callers combine the per-rank totals themselves. Container nodes
    sum their children; the default arm returns 0 so recursion never descends
    into index/header/data_linked groups.
    """
    match schema.type:
        case FileEntry.DATASET | FileEntry.HEALPIX_MAP:
            return data_group_length(schema)
        case FileEntry.LIGHTCONE:
            if "data" in schema.children:  # single-dataset lightcone
                return data_group_length(schema)
            return sum(schema_data_length(c) for c in schema.children.values())
        case FileEntry.STRUCTURE_COLLECTION | FileEntry.SIMULATION_COLLECTION:
            return sum(schema_data_length(c) for c in schema.children.values())
        case _:
            return 0


def verify_column_group(schema: Schema):
    """
    Verify that a given data group is valid. This requires that:
    1. All column writers have the same length
    2. All columns have the same combine strategy
    3. All columns are in the same group

    Note this is a structural check only: a zero-length group is valid.
    """
    column_names = set()
    group_names = set()
    column_lengths = {}
    column_strategies = set()
    for column_path, column_writer in schema.columns.items():
        try:
            group_name, column_name = column_path.rsplit("/", 1)
        except ValueError:
            group_name = ""
            column_name = column_path
        group_names.add(group_name)
        column_names.add(column_name)
        column_lengths[column_path] = len(column_writer)
        column_strategies.add(column_writer.combine_strategy)

    all_column_lengths = set(column_lengths.values())

    if len(all_column_lengths) != 1:
        raise ValueError(
            "Columns within a single group should always have the same length!"
        )
    group_length = all_column_lengths.pop()

    if len(column_strategies) != 1:
        raise ValueError(
            "Columns within a single group should always have the same combine strategy!"
        )

    return (group_names.pop(), group_length, column_strategies.pop())


def verify_dataset_data(schema: Schema, has_index=True):
    """
    Verify a given dataset is valid. Requiring:
    1. It has a data group
    2. It has a spatial index group (if has_index = True)
    3. Any sibling column group (e.g. /data_linked) is the same length and
       combine strategy as the data group

    Once this is verified, we delegate to verify_column_group to ensure
    individual column groups are valid.
    """
    children = schema.children

    if "data" not in children or ("index" not in children and has_index):
        raise ValueError("Datasets must have at least a data group and a index group")

    sibling_groups = [
        child
        for name, child in schema.children.items()
        if name not in ["data", "index"] and child.type == FileEntry.COLUMNS
    ]

    _, data_length, data_combine_strategy = verify_column_group(schema.children["data"])
    if has_index:
        for child in schema.children["index"].children.values():
            verify_column_group(child)
    for sibling in sibling_groups:
        _, sib_length, sib_combine_strategy = verify_column_group(sibling)
        if sib_length != data_length or sib_combine_strategy != data_combine_strategy:
            raise ValueError(
                "Sibling column groups must be the same length and have the same "
                "combine strategy as the data group!"
            )


def verify_lightcone_collection_schema(schema: Schema):
    """
    Verify a lightcone collection. Note that a single dataset
    can also technically be a lighcone collection, if is_lightcone is
    set to true in its header. Mostly just delegates to underlying
    dataset checks.

    A lightcone with no children is structurally valid: it is how an empty
    partition (a rank that received none of the sampled structures) is
    represented. Whether it actually has data is decided by
    ``schema_data_length``, not here.
    """
    if "data" in schema.children:
        # Single-dataset lightcone
        return verify_dataset_data(schema)
    for key, child_schema in schema.children.items():
        verify_dataset_data(child_schema)


def verify_structure_collection_data(schema: Schema):
    """
    Structure collections have a lot going on, but they are mostly just datasets. The only
    thing we have to check is that we have an explicit "data_linked" group in any link
    holders, and we then verify the individual dataset.

    """
    if "halo_properties" in schema.children:
        link_holder = "halo_properties"
    elif "galaxy_properties" in schema.children:
        link_holder = "galaxy_properties"
    else:
        raise ValueError("No valid link holder found in schema!")

    for child_name, child_schema in schema.children.items():
        if child_schema.type == FileEntry.LIGHTCONE and child_name == link_holder:
            for grandchild_name, grandchild_schema in child_schema.children.items():
                has_link = any(
                    map(
                        lambda cn: "data_linked" in cn,
                        grandchild_schema.children.keys(),
                    )
                )
                if not has_link:
                    raise ValueError(
                        f'Source dataset {child_name}/{grandchild_name} does not have expected "data_linked" group'
                    )

        elif child_name == link_holder:
            has_link = any(
                map(lambda cn: "data_linked" in cn, child_schema.children.keys())
            )
            if not has_link:
                raise ValueError(
                    f'Source dataset {child_name} does not have expected "data_linked" group'
                )

        match child_schema.type:
            case FileEntry.DATASET:
                verify_dataset_data(child_schema)
            case FileEntry.STRUCTURE_COLLECTION:
                verify_structure_collection_data(child_schema)
            case FileEntry.LIGHTCONE:
                verify_lightcone_collection_schema(child_schema)
            case _:
                raise ValueError("Got an unknown child for structure collection!")
