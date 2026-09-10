from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any

import rustworkx as rx

from opencosmo.column.column import RawColumn

if TYPE_CHECKING:
    from uuid import UUID

    import astropy.units as u

    from opencosmo.column.column import ConstructedColumn
    from opencosmo.units.handler import UnitHandler


def get_all_required_pairs(
    producers: list[ConstructedColumn],
    columns_to_uuid: dict[str, UUID],
) -> set[tuple[UUID, str]]:
    """
    Return the full set of (producer_uuid, column_name) pairs needed to
    produce the requested columns, including all transitive dependencies.
    """
    dependency_graph = build_dependency_graph(producers)
    uuid_to_node: dict[UUID, int] = {
        dependency_graph[i].uuid: i for i in range(dependency_graph.num_nodes())
    }
    required_nodes: set[int] = set()
    for uuid in columns_to_uuid.values():
        if uuid in uuid_to_node:
            node_idx = uuid_to_node[uuid]
            required_nodes.add(node_idx)
            required_nodes.update(rx.ancestors(dependency_graph, node_idx))

    pairs: set[tuple[UUID, str]] = {
        (uuid, name) for name, uuid in columns_to_uuid.items()
    }
    for node_idx in required_nodes:
        producer = dependency_graph[node_idx]
        for name in producer.produces:
            pairs.add((producer.uuid, name))
    return pairs


def validate_column_producers(
    producers: list[ConstructedColumn], unit_handler: UnitHandler
):
    """
    Validate the network of column producers.
    """
    dependency_graph = build_dependency_graph(producers)

    if cycle := rx.digraph_find_cycle(dependency_graph):
        all_nodes: set[int] = reduce(
            lambda known, edge: known.union(edge), cycle, set()
        )
        names = [dependency_graph[i].produces for i in all_nodes]
        raise ValueError(f"Found columns that depend on each other! Columns: {names}")

    for i in range(dependency_graph.num_nodes()):
        if dependency_graph.in_degree(i):
            continue
        node = dependency_graph[i]
        if not isinstance(node, RawColumn):
            raise ValueError(
                f"Tried to derive columns from unknown columns: {node.produces}"
            )

    return get_derived_units(dependency_graph, unit_handler.current_units)


def build_dependency_graph(
    producers: list[ConstructedColumn],
) -> rx.PyDiGraph:
    graph = rx.PyDiGraph()
    uuid_to_node: dict[UUID, int] = {}

    for producer in producers:
        node_idx = graph.add_node(producer)
        uuid = producer.uuid
        assert uuid is not None
        uuid_to_node[uuid] = node_idx

    known_uuids = set(uuid_to_node)

    for producer in producers:
        uuid = producer.uuid
        assert uuid is not None
        produces_idx = uuid_to_node[uuid]
        if not producer.requires.issubset(known_uuids):
            raise ValueError(
                f"Producer {producer.produces} depends on an unknown producer UUID."
            )
        new_edges = (
            (uuid_to_node[dep_uuid], produces_idx) for dep_uuid in producer.requires
        )
        graph.add_edges_from_no_data(new_edges)

    return graph


def evaluate_producers(
    producers: list[ConstructedColumn],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate a list of producers in topological order against a flat
    name-keyed input mapping, returning the produced columns by name.

    Each producer reads its dependencies by name from ``inputs`` (extended
    with what earlier producers in the graph have already produced). The
    flat-mapping form is intentional: callers that don't carry a full
    UUID-keyed cache (lightcone scope, ad-hoc evaluation against a
    materialized table) can just hand over the columns they have.

    Used by Lightcone.get_data to materialize scope-owned derived columns
    against the vstacked per-child data.
    """
    graph = build_dependency_graph(producers)
    outputs: dict[str, Any] = {}
    for node_idx in rx.topological_sort(graph):
        producer = graph[node_idx]
        if isinstance(producer, RawColumn):
            continue
        all_data = {**inputs, **outputs}
        result = producer.evaluate(all_data, None)
        if not isinstance(result, dict):
            result = {next(iter(producer.produces)): result}
        outputs.update(result)
    return outputs


def get_derived_units(
    dependency_graph: rx.PyDiGraph,
    units: dict[str, u.Unit],
):
    new_units: dict[str, u.Unit | None] = {}
    for node_idx in rx.topological_sort(dependency_graph):
        node = dependency_graph[node_idx]
        if isinstance(node, RawColumn):
            continue
        column_units = node.get_units(units | new_units)
        if not isinstance(column_units, dict):
            column_units = {prod: column_units for prod in node.produces}
        new_units |= column_units
    return new_units
