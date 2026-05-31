"""Writing a resolved graph into Neo4j.

Batched and idempotent: every node and edge is written with APOC's ``apoc.merge.*``
procedures rather than plain ``MERGE``, because plain Cypher needs a node's labels and
an edge's relationship type spelled out as literal tokens in the query text, and both
vary per row here (an ``Item`` that is also a ``Station``, six different relationship
types). ``onCreateProps`` and ``onMatchProps`` are the same dict, so a re-run always
refreshes every property to match the latest extraction rather than only setting it
once.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from neo4j import Driver

from ragtorio.ontology.models import ResolvedEdge, ResolvedGraph, ResolvedNode

#: Every label a node might carry, used so an edge lookup can use each label's index
#: rather than an unlabelled property scan.
_ANY_LABEL = "Item|Fluid|Recipe|Station|Unlock"

_MERGE_NODES = """
UNWIND $rows AS row
CALL apoc.merge.node(row.labels, {id: row.id}, row.props, row.props) YIELD node
RETURN count(node) AS n
"""

_MERGE_EDGES = f"""
UNWIND $rows AS row
MATCH (a:{_ANY_LABEL} {{id: row.from_id}})
MATCH (b:{_ANY_LABEL} {{id: row.to_id}})
CALL apoc.merge.relationship(a, row.rel_type, {{}}, row.props, b, row.props) YIELD rel
RETURN count(rel) AS n
"""

#: The API cap for batches was a real constraint for the harvester; here it is just a
#: sane chunk size so one query's parameter payload stays small.
DEFAULT_BATCH_SIZE = 500


class GraphLoader:
    """Loads a :class:`~ragtorio.ontology.models.ResolvedGraph`. The caller owns the driver."""

    def __init__(self, driver: Driver, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._driver = driver
        self._batch_size = batch_size

    def load(self, graph: ResolvedGraph) -> None:
        with self._driver.session() as session:
            for batch in _chunks(graph.nodes, self._batch_size):
                session.run(_MERGE_NODES, rows=[_node_row(n) for n in batch])
            for batch in _chunks(graph.edges, self._batch_size):
                session.run(_MERGE_EDGES, rows=[_edge_row(e) for e in batch])


def _node_row(node: ResolvedNode) -> dict[str, Any]:
    return {"labels": list(node.labels), "id": node.id, "props": node.props}


def _edge_row(edge: ResolvedEdge) -> dict[str, Any]:
    return {
        "from_id": edge.from_id,
        "to_id": edge.to_id,
        "rel_type": edge.rel_type,
        "props": edge.props,
    }


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
