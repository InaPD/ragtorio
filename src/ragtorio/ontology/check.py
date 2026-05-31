"""``ragtorio graph check``: what to look at before trusting the graph.

Structural checks against the loaded graph. Unresolved references are not this
module's concern - they come straight from the last :meth:`EntityResolver.resolve`
call, since resolution is cheap and fully recomputable, exactly like the crawl and
the extraction it builds on.
"""

from __future__ import annotations

from neo4j import Driver
from pydantic import BaseModel, ConfigDict

from ragtorio.ontology.models import UnresolvedReference

_ORPHANS = "MATCH (n) WHERE NOT (n)--() RETURN n.id AS id ORDER BY id"

_MISSING_INPUTS = """
MATCH (r:Recipe) WHERE NOT (r)-[:CONSUMES]->() RETURN r.id AS id ORDER BY id
"""

_MISSING_OUTPUTS = """
MATCH (r:Recipe) WHERE NOT (r)-[:PRODUCES]->() RETURN r.id AS id ORDER BY id
"""

#: A recipe that consumes an item it also produces: a legitimate one-step cycle
#: (Kovarex enrichment process nets uranium-235 from uranium-238 this way). Longer,
#: multi-recipe cycles are not detected; this catches the shape the plan names.
_SELF_CYCLES = """
MATCH (r:Recipe)-[:CONSUMES]->(i)<-[:PRODUCES]-(r)
RETURN r.id AS recipe_id, i.id AS item_id ORDER BY recipe_id
"""


class GraphCheckReport(BaseModel):
    """Everything worth a human's attention before trusting the graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    orphans: tuple[str, ...] = ()
    recipes_missing_inputs: tuple[str, ...] = ()
    recipes_missing_outputs: tuple[str, ...] = ()
    self_cycles: tuple[tuple[str, str], ...] = ()
    unresolved: tuple[UnresolvedReference, ...] = ()


def check_graph(
    driver: Driver, unresolved: tuple[UnresolvedReference, ...] = ()
) -> GraphCheckReport:
    """Query the loaded graph for orphans, incomplete recipes and self-cycles."""
    with driver.session() as session:
        orphans = tuple(r["id"] for r in session.run(_ORPHANS))
        missing_inputs = tuple(r["id"] for r in session.run(_MISSING_INPUTS))
        missing_outputs = tuple(r["id"] for r in session.run(_MISSING_OUTPUTS))
        cycles = tuple((r["recipe_id"], r["item_id"]) for r in session.run(_SELF_CYCLES))
    return GraphCheckReport(
        orphans=orphans,
        recipes_missing_inputs=missing_inputs,
        recipes_missing_outputs=missing_outputs,
        self_cycles=cycles,
        unresolved=unresolved,
    )
