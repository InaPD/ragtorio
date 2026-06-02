"""GraphLoader, check_graph and recipe_tree against a real Neo4j.

Idempotent writes, the per-label constraints, and a real multi-level traversal are
exactly what an in-memory fake would get wrong. Skipped, not failed, when no Neo4j
is reachable.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from neo4j import Driver, GraphDatabase

from ragtorio.db.neo4j import apply_schema, connect
from ragtorio.ontology.check import check_graph
from ragtorio.ontology.load import GraphLoader
from ragtorio.ontology.models import ResolvedEdge, ResolvedGraph, ResolvedNode
from ragtorio.ontology.recipe_tree import raw_totals, recipe_tree

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "ragtorio")


def _reachable() -> bool:
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason=f"no Neo4j at {URI}"),
]


@pytest.fixture
def driver() -> Iterator[Driver]:
    with connect(URI, *AUTH) as d:
        with d.session() as session:
            session.run(
                # Scoped to this wiki: a full wipe here would take a real
                # crawl with it.
                "MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p="t:",
            )
        apply_schema(d)
        yield d


def node(node_id: str, label: str, **props: object) -> ResolvedNode:
    return ResolvedNode(id=node_id, labels=(label,), props={"title": node_id, **props})


def test_load_is_idempotent(driver: Driver) -> None:
    graph = ResolvedGraph(
        nodes=(node("t:Gear", "Item", stack_size=100.0),),
        edges=(),
        unresolved=(),
    )
    GraphLoader(driver).load(graph)
    GraphLoader(driver).load(graph)  # same data again: no duplicate node
    with driver.session() as session:
        count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    assert count == 1


def test_load_a_multi_label_node_then_check_finds_orphans_and_gaps(driver: Driver) -> None:
    """Station is resolve.py's own job (tested there); this checks that GraphLoader
    and check_graph handle a node carrying two labels correctly, since that is what
    resolve.py hands them once it has done that work."""
    graph = ResolvedGraph(
        nodes=(
            node("t:Gear (recipe)", "Recipe", crafting_time=1.0),
            ResolvedNode(
                id="t:Assembler", labels=("Item", "Station"), props={"title": "Assembler"}
            ),
            node("t:Lonely", "Item"),  # no edges at all: an orphan
        ),
        edges=(
            ResolvedEdge(from_id="t:Gear (recipe)", rel_type="CRAFTED_AT", to_id="t:Assembler"),
        ),
        unresolved=(),
    )
    GraphLoader(driver).load(graph)

    with driver.session() as session:
        labels = session.run("MATCH (n {id: 't:Assembler'}) RETURN labels(n) AS labels").single()[
            "labels"
        ]
    assert set(labels) == {"Item", "Station"}

    report = check_graph(driver)
    assert "t:Lonely" in report.orphans
    assert "t:Gear (recipe)" in report.recipes_missing_inputs
    assert "t:Gear (recipe)" in report.recipes_missing_outputs


def test_check_finds_a_self_cycle(driver: Driver) -> None:
    graph = ResolvedGraph(
        nodes=(node("t:Kovarex", "Recipe"), node("t:Uranium-235", "Item")),
        edges=(
            ResolvedEdge(from_id="t:Kovarex", rel_type="CONSUMES", to_id="t:Uranium-235"),
            ResolvedEdge(from_id="t:Kovarex", rel_type="PRODUCES", to_id="t:Uranium-235"),
        ),
        unresolved=(),
    )
    GraphLoader(driver).load(graph)
    report = check_graph(driver)
    assert report.self_cycles == (("t:Kovarex", "t:Uranium-235"),)


def test_recipe_tree_multiplies_amounts_level_by_level(driver: Driver) -> None:
    # 5 Widget needs 2x Gear + 1x Plate each; Gear needs 3x Ore (raw); Plate is raw.
    with driver.session() as session:
        session.run(
            """
            CREATE (w:Item {id:'t:Widget', title:'Widget'})
            CREATE (wr:Recipe {id:'t:Widget (recipe)', title:'Widget (recipe)'})
            CREATE (g:Item {id:'t:Gear', title:'Gear'})
            CREATE (gr:Recipe {id:'t:Gear (recipe)', title:'Gear (recipe)'})
            CREATE (p:Item {id:'t:Plate', title:'Plate'})
            CREATE (o:Item {id:'t:Ore', title:'Ore'})
            CREATE (wr)-[:PRODUCES {amount:1.0, probability:1.0}]->(w)
            CREATE (wr)-[:CONSUMES {amount:2.0}]->(g)
            CREATE (wr)-[:CONSUMES {amount:1.0}]->(p)
            CREATE (gr)-[:PRODUCES {amount:1.0, probability:1.0}]->(g)
            CREATE (gr)-[:CONSUMES {amount:3.0}]->(o)
            """
        )
    tree = recipe_tree(driver, "t", "Widget", quantity=5.0)
    assert raw_totals(tree) == {"Ore": 30.0, "Plate": 5.0}


def test_recipe_tree_treats_a_fractional_output_as_a_probability(driver: Driver) -> None:
    """Uranium-processing-shaped: an expected-value calculation, not a literal amount."""
    with driver.session() as session:
        session.run(
            """
            CREATE (sh:Item {id:'t:Shard', title:'Shard'})
            CREATE (shr:Recipe {id:'t:Shard (recipe)', title:'Shard (recipe)'})
            CREATE (du:Item {id:'t:Dust', title:'Dust'})
            CREATE (shr)-[:PRODUCES {amount:1.0, probability:0.5}]->(sh)
            CREATE (shr)-[:CONSUMES {amount:1.0}]->(du)
            """
        )
    tree = recipe_tree(driver, "t", "Shard", quantity=2.0)
    assert raw_totals(tree) == {"Dust": 4.0}  # half the batches succeed, so twice the dust


def test_recipe_tree_truncates_a_cycle_instead_of_looping_forever(driver: Driver) -> None:
    """Kovarex-shaped: a recipe that consumes and produces the same item."""
    with driver.session() as session:
        session.run(
            """
            CREATE (l:Item {id:'t:Loopy', title:'Loopy'})
            CREATE (lr:Recipe {id:'t:Loopy (recipe)', title:'Loopy (recipe)'})
            CREATE (lr)-[:PRODUCES {amount:2.0, probability:1.0}]->(l)
            CREATE (lr)-[:CONSUMES {amount:1.0}]->(l)
            """
        )
    tree = recipe_tree(driver, "t", "Loopy", quantity=1.0, max_depth=5)
    deepest = tree
    while deepest.children:
        deepest = deepest.children[0]
    assert deepest.truncated is True
