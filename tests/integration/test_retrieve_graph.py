"""The four Cypher templates against a real Neo4j, on a graph built for the purpose.

A fixture graph rather than the live crawl, because these tests have to prove the
*query* and a crawl would let a data gap read as a query bug (and vice versa). It is
shaped like the real thing where the shape is what the query depends on: an item and
its separate ``(recipe)`` node, a technology that gates a recipe rather than the item,
a ``CONSUMES`` edge with no amount, and nodes carrying the wiki's own categories.

The same twelve checks against the live graph are in
``docs/coverage/phase5-retrieval.md``; those prove the data, these prove the Cypher.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from ragtorio.config import RetrievalConfig
from ragtorio.db.neo4j import apply_schema, connect
from ragtorio.ontology.load import GraphLoader
from ragtorio.ontology.models import ResolvedEdge, ResolvedGraph, ResolvedNode
from ragtorio.retrieve.entities import GraphEntityResolver
from ragtorio.retrieve.graph import GraphRetriever
from ragtorio.retrieve.models import Route

URI = os.environ.get("RAGTORIO_TEST_NEO4J_URI", "bolt://localhost:7687")
AUTH = ("neo4j", os.environ.get("RAGTORIO_TEST_NEO4J_PASSWORD", "ragtorio"))
WIKI = "testwiki"


def _reachable() -> bool:
    try:
        with connect(URI, *AUTH) as driver:
            driver.verify_connectivity()
            return True
    except (ServiceUnavailable, Neo4jError, OSError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason=f"no Neo4j at {URI}"),
]


def node(name: str, *labels: str, **props: object) -> ResolvedNode:
    return ResolvedNode(id=f"{WIKI}:{name}", labels=labels, props={"title": name, **props})


def edge(source: str, rel: str, target: str, **props: object) -> ResolvedEdge:
    return ResolvedEdge(
        from_id=f"{WIKI}:{source}", rel_type=rel, to_id=f"{WIKI}:{target}", props=props
    )


#: Two crafting chains, a three-deep technology tree, and the awkward cases: a recipe
#: gated only through the recipe node, and a CONSUMES edge with no amount on it.
FIXTURE = ResolvedGraph(
    nodes=(
        node("Iron ore", "Item", categories=["Resources"], stack_size=50, mining_time=1.0),
        node("Copper ore", "Item", categories=["Resources"], stack_size=50, mining_time=2.0),
        node("Coal", "Item", categories=["Resources"], stack_size=50, mining_time=0.5),
        node("Iron plate", "Item", categories=["Intermediates"], stack_size=100),
        node("Iron plate (recipe)", "Recipe"),
        node("Copper plate", "Item", categories=["Intermediates"], stack_size=100),
        node("Copper plate (recipe)", "Recipe"),
        node("Gear", "Item", categories=["Intermediates"], stack_size=200),
        node("Gear (recipe)", "Recipe"),
        node("Widget", "Item", categories=["Intermediates"], stack_size=10),
        node("Widget (recipe)", "Recipe"),
        node("Furnace", "Item", "Station"),
        # A resource the wiki also documents a recipe for, like Factorio's coal.
        node("Stone", "Item", categories=["Resources"], stack_size=50),
        node("Stone synthesis", "Recipe"),
        node("Dust", "Item", categories=["Intermediates"]),
        # A legitimate cycle: the recipe consumes what it also produces.
        node("Enriched (recipe)", "Recipe"),
        node("Enriched", "Item", categories=["Intermediates"]),
        node("Smelting (research)", "Unlock"),
        node("Automation (research)", "Unlock"),
        node("Widgetry (research)", "Unlock"),
    ),
    edges=(
        edge("Iron plate (recipe)", "PRODUCES", "Iron plate", amount=1.0),
        edge("Iron plate (recipe)", "CONSUMES", "Iron ore", amount=2.0),
        edge("Iron plate (recipe)", "CRAFTED_AT", "Furnace"),
        edge("Copper plate (recipe)", "PRODUCES", "Copper plate", amount=1.0),
        edge("Copper plate (recipe)", "CONSUMES", "Copper ore", amount=3.0),
        edge("Gear (recipe)", "PRODUCES", "Gear", amount=1.0),
        edge("Gear (recipe)", "CONSUMES", "Iron plate", amount=2.0),
        edge("Widget (recipe)", "PRODUCES", "Widget", amount=1.0),
        edge("Widget (recipe)", "CONSUMES", "Gear", amount=4.0),
        edge("Widget (recipe)", "CONSUMES", "Copper plate", amount=1.0),
        # No amount: the shape an item's `consumers` field produces.
        edge("Widget (recipe)", "CONSUMES", "Coal"),
        edge("Stone synthesis", "PRODUCES", "Stone", amount=1.0),
        edge("Stone synthesis", "CONSUMES", "Dust", amount=100.0),
        edge("Enriched (recipe)", "PRODUCES", "Enriched", amount=2.0),
        edge("Enriched (recipe)", "CONSUMES", "Enriched", amount=1.0),
        edge("Enriched (recipe)", "CONSUMES", "Iron plate", amount=1.0),
        edge("Smelting (research)", "UNLOCKS", "Iron plate (recipe)"),
        edge("Widgetry (research)", "UNLOCKS", "Widget (recipe)"),
        edge("Automation (research)", "REQUIRES", "Smelting (research)"),
        edge("Widgetry (research)", "REQUIRES", "Automation (research)"),
    ),
    unresolved=(),
)


@pytest.fixture(scope="module")
def driver() -> Iterator[Driver]:
    with connect(URI, *AUTH) as connected:
        apply_schema(connected)
        with connected.session() as session:
            session.run("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=f"{WIKI}:")
        GraphLoader(connected).load(FIXTURE)
        yield connected
        with connected.session() as session:
            session.run("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=f"{WIKI}:")


#: First entry is the default when a question names no property.
RETRIEVAL = RetrievalConfig(comparable_properties=["stack_size", "mining_time"])


@pytest.fixture
def retriever(driver: Driver) -> GraphRetriever:
    return GraphRetriever(driver, WIKI, retrieval=RETRIEVAL)


@pytest.fixture
def entities(driver: Driver) -> GraphEntityResolver:
    return GraphEntityResolver(driver, WIKI)


def run(
    retriever: GraphRetriever,
    entities: GraphEntityResolver,
    template: str,
    name: str,
    **kwargs: object,
) -> tuple[str, ...]:
    resolved, _ = entities.resolve_all([name])
    route = Route.model_validate(
        {
            "question": "q",
            "intent": "graph",
            "template": template,
            "entities": tuple(resolved),
            **kwargs,
        }
    )
    return retriever.run(route).lines


# -- recipe_tree ------------------------------------------------------------------


def test_recipe_tree_walks_to_raw_materials(retriever, entities):
    lines = run(retriever, entities, "recipe_tree", "Gear")
    assert "Gear: 1" in lines[0]
    assert any("Iron ore: 4 (raw)" in line for line in lines)


def test_recipe_tree_multiplies_amounts_level_by_level(retriever, entities):
    """One widget: 4 gears, 8 iron plates, 16 iron ore, plus 3 copper ore."""
    totals = run(retriever, entities, "recipe_tree", "Widget")[-1]
    assert "Copper ore 3" in totals
    assert "Iron ore 16" in totals


def test_recipe_tree_skips_an_ingredient_with_no_stated_amount(retriever, entities):
    """Multiplying by an invented 1.0 would put a wrong number in the total."""
    lines = run(retriever, entities, "recipe_tree", "Widget")
    assert not any("Coal" in line for line in lines)


def test_recipe_tree_stops_at_a_cycle_instead_of_multiplying_through_it(retriever, entities):
    """A recipe consuming what it produces is legitimate (Kovarex). Expanding it again
    multiplies a quantity by a chain leading back to itself, which is how one
    processing unit came to cost twenty million sulfuric acid."""
    lines = run(retriever, entities, "recipe_tree", "Enriched")
    assert any("(cycle, not followed)" in line for line in lines)
    assert len(lines) < 10


def test_a_profile_can_declare_an_item_raw_whatever_the_graph_says(driver, entities):
    """The ontology's 'no inbound PRODUCES means raw' rule predates the game adding
    synthesis recipes for things you mine."""
    plain = GraphRetriever(driver, WIKI)
    declared = GraphRetriever(driver, WIKI, raw_items=frozenset({"stone"}))
    resolved, _ = entities.resolve_all(["Stone"])
    route = Route(question="q", intent="graph", template="recipe_tree", entities=tuple(resolved))

    assert any("Dust" in line for line in plain.run(route).lines)
    assert declared.run(route).lines == ("Stone: 1 (raw)", "Raw materials for 1 Stone: Stone 1.")


# -- unlock_chain -----------------------------------------------------------------


def test_unlock_chain_finds_the_technology_gating_an_item(retriever, entities):
    """UNLOCKS points at the recipe, never at the item it produces."""
    lines = run(retriever, entities, "unlock_chain", "Iron plate")
    assert "gated by the technology Smelting (research)" in lines[0]


def test_unlock_chain_returns_the_whole_prerequisite_chain_in_order(retriever, entities):
    lines = run(retriever, entities, "unlock_chain", "Widget")
    assert lines[1] == (
        "Research order: Smelting (research) -> Automation (research) -> Widgetry (research)."
    )


def test_unlock_chain_accepts_a_technology_as_the_entity(retriever, entities):
    """ "What comes before widgetry" and "what gates a widget" are the same question."""
    lines = run(retriever, entities, "unlock_chain", "Widgetry (research)")
    assert "Smelting (research) -> Automation (research)" in lines[1]


# -- consumers_of -----------------------------------------------------------------


def test_consumers_of_finds_every_recipe_taking_an_item(retriever, entities):
    lines = run(retriever, entities, "consumers_of", "Iron plate")
    assert any("Gear (recipe) consumes 2 of it" in line for line in lines)


def test_consumers_of_reports_what_the_consumer_produces_and_where(retriever, entities):
    lines = run(retriever, entities, "consumers_of", "Iron ore")
    assert "at Furnace" in lines[0]
    assert "produces Iron plate" in lines[0]


def test_consumers_of_says_unstated_rather_than_inventing_an_amount(retriever, entities):
    """The half of the relationship that recipe_tree cannot use is still worth having."""
    lines = run(retriever, entities, "consumers_of", "Coal")
    assert "consumes an unstated amount of it" in lines[0]


# -- tier_compare -----------------------------------------------------------------


def test_tier_compare_orders_peers_sharing_a_category(retriever, entities):
    lines = run(retriever, entities, "tier_compare", "Iron ore", compare_by="mining_time")
    assert [line.split(":")[0] for line in lines] == ["Copper ore", "Iron ore", "Coal"]


def test_tier_compare_does_not_cross_categories(retriever, entities):
    """An ore and an intermediate are not the same kind of thing."""
    lines = run(retriever, entities, "tier_compare", "Iron ore", compare_by="stack_size")
    assert not any("Gear" in line for line in lines)


def test_tier_compare_falls_back_to_the_default_for_a_property_it_does_not_allow(
    retriever, entities
):
    """The one place a template needs a column name at runtime is a closed set, and
    the set is the profile's - not a constant naming one game's field names."""
    lines = run(retriever, entities, "tier_compare", "Gear", compare_by="'; DROP DATABASE --")
    assert lines[0].startswith("Gear: 200")


def test_tier_compare_returns_nothing_for_a_wiki_that_declares_no_properties(driver, entities):
    """Rather than crashing on a missing default, or silently ordering by whichever
    column a Factorio-shaped constant happened to name."""
    bare = GraphRetriever(driver, WIKI)
    resolved, _ = entities.resolve_all(["Gear"])
    route = Route(question="q", intent="graph", template="tier_compare", entities=tuple(resolved))
    assert bare.run(route).lines == ()


# -- entity resolution -------------------------------------------------------------


def test_an_exact_title_beats_everything_else(entities):
    resolved, unresolved = entities.resolve_all(["iron plate"])
    assert (resolved[0].id, resolved[0].matched_by) == (f"{WIKI}:Iron plate", "title")
    assert unresolved == []


def test_the_same_thing_named_twice_resolves_once(entities):
    resolved, _ = entities.resolve_all(["Iron plate", "iron plate"])
    assert len(resolved) == 1


def test_a_blank_mention_is_neither_resolved_nor_logged(entities):
    assert entities.resolve_all(["   "]) == ([], [])


def test_lucene_syntax_in_a_mention_does_not_break_the_query(entities):
    """A model that wraps an entity in quotes should miss, not raise."""
    resolved, unresolved = entities.resolve_all(['"Iron plate" AND (nonsense^2)'])
    assert resolved or unresolved  # either outcome is fine; crashing is not
