"""The resolver: canonical ids, the Item/Recipe split, and the edge-routing rules
that are not directly stated by the ontology (CONSUMED_BY, Station, REQUIRES,
UNLOCKS). See ``resolve.py``'s own module docstring for the reasoning behind each.
"""

from __future__ import annotations

from typing import Any

from ragtorio.extract.models import Fact, Provenance
from ragtorio.harvest.models import RawRedirect
from ragtorio.ontology.resolve import EntityResolver

PROV = Provenance(wiki="factorio", page_id=1, revision_id=1, field="f")


def fact(subject: str, labels: tuple[str, ...], predicate: str, obj: Any, **props: Any) -> Fact:
    return Fact(
        subject=subject,
        subject_labels=labels,
        predicate=predicate,
        object=obj,
        props=props,
        provenance=PROV,
    )


def resolve(
    facts: list[Fact],
    redirects: list[RawRedirect] | None = None,
    aliases: dict[str, str] | None = None,
    archived: frozenset[str] = frozenset(),
) -> Any:
    return EntityResolver("factorio", facts, redirects or [], aliases or {}, archived).resolve()


def by_id(graph: Any, node_id: str) -> Any:
    return next(n for n in graph.nodes if n.id == node_id)


def test_item_with_its_own_recipe_splits_into_two_nodes() -> None:
    """Iron gear wheel-shaped: an Item page carrying a recipe field."""
    graph = resolve(
        [
            fact("Widget", ("Item",), "prop.stack_size", 100.0),
            fact("Widget", ("Item",), "prop.crafting_time", 0.5),
            fact("Widget", ("Item",), "rel.PRODUCES", "Widget", amount=1.0, probability=1.0),
        ]
    )
    item = by_id(graph, "factorio:Widget")
    recipe = by_id(graph, "factorio:Widget (recipe)")
    assert item.labels == ("Item",)
    assert item.props["stack_size"] == 100.0
    assert recipe.labels == ("Recipe",)
    assert recipe.props["crafting_time"] == 0.5
    assert any(e.from_id == recipe.id and e.to_id == item.id for e in graph.edges)


def test_standalone_recipe_page_is_not_split() -> None:
    """Kovarex-shaped: prototype-type=recipe already, one node carries everything."""
    graph = resolve(
        [
            fact("Kovarex", ("Recipe",), "prop.crafting_time", 60.0),
            fact("Kovarex", ("Recipe",), "rel.CONSUMES", "Uranium-235", amount=40.0),
        ]
    )
    assert [n.id for n in graph.nodes if "(recipe)" in n.id] == []
    assert by_id(graph, "factorio:Kovarex").labels == ("Recipe",)


def test_consumed_by_becomes_consumes_from_the_consumers_recipe_aspect() -> None:
    """'X's consumers list Y' -> Y's recipe CONSUMES X, merged with any real amount
    Y's own recipe field already produced for the same edge."""
    graph = resolve(
        [
            fact("Iron plate", ("Item",), "rel.CONSUMED_BY", "Gear"),
            fact("Gear", ("Item",), "prop.crafting_time", 1.0),
            fact("Gear", ("Item",), "rel.CONSUMES", "Iron plate", amount=2.0),
        ]
    )
    consumes = [e for e in graph.edges if e.rel_type == "CONSUMES"]
    assert len(consumes) == 1  # merged, not duplicated
    assert consumes[0].props == {"amount": 2.0}  # the real amount won, not the blank one


def test_station_label_is_added_to_whatever_crafted_at_points_at() -> None:
    graph = resolve(
        [
            fact("Assembler", ("Item",), "prop.stack_size", 50.0),
            fact("Gear", ("Item",), "prop.crafting_time", 0.5),
            fact("Gear", ("Item",), "rel.CRAFTED_AT", "Assembler"),
        ]
    )
    assert set(by_id(graph, "factorio:Assembler").labels) == {"Item", "Station"}


def test_requires_only_becomes_an_edge_between_two_unlocks() -> None:
    graph = resolve(
        [
            fact("Logistics", ("Unlock",), "rel.REQUIRES", "Automation"),
            fact("Automation", ("Unlock",), "prop.internal_name", "automation"),
            fact("Electronic circuit", ("Item",), "rel.REQUIRES", "Electronics"),
            fact("Electronics", ("Unlock",), "prop.internal_name", "electronics"),
        ]
    )
    requires = [e for e in graph.edges if e.rel_type == "REQUIRES"]
    assert len(requires) == 1
    assert requires[0].from_id == "factorio:Logistics"
    assert requires[0].to_id == "factorio:Automation"


def test_unlocks_prefers_the_targets_recipe_aspect() -> None:
    graph = resolve(
        [
            fact("Automation", ("Unlock",), "rel.UNLOCKS", "Assembler"),
            fact("Assembler", ("Item",), "prop.crafting_time", 1.0),
            fact("Assembler", ("Item",), "rel.PRODUCES", "Assembler", amount=1.0),
        ]
    )
    unlocks = next(e for e in graph.edges if e.rel_type == "UNLOCKS")
    assert unlocks.to_id == "factorio:Assembler (recipe)"


def test_redirects_and_shorthand_resolve_both_subjects_and_objects() -> None:
    graph = resolve(
        [
            fact("Gear", ("Item",), "rel.CONSUMES", "Green circuit", amount=1.0),
            fact("Electronic circuit", ("Item",), "prop.stack_size", 200.0),
        ],
        redirects=[RawRedirect(wiki="factorio", from_title="Iron gear wheel", to_title="Gear")],
        aliases={"Green circuit": "Electronic circuit"},
    )
    edge = next(e for e in graph.edges if e.rel_type == "CONSUMES")
    assert edge.to_id == "factorio:Electronic circuit"
    node = by_id(graph, "factorio:Gear")
    assert node.props["aliases"] == ["Iron gear wheel"]


def test_unresolved_objects_are_logged_not_invented() -> None:
    """COSTS, not CONSUMES: a predicate that does not itself imply a recipe split,
    to isolate "no node invented for an unknown target" from that other behaviour."""
    graph = resolve([fact("Automation", ("Unlock",), "rel.COSTS", "Nonexistent pack")])
    assert graph.nodes == (by_id(graph, "factorio:Automation"),)
    assert len(graph.unresolved) == 1
    assert graph.unresolved[0].subject == "Automation"
    assert graph.unresolved[0].object == "Nonexistent pack"


def test_archived_flag_and_introduced_in_from_history() -> None:
    graph = resolve(
        [
            fact("Old item", ("Item",), "prop.stack_size", 50.0),
            fact("Old item", ("Item",), "prop.version_event", None, version="2.0.0", text="Nerfed"),
            fact(
                "Old item",
                ("Item",),
                "prop.version_event",
                None,
                version="0.1.0",
                text="Introduced",
            ),
        ],
        archived=frozenset({"Old item"}),
    )
    node = by_id(graph, "factorio:Old item")
    assert node.props["is_archived"] is True
    assert node.props["introduced_in"] == "0.1.0"
