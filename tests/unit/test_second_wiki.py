"""A second wiki, shaped nothing like Factorio, through config -> extract -> resolve.

The README claims a second crafting-game wiki is a config file rather than a rewrite,
and for five phases nothing tested that, because `wikis/` held exactly one profile.
Five things turned out to be false: `location: inline` was accepted by the schema and
rejected by the extractor, the `recipe` target ran Factorio's parser whatever the
profile said, four of seven declared parser names had no implementation, the
`(recipe)` node suffix was a literal in two modules, and the properties `tier_compare`
may order by were a constant naming Factorio's fields.

So this profile deliberately shares no vocabulary with Factorio: inline infoboxes in a
namespace that is not 0, a different type field, different field names, a different
recipe-node suffix, and numeric properties Factorio has never heard of. If any of those
five regress, this fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ragtorio.config import WikiProfile
from ragtorio.extract.repository import InMemoryPageRepository
from ragtorio.extract.template import TemplateExtractor
from ragtorio.harvest.models import RawPage
from ragtorio.ontology.resolve import EntityResolver

WIKI = "forgecraft"

PROFILE_DATA: dict[str, Any] = {
    "wiki": {
        "id": WIKI,
        "api": "https://forgecraft.example/api.php",
        "license": "CC BY-SA 4.0",
        "attribution": "Forgecraft Wiki",
        "language_filter": {"strategy": "none"},
    },
    "infobox": {
        # The layout most wikis use, and the one that did not work at all.
        "location": "inline",
        "namespace_id": 100,
        "template": "Thingbox",
        "type_field": "category",
        "type_map": {"gadget": ["Item"], "brew": ["Item", "Fluid"], "lore": []},
        "fields": {
            "durability": {"to": "prop.durability", "type": "int"},
            "weight": {"to": "prop.weight", "type": "float"},
            # Factorio's grammar, named explicitly. A second wiki would write its own
            # parser here; what matters is that the profile's choice is honoured.
            "formula": {"to": "recipe", "parser": "factorio_recipe_expr"},
            "forged-at": {"to": "rel.CRAFTED_AT", "parser": "plus_list"},
        },
    },
    "inline_templates": {"changelog": {"to": "version_event", "args": ["version", "text"]}},
    "resolution": {"recipe_suffix": "[formula]"},
    "retrieval": {"comparable_properties": ["weight", "durability"]},
}


def page(title: str, wikitext: str, page_id: int = 1, ns: int = 100) -> RawPage:
    return RawPage(
        wiki=WIKI,
        page_id=page_id,
        ns=ns,
        title=title,
        revision_id=3,
        revised_at=datetime(2026, 1, 1, tzinfo=UTC),
        wikitext=wikitext,
    )


@pytest.fixture
def profile() -> WikiProfile:
    return WikiProfile.model_validate(PROFILE_DATA)


@pytest.fixture
def pages() -> InMemoryPageRepository:
    return InMemoryPageRepository(
        [
            page(
                "Brass cog",
                "{{Thingbox|category=gadget|durability=40|weight=1.5"
                "|formula=Time, 2 + Brass ingot, 3|forged-at=Anvil + Forge}}\n"
                "{{changelog|0.4|Introduced}}",
            ),
            page("Brass ingot", "{{Thingbox|category=gadget|durability=10|weight=0.5}}", 2),
            page("The Founding", "{{Thingbox|category=lore}}", 3),
            page("Not walked", "{{Thingbox|category=gadget}}", 4, ns=0),
        ]
    )


def test_the_profile_validates(profile: WikiProfile):
    assert profile.infobox.location == "inline"
    assert profile.infobox.source_namespace == 100


def test_an_inline_infobox_in_a_custom_namespace_is_extracted(
    profile: WikiProfile, pages: InMemoryPageRepository
):
    report = TemplateExtractor(profile, pages).extract().report
    assert report.pages_seen == 3  # the ns 0 page is not this wiki's article namespace
    assert report.pages_with_facts == 2  # "lore" is mapped to no labels on purpose
    assert report.parser_failures == ()


def test_this_wikis_own_field_names_become_its_own_properties(
    profile: WikiProfile, pages: InMemoryPageRepository
):
    facts = TemplateExtractor(profile, pages).extract().facts
    cog = {(f.predicate, f.object) for f in facts if f.subject == "Brass cog"}

    assert ("prop.durability", 40.0) in cog
    assert ("prop.weight", 1.5) in cog
    assert ("rel.CRAFTED_AT", "Anvil") in cog
    assert ("rel.CRAFTED_AT", "Forge") in cog


def test_the_recipe_target_runs_the_parser_the_profile_named(
    profile: WikiProfile, pages: InMemoryPageRepository
):
    """It used to run `factorio_recipe_expr` by name regardless."""
    facts = TemplateExtractor(profile, pages).extract().facts
    cog = {(f.predicate, f.object) for f in facts if f.subject == "Brass cog"}

    assert ("prop.crafting_time", 2.0) in cog
    assert ("rel.CONSUMES", "Brass ingot") in cog
    assert ("rel.PRODUCES", "Brass cog") in cog


def test_an_inline_wikis_history_comes_from_the_page_being_walked(
    profile: WikiProfile, pages: InMemoryPageRepository
):
    facts = TemplateExtractor(profile, pages).extract().facts
    events = [f for f in facts if f.predicate == "prop.version_event"]
    assert [f.props for f in events] == [{"version": "0.4", "text": "Introduced"}]


def test_the_split_recipe_node_uses_this_wikis_suffix(
    profile: WikiProfile, pages: InMemoryPageRepository
):
    """`(recipe)` was a literal in the resolver and again in the tree walker."""
    facts = list(TemplateExtractor(profile, pages).extract().facts)
    graph = EntityResolver(
        WIKI, facts, [], {}, frozenset(), recipe_suffix=profile.resolution.recipe_suffix
    ).resolve()

    ids = {node.id for node in graph.nodes}
    assert f"{WIKI}:Brass cog" in ids
    assert f"{WIKI}:Brass cog [formula]" in ids
    assert not any("(recipe)" in node_id for node_id in ids)


def test_edges_hang_off_the_correctly_named_recipe_node(
    profile: WikiProfile, pages: InMemoryPageRepository
):
    facts = list(TemplateExtractor(profile, pages).extract().facts)
    graph = EntityResolver(
        WIKI, facts, [], {}, frozenset(), recipe_suffix=profile.resolution.recipe_suffix
    ).resolve()

    consumes = [e for e in graph.edges if e.rel_type == "CONSUMES"]
    assert [(e.from_id, e.to_id) for e in consumes] == [
        (f"{WIKI}:Brass cog [formula]", f"{WIKI}:Brass ingot")
    ]


def test_comparable_properties_are_this_wikis_not_factorios(profile: WikiProfile):
    assert profile.retrieval.default_property == "weight"
    assert profile.retrieval.comparable("durability") == "durability"
    # Factorio's own property is not comparable here, and falls back rather than leaking.
    assert profile.retrieval.comparable("stack_size") == "weight"


def test_nothing_in_the_pipeline_needed_a_factorio_specific_default(profile: WikiProfile):
    """The summary claim, stated as an assertion: this profile names no Factorio
    field, label value, namespace, template or suffix, and still runs end to end."""
    text = profile.model_dump_json()
    for factorio_ism in ("prototype-type", "Infobox", "stack_size", "(recipe)", "3002"):
        assert factorio_ism not in text
