"""The extractor, run against the real committed fixtures - the closest thing to a
live crawl the test suite has.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragtorio.config import load_profile
from ragtorio.extract.repository import InMemoryPageRepository
from ragtorio.extract.template import TemplateExtractor
from ragtorio.harvest.models import RawPage


@pytest.fixture
def extractor(factorio_pages: list[RawPage]) -> TemplateExtractor:
    profile = load_profile("factorio")
    return TemplateExtractor(profile, InMemoryPageRepository(factorio_pages))


def facts_for(extractor: TemplateExtractor, subject: str) -> list[Any]:
    return [f for f in extractor.extract().facts if f.subject == subject]


def test_simple_and_multi_ingredient_recipes(extractor: TemplateExtractor) -> None:
    facts = facts_for(extractor, "Iron gear wheel")
    assert any(f.predicate == "prop.internal_name" and f.object == "iron-gear-wheel" for f in facts)
    consumed = {f.object: f.props["amount"] for f in facts if f.predicate == "rel.CONSUMES"}
    assert consumed == {"Iron plate": 2.0}
    # No explicit output side: the recipe implicitly produces the item itself.
    assert any(f.predicate == "rel.PRODUCES" and f.object == "Iron gear wheel" for f in facts)

    circuit = facts_for(extractor, "Electronic circuit")
    consumed = {f.object: f.props["amount"] for f in circuit if f.predicate == "rel.CONSUMES"}
    assert consumed == {"Copper cable": 3.0, "Iron plate": 1.0}


def test_recipe_page_classification_and_self_feeding_output(extractor: TemplateExtractor) -> None:
    """Kovarex enrichment process: prototype-type=recipe, and its output feeds its own input."""
    facts = facts_for(extractor, "Kovarex enrichment process")
    assert facts[0].subject_labels == ("Recipe",)
    produced = {f.object: f.props["amount"] for f in facts if f.predicate == "rel.PRODUCES"}
    assert produced == {"Uranium-235": 41.0, "Uranium-238": 2.0}


def test_probability_based_recipe(extractor: TemplateExtractor) -> None:
    facts = facts_for(extractor, "Uranium processing")
    produced = {f.object: f.props for f in facts if f.predicate == "rel.PRODUCES"}
    assert produced["Uranium-235"] == {"amount": 1.0, "probability": 0.007}
    assert produced["Uranium-238"] == {"amount": 1.0, "probability": 0.993}


def test_technology_classified_as_unlock(extractor: TemplateExtractor) -> None:
    """Also proves the lowercase '{{infobox' template variant is still found."""
    facts = facts_for(extractor, "Automation (research)")
    assert facts[0].subject_labels == ("Unlock",)
    assert {f.object for f in facts if f.predicate == "rel.UNLOCKS"} == {
        "Assembling machine 1",
        "Long-handed inserter",
    }
    costs = [f for f in facts if f.predicate == "rel.COSTS"]
    assert costs[0].object == "Automation science pack"
    assert costs[0].props == {"amount": 1.0}


def test_archived_item_with_explicit_bare_output(extractor: TemplateExtractor) -> None:
    """Iron axe: no prototype-type at all (classified via type_rules); output side
    names a page with no amount."""
    facts = facts_for(extractor, "Iron axe")
    assert facts[0].subject_labels == ("Item",)
    produced = [f for f in facts if f.predicate == "rel.PRODUCES"]
    assert produced[0].object == "Archive:Iron axe"
    assert produced[0].props["amount"] == 1.0


def test_world_entity_with_no_recipe_is_unclassified(factorio_pages: list[RawPage]) -> None:
    """Yumako tree has no prototype-type, no recipe, no cost, no allows: not in the
    ontology at all, and that is a correct outcome, not a failure."""
    profile = load_profile("factorio")
    result = TemplateExtractor(profile, InMemoryPageRepository(factorio_pages)).extract()
    assert "Infobox:Yumako tree" in result.report.unclassified_pages
    assert not any(f.subject == "Yumako tree" for f in result.facts)


def test_history_events_are_walked_from_the_article_not_the_infobox(
    extractor: TemplateExtractor,
) -> None:
    """Also covers a capital-H '{{History}}' template (Kovarex's article uses it)."""
    facts = facts_for(extractor, "Iron gear wheel")
    events = [f for f in facts if f.predicate == "prop.version_event"]
    assert {e.props["version"] for e in events} == {"2.0.7", "0.10.0", "0.7.1", "0.1.0"}
    assert events[0].object is None

    kovarex_events = [
        f
        for f in facts_for(extractor, "Kovarex enrichment process")
        if f.predicate == "prop.version_event"
    ]
    assert len(kovarex_events) == 3


def test_coverage_report_and_unknown_param_bookkeeping(extractor: TemplateExtractor) -> None:
    report = extractor.extract().report
    assert report.pages_seen == 8  # every Infobox: page in factorio_pages
    assert report.pages_with_facts == 7  # every one except Yumako tree
    assert report.parser_failures == ()
    # "category" is present but not a mapped field: a real unknown parameter.
    assert report.unknown_params["category"] == 7
    # "allows" is present and used by a type_rule, so it must not count as unknown
    # even though it is deliberately never mapped to an edge.
    assert "allows" not in report.unknown_params
    assert "prototype-type" not in report.unknown_params
