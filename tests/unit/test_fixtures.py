"""Guard the committed wikitext fixtures.

The extractor tests in Phase 2 are only meaningful if these files still contain the
shapes they were chosen for. A silent refetch that flattened an edge case would
otherwise leave the parser tests passing against nothing.
"""

from __future__ import annotations

import re

import pytest

EXPECTED = {
    "Iron_gear_wheel",
    "Infobox__Iron_gear_wheel",
    "Electronic_circuit",
    "Infobox__Electronic_circuit",
    "Oil_processing",
    "Uranium_processing",
    "Infobox__Uranium_processing",
    "Kovarex_enrichment_process",
    "Infobox__Kovarex_enrichment_process",
    "Automation_research",
    "Infobox__Automation_research",
    "Assembling_machine_2",
    "Infobox__Assembling_machine_2",
    "Infobox__Iron_axe",
    "Infobox__Yumako_tree",
}


def test_every_expected_fixture_is_present(wikitext: dict[str, str]) -> None:
    assert set(wikitext) >= EXPECTED


def test_article_transcludes_its_infobox(wikitext: dict[str, str]) -> None:
    assert "{{:Infobox:Iron gear wheel}}" in wikitext["Iron_gear_wheel"]


def test_simple_recipe_shape_survives(wikitext: dict[str, str]) -> None:
    assert "|recipe = Time, 0.5 + Iron plate, 2" in wikitext["Infobox__Iron_gear_wheel"]


def test_multi_output_recipe_with_explicit_outputs(wikitext: dict[str, str]) -> None:
    recipe = wikitext["Infobox__Kovarex_enrichment_process"]
    assert "= Uranium-235, 41 + Uranium-238, 2" in recipe, "explicit output side"


def test_fractional_amounts_are_probabilities(wikitext: dict[str, str]) -> None:
    assert "Uranium-235, 0.007" in wikitext["Infobox__Uranium_processing"]


def test_technology_unlocks_live_in_effects(wikitext: dict[str, str]) -> None:
    tech = wikitext["Infobox__Automation_research"]
    assert "|effects = Assembling machine 1 + Long-handed inserter" in tech
    assert "|cost = Time, 10 + Automation science pack, 1" in tech


def test_lowercase_infobox_template_is_represented(wikitext: dict[str, str]) -> None:
    # 21 of 590 infobox pages open with a lowercase {{infobox, so the parser
    # must not match on capitalisation.
    assert wikitext["Infobox__Automation_research"].lstrip().startswith("{{infobox")


def test_archived_page_without_prototype_type_keeps_its_recipe(
    wikitext: dict[str, str],
) -> None:
    axe = wikitext["Infobox__Iron_axe"]
    assert "prototype-type" not in axe
    assert "recipe = Time, 0.5 + Iron plate, 3 + Iron stick, 2 = Archive:Iron axe" in axe


def test_world_entity_yields_resources(wikitext: dict[str, str]) -> None:
    tree = wikitext["Infobox__Yumako_tree"]
    assert "prototype-type" not in tree
    assert "expected-resources" in tree


def test_redirect_page_is_not_mistaken_for_content(wikitext: dict[str, str]) -> None:
    # "Advanced oil processing" turned out to be a redirect; "Oil processing" is
    # the real article. Guard against re-adding a redirect as a fixture.
    for name, text in wikitext.items():
        assert not text.lstrip().upper().startswith("#REDIRECT"), f"{name} is a redirect"


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_fixtures_are_non_trivial(wikitext: dict[str, str], stem: str) -> None:
    assert len(wikitext[stem]) > 200, f"{stem} looks truncated"


def test_infobox_fixtures_open_with_the_infobox_template(wikitext: dict[str, str]) -> None:
    for stem, text in wikitext.items():
        if stem.startswith("Infobox__"):
            assert re.match(r"\{\{\s*infobox", text.strip(), re.IGNORECASE), stem
