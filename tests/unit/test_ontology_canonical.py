"""Title canonicalisation: the one thing resolution and indexing must agree on."""

from __future__ import annotations

from ragtorio.harvest.models import RawRedirect
from ragtorio.ontology.canonical import TitleCanonicalizer


def redirect(from_title: str, to_title: str) -> RawRedirect:
    return RawRedirect(wiki="factorio", from_title=from_title, to_title=to_title)


def test_redirects_and_aliases_resolve_through_one_lookup():
    titles = TitleCanonicalizer(
        [redirect("Green circuits", "Electronic circuit")],
        {"green circuit": "Electronic circuit"},
    )
    assert titles.canonical("Green circuits") == "Electronic circuit"
    assert titles.canonical("green circuit") == "Electronic circuit"


def test_canonical_is_case_insensitive_but_preserves_the_target_casing():
    titles = TitleCanonicalizer(aliases={"Red Belt": "Fast transport belt"})
    assert titles.canonical("red belt") == "Fast transport belt"


def test_an_unknown_title_comes_back_unchanged():
    """Logging what failed to resolve needs the title someone actually wrote."""
    assert TitleCanonicalizer().canonical("Nothing here") == "Nothing here"


def test_link_normalisation_follows_mediawiki_title_rules():
    titles = TitleCanonicalizer()
    assert titles.normalize_link("crude oil") == "Crude oil"
    assert titles.normalize_link("Crude_oil") == "Crude oil"
    assert titles.normalize_link("crude   oil ") == "Crude oil"
    assert titles.normalize_link("Oil processing#Cracking") == "Oil processing"


def test_resolve_link_normalises_before_looking_up_the_alias():
    """A wikilink writes 'green_circuit'; the alias table knows 'green circuit'."""
    titles = TitleCanonicalizer(aliases={"green circuit": "Electronic circuit"})
    assert titles.resolve_link("green_circuit") == "Electronic circuit"


def test_aliases_of_collects_every_source_pointing_at_a_title():
    titles = TitleCanonicalizer(
        [redirect("Green circuits", "Electronic circuit")],
        {"green circuit": "Electronic circuit"},
    )
    assert titles.aliases_of("Electronic circuit") == ["Green circuits", "green circuit"]
    assert titles.aliases_of("Iron plate") == []
