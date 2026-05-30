"""A few behaviours the real fixtures don't happen to exercise: the construction-time
guards, and that one field's parser failure doesn't block the rest of the page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ragtorio.config import WikiProfile
from ragtorio.extract.repository import InMemoryPageRepository
from ragtorio.extract.template import TemplateExtractor
from ragtorio.harvest.models import RawPage

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def infobox_page(wikitext: str) -> RawPage:
    return RawPage(
        wiki="testwiki",
        page_id=1,
        ns=3002,
        title="Infobox:Widget",
        revision_id=1,
        revised_at=NOW,
        wikitext=wikitext,
    )


def build_profile(minimal_profile_data: dict[str, Any], **infobox_overrides: Any) -> WikiProfile:
    data = {
        **minimal_profile_data,
        "wiki": {**minimal_profile_data["wiki"], "id": "testwiki"},
        "infobox": {
            "location": "separate_namespace",
            "namespace_id": 3002,
            "template": "Infobox",
            "type_field": "kind",
            "type_map": {"widget": ["Item"]},
            **infobox_overrides,
        },
    }
    return WikiProfile.model_validate(data)


def test_inline_location_is_rejected(minimal_profile_data: dict[str, Any]) -> None:
    """Phase 2 only implements the separate-namespace walk; must fail loudly, not
    silently find zero pages."""
    profile = WikiProfile.model_validate(minimal_profile_data)  # location: inline
    with pytest.raises(ValueError, match="separate_namespace"):
        TemplateExtractor(profile, InMemoryPageRepository())


def test_unimplemented_parser_name_is_rejected_at_construction(
    minimal_profile_data: dict[str, Any],
) -> None:
    profile = build_profile(
        minimal_profile_data, fields={"foo": {"to": "prop.foo", "parser": "name_template"}}
    )
    with pytest.raises(ValueError, match="name_template"):
        TemplateExtractor(profile, InMemoryPageRepository())


def test_one_failing_field_does_not_block_the_others_on_the_same_page(
    minimal_profile_data: dict[str, Any],
) -> None:
    profile = build_profile(
        minimal_profile_data,
        fields={
            "recipe": {"to": "recipe", "parser": "factorio_recipe_expr"},
            "internal-name": {"to": "prop.internal_name"},
        },
    )
    pages = InMemoryPageRepository(
        [infobox_page("{{Infobox|kind=widget|recipe=garbage|internal-name=widget-1}}")]
    )
    result = TemplateExtractor(profile, pages).extract()
    assert any(f.predicate == "prop.internal_name" for f in result.facts)
    (failure,) = result.report.parser_failures
    assert failure.field == "recipe"
    assert "does not start with 'Time" in failure.error
