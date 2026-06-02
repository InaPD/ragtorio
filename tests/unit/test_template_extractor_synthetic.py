"""A few behaviours the real fixtures don't happen to exercise: the construction-time
guards, and that one field's parser failure doesn't block the rest of the page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

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


def inline_page(title: str, wikitext: str, page_id: int = 1, ns: int = 0) -> RawPage:
    return RawPage(
        wiki="testwiki",
        page_id=page_id,
        ns=ns,
        title=title,
        revision_id=1,
        revised_at=datetime(2026, 1, 1, tzinfo=UTC),
        wikitext=wikitext,
    )


class TestInlineInfoboxes:
    """The layout most wikis use: the infobox is a template call in the article.

    Factorio's separate `Infobox:` namespace is the unusual case, and for five phases
    `inline` was a value the profile schema accepted and the extractor rejected.
    """

    def profile(self, minimal_profile_data: dict[str, Any], **overrides: Any) -> WikiProfile:
        data = {
            **minimal_profile_data,
            "wiki": {**minimal_profile_data["wiki"], "id": "testwiki"},
            "infobox": {
                "location": "inline",
                "template": "Infobox",
                "type_field": "kind",
                "type_map": {"widget": ["Item"]},
                **overrides,
            },
        }
        return WikiProfile.model_validate(data)

    def test_the_article_is_walked_and_its_own_title_is_the_subject(
        self, minimal_profile_data: dict[str, Any]
    ) -> None:
        profile = self.profile(
            minimal_profile_data, fields={"size": {"to": "prop.size", "type": "int"}}
        )
        pages = InMemoryPageRepository([inline_page("Widget", "{{Infobox|kind=widget|size=5}}")])

        result = TemplateExtractor(profile, pages).extract()

        assert result.report.pages_seen == 1
        assert [(f.subject, f.predicate, f.object) for f in result.facts] == [
            ("Widget", "prop.size", 5.0)
        ]

    def test_a_title_containing_a_colon_keeps_it(
        self, minimal_profile_data: dict[str, Any]
    ) -> None:
        """`_strip_namespace` is right for `Infobox:Widget` and wrong for a real title
        like `Tutorial:Circuit network cookbook`."""
        profile = self.profile(minimal_profile_data)
        pages = InMemoryPageRepository([inline_page("Tutorial:Widgets", "{{Infobox|kind=widget}}")])

        result = TemplateExtractor(profile, pages).extract()
        assert result.report.pages_seen == 1
        assert result.report.unclassified_pages == ()

    def test_history_comes_from_the_same_page_with_no_second_lookup(
        self, minimal_profile_data: dict[str, Any]
    ) -> None:
        data = {
            **minimal_profile_data,
            "wiki": {**minimal_profile_data["wiki"], "id": "testwiki"},
            "infobox": {
                "location": "inline",
                "template": "Infobox",
                "type_field": "kind",
                "type_map": {"widget": ["Item"]},
            },
            "inline_templates": {"history": {"to": "version_event", "args": ["version", "text"]}},
        }
        profile = WikiProfile.model_validate(data)
        pages = InMemoryPageRepository(
            [inline_page("Widget", "{{Infobox|kind=widget}}\n{{history|1.2|Introduced}}")]
        )

        facts = TemplateExtractor(profile, pages).extract().facts
        events = [f for f in facts if f.predicate == "prop.version_event"]
        assert [f.props for f in events] == [{"version": "1.2", "text": "Introduced"}]

    def test_the_article_namespace_is_mainspace_unless_the_profile_says_otherwise(
        self, minimal_profile_data: dict[str, Any]
    ) -> None:
        profile = self.profile(minimal_profile_data, namespace_id=14)
        pages = InMemoryPageRepository(
            [
                inline_page("Ignored", "{{Infobox|kind=widget}}", page_id=1, ns=0),
                inline_page("Walked", "{{Infobox|kind=widget}}", page_id=2, ns=14),
            ]
        )

        result = TemplateExtractor(profile, pages).extract()
        assert result.report.pages_seen == 1
        assert {f.subject for f in result.facts} == set()  # no fields mapped, but it was seen


def test_an_unknown_parser_name_is_rejected_when_the_profile_loads(
    minimal_profile_data: dict[str, Any],
) -> None:
    """Earlier this was caught one step later, at extractor construction, because
    config kept its own list of parser names that had drifted from the registry."""
    with pytest.raises(ValidationError, match="unknown parser"):
        build_profile(
            minimal_profile_data,
            fields={"foo": {"to": "prop.foo", "parser": "no_such_parser"}},
        )


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
