"""The translation filter.

Getting this wrong is expensive in both directions: too loose and the corpus is twenty
times too big, too tight and real English pages vanish from the graph.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from ragtorio.config import WikiProfile
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.language import (
    TranslationFilter,
    fetch_language_codes,
    for_profile,
)

API = "https://example.test/api.php"
CODES = frozenset({"de", "fr", "pt-br", "zh"})


def make_client() -> MediaWikiClient:
    return MediaWikiClient(API, "ua", rps=1000.0, max_retries=0, sleep=lambda _: None)


class TestTranslationFilter:
    def test_known_language_suffix_is_a_translation(self) -> None:
        assert TranslationFilter("en", CODES).is_translation("Iron plate/de")

    def test_hyphenated_code_is_a_translation(self) -> None:
        assert TranslationFilter("en", CODES).is_translation("Iron plate/pt-br")

    def test_plain_title_is_kept(self) -> None:
        assert not TranslationFilter("en", CODES).is_translation("Iron plate")

    def test_the_kept_language_is_not_a_translation(self) -> None:
        assert not TranslationFilter("en", CODES | {"en"}).is_translation("Iron plate/en")

    def test_subpage_that_is_not_a_language_is_kept(self) -> None:
        """The whole reason for consulting siteinfo: /tips is not a language."""
        assert not TranslationFilter("en", CODES).is_translation("Blueprint/tips")

    def test_infobox_namespace_prefix_is_not_a_suffix(self) -> None:
        assert not TranslationFilter("en", CODES).is_translation("Infobox:Iron plate")

    def test_strategy_none_keeps_everything(self) -> None:
        assert not TranslationFilter(None).is_translation("Iron plate/de")

    def test_without_codes_falls_back_to_shape(self) -> None:
        """When siteinfo says nothing, over-filter rather than let 20 languages in."""
        filter_ = TranslationFilter("en", frozenset())
        assert filter_.is_translation("Iron plate/de")
        assert not filter_.is_translation("Iron plate")


class TestFetchLanguageCodes:
    @respx.mock
    def test_reads_codes_from_siteinfo(self) -> None:
        respx.get(API).mock(
            return_value=httpx.Response(
                200,
                json={"query": {"languages": [{"code": "de", "name": "Deutsch"}, {"code": "fr"}]}},
            )
        )
        with make_client() as client:
            assert fetch_language_codes(client) == frozenset({"de", "fr"})

    @respx.mock
    def test_malformed_entries_are_skipped(self) -> None:
        respx.get(API).mock(
            return_value=httpx.Response(
                200, json={"query": {"languages": [{"code": "de"}, {"name": "no code"}, "junk"]}}
            )
        )
        with make_client() as client:
            assert fetch_language_codes(client) == frozenset({"de"})

    @respx.mock
    def test_api_failure_yields_empty_set(self) -> None:
        respx.get(API).mock(return_value=httpx.Response(503))
        with make_client() as client:
            assert fetch_language_codes(client) == frozenset()

    @respx.mock
    def test_api_error_yields_empty_set(self) -> None:
        respx.get(API).mock(
            return_value=httpx.Response(
                200, json={"error": {"code": "unknown_siprop", "info": "nope"}}
            )
        )
        with make_client() as client:
            assert fetch_language_codes(client) == frozenset()


class TestForProfile:
    @respx.mock
    def test_subpage_strategy_consults_the_wiki(
        self, minimal_profile_data: dict[str, object]
    ) -> None:
        data = dict(minimal_profile_data)
        wiki = dict(data["wiki"])  # type: ignore[arg-type]
        wiki["language_filter"] = {"strategy": "subpage_suffix", "keep": "en"}
        data["wiki"] = wiki
        respx.get(API).mock(
            return_value=httpx.Response(200, json={"query": {"languages": [{"code": "de"}]}})
        )
        with make_client() as client:
            built = for_profile(WikiProfile.model_validate(data), client)
        assert built.keep == "en"
        assert built.is_translation("X/de")

    @respx.mock
    def test_none_strategy_asks_the_wiki_nothing(
        self, minimal_profile_data: dict[str, object]
    ) -> None:
        route = respx.get(API).mock(return_value=httpx.Response(200, json={"query": {}}))
        with make_client() as client:
            built = for_profile(WikiProfile.model_validate(minimal_profile_data), client)
        assert built.keep is None
        assert not route.called


@pytest.mark.parametrize(
    "title",
    ["Iron plate/de", "Assembling machine 2/fr", "Nuclear power/zh"],
)
def test_real_factorio_translation_titles_are_dropped(title: str) -> None:
    assert TranslationFilter("en", CODES).is_translation(title)
