"""One build, end to end over the committed fixtures, with no model and no database."""

from __future__ import annotations

import pytest

from ragtorio.config import WikiProfile, load_profile
from ragtorio.harvest.models import RawPage, RawRedirect
from ragtorio.index.build import build_index
from ragtorio.index.embed import HashingEmbedder
from ragtorio.index.repository import InMemoryChunkSourceRepository


@pytest.fixture
def profile() -> WikiProfile:
    return load_profile("factorio")


@pytest.fixture
def articles(factorio_pages: list[RawPage]) -> list[RawPage]:
    return [page for page in factorio_pages if page.ns == 0]


def test_a_build_chunks_every_article_and_embeds_each_chunk(
    profile: WikiProfile, articles: list[RawPage]
):
    source = InMemoryChunkSourceRepository(articles)
    result = build_index(profile, source, HashingEmbedder(dimensions=64))

    assert result.report.pages_seen == len(articles)
    assert result.report.chunks == len(result.chunks)
    assert all(len(stored.embedding) == 64 for stored in result.chunks)


def test_only_the_configured_namespace_is_indexed(
    profile: WikiProfile, factorio_pages: list[RawPage]
):
    """Infobox pages are the graph's input, not prose, and would be noise here."""
    source = InMemoryChunkSourceRepository(factorio_pages)
    result = build_index(profile, source, HashingEmbedder(dimensions=32))

    assert result.report.pages_seen == sum(1 for page in factorio_pages if page.ns == 0)
    assert not any(":Infobox:" in stored.chunk.title for stored in result.chunks)


def test_the_limit_stops_the_build_early(profile: WikiProfile, articles: list[RawPage]):
    source = InMemoryChunkSourceRepository(articles)
    result = build_index(profile, source, HashingEmbedder(dimensions=32), limit=2)
    assert result.report.pages_seen == 2


def test_mentions_are_restricted_to_titles_the_graph_knows(
    profile: WikiProfile, articles: list[RawPage]
):
    source = InMemoryChunkSourceRepository(articles, entity_titles=frozenset({"Crude oil"}))
    result = build_index(profile, source, HashingEmbedder(dimensions=32))

    mentioned = {
        entity_id for stored in result.chunks for entity_id in stored.chunk.mentioned_entity_ids
    }
    assert mentioned == {"factorio:Crude oil"}


def test_a_redirect_in_the_crawl_resolves_a_link_to_its_target(
    profile: WikiProfile, articles: list[RawPage]
):
    source = InMemoryChunkSourceRepository(
        articles,
        redirects=[RawRedirect(wiki="factorio", from_title="Oil refinery", to_title="Refinery")],
        entity_titles=frozenset({"Refinery"}),
    )
    result = build_index(profile, source, HashingEmbedder(dimensions=32))

    mentioned = {
        entity_id for stored in result.chunks for entity_id in stored.chunk.mentioned_entity_ids
    }
    assert mentioned == {"factorio:Refinery"}


def test_the_report_counts_articles_that_yielded_nothing(profile: WikiProfile):
    """Half the mainspace producing no chunks is a chunker bug; this is how it shows."""
    source = InMemoryChunkSourceRepository([_stub(profile)])
    result = build_index(profile, source, HashingEmbedder(dimensions=32))

    assert result.report.pages_seen == 1
    assert result.report.pages_without_chunks == 1
    assert result.report.chunks == 0
    assert result.report.mention_coverage == 0.0
    assert result.report.mean_tokens == 0.0


def test_the_report_measures_mention_coverage(profile: WikiProfile, articles: list[RawPage]):
    source = InMemoryChunkSourceRepository(articles)
    report = build_index(profile, source, HashingEmbedder(dimensions=32)).report

    assert 0.0 < report.mention_coverage <= 1.0
    assert report.mean_tokens > 0


def test_the_report_names_the_provider_an_index_was_built_with(
    profile: WikiProfile, articles: list[RawPage]
):
    """An index is only comparable with itself: knowing which model wrote it matters."""
    source = InMemoryChunkSourceRepository(articles)
    report = build_index(profile, source, HashingEmbedder(dimensions=64)).report
    assert report.provider == "hashing/64"
    assert report.dimensions == 64


def test_progress_is_reported_per_batch(profile: WikiProfile, articles: list[RawPage]):
    messages: list[str] = []
    source = InMemoryChunkSourceRepository(articles)
    build_index(profile, source, HashingEmbedder(dimensions=32), progress=messages.append)
    assert messages and all("embedded" in message for message in messages)


def test_a_provider_that_breaks_its_own_promise_fails_at_the_first_chunk(
    profile: WikiProfile, articles: list[RawPage]
):
    """Otherwise this surfaces thousands of rows later as a column type error."""

    class Liar(HashingEmbedder):
        @property
        def dimensions(self) -> int:
            return 999

    source = InMemoryChunkSourceRepository(articles)
    with pytest.raises(ValueError, match="advertises 999 dimensions"):
        build_index(profile, source, Liar(dimensions=32))


def _stub(profile: WikiProfile) -> RawPage:
    """An article that is nothing but a navigation template."""
    from datetime import UTC, datetime

    return RawPage(
        wiki=profile.wiki.id,
        page_id=99,
        ns=profile.index.namespace_id,
        title="Stub",
        revision_id=1,
        revised_at=datetime(2026, 1, 1, tzinfo=UTC),
        wikitext="{{Languages}}\n{{IntermediateNav}}\n",
    )


def test_pages_excluded_by_title_are_counted_and_not_indexed(profile: WikiProfile):
    """23 changelog pages produced 42% of the first real index built here."""
    from datetime import UTC, datetime

    changelog = RawPage(
        wiki="factorio",
        page_id=50,
        ns=0,
        title="Version history/1.1.0",
        revision_id=1,
        revised_at=datetime(2026, 1, 1, tzinfo=UTC),
        wikitext="Bugfixes: fixed a crash when a train stopped at a station in the rain.",
    )
    source = InMemoryChunkSourceRepository([changelog])
    report = build_index(profile, source, HashingEmbedder(dimensions=32)).report

    assert report.pages_seen == 1
    assert report.pages_excluded == 1
    assert report.chunks == 0
