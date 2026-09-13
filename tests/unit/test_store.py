"""The in-memory store.

It backs ``--dry-run`` as well as the crawler's tests, so its behaviour has to match
the Postgres one closely enough that a dry run means something.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ragtorio.harvest.models import CrawlStats, RawPage, RawRedirect
from ragtorio.harvest.postgres import _as_int
from ragtorio.harvest.store import InMemoryHarvestStore

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def page(page_id: int, *, wiki: str = "factorio", ns: int = 0, revision_id: int = 1) -> RawPage:
    return RawPage(
        wiki=wiki,
        page_id=page_id,
        ns=ns,
        title=f"Page {page_id}",
        revision_id=revision_id,
        revised_at=NOW,
        wikitext="body",
    )


class TestInMemoryHarvestStore:
    def test_runs_get_sequential_ids(self) -> None:
        store = InMemoryHarvestStore()
        assert store.start_run("factorio") == 1
        assert store.start_run("factorio") == 2

    def test_finish_run_records_stats_and_error(self) -> None:
        store = InMemoryHarvestStore()
        run_id = store.start_run("factorio")
        stats = CrawlStats(listed=3)
        store.finish_run(run_id, stats, error="boom")
        assert store.runs[0]["stats"] == stats
        assert store.runs[0]["error"] == "boom"

    def test_save_pages_upserts_on_page_id(self) -> None:
        store = InMemoryHarvestStore()
        store.save_pages([page(1, revision_id=1)])
        store.save_pages([page(1, revision_id=2)])
        assert len(store.pages) == 1
        assert store.pages[("factorio", 1)].revision_id == 2

    def test_known_revisions_is_scoped_to_one_wiki(self) -> None:
        store = InMemoryHarvestStore()
        store.save_pages([page(1, revision_id=5), page(2, wiki="stardew", revision_id=9)])
        assert store.known_revisions("factorio") == {1: 5}

    def test_namespace_counts(self) -> None:
        store = InMemoryHarvestStore()
        store.save_pages([page(1, ns=0), page(2, ns=0), page(3, ns=3002)])
        assert store.namespace_counts("factorio") == {0: 2, 3002: 1}

    def test_redirects_upsert_on_source_title(self) -> None:
        store = InMemoryHarvestStore()
        store.save_redirects([RawRedirect(wiki="factorio", from_title="A", to_title="B")])
        store.save_redirects([RawRedirect(wiki="factorio", from_title="A", to_title="C")])
        assert len(store.redirects) == 1
        assert store.redirects[("factorio", "A")].to_title == "C"


class TestCrawlStatsMerge:
    def test_counters_add_and_namespaces_union(self) -> None:
        merged = CrawlStats(listed=2, fetched=1, by_namespace={0: 2}).merge(
            CrawlStats(listed=3, fetched=3, by_namespace={3002: 3})
        )
        assert merged.listed == 5
        assert merged.fetched == 4
        assert merged.by_namespace == {0: 2, 3002: 3}


class TestAsInt:
    def test_passes_integers_through(self) -> None:
        assert _as_int(7) == 7

    def test_rejects_anything_else(self) -> None:
        with pytest.raises(TypeError, match="expected an integer column"):
            _as_int("7")
