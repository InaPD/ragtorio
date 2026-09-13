"""Where a crawl's output goes.

The crawler talks to this protocol, not to Postgres. That keeps the network logic
testable without a database, and it is what lets ``ragtorio harvest --dry-run`` exercise
the real crawl path against an in-memory store before anyone waits 40 minutes for a
full run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragtorio.harvest.models import CrawlStats, RawPage, RawRedirect


class HarvestStore(Protocol):
    """The persistence a crawl needs, and nothing more."""

    def start_run(self, wiki: str) -> int:
        """Open a ``crawl_run`` row and return its id."""
        ...

    def finish_run(self, run_id: int, stats: CrawlStats, error: str | None = None) -> None:
        """Close the run, recording its tally and how it ended."""
        ...

    def known_revisions(self, wiki: str) -> dict[int, int]:
        """Page id to stored revision id, so the crawler can skip what has not changed."""
        ...

    def save_pages(self, pages: Sequence[RawPage]) -> None:
        """Upsert pages and replace their category rows."""
        ...

    def save_redirects(self, redirects: Sequence[RawRedirect]) -> None:
        """Upsert redirects."""
        ...

    def namespace_counts(self, wiki: str) -> dict[int, int]:
        """Stored page count per namespace, for the exit-criteria report."""
        ...


class InMemoryHarvestStore:
    """A store that keeps everything in dicts.

    Used by ``--dry-run`` and by the crawler's tests. It is deliberately in ``src`` and
    not in ``tests``: a dry run is a real feature, and one implementation serving both
    means the tested path is the shipped path.
    """

    def __init__(self) -> None:
        self.pages: dict[tuple[str, int], RawPage] = {}
        self.redirects: dict[tuple[str, str], RawRedirect] = {}
        self.runs: list[dict[str, object]] = []

    def start_run(self, wiki: str) -> int:
        self.runs.append({"wiki": wiki, "stats": None, "error": None})
        return len(self.runs)

    def finish_run(self, run_id: int, stats: CrawlStats, error: str | None = None) -> None:
        self.runs[run_id - 1]["stats"] = stats
        self.runs[run_id - 1]["error"] = error

    def known_revisions(self, wiki: str) -> dict[int, int]:
        return {
            page_id: page.revision_id
            for (stored_wiki, page_id), page in self.pages.items()
            if stored_wiki == wiki
        }

    def save_pages(self, pages: Sequence[RawPage]) -> None:
        for page in pages:
            self.pages[(page.wiki, page.page_id)] = page

    def save_redirects(self, redirects: Sequence[RawRedirect]) -> None:
        for redirect in redirects:
            self.redirects[(redirect.wiki, redirect.from_title)] = redirect

    def namespace_counts(self, wiki: str) -> dict[int, int]:
        counts: dict[int, int] = {}
        for (stored_wiki, _), page in self.pages.items():
            if stored_wiki == wiki:
                counts[page.ns] = counts.get(page.ns, 0) + 1
        return counts
