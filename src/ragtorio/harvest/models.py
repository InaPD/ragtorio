"""What a crawl produces.

These are validated rather than plain dataclasses because every field arrives from
someone else's API. A page with no revision, a title that is not a string, a revision id
that came back as ``null`` -- all of that is cheaper to reject here than to debug three
phases later when the graph is mysteriously missing a branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PageStub(BaseModel):
    """A page's identity and current revision, without its text.

    The catalogue pass collects these for every page so the crawler can compare
    revision ids and fetch content only for what changed.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    page_id: int
    ns: int
    title: str
    revision_id: int
    revised_at: datetime


class RawPage(BaseModel):
    """One page's current revision, as fetched."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    wiki: str
    page_id: int
    ns: int
    title: str
    revision_id: int
    revised_at: datetime
    wikitext: str
    categories: tuple[str, ...] = ()


class RawRedirect(BaseModel):
    """A redirect, which Phase 3 turns into a node alias."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    wiki: str
    from_title: str
    to_title: str


class CrawlStats(BaseModel):
    """The tally a crawl reports, and the evidence for the phase exit criteria.

    ``unchanged`` is the interesting one on a re-run: it should account for everything
    listed, and ``fetched`` should be zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    listed: int = 0
    translations: int = 0
    unchanged: int = 0
    fetched: int = 0
    redirects: int = 0
    by_namespace: Mapping[int, int] = Field(default_factory=dict)

    def merge(self, other: CrawlStats) -> CrawlStats:
        """Combine two namespaces' tallies. Namespace counts are disjoint, so they union."""
        return CrawlStats(
            listed=self.listed + other.listed,
            translations=self.translations + other.translations,
            unchanged=self.unchanged + other.unchanged,
            fetched=self.fetched + other.fetched,
            redirects=self.redirects + other.redirects,
            by_namespace={**self.by_namespace, **other.by_namespace},
        )
