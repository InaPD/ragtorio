"""Read access to the raw crawl, as extraction needs it.

Extraction only ever asks for pages by namespace (to walk every infobox) or by exact
title (to find an article's own page for its ``{{history}}`` templates). It never
writes, which is why this is a separate, narrower protocol from
:class:`~ragtorio.harvest.store.HarvestStore` rather than that same interface reused.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import psycopg

from ragtorio.harvest.models import RawPage


class PageRepository(Protocol):
    """What extraction needs to read from the crawl. Read-only, by design."""

    def by_namespace(self, wiki: str, ns: int) -> Iterator[RawPage]:
        """Every page stored for one namespace, in no particular order."""
        ...

    def by_title(self, wiki: str, title: str) -> RawPage | None:
        """One page by its exact title, or ``None`` if nothing was crawled there."""
        ...


class InMemoryPageRepository:
    """Pages held in a list. Backs extractor tests without a database."""

    def __init__(self, pages: Sequence[RawPage] = ()) -> None:
        self._by_title: dict[tuple[str, str], RawPage] = {(p.wiki, p.title): p for p in pages}

    def by_namespace(self, wiki: str, ns: int) -> Iterator[RawPage]:
        return (p for p in self._by_title.values() if p.wiki == wiki and p.ns == ns)

    def by_title(self, wiki: str, title: str) -> RawPage | None:
        return self._by_title.get((wiki, title))


class PostgresPageRepository:
    """Reads ``raw_page``. The caller owns the connection."""

    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn

    def by_namespace(self, wiki: str, ns: int) -> Iterator[RawPage]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT page_id, ns, title, revision_id, revised_at, wikitext
                  FROM raw_page WHERE wiki = %s AND ns = %s
                """,
                (wiki, ns),
            )
            rows = cur.fetchall()
        for page_id, page_ns, title, revision_id, revised_at, wikitext in rows:
            # model_validate coerces the untyped columns psycopg hands back; RawPage's
            # own field types are the single source of truth for what "correct" means.
            yield RawPage.model_validate(
                {
                    "wiki": wiki,
                    "page_id": page_id,
                    "ns": page_ns,
                    "title": title,
                    "revision_id": revision_id,
                    "revised_at": revised_at,
                    "wikitext": wikitext,
                }
            )

    def by_title(self, wiki: str, title: str) -> RawPage | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT page_id, ns, revision_id, revised_at, wikitext
                  FROM raw_page WHERE wiki = %s AND title = %s
                """,
                (wiki, title),
            )
            row = cur.fetchone()
        if row is None:
            return None
        page_id, ns, revision_id, revised_at, wikitext = row
        return RawPage.model_validate(
            {
                "wiki": wiki,
                "page_id": page_id,
                "ns": ns,
                "title": title,
                "revision_id": revision_id,
                "revised_at": revised_at,
                "wikitext": wikitext,
            }
        )
