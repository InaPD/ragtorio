"""Read access to what indexing needs: articles, redirects, and the graph's vocabulary.

A third read-only protocol alongside ``extract``'s and ``ontology``'s, for the same
reason they are separate from each other: indexing reads one namespace of articles,
the redirect pairs it needs to canonicalise a link, and the set of titles that actually
produced facts. Widening an existing protocol to cover it would give every caller a
method it must not use, and hide which part of the crawl each phase really depends on.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

import psycopg

from ragtorio.extract.repository import PostgresPageRepository
from ragtorio.harvest.models import RawPage, RawRedirect


class ChunkSourceRepository(Protocol):
    """What indexing needs to read. Read-only, by design."""

    def articles(self, wiki: str, ns: int) -> Iterator[RawPage]:
        """Every crawled page in the article namespace."""
        ...

    def redirects(self, wiki: str) -> list[RawRedirect]:
        """Every redirect pair, so a link to an alias resolves to the canonical title."""
        ...

    def entity_titles(self, wiki: str) -> frozenset[str]:
        """Canonical titles that produced at least one fact.

        This is the graph's vocabulary. A mention that is not in it names nothing the
        graph can be queried about, which is exactly when it should not be recorded
        as a mention.
        """
        ...


class InMemoryChunkSourceRepository:
    """Pages and redirects held in memory. Backs chunker tests without a database."""

    def __init__(
        self,
        pages: Sequence[RawPage] = (),
        redirects: Sequence[RawRedirect] = (),
        entity_titles: frozenset[str] = frozenset(),
    ) -> None:
        self._pages = list(pages)
        self._redirects = list(redirects)
        self._entity_titles = entity_titles

    def articles(self, wiki: str, ns: int) -> Iterator[RawPage]:
        return (p for p in self._pages if p.wiki == wiki and p.ns == ns)

    def redirects(self, wiki: str) -> list[RawRedirect]:
        return [r for r in self._redirects if r.wiki == wiki]

    def entity_titles(self, wiki: str) -> frozenset[str]:  # noqa: ARG002 - protocol shape
        return self._entity_titles


class PostgresChunkSourceRepository:
    """Reads ``raw_page``, ``raw_redirect`` and ``fact``. The caller owns the connection."""

    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn
        self._pages = PostgresPageRepository(conn)

    def articles(self, wiki: str, ns: int) -> Iterator[RawPage]:
        return self._pages.by_namespace(wiki, ns)

    def redirects(self, wiki: str) -> list[RawRedirect]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT from_title, to_title FROM raw_redirect WHERE wiki = %s", (wiki,))
            rows = cur.fetchall()
        return [
            RawRedirect.model_validate({"wiki": wiki, "from_title": f, "to_title": t})
            for f, t in rows
        ]

    def entity_titles(self, wiki: str) -> frozenset[str]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT subject FROM fact WHERE wiki = %s", (wiki,))
            return frozenset(str(row[0]) for row in cur.fetchall())
