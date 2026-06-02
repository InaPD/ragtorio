"""Read access to facts and the crawl metadata entity resolution needs.

Narrower than either ``harvest``'s or ``extract``'s repositories: resolution reads
facts (the ``fact`` table), redirect pairs, and the two ways a page can be marked
archived - a dedicated namespace or a category - and nothing else.
"""

from __future__ import annotations

from typing import Protocol

import psycopg

from ragtorio.extract.models import Fact
from ragtorio.harvest.models import RawRedirect


class GraphSourceRepository(Protocol):
    """What resolution needs to read. Read-only, by design."""

    def facts(self, wiki: str) -> list[Fact]:
        """Every fact extracted for this wiki."""
        ...

    def redirects(self, wiki: str) -> list[RawRedirect]:
        """Every redirect pair, which becomes node aliases."""
        ...

    def titles_in_namespace(self, wiki: str, ns: int) -> frozenset[str]:
        """Titles crawled in one namespace, e.g. the archived-content namespace."""
        ...

    def titles_in_category(self, wiki: str, category: str) -> frozenset[str]:
        """Titles carrying one category, given its bare name (``Archived``, not
        ``Category:Archived`` - the ``Category:`` prefix is assumed English)."""
        ...

    def categories_by_title(self, wiki: str) -> dict[str, list[str]]:
        """Every page's categories, bare names, keyed by title.

        Phase 5's ``tier_compare`` is "items sharing a category, ordered by a numeric
        property", so the category has to be on the node: it is the only grouping the
        wiki gives that a player would recognise ("Intermediate products",
        "Logistics"), and recomputing it from Postgres at query time would put a
        second database in the path of a Cypher template.
        """
        ...


class PostgresGraphSourceRepository:
    """Reads ``fact``, ``raw_redirect``, ``raw_page`` and ``raw_category``."""

    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn

    def facts(self, wiki: str) -> list[Fact]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT subject, subject_labels, predicate, object, object_labels,
                       props, source_page_id, source_revision_id, source_field
                  FROM fact WHERE wiki = %s
                """,
                (wiki,),
            )
            rows = cur.fetchall()
        return [
            Fact.model_validate(
                {
                    "subject": subject,
                    "subject_labels": subject_labels,
                    "predicate": predicate,
                    "object": obj,
                    "object_labels": object_labels,
                    "props": props,
                    "provenance": {
                        "wiki": wiki,
                        "page_id": page_id,
                        "revision_id": revision_id,
                        "field": field,
                    },
                }
            )
            for (
                subject,
                subject_labels,
                predicate,
                obj,
                object_labels,
                props,
                page_id,
                revision_id,
                field,
            ) in rows
        ]

    def redirects(self, wiki: str) -> list[RawRedirect]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT from_title, to_title FROM raw_redirect WHERE wiki = %s", (wiki,))
            rows = cur.fetchall()
        return [
            RawRedirect.model_validate({"wiki": wiki, "from_title": f, "to_title": t})
            for f, t in rows
        ]

    def titles_in_namespace(self, wiki: str, ns: int) -> frozenset[str]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT title FROM raw_page WHERE wiki = %s AND ns = %s", (wiki, ns))
            return frozenset(str(row[0]) for row in cur.fetchall())

    def titles_in_category(self, wiki: str, category: str) -> frozenset[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT rp.title FROM raw_page rp
                JOIN raw_category rc ON rc.wiki = rp.wiki AND rc.page_id = rp.page_id
                WHERE rp.wiki = %s AND rc.category = %s
                """,
                (wiki, f"Category:{category}"),
            )
            return frozenset(str(row[0]) for row in cur.fetchall())

    def categories_by_title(self, wiki: str) -> dict[str, list[str]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT rp.title, rc.category FROM raw_page rp
                JOIN raw_category rc ON rc.wiki = rp.wiki AND rc.page_id = rp.page_id
                WHERE rp.wiki = %s
                ORDER BY rp.title, rc.category
                """,
                (wiki,),
            )
            rows = cur.fetchall()
        grouped: dict[str, list[str]] = {}
        for title, category in rows:
            name = str(category).removeprefix("Category:")
            grouped.setdefault(str(title), []).append(name)
        return grouped
