"""The Postgres implementation of :class:`~ragtorio.extract.store.FactStore`.

``object`` and ``props`` are stored as ``jsonb`` so a fact's value keeps its real type
on the way in: a string stays a string, a number stays a number, and ``None`` stays
SQL ``null`` rather than everything collapsing to text.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg
from psycopg.types.json import Json

from ragtorio.extract.models import Fact

_DELETE = "DELETE FROM fact WHERE wiki = %s"

_INSERT = """
INSERT INTO fact (
    wiki, subject, subject_labels, predicate, object, object_labels, props,
    source_page_id, source_revision_id, source_field
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class PostgresFactStore:
    """Persists facts. The caller owns the connection and its lifetime."""

    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn

    def replace_all(self, wiki: str, facts: Sequence[Fact]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_DELETE, (wiki,))
            if facts:
                cur.executemany(_INSERT, [_row(wiki, fact) for fact in facts])
        self._conn.commit()

    def count(self, wiki: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM fact WHERE wiki = %s", (wiki,))
            row = cur.fetchone()
        assert row is not None  # pragma: no cover - count() always returns one row
        value = row[0]
        if not isinstance(value, int):
            raise TypeError(f"expected an integer column, got {type(value).__name__}")
        return value


def _row(wiki: str, fact: Fact) -> tuple[object, ...]:
    return (
        wiki,
        fact.subject,
        list(fact.subject_labels),
        fact.predicate,
        Json(fact.object),
        list(fact.object_labels),
        Json(fact.props),
        fact.provenance.page_id,
        fact.provenance.revision_id,
        fact.provenance.field,
    )
