"""Recording what the router decided.

Separate from the retrieval path that produces it, and behind a protocol, for one
practical reason: the ``ask`` command must still answer when Postgres is unavailable.
A logging failure is not a reason to fail a question, so the in-memory implementation
is a real fallback rather than only a test double.
"""

from __future__ import annotations

from typing import Protocol

import psycopg

from ragtorio.retrieve.models import RetrievedContext

_INSERT = """
INSERT INTO routing_log (
    wiki, question, intent, template, entities, unresolved,
    confidence, downgraded, chunk_ids, latency_ms
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class RoutingLog(Protocol):
    """Where a routing decision goes."""

    def record(self, wiki: str, context: RetrievedContext) -> None:
        """Store one question's route and what it retrieved."""
        ...

    def count(self, wiki: str) -> int:
        """How many decisions are stored for ``wiki``."""
        ...


class InMemoryRoutingLog:
    """Decisions in a list. Backs tests and the no-database path."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, RetrievedContext]] = []

    def record(self, wiki: str, context: RetrievedContext) -> None:
        self.entries.append((wiki, context))

    def count(self, wiki: str) -> int:
        return sum(1 for stored_wiki, _ in self.entries if stored_wiki == wiki)


class PostgresRoutingLog:
    """Writes ``routing_log``. The caller owns the connection."""

    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn

    def record(self, wiki: str, context: RetrievedContext) -> None:
        route = context.route
        with self._conn.cursor() as cur:
            cur.execute(
                _INSERT,
                (
                    wiki,
                    route.question,
                    route.intent,
                    route.template,
                    [entity.id for entity in route.entities],
                    list(route.unresolved),
                    route.confidence,
                    route.downgraded,
                    list(context.chunk_ids),
                    context.latency_ms,
                ),
            )
        self._conn.commit()

    def count(self, wiki: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM routing_log WHERE wiki = %s", (wiki,))
            row = cur.fetchone()
        assert row is not None  # pragma: no cover - count() always returns one row
        value = row[0]
        if not isinstance(value, int):
            raise TypeError(f"expected an integer column, got {type(value).__name__}")
        return value
