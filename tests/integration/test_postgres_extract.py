"""The Postgres-backed repository and fact store, against a real database.

The type coercion at the read boundary and the jsonb round-trip for ``object``/
``props`` are exactly what a fake would get wrong. Skipped, not failed, when no
database is reachable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import psycopg
import pytest

from ragtorio.db.connect import apply_schema, connect
from ragtorio.extract.models import Fact, Provenance
from ragtorio.extract.postgres import PostgresFactStore
from ragtorio.extract.repository import PostgresPageRepository
from ragtorio.harvest.models import RawPage
from ragtorio.harvest.postgres import PostgresHarvestStore

#: A database of its own. These tests TRUNCATE, and pointing them at the one
#: `ragtorio harvest` writes to means a `make test` silently destroys a crawl.
DSN = os.environ.get(
    "RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio_test"
)
NOW = datetime(2026, 1, 1, tzinfo=UTC)
TABLES = ("fact", "raw_category", "raw_redirect", "raw_page", "crawl_run")

type Conn = psycopg.Connection[tuple[object, ...]]


def _reachable() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason=f"no Postgres at {DSN}"),
]


@pytest.fixture
def conn() -> Iterator[Conn]:
    with connect(DSN) as connection:
        apply_schema(connection)
        with connection.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY")
        connection.commit()
        yield connection


def page(page_id: int, title: str, ns: int = 0) -> RawPage:
    return RawPage(
        wiki="factorio",
        page_id=page_id,
        ns=ns,
        title=title,
        revision_id=1,
        revised_at=NOW,
        wikitext=f"wikitext for {title}",
    )


def test_page_repository_reads_by_namespace_and_title(conn: Conn) -> None:
    PostgresHarvestStore(conn).save_pages(
        [page(1, "Iron plate", ns=0), page(2, "Infobox:Iron plate", ns=3002)]
    )
    repo = PostgresPageRepository(conn)
    assert [p.title for p in repo.by_namespace("factorio", 3002)] == ["Infobox:Iron plate"]
    assert repo.by_title("factorio", "Iron plate") is not None
    assert repo.by_title("factorio", "Nothing here") is None


def test_fact_store_round_trips_object_and_props_types_and_replaces_on_rerun(conn: Conn) -> None:
    facts = [
        Fact(
            subject="Iron plate",
            subject_labels=("Item",),
            predicate="prop.stack_size",
            object=100.0,
            provenance=Provenance(wiki="factorio", page_id=1, revision_id=1, field="s"),
        ),
        Fact(
            subject="X",
            subject_labels=("Item",),
            predicate="prop.version_event",
            object=None,
            props={"version": "1.0", "text": "intro"},
            provenance=Provenance(wiki="factorio", page_id=2, revision_id=1, field="h"),
        ),
    ]
    store = PostgresFactStore(conn)
    store.replace_all("factorio", facts)

    with conn.cursor() as cur:
        cur.execute("SELECT subject, object, props FROM fact ORDER BY fact_id")
        rows = cur.fetchall()
    assert rows[0] == ("Iron plate", 100.0, {})
    assert rows[1] == ("X", None, {"version": "1.0", "text": "intro"})
    assert store.count("factorio") == 2

    store.replace_all("factorio", facts[:1])
    assert store.count("factorio") == 1
