"""The Postgres store, against a real database.

Upserts, the unique title index and category replacement are exactly the things a
fake would get wrong, so these run against the container from ``docker compose up``.
They are skipped, not failed, when no database is reachable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import psycopg
import pytest

from ragtorio.db.connect import apply_schema, connect
from ragtorio.harvest.models import CrawlStats, RawPage, RawRedirect
from ragtorio.harvest.postgres import PostgresHarvestStore, _as_int

DSN = os.environ.get("RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio")
NOW = datetime(2026, 1, 1, tzinfo=UTC)

#: Spelled out once so test signatures stay readable.
type Conn = psycopg.Connection[tuple[object, ...]]
TABLES = ("raw_category", "raw_redirect", "raw_page", "crawl_run")


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
    """A connection on a clean schema. Tables are truncated, not dropped, between tests."""
    with connect(DSN) as connection:
        apply_schema(connection)
        with connection.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY")
        connection.commit()
        yield connection


@pytest.fixture
def store(conn: Conn) -> PostgresHarvestStore:
    return PostgresHarvestStore(conn)


def page(
    page_id: int,
    *,
    title: str | None = None,
    ns: int = 0,
    revision_id: int = 1,
    text: str = "body",
    categories: tuple[str, ...] = (),
) -> RawPage:
    return RawPage(
        wiki="factorio",
        page_id=page_id,
        ns=ns,
        title=title or f"Page {page_id}",
        revision_id=revision_id,
        revised_at=NOW,
        wikitext=text,
        categories=categories,
    )


def count(conn: Conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        row = cur.fetchone()
    assert row is not None
    return _as_int(row[0])


def test_schema_is_idempotent() -> None:
    with connect(DSN) as conn:
        apply_schema(conn)
        apply_schema(conn)


def test_run_lifecycle_is_recorded(store: PostgresHarvestStore, conn: Conn) -> None:
    run_id = store.start_run("factorio")
    store.finish_run(run_id, CrawlStats(listed=10, fetched=4, redirects=2))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT wiki, listed, fetched, redirects, error, finished_at IS NOT NULL "
            "FROM crawl_run WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    assert row == ("factorio", 10, 4, 2, None, True)


def test_failed_run_keeps_the_error(store: PostgresHarvestStore, conn: Conn) -> None:
    run_id = store.start_run("factorio")
    store.finish_run(run_id, CrawlStats(), error="MediaWikiError: nope")

    with conn.cursor() as cur:
        cur.execute("SELECT error FROM crawl_run WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "MediaWikiError: nope"


def test_pages_round_trip(store: PostgresHarvestStore, conn: Conn) -> None:
    store.save_pages([page(1, text="== Uses ==", categories=("Category:Items",))])
    assert store.known_revisions("factorio") == {1: 1}

    with conn.cursor() as cur:
        cur.execute("SELECT title, wikitext FROM raw_page WHERE page_id = 1")
        row = cur.fetchone()
    assert row == ("Page 1", "== Uses ==")


def test_resaving_a_page_updates_rather_than_duplicates(
    store: PostgresHarvestStore, conn: Conn
) -> None:
    store.save_pages([page(1, revision_id=1, text="old")])
    store.save_pages([page(1, revision_id=2, text="new")])

    assert count(conn, "raw_page") == 1
    assert store.known_revisions("factorio") == {1: 2}


def test_categories_are_replaced_not_appended(store: PostgresHarvestStore, conn: Conn) -> None:
    store.save_pages([page(1, categories=("Category:A", "Category:B"))])
    store.save_pages([page(1, revision_id=2, categories=("Category:B",))])

    with conn.cursor() as cur:
        cur.execute("SELECT category FROM raw_category WHERE page_id = 1 ORDER BY category")
        assert [r[0] for r in cur.fetchall()] == ["Category:B"]


def test_a_moved_page_does_not_collide_on_the_title_index(
    store: PostgresHarvestStore, conn: Conn
) -> None:
    """A rename frees a title that another page may take in the same crawl."""
    store.save_pages([page(1, title="Old name")])
    store.save_pages([page(2, title="Old name")])

    assert count(conn, "raw_page") == 1
    assert store.known_revisions("factorio") == {2: 1}


def test_namespace_counts(store: PostgresHarvestStore) -> None:
    store.save_pages([page(1, ns=0), page(2, ns=0), page(3, ns=3002)])
    assert store.namespace_counts("factorio") == {0: 2, 3002: 1}


def test_redirects_upsert(store: PostgresHarvestStore, conn: Conn) -> None:
    store.save_redirects([RawRedirect(wiki="factorio", from_title="A", to_title="B")])
    store.save_redirects([RawRedirect(wiki="factorio", from_title="A", to_title="C")])

    with conn.cursor() as cur:
        cur.execute("SELECT to_title FROM raw_redirect WHERE from_title = 'A'")
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "C"
    assert count(conn, "raw_redirect") == 1


def test_empty_saves_are_no_ops(store: PostgresHarvestStore, conn: Conn) -> None:
    store.save_pages([])
    store.save_redirects([])
    assert count(conn, "raw_page") == 0
    assert count(conn, "raw_redirect") == 0


def test_known_revisions_is_scoped_to_one_wiki(store: PostgresHarvestStore) -> None:
    store.save_pages([page(1)])
    other = RawPage(
        wiki="stardew",
        page_id=2,
        ns=0,
        title="Parsnip",
        revision_id=9,
        revised_at=NOW,
        wikitext="body",
    )
    store.save_pages([other])
    assert store.known_revisions("factorio") == {1: 1}
    assert store.known_revisions("stardew") == {2: 9}
