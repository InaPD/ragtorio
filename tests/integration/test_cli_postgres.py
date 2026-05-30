"""The CLI's database paths, against a real Postgres.

``init-db`` and a non-dry harvest are the two commands that only work with a database,
so they are the two that a fake cannot prove anything about.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import httpx
import psycopg
import pytest
import respx
from typer.testing import CliRunner

from ragtorio.cli import app
from ragtorio.db.connect import apply_schema, connect
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.models import RawPage
from ragtorio.harvest.postgres import PostgresHarvestStore

DSN = os.environ.get("RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio")
FACTORIO_API = "https://wiki.factorio.com/api.php"
TABLES = ("fact", "raw_category", "raw_redirect", "raw_page", "crawl_run")

type Conn = psycopg.Connection[tuple[object, ...]]

runner = CliRunner()


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


def test_init_db_applies_the_schema_and_redacts_the_dsn() -> None:
    result = runner.invoke(app, ["init-db", "--dsn", DSN])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "ragtorio:***@" in result.output
    assert "ragtorio:ragtorio@" not in result.output


@respx.mock
def test_harvest_writes_pages_redirects_and_a_run(
    conn: Conn,
    monkeypatch: pytest.MonkeyPatch,
    fast_client: Callable[..., MediaWikiClient],
    factorio_responder: Callable[[httpx.Request], httpx.Response],
) -> None:
    monkeypatch.setattr("ragtorio.cli.MediaWikiClient", fast_client)
    respx.get(FACTORIO_API).mock(side_effect=factorio_responder)

    result = runner.invoke(app, ["harvest", "factorio", "--dsn", DSN])
    assert result.exit_code == 0, result.output

    with conn.cursor() as cur:
        cur.execute("SELECT title, wikitext FROM raw_page")
        assert cur.fetchall() == [("Iron plate", "body")]
        cur.execute("SELECT from_title, to_title FROM raw_redirect")
        assert cur.fetchall() == [("Green circuit", "Iron plate")]
        cur.execute("SELECT category FROM raw_category")
        assert cur.fetchall() == [("Category:Items",)]
        cur.execute("SELECT wiki, fetched, finished_at IS NOT NULL FROM crawl_run")
        assert cur.fetchall() == [("factorio", 1, True)]


@respx.mock
def test_second_harvest_fetches_nothing(
    conn: Conn,
    monkeypatch: pytest.MonkeyPatch,
    fast_client: Callable[..., MediaWikiClient],
    factorio_responder: Callable[[httpx.Request], httpx.Response],
) -> None:
    """The Phase 1 exit criterion, end to end through the CLI and a real database."""
    monkeypatch.setattr("ragtorio.cli.MediaWikiClient", fast_client)
    respx.get(FACTORIO_API).mock(side_effect=factorio_responder)

    runner.invoke(app, ["harvest", "factorio", "--dsn", DSN])
    second = runner.invoke(app, ["harvest", "factorio", "--dsn", DSN])
    assert second.exit_code == 0, second.output

    with conn.cursor() as cur:
        cur.execute("SELECT fetched, unchanged FROM crawl_run ORDER BY run_id")
        assert cur.fetchall() == [(1, 0), (0, 1)]
        cur.execute("SELECT count(*) FROM raw_page")
        assert cur.fetchone() == (1,)


def _seed_factorio_pages(conn: Conn, pages: list[RawPage]) -> None:
    PostgresHarvestStore(conn).save_pages(pages)


def test_extract_dry_run_writes_nothing_then_a_real_run_writes_facts(
    conn: Conn, factorio_pages: list[RawPage]
) -> None:
    _seed_factorio_pages(conn, factorio_pages)

    dry = runner.invoke(app, ["extract", "factorio", "--dry-run", "--dsn", DSN])
    assert dry.exit_code == 0, dry.output
    assert "dry run" in dry.output
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fact")
        assert cur.fetchone() == (0,)

    result = runner.invoke(app, ["extract", "factorio", "--dsn", DSN])
    assert result.exit_code == 0, result.output
    assert "coverage" in result.output
    assert "wrote 168 facts" in result.output
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fact WHERE wiki = 'factorio'")
        assert cur.fetchone() == (168,)


def test_extract_replaces_facts_on_a_second_run(conn: Conn, factorio_pages: list[RawPage]) -> None:
    """Facts are recomputable, so a second run must not accumulate duplicates."""
    _seed_factorio_pages(conn, factorio_pages)

    runner.invoke(app, ["extract", "factorio", "--dsn", DSN])
    second = runner.invoke(app, ["extract", "factorio", "--dsn", DSN])

    assert second.exit_code == 0, second.output
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fact WHERE wiki = 'factorio'")
        assert cur.fetchone() == (168,)
