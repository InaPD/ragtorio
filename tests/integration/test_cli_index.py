"""``ragtorio index build`` and ``ragtorio index recall`` against a real Postgres.

The hashing provider stands in for the local model throughout: what these tests are
checking is the wiring - profile to chunker to store to recall report - not the
quality of anyone's embeddings, and pulling PyTorch into CI to assert on an exit code
would be a poor trade.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
import yaml
from typer.testing import CliRunner

from ragtorio.cli import app
from ragtorio.db.connect import apply_schema, connect
from ragtorio.harvest.models import RawPage
from ragtorio.harvest.postgres import PostgresHarvestStore
from ragtorio.index.postgres import ensure_embedding_dimension

DSN = os.environ.get("RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio")
FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "wikitext" / "factorio"
TABLES = ("chunk", "fact", "raw_category", "raw_redirect", "raw_page", "crawl_run")
NOW = datetime(2026, 1, 1, tzinfo=UTC)

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
        ensure_embedding_dimension(connection, 768)
        yield connection


@pytest.fixture
def crawled(conn: Conn) -> Conn:
    """Three real articles in the crawl, as ``ragtorio harvest`` would leave them."""
    pages = [
        ("Oil_processing", "Oil processing"),
        ("Electronic_circuit", "Electronic circuit"),
        ("Kovarex_enrichment_process", "Kovarex enrichment process"),
    ]
    PostgresHarvestStore(conn).save_pages(
        [
            RawPage(
                wiki="factorio",
                page_id=page_id,
                ns=0,
                title=title,
                revision_id=7,
                revised_at=NOW,
                wikitext=(FIXTURE_DIR / f"{stem}.txt").read_text(encoding="utf-8"),
            )
            for page_id, (stem, title) in enumerate(pages, start=1)
        ]
    )
    return conn


def test_build_writes_chunks_and_reports_what_it_produced(crawled: Conn) -> None:
    result = runner.invoke(
        app, ["index", "build", "factorio", "--provider", "hashing", "--dsn", DSN]
    )
    assert result.exit_code == 0, result.output
    assert "articles seen" in result.output
    assert "wrote" in result.output

    with crawled.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunk WHERE wiki = 'factorio'")
        row = cur.fetchone()
    assert row is not None and row[0] > 0


def test_a_dry_run_reports_but_writes_nothing(crawled: Conn) -> None:
    result = runner.invoke(
        app, ["index", "build", "factorio", "--provider", "hashing", "--dry-run", "--dsn", DSN]
    )
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output

    with crawled.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunk")
        row = cur.fetchone()
    assert row is not None and row[0] == 0


def test_a_rebuild_replaces_rather_than_accumulates(crawled: Conn) -> None:
    for _ in range(2):
        result = runner.invoke(
            app, ["index", "build", "factorio", "--provider", "hashing", "--dsn", DSN]
        )
        assert result.exit_code == 0, result.output

    with crawled.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT chunk_id) FROM chunk")
        row = cur.fetchone()
    assert row is not None and row[0] == row[1]


def test_an_unknown_provider_exits_cleanly(crawled: Conn) -> None:
    result = runner.invoke(
        app, ["index", "build", "factorio", "--provider", "word2vec", "--dsn", DSN]
    )
    assert result.exit_code == 1
    assert "unknown embedding provider" in result.output


def test_recall_refuses_to_score_an_index_that_does_not_exist(conn: Conn) -> None:
    """Otherwise it reports a confident 0% and looks like a retrieval problem."""
    result = runner.invoke(
        app, ["index", "recall", "factorio", "--provider", "hashing", "--dsn", DSN]
    )
    assert result.exit_code == 1
    assert "no chunks stored" in result.output


def test_recall_scores_the_index_and_flags_labels_it_cannot_find(crawled: Conn) -> None:
    """Three articles cannot answer thirty questions; the point is that it says so
    rather than reporting a number built on absent pages."""
    runner.invoke(app, ["index", "build", "factorio", "--provider", "hashing", "--dsn", DSN])
    result = runner.invoke(
        app, ["index", "recall", "factorio", "--provider", "hashing", "--dsn", DSN]
    )
    assert result.exit_code == 0, result.output
    assert "recall@10" in result.output
    assert "queries not scored" in result.output


def test_the_ef_search_sweep_compares_settings(crawled: Conn, tmp_path: Path) -> None:
    runner.invoke(app, ["index", "build", "factorio", "--provider", "hashing", "--dsn", DSN])
    result = runner.invoke(
        app,
        [
            "index",
            "recall",
            "factorio",
            "--provider",
            "hashing",
            "--ef-search",
            "40,200",
            "--dsn",
            DSN,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ef_search" in result.output
    assert "40" in result.output and "200" in result.output


def test_a_malformed_ef_search_option_exits_cleanly(crawled: Conn) -> None:
    runner.invoke(app, ["index", "build", "factorio", "--provider", "hashing", "--dsn", DSN])
    result = runner.invoke(
        app,
        [
            "index",
            "recall",
            "factorio",
            "--provider",
            "hashing",
            "--ef-search",
            "fast",
            "--dsn",
            DSN,
        ],
    )
    assert result.exit_code == 1
    assert "comma-separated integers" in result.output


def test_an_unknown_wiki_exits_cleanly() -> None:
    result = runner.invoke(app, ["index", "build", "nowhere", "--provider", "hashing"])
    assert result.exit_code == 1
    assert "no profile at" in result.output


def test_a_wiki_with_no_labeled_set_says_what_is_missing(
    crawled: Conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ragtorio.index.recall.profiles_dir", lambda: tmp_path)
    (tmp_path / "factorio.yaml").write_text(
        yaml.safe_dump({"queries": []}), encoding="utf-8"
    )  # a profile-shaped file, but no <id>.recall.yaml next to it
    runner.invoke(app, ["index", "build", "factorio", "--provider", "hashing", "--dsn", DSN])
    result = runner.invoke(
        app, ["index", "recall", "factorio", "--provider", "hashing", "--dsn", DSN]
    )
    assert result.exit_code == 1
    assert "no labeled query set" in result.output
