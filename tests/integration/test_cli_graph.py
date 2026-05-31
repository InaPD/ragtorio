"""``graph load``, ``graph check`` and ``ask --graph-only``, end to end through
Postgres and Neo4j. Skipped, not failed, when either is unreachable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest
from neo4j import GraphDatabase
from typer.testing import CliRunner

from ragtorio.cli import app
from ragtorio.db.connect import apply_schema, connect
from ragtorio.extract.models import Fact, Provenance
from ragtorio.extract.postgres import PostgresFactStore

PG_DSN = os.environ.get(
    "RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio"
)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "ragtorio")
TABLES = ("fact", "raw_category", "raw_redirect", "raw_page", "crawl_run")

type Conn = psycopg.Connection[tuple[object, ...]]

runner = CliRunner()


def _reachable() -> bool:
    try:
        with psycopg.connect(PG_DSN, connect_timeout=2):
            pass
        with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason="needs both Postgres and Neo4j"),
]


def _clear_neo4j() -> None:
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver, driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def conn() -> Iterator[Conn]:
    _clear_neo4j()  # other integration test files leave their own data behind
    with connect(PG_DSN) as connection:
        apply_schema(connection)
        with connection.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY")
        connection.commit()
        yield connection
    _clear_neo4j()


def _seed_widget_chain(conn: Conn) -> None:
    """Widget (recipe) consumes 2x Gear; Gear has no recipe of its own (raw).

    Gear needs its own fact (even a trivial one) or it never becomes a resolvable
    node at all, and the CONSUMES edge to it is logged as unresolved instead of
    created - the same thing the resolver's own unit tests have to watch for.
    """
    prov = Provenance(wiki="factorio", page_id=1, revision_id=1, field="recipe")
    facts = [
        Fact(
            subject="Widget",
            subject_labels=("Item",),
            predicate="prop.crafting_time",
            object=1.0,
            provenance=prov,
        ),
        Fact(
            subject="Widget",
            subject_labels=("Item",),
            predicate="rel.CONSUMES",
            object="Gear",
            props={"amount": 2.0},
            provenance=prov,
        ),
        Fact(
            subject="Widget",
            subject_labels=("Item",),
            predicate="rel.PRODUCES",
            object="Widget",
            props={"amount": 1.0, "probability": 1.0},
            provenance=prov,
        ),
        Fact(
            subject="Gear",
            subject_labels=("Item",),
            predicate="prop.stack_size",
            object=100.0,
            provenance=prov,
        ),
    ]
    PostgresFactStore(conn).replace_all("factorio", facts)


def test_graph_load_then_check_report_the_widget_chain(conn: Conn) -> None:
    _seed_widget_chain(conn)

    load_result = runner.invoke(app, ["graph", "load", "factorio", "--dsn", PG_DSN])
    assert load_result.exit_code == 0, load_result.output
    assert "nodes" in load_result.output

    check_result = runner.invoke(app, ["graph", "check", "factorio", "--dsn", PG_DSN])
    assert check_result.exit_code == 0, check_result.output
    # A complete little chain: nothing orphaned, nothing missing, nothing unresolved.
    assert "orphans                   0" in check_result.output
    assert "unresolved references     0" in check_result.output


def test_ask_graph_only_answers_the_widget_chain(conn: Conn) -> None:
    _seed_widget_chain(conn)
    runner.invoke(app, ["graph", "load", "factorio", "--dsn", PG_DSN])

    result = runner.invoke(app, ["ask", "factorio", "raw ore for one widget", "--graph-only"])

    assert result.exit_code == 0, result.output
    assert "Widget" in result.output
    assert "Gear: 2" in result.output
    assert "raw materials" in result.output
    assert "Gear: 2" in result.output.split("raw materials")[1]


def test_ask_without_graph_only_is_not_yet_implemented() -> None:
    result = runner.invoke(app, ["ask", "factorio", "anything"])
    assert result.exit_code == 1
    assert "Phase 5" in " ".join(result.output.split())  # normalize rich's line wrap


def test_ask_with_an_unrecognised_question_shape_exits_cleanly() -> None:
    result = runner.invoke(app, ["ask", "factorio", "what is love", "--graph-only"])
    assert result.exit_code == 1
    assert "could not find an item name" in result.output


def test_ask_for_an_entity_the_graph_does_not_have_exits_cleanly(conn: Conn) -> None:
    """The question parses fine; nothing in the (empty) graph matches it."""
    result = runner.invoke(
        app, ["ask", "factorio", "raw ore for one nonexistent widget", "--graph-only"]
    )
    assert result.exit_code == 1
    assert "no entity matching" in result.output
