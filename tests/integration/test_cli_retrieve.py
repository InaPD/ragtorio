"""``ragtorio ask`` and ``ragtorio route eval`` end to end, with the models stubbed.

The router and the answering model are the two components that need credentials, so
they are the two components these tests replace. Everything else is real: a real
Postgres with a real chunk index, a real Neo4j with a real graph, and the merge,
budget, generation and citation validation in between. What is being checked is the
wiring - that a route reaches both stores, that the context comes back labeled, that
an answer's citations resolve to real revisions, and that every question lands in
``routing_log``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from typer.testing import CliRunner

from ragtorio import cli
from ragtorio.answer.models import Generation
from ragtorio.cli import app
from ragtorio.db.connect import apply_schema, connect
from ragtorio.db.neo4j import apply_schema as apply_neo4j_schema
from ragtorio.db.neo4j import connect as neo4j_connect
from ragtorio.index.embed import HashingEmbedder
from ragtorio.index.models import Chunk, EmbeddedChunk
from ragtorio.index.postgres import PostgresChunkStore, ensure_embedding_dimension
from ragtorio.ontology.load import GraphLoader
from ragtorio.ontology.models import ResolvedEdge, ResolvedGraph, ResolvedNode
from ragtorio.retrieve.models import ResolvedEntity, Route

#: A database of its own. These tests TRUNCATE, and pointing them at the one
#: `ragtorio harvest` writes to means a `make test` silently destroys a crawl.
DSN = os.environ.get(
    "RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio_test"
)
NEO4J_URI = os.environ.get("RAGTORIO_TEST_NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("RAGTORIO_TEST_NEO4J_PASSWORD", "ragtorio"))
TABLES = ("routing_log", "chunk", "fact", "raw_category", "raw_redirect", "raw_page", "crawl_run")
WIKI = "factorio"

type Conn = psycopg.Connection[tuple[object, ...]]

runner = CliRunner()


def _postgres_up() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


def _neo4j_up() -> bool:
    try:
        with neo4j_connect(NEO4J_URI, *NEO4J_AUTH) as driver:
            driver.verify_connectivity()
            return True
    except (ServiceUnavailable, Neo4jError, OSError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_postgres_up() and _neo4j_up()), reason="needs both Postgres and Neo4j"
    ),
]

GRAPH = ResolvedGraph(
    nodes=(
        ResolvedNode(
            id=f"{WIKI}:Testium ore",
            labels=("Item",),
            props={"title": "Testium ore", "stack_size": 50},
        ),
        ResolvedNode(
            id=f"{WIKI}:Testium plate",
            labels=("Item",),
            props={"title": "Testium plate", "aliases": ["plate"]},
        ),
        ResolvedNode(
            id=f"{WIKI}:Testium plate (recipe)",
            labels=("Recipe",),
            props={"title": "Testium plate (recipe)"},
        ),
    ),
    edges=(
        ResolvedEdge(
            from_id=f"{WIKI}:Testium plate (recipe)",
            rel_type="PRODUCES",
            to_id=f"{WIKI}:Testium plate",
            props={"amount": 1.0},
        ),
        ResolvedEdge(
            from_id=f"{WIKI}:Testium plate (recipe)",
            rel_type="CONSUMES",
            to_id=f"{WIKI}:Testium ore",
            props={"amount": 2.0},
        ),
    ),
    unresolved=(),
)


class StubRouter:
    """Returns a fixed route, with the question filled in."""

    def __init__(self, route: Route) -> None:
        self._route = route

    def route(self, question: str) -> Route:
        return self._route.model_copy(update={"question": question})


def stub_route(intent: str, template: str | None = None, entity: str | None = None) -> Route:
    entities = (
        (
            ResolvedEntity(
                mention=entity,
                id=f"{WIKI}:{entity}",
                title=entity,
                labels=("Item",),
                matched_by="title",
            ),
        )
        if entity
        else ()
    )
    return Route.model_validate(
        {"question": "q", "intent": intent, "template": template, "entities": entities}
    )


@pytest.fixture
def conn() -> Iterator[Conn]:
    with connect(DSN) as connection:
        apply_schema(connection)
        with connection.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY")
        connection.commit()
        ensure_embedding_dimension(connection, 768)
        embedder = HashingEmbedder()
        text = "testium plate is smelted from testium ore in a furnace over time"
        PostgresChunkStore(connection).replace_all(
            WIKI,
            [
                EmbeddedChunk(
                    chunk=Chunk(
                        chunk_id=f"{WIKI}:1:0000",
                        wiki=WIKI,
                        page_id=1,
                        title="Testium plate",
                        revision_id=7,
                        text=text,
                        mentioned_entity_ids=(f"{WIKI}:Testium plate",),
                    ),
                    embedding=embedder.embed_documents([text])[0],
                )
            ],
        )
        yield connection


@pytest.fixture
def graph() -> Iterator[Driver]:
    with neo4j_connect(NEO4J_URI, *NEO4J_AUTH) as driver:
        apply_neo4j_schema(driver)
        # By id, not by prefix: the wiki id has to be a real one for the CLI to find
        # a profile, so a prefix wipe here would delete an actual loaded graph.
        ids = [node.id for node in GRAPH.nodes]
        with driver.session() as session:
            session.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=ids)
        GraphLoader(driver).load(GRAPH)
        yield driver
        with driver.session() as session:
            session.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=ids)


@pytest.fixture
def stub_the_router(monkeypatch: pytest.MonkeyPatch):
    def install(route: Route) -> None:
        monkeypatch.setattr(cli, "_router_or_exit", lambda driver, wiki_id: StubRouter(route))

    return install


class StubAnswerer:
    """Answers with whatever it was handed, and records the prompt it was given."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, messages, on_text=None) -> Generation:  # type: ignore[no-untyped-def]
        self.prompts.append(str(messages[0]["content"]))
        return Generation(text=self.text, model="stub", stop_reason="end_turn")


@pytest.fixture
def stub_the_answerer(monkeypatch: pytest.MonkeyPatch):
    def install(text: str) -> StubAnswerer:
        answerer = StubAnswerer(text)
        monkeypatch.setattr(cli, "AnthropicAnswerer", lambda: answerer)
        return answerer

    return install


def ask(*args: str) -> object:
    """Retrieval only. Answering needs the model stubbed, which ``answer`` does."""
    return answer(*args, "--retrieve-only")


def answer(*args: str) -> object:
    return runner.invoke(
        app,
        [
            "ask",
            WIKI,
            *args,
            "--provider",
            "hashing",
            "--dsn",
            DSN,
            "--neo4j-uri",
            NEO4J_URI,
        ],
    )


def test_a_graph_question_returns_graph_facts(conn, graph, stub_the_router):
    stub_the_router(stub_route("graph", "recipe_tree", "Testium plate"))
    result = ask("what does a testium plate cost")

    assert result.exit_code == 0, result.output
    assert "graph facts" in result.output
    assert "Testium ore: 2 (raw)" in result.output


def test_a_prose_question_returns_passages(conn, graph, stub_the_router):
    stub_the_router(stub_route("vector"))
    result = ask("how is testium smelted")

    assert result.exit_code == 0, result.output
    assert f"{WIKI}:1:0000" in result.output


def test_both_returns_both_halves_labeled(conn, graph, stub_the_router):
    stub_the_router(stub_route("both", "recipe_tree", "Testium plate"))
    result = ask("what is a testium plate and what does it cost")

    assert result.exit_code == 0, result.output
    assert "graph facts" in result.output
    assert "passages" in result.output


def test_show_context_prints_the_passage_text(conn, graph, stub_the_router):
    stub_the_router(stub_route("vector"))
    assert "smelted from testium ore" in ask("how is testium smelted", "--show-context").output


def test_forcing_an_intent_overrides_the_router(conn, graph, stub_the_router):
    """How the benchmark's two baselines are run."""
    stub_the_router(stub_route("both", "recipe_tree", "Testium plate"))
    result = ask("anything", "--intent", "graph")

    assert result.exit_code == 0, result.output
    assert "passages" not in result.output


def test_an_unknown_intent_exits_cleanly(conn, graph, stub_the_router):
    stub_the_router(stub_route("both"))
    result = ask("anything", "--intent", "sideways")

    assert result.exit_code == 1
    assert "must be graph, vector or both" in result.output


def test_every_question_is_written_to_the_routing_log(conn, graph, stub_the_router):
    stub_the_router(stub_route("both", "recipe_tree", "Testium plate"))
    ask("what does a testium plate cost")

    with conn.cursor() as cur:
        cur.execute("SELECT question, intent, template, entities, chunk_ids FROM routing_log")
        row = cur.fetchone()
    assert row is not None
    question, intent, template, entities, chunk_ids = row
    assert (question, intent, template) == ("what does a testium plate cost", "both", "recipe_tree")
    assert entities == [f"{WIKI}:Testium plate"]
    assert chunk_ids == [f"{WIKI}:1:0000"]


def test_an_answer_cites_a_retrieved_chunk_and_links_to_its_revision(
    conn, graph, stub_the_router, stub_the_answerer
):
    """The Phase 6 exit criterion, end to end: the citation in the answer resolves to
    a chunk that was retrieved, and renders as a revision-pinned wiki URL."""
    stub_the_router(stub_route("vector"))
    stub_the_answerer(f"Testium plate is smelted from testium ore [{WIKI}:1:0000].")
    result = answer("how is testium smelted")

    assert result.exit_code == 0, result.output
    assert "index.php?title=Testium_plate&oldid=7" in result.output
    assert "unverified" not in result.output


def test_a_fabricated_citation_is_retried_once_and_then_reported(
    conn, graph, stub_the_router, stub_the_answerer
):
    """The stub answers the same way both times, which is the case the plan cares
    about: the claim ships labeled rather than silently stripped."""
    stub_the_router(stub_route("vector"))
    stub_the_answerer(f"Smelted from ore [{WIKI}:9:9999].")
    result = answer("how is testium smelted")

    assert result.exit_code == 0, result.output
    assert "unverified" in result.output
    assert "generations" in result.output


def test_the_evidence_reaches_the_answering_model_labeled(
    conn, graph, stub_the_router, stub_the_answerer
):
    stub_the_router(stub_route("both", "recipe_tree", "Testium plate"))
    answerer = stub_the_answerer("Two testium ore [graph].")
    answer("what does a testium plate cost")

    prompt = answerer.prompts[0]
    assert "<graph_facts>" in prompt and "<passages>" in prompt
    assert "Testium ore" in prompt


def test_route_eval_scores_the_labeled_set(graph, stub_the_router):
    """Every question routed the same way, so the accuracy is arithmetic we can check:
    the shipped set is graph-heavy, so an all-graph router scores well on intent."""
    stub_the_router(stub_route("graph", "recipe_tree", "Testium plate"))
    result = runner.invoke(app, ["route", "eval", WIKI, "--neo4j-uri", NEO4J_URI])

    assert result.exit_code == 0, result.output
    assert "intent accuracy" in result.output
    assert "template accuracy" in result.output


def test_route_eval_on_a_wiki_with_no_labeled_set_exits_cleanly(graph):
    result = runner.invoke(app, ["route", "eval", "nowhere", "--neo4j-uri", NEO4J_URI])
    assert result.exit_code == 1
    assert "no profile at" in result.output or "no labeled question set" in result.output
