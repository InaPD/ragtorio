"""The pgvector store against a real database.

The vector literal, the HNSW ordering, the array containment filter and the column
resize are exactly the things an in-memory fake would get right by construction and
Postgres might not. Skipped, not failed, when no database is reachable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import psycopg
import pytest

from ragtorio.db.connect import apply_schema, connect
from ragtorio.harvest.models import RawPage
from ragtorio.harvest.postgres import PostgresHarvestStore
from ragtorio.index.embed import HashingEmbedder
from ragtorio.index.models import Chunk, EmbeddedChunk
from ragtorio.index.postgres import PostgresChunkStore, ensure_embedding_dimension
from ragtorio.index.recall import LabeledQuery, evaluate, sweep
from ragtorio.index.repository import PostgresChunkSourceRepository

#: A database of its own. These tests TRUNCATE, and pointing them at the one
#: `ragtorio harvest` writes to means a `make test` silently destroys a crawl.
DSN = os.environ.get(
    "RAGTORIO_TEST_DSN", "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio_test"
)
NOW = datetime(2026, 1, 1, tzinfo=UTC)
TABLES = ("chunk", "fact", "raw_category", "raw_redirect", "raw_page", "crawl_run")
DIMENSIONS = 768

PASSAGES = {
    "Oil processing": "heavy oil can be cracked into light oil in a chemical plant",
    "Pollution": "pollution spreads across chunks and attracts biters to the base",
    "Railway": "trains move between stations along rails and need fuel to run",
}

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
        # A previous test may have left the column at another width.
        ensure_embedding_dimension(connection, DIMENSIONS)
        yield connection


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(DIMENSIONS)


@pytest.fixture
def store(conn: Conn) -> PostgresChunkStore:
    return PostgresChunkStore(conn)


def chunk(page_id: int, title: str, text: str, mentions: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        chunk_id=f"factorio:{page_id}:0000",
        wiki="factorio",
        page_id=page_id,
        title=title,
        revision_id=7,
        section_path=("Overview", "Tips"),
        text=text,
        mentioned_entity_ids=mentions,
    )


@pytest.fixture
def filled(store: PostgresChunkStore, embedder: HashingEmbedder) -> PostgresChunkStore:
    store.replace_all(
        "factorio",
        [
            EmbeddedChunk(
                chunk=chunk(i, title, text, (f"factorio:{title}",)),
                embedding=embedder.embed_documents([text])[0],
            )
            for i, (title, text) in enumerate(PASSAGES.items(), start=1)
        ],
    )
    return store


def test_a_chunk_survives_the_round_trip_intact(
    filled: PostgresChunkStore, embedder: HashingEmbedder
):
    """Including the section path array, which is the part a text column would flatten."""
    query = embedder.embed_query("heavy oil cracked in a chemical plant")
    top = filled.search("factorio", query, limit=1)[0].chunk

    assert top.title == "Oil processing"
    assert top.section_path == ("Overview", "Tips")
    assert top.mentioned_entity_ids == ("factorio:Oil processing",)
    assert top.revision_id == 7


def test_results_come_back_most_similar_first(
    filled: PostgresChunkStore, embedder: HashingEmbedder
):
    query = embedder.embed_query("trains stations rails fuel")
    matches = filled.search("factorio", query, limit=3)

    assert matches[0].chunk.title == "Railway"
    assert matches[0].score > matches[-1].score


def test_similarity_is_reported_not_distance(filled: PostgresChunkStore, embedder: HashingEmbedder):
    """1.0 for an identical vector, so a bigger number is always a better result."""
    query = embedder.embed_documents([PASSAGES["Pollution"]])[0]
    assert filled.search("factorio", query, limit=1)[0].score == pytest.approx(1.0)


def test_the_entity_filter_restricts_what_can_come_back(
    filled: PostgresChunkStore, embedder: HashingEmbedder
):
    query = embedder.embed_query("heavy oil cracked in a chemical plant")
    matches = filled.search("factorio", query, limit=10, entity_ids=["factorio:Railway"])
    assert [match.chunk.title for match in matches] == ["Railway"]


def test_an_empty_entity_filter_means_no_filter(
    filled: PostgresChunkStore, embedder: HashingEmbedder
):
    query = embedder.embed_query("anything at all")
    assert len(filled.search("factorio", query, limit=10)) == len(PASSAGES)


def test_ef_search_is_accepted_and_does_not_leak_into_the_next_query(
    filled: PostgresChunkStore, embedder: HashingEmbedder
):
    query = embedder.embed_query("trains stations rails fuel")
    tuned = filled.search("factorio", query, limit=3, ef_search=200)
    plain = filled.search("factorio", query, limit=3)
    assert [m.chunk.chunk_id for m in tuned] == [m.chunk.chunk_id for m in plain]


def test_a_second_run_replaces_the_wikis_chunks_rather_than_adding_to_them(
    filled: PostgresChunkStore, embedder: HashingEmbedder
):
    filled.replace_all(
        "factorio",
        [
            EmbeddedChunk(
                chunk=chunk(9, "New", "text"), embedding=embedder.embed_documents(["t"])[0]
            )
        ],
    )
    assert filled.count("factorio") == 1
    assert filled.titles("factorio") == {"New"}


def test_another_wikis_chunks_are_left_alone(filled: PostgresChunkStore, embedder: HashingEmbedder):
    other = EmbeddedChunk(
        chunk=chunk(1, "Stardew", "text").model_copy(
            update={"wiki": "stardew", "chunk_id": "stardew:1:0000"}
        ),
        embedding=embedder.embed_documents(["text"])[0],
    )
    filled.replace_all("stardew", [other])
    filled.replace_all("factorio", [])

    assert filled.count("factorio") == 0
    assert filled.count("stardew") == 1


def test_the_column_is_resized_for_a_provider_of_another_width(conn: Conn):
    """Voyage's models offer 1024, never 768; switching should not need a migration."""
    assert ensure_embedding_dimension(conn, 1024) is True
    assert ensure_embedding_dimension(conn, 1024) is False

    store = PostgresChunkStore(conn)
    embedder = HashingEmbedder(1024)
    store.replace_all(
        "factorio",
        [
            EmbeddedChunk(
                chunk=chunk(1, "Wide", "text"), embedding=embedder.embed_documents(["t"])[0]
            )
        ],
    )
    assert store.count("factorio") == 1


def test_resizing_refuses_to_throw_away_an_existing_index(filled: PostgresChunkStore, conn: Conn):
    with pytest.raises(RuntimeError, match="already stored"):
        ensure_embedding_dimension(conn, 256)


def test_recall_runs_against_the_real_index(filled: PostgresChunkStore, embedder: HashingEmbedder):
    """The harness and the HNSW path together, which is what `index recall` really is."""
    queries = [LabeledQuery(query=text, relevant=(title,)) for title, text in PASSAGES.items()]
    report = evaluate(
        filled, embedder, "factorio", queries, indexed_titles=filled.titles("factorio")
    )
    assert report.recall_at(1) == 1.0
    assert report.p95_ms > 0

    sweep_reports = sweep(filled, embedder, "factorio", queries, [40, 200])
    assert [r.recall_at(1) for r in sweep_reports] == [1.0, 1.0]


def test_the_source_repository_reads_articles_redirects_and_the_graphs_vocabulary(conn: Conn):
    harvest = PostgresHarvestStore(conn)
    harvest.save_pages(
        [
            RawPage(
                wiki="factorio",
                page_id=1,
                ns=0,
                title="Oil processing",
                revision_id=7,
                revised_at=NOW,
                wikitext="body",
            )
        ]
    )
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO raw_redirect (wiki, from_title, to_title) VALUES (%s, %s, %s)",
            ("factorio", "Oil", "Oil processing"),
        )
        cur.execute(
            """
            INSERT INTO fact (wiki, subject, subject_labels, predicate, source_page_id,
                              source_revision_id, source_field)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            ("factorio", "Crude oil", ["Item"], "prop.stack_size", 1, 7, "stack-size"),
        )
    conn.commit()

    source = PostgresChunkSourceRepository(conn)
    assert [page.title for page in source.articles("factorio", 0)] == ["Oil processing"]
    assert [r.from_title for r in source.redirects("factorio")] == ["Oil"]
    assert source.entity_titles("factorio") == frozenset({"Crude oil"})
