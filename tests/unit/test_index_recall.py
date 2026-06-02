"""The recall harness: the scoring rules, and the loader for the labeled set.

The shipped ``wikis/factorio.recall.yaml`` is validated here too. A labeled set with a
typo'd key or a query carrying no relevant page would otherwise only fail at the end of
an index build, which is the wrong moment to discover it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragtorio.index.embed import HashingEmbedder
from ragtorio.index.models import Chunk, EmbeddedChunk
from ragtorio.index.recall import (
    LabeledQuery,
    QueryOutcome,
    RecallReport,
    evaluate,
    load_queries,
    sweep,
)
from ragtorio.index.store import InMemoryChunkStore

PASSAGES = {
    "Oil processing": "heavy oil can be cracked into light oil in a chemical plant",
    "Pollution": "pollution spreads across chunks and attracts biters to the base",
    "Railway": "trains move between stations along rails and need fuel to run",
    "Beacon": "a beacon transmits module effects to every machine in its range",
}


@pytest.fixture
def store() -> InMemoryChunkStore:
    embedder = HashingEmbedder()
    filled = InMemoryChunkStore()
    filled.replace_all(
        "factorio",
        [
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=f"factorio:{i}:0000",
                    wiki="factorio",
                    page_id=i,
                    title=title,
                    revision_id=1,
                    text=text,
                ),
                embedding=embedder.embed_documents([text])[0],
            )
            for i, (title, text) in enumerate(PASSAGES.items(), start=1)
        ],
    )
    return filled


def outcome(hit_rank: int | None, labels_in_index: bool = True) -> QueryOutcome:
    return QueryOutcome(
        query="q",
        relevant=("Oil processing",),
        retrieved=(),
        hit_rank=hit_rank,
        labels_in_index=labels_in_index,
    )


def test_a_query_hits_when_any_labeled_page_comes_back(store: InMemoryChunkStore):
    queries = [LabeledQuery(query="how do I crack heavy oil", relevant=("Oil processing",))]
    report = evaluate(store, HashingEmbedder(), "factorio", queries)
    assert report.recall_at(10) == 1.0
    assert report.outcomes[0].hit_rank == 1


def test_a_miss_is_reported_with_what_came_back_instead(store: InMemoryChunkStore):
    """At k=1 this corpus can actually miss; at k=10 it holds fewer pages than that."""
    queries = [LabeledQuery(query="biters attracted to pollution", relevant=("Railway",))]
    report = evaluate(store, HashingEmbedder(), "factorio", queries, k_values=(1,))
    assert report.recall_at(1) == 0.0
    assert report.misses[0].retrieved == ("Pollution",)


def test_recall_at_a_smaller_k_is_a_prefix_of_the_larger_one():
    report = RecallReport(provider="test", outcomes=(outcome(1), outcome(4), outcome(None)))
    assert report.recalls == {1: 1 / 3, 3: 1 / 3, 5: 2 / 3, 10: 2 / 3}


def test_the_phase_target_is_recall_at_ten():
    hits = [outcome(1) for _ in range(8)]
    below = RecallReport(provider="test", outcomes=(*hits, outcome(None), outcome(None)))
    at_target = RecallReport(provider="test", outcomes=(*hits, outcome(1), outcome(None)))
    assert not below.meets_target
    assert at_target.meets_target


def test_a_query_whose_pages_are_absent_is_excluded_rather_than_scored_as_a_miss(
    store: InMemoryChunkStore,
):
    """A bad label and a retrieval failure look identical in the number; they are not."""
    queries = [
        LabeledQuery(query="how do I crack heavy oil", relevant=("Oil processing",)),
        LabeledQuery(query="what does a fusion reactor do", relevant=("Fusion reactor",)),
    ]
    report = evaluate(
        store, HashingEmbedder(), "factorio", queries, indexed_titles=store.titles("factorio")
    )
    assert len(report.scored) == 1
    assert report.recall_at(10) == 1.0
    assert [o.query for o in report.unlabeled] == ["what does a fusion reactor do"]


def test_without_an_index_to_check_against_every_label_is_taken_at_face_value(
    store: InMemoryChunkStore,
):
    queries = [LabeledQuery(query="anything", relevant=("Fusion reactor",))]
    report = evaluate(store, HashingEmbedder(), "factorio", queries)
    assert len(report.scored) == 1


def test_an_empty_report_scores_zero_rather_than_dividing_by_nothing():
    report = RecallReport(provider="test")
    assert report.recall_at(10) == 0.0
    assert report.p50_ms == 0.0
    assert report.p95_ms == 0.0


def test_titles_are_matched_case_insensitively():
    assert LabeledQuery(query="q", relevant=("Oil processing",)).matches("oil PROCESSING")


def test_latency_percentiles_come_from_the_measured_searches(store: InMemoryChunkStore):
    queries = [LabeledQuery(query=text, relevant=(title,)) for title, text in PASSAGES.items()]
    report = evaluate(store, HashingEmbedder(), "factorio", queries)
    assert report.p95_ms >= report.p50_ms > 0


def test_a_sweep_evaluates_once_per_ef_search_value(store: InMemoryChunkStore):
    queries = [LabeledQuery(query="how do I crack heavy oil", relevant=("Oil processing",))]
    reports = sweep(store, HashingEmbedder(), "factorio", queries, [40, 100])
    assert [report.ef_search for report in reports] == [40, 100]


def test_a_missing_labeled_set_says_so_rather_than_reporting_zero(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no labeled query set"):
        load_queries("nowhere", tmp_path)


def test_a_query_must_name_at_least_one_relevant_page(tmp_path: Path):
    (tmp_path / "w.recall.yaml").write_text(
        yaml.safe_dump({"queries": [{"query": "q", "relevant": []}]}), encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_queries("w", tmp_path)


def test_the_shipped_factorio_set_is_valid_and_large_enough():
    """The plan calls for thirty hand-labeled queries; fewer is not a measurement."""
    queries = load_queries("factorio")
    assert len(queries) >= 30
    assert len({query.query for query in queries}) == len(queries)
    assert all(query.relevant for query in queries)
