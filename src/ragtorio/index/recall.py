"""Recall@k on a hand-labeled query set, and the ``ef_search`` sweep that uses it.

**Why recall and not a ranking metric.** This index is one input to a grounded answer,
not a search results page. Whether the right passage came back first or fifth barely
matters once ten of them are concatenated into a prompt; whether it came back at all
decides whether the answer can be grounded. Recall@10 is therefore the number Phase 4
is judged on, and NDCG would be measuring a quality nothing downstream consumes.

**Why labels are page titles, not chunk ids.** A chunk id is an artifact of the current
chunker. Relabelling thirty queries every time the token budget changes would make the
labeled set a hostage to the code it exists to measure, and the labels would quietly
rot into agreement with whatever the chunker last did. A title is what a person can
actually judge: "the answer to this question is on the Oil processing page" stays true
across any chunking.

**Labels are checked against the index, not trusted.** A query whose labeled titles are
not in the index at all would otherwise count as a miss and quietly drag the number
down, looking exactly like a retrieval failure. Those are reported separately, as a
labelling bug, which is what they are.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ragtorio.config import profiles_dir
from ragtorio.index.embed import EmbeddingProvider
from ragtorio.index.store import ChunkStore

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

#: The phase's exit criterion, kept next to the thing that measures it.
TARGET_RECALL_AT_10 = 0.9


class LabeledQuery(BaseModel):
    """One question and the pages whose prose answers it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    relevant: tuple[str, ...] = Field(min_length=1)

    def matches(self, title: str) -> bool:
        """Whether a retrieved page counts as a hit, comparing titles case-insensitively."""
        return any(title.casefold() == relevant.casefold() for relevant in self.relevant)


class QueryOutcome(BaseModel):
    """What one query retrieved, kept so a miss can be read rather than guessed at."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    relevant: tuple[str, ...]
    retrieved: tuple[str, ...]
    hit_rank: int | None = None
    labels_in_index: bool = True

    def hit_at(self, k: int) -> bool:
        return self.hit_rank is not None and self.hit_rank <= k


class RecallReport(BaseModel):
    """Recall at each k, plus the latency and the misses behind the number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    ef_search: int | None = None
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    outcomes: tuple[QueryOutcome, ...] = ()
    latencies_ms: tuple[float, ...] = ()

    @property
    def scored(self) -> tuple[QueryOutcome, ...]:
        """Queries that count: the ones whose labeled pages are actually indexed."""
        return tuple(outcome for outcome in self.outcomes if outcome.labels_in_index)

    def recall_at(self, k: int) -> float:
        scored = self.scored
        if not scored:
            return 0.0
        return sum(1 for outcome in scored if outcome.hit_at(k)) / len(scored)

    @property
    def recalls(self) -> dict[int, float]:
        return {k: self.recall_at(k) for k in self.k_values}

    @property
    def misses(self) -> tuple[QueryOutcome, ...]:
        """Scored queries with no relevant page in the top k, at the largest k."""
        largest = max(self.k_values)
        return tuple(o for o in self.scored if not o.hit_at(largest))

    @property
    def unlabeled(self) -> tuple[QueryOutcome, ...]:
        """Queries excluded because none of their labeled pages is in the index."""
        return tuple(outcome for outcome in self.outcomes if not outcome.labels_in_index)

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 0.50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 0.95)

    @property
    def meets_target(self) -> bool:
        """The Phase 4 exit criterion: recall@10 at or above 0.9."""
        return self.recall_at(10) >= TARGET_RECALL_AT_10


def load_queries(wiki_id: str, directory: Path | None = None) -> tuple[LabeledQuery, ...]:
    """Load ``wikis/<wiki_id>.recall.yaml``.

    Raises:
        FileNotFoundError: if the wiki has no labeled set. Unlike aliases, an absent
            file is not a valid state here - a recall run with no queries would report
            a confident 0.0 rather than saying what is missing.
    """
    base = directory or profiles_dir()
    path = base / f"{wiki_id}.recall.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no labeled query set at {path}. Recall is measured against hand-labeled "
            "queries; write that file before running `ragtorio index recall`."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(LabeledQuery.model_validate(entry) for entry in data.get("queries", []))


def evaluate(
    store: ChunkStore,
    provider: EmbeddingProvider,
    wiki: str,
    queries: Sequence[LabeledQuery],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    ef_search: int | None = None,
    indexed_titles: frozenset[str] | None = None,
) -> RecallReport:
    """Run every labeled query and score the results.

    One search per query, at the largest k, sliced for the smaller ones: recall@3 is
    by definition a prefix of recall@10, and searching four times would measure the
    latency of a query pattern nothing actually uses.
    """
    largest = max(k_values)
    outcomes: list[QueryOutcome] = []
    latencies: list[float] = []

    for labeled in queries:
        embedding = provider.embed_query(labeled.query)
        started = time.perf_counter()
        matches = store.search(wiki, embedding, limit=largest, ef_search=ef_search)
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved = tuple(match.chunk.title for match in matches)
        hit_rank = next(
            (rank for rank, title in enumerate(retrieved, start=1) if labeled.matches(title)),
            None,
        )
        outcomes.append(
            QueryOutcome(
                query=labeled.query,
                relevant=labeled.relevant,
                retrieved=retrieved,
                hit_rank=hit_rank,
                labels_in_index=_labels_indexed(labeled, indexed_titles),
            )
        )

    return RecallReport(
        provider=provider.name,
        ef_search=ef_search,
        k_values=tuple(k_values),
        outcomes=tuple(outcomes),
        latencies_ms=tuple(latencies),
    )


def sweep(
    store: ChunkStore,
    provider: EmbeddingProvider,
    wiki: str,
    queries: Sequence[LabeledQuery],
    ef_values: Sequence[int],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    indexed_titles: frozenset[str] | None = None,
) -> list[RecallReport]:
    """Evaluate at each ``ef_search``, which is how the setting gets chosen.

    HNSW trades recall for latency at query time through this one number, and the
    right value is a property of this index and this query set, not something to copy
    from a blog post. The sweep is the measurement; the profile keeps the answer.
    """
    return [
        evaluate(
            store,
            provider,
            wiki,
            queries,
            k_values=k_values,
            ef_search=ef_value,
            indexed_titles=indexed_titles,
        )
        for ef_value in ef_values
    ]


def _labels_indexed(labeled: LabeledQuery, indexed_titles: frozenset[str] | None) -> bool:
    if indexed_titles is None:
        return True
    folded = {title.casefold() for title in indexed_titles}
    return any(relevant.casefold() in folded for relevant in labeled.relevant)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Exact enough for thirty measurements, and no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]
