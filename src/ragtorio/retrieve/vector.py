"""The vector half of a route: top-k passages, optionally narrowed to known entities.

The entity filter is the reason routing resolves entities before retrieving anything.
When the router has established that a question is about sulfuric acid, and the index
already records which chunks mention sulfuric acid, restricting to those beats hoping
the embedding agrees - and it makes the graph's vocabulary do work on the prose side.

It is a preference, not a constraint. A filtered search that comes back short falls
back to an unfiltered one, because a question whose entity appears in no passage
("what is the fastest belt") is a routing success and a retrieval failure only if the
filter is allowed to be the last word.
"""

from __future__ import annotations

from ragtorio.index.embed import EmbeddingProvider
from ragtorio.index.models import ChunkMatch
from ragtorio.index.store import ChunkStore
from ragtorio.retrieve.models import Route

DEFAULT_TOP_K = 8

#: Below this many filtered hits, widen. One passage is not a retrieval; it is the
#: filter having been too narrow.
MIN_FILTERED_HITS = 3


class VectorRetriever:
    """Searches the chunk index for a route. The caller owns the store and provider."""

    def __init__(
        self,
        store: ChunkStore,
        provider: EmbeddingProvider,
        wiki: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        ef_search: int | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._wiki = wiki
        self._top_k = top_k
        self._ef_search = ef_search

    def run(self, route: Route) -> list[ChunkMatch]:
        """Passages for a question, most similar first."""
        embedding = self._provider.embed_query(route.question)
        entity_ids = route.entity_ids

        if entity_ids:
            narrowed = self._search(embedding, entity_ids)
            if len(narrowed) >= MIN_FILTERED_HITS:
                return narrowed
            # Widen, then put the filtered hits first: they are the ones the router's
            # own entity resolution vouched for.
            return _merge_keeping_order(narrowed, self._search(embedding, ()), self._top_k)
        return self._search(embedding, ())

    def _search(
        self, embedding: tuple[float, ...], entity_ids: tuple[str, ...]
    ) -> list[ChunkMatch]:
        return self._store.search(
            self._wiki,
            embedding,
            limit=self._top_k,
            entity_ids=entity_ids,
            ef_search=self._ef_search,
        )


def _merge_keeping_order(
    preferred: list[ChunkMatch], rest: list[ChunkMatch], limit: int
) -> list[ChunkMatch]:
    """Preferred matches first, then the rest, deduplicated by chunk id."""
    seen = {match.chunk.chunk_id for match in preferred}
    merged = list(preferred)
    for match in rest:
        if match.chunk.chunk_id not in seen:
            seen.add(match.chunk.chunk_id)
            merged.append(match)
    return merged[:limit]
