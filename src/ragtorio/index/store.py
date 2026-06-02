"""Where chunks live, and the one query asked of them.

The same replace-the-wiki's-rows rule as ``extract/store.py``, for the same reason:
a chunk is fully recomputable from ``raw_page``, so after a chunker or model change the
old rows are not stale, they are wrong, and leaving them mixed in with new ones would
put two incompatible vector spaces behind one index.

``search`` takes an optional entity filter because Phase 5's router resolves a
question's entities before retrieving anything. When it knows the question is about
sulfuric acid, restricting to chunks that mention sulfuric acid is strictly better
than hoping the embedding agrees.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragtorio.index.embed import cosine
from ragtorio.index.models import ChunkMatch, EmbeddedChunk


class ChunkStore(Protocol):
    """The persistence the vector index needs, and nothing more."""

    def replace_all(self, wiki: str, chunks: Sequence[EmbeddedChunk]) -> None:
        """Delete every chunk stored for ``wiki`` and insert this run's instead."""
        ...

    def count(self, wiki: str) -> int:
        """How many chunks are currently stored for ``wiki``."""
        ...

    def titles(self, wiki: str) -> frozenset[str]:
        """Distinct page titles present in the index.

        The recall harness needs this to tell a retrieval failure apart from a label
        naming a page the crawl never reached.
        """
        ...

    def search(
        self,
        wiki: str,
        embedding: Sequence[float],
        limit: int = 10,
        entity_ids: Sequence[str] = (),
        ef_search: int | None = None,
    ) -> list[ChunkMatch]:
        """The nearest chunks by cosine similarity, most similar first.

        ``entity_ids`` keeps only chunks mentioning at least one of them; empty means
        no filter. ``ef_search`` tunes the HNSW index's accuracy/latency trade-off and
        is ignored by stores that do an exact scan.
        """
        ...


class InMemoryChunkStore:
    """Chunks in a list, searched by exact cosine.

    Backs ``--dry-run`` and every test that needs real ranking without a database.
    Being exact is the point: when a recall number differs between this and Postgres,
    the difference is the HNSW index's approximation, which is exactly what
    ``ragtorio index recall --sweep`` is for.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, tuple[EmbeddedChunk, ...]] = {}

    def replace_all(self, wiki: str, chunks: Sequence[EmbeddedChunk]) -> None:
        self.chunks[wiki] = tuple(chunks)

    def count(self, wiki: str) -> int:
        return len(self.chunks.get(wiki, ()))

    def titles(self, wiki: str) -> frozenset[str]:
        return frozenset(stored.chunk.title for stored in self.chunks.get(wiki, ()))

    def search(
        self,
        wiki: str,
        embedding: Sequence[float],
        limit: int = 10,
        entity_ids: Sequence[str] = (),
        ef_search: int | None = None,  # noqa: ARG002 - an exact scan has nothing to tune
    ) -> list[ChunkMatch]:
        wanted = set(entity_ids)
        matches = [
            ChunkMatch(chunk=stored.chunk, score=cosine(embedding, stored.embedding))
            for stored in self.chunks.get(wiki, ())
            if not wanted or wanted.intersection(stored.chunk.mentioned_entity_ids)
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]
