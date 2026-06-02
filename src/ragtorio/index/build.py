"""One index build: articles in, embedded chunks out, with a report to check it by.

Kept apart from the CLI so the whole pipeline can be driven from a test with an
in-memory repository and store, and apart from the chunker so that a bad build can be
diagnosed as either "the chunks are wrong" or "the vectors are wrong" and never both
at once. The report is the evidence for that split: it says how many pages yielded
nothing and how many chunks carry no entity mention at all, which are the two ways
this stage fails quietly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from ragtorio.config import WikiProfile
from ragtorio.index.chunk import Chunker, estimate_tokens
from ragtorio.index.embed import EmbeddingProvider
from ragtorio.index.models import Chunk, EmbeddedChunk
from ragtorio.index.repository import ChunkSourceRepository
from ragtorio.ontology.canonical import TitleCanonicalizer

#: Texts per ``embed_documents`` call. Large enough that a local model's per-call
#: overhead disappears, small enough to stay well inside a hosted provider's request
#: limit and to give progress reporting something to report.
_BATCH = 64


class IndexReport(BaseModel):
    """What one build produced, and the two numbers that say whether it worked.

    ``pages_without_chunks`` catches a chunker that silently ate a whole class of
    article - a redirect-shaped stub is fine, half the mainspace is not.
    ``chunks_without_mentions`` catches the opposite failure: chunks that exist but
    that Phase 5's entity filter can never reach. It was the second of those that
    showed 23 changelog pages producing 42% of the Factorio index.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    dimensions: int
    pages_seen: int = 0
    pages_excluded: int = 0
    pages_without_chunks: int = 0
    chunks: int = 0
    chunks_without_mentions: int = 0
    total_tokens: int = 0

    @property
    def mention_coverage(self) -> float:
        """Share of chunks carrying at least one entity id."""
        if not self.chunks:
            return 0.0
        return 1 - self.chunks_without_mentions / self.chunks

    @property
    def mean_tokens(self) -> float:
        """Average chunk size, the number to sanity-check a budget change against."""
        return self.total_tokens / self.chunks if self.chunks else 0.0


class IndexResult(BaseModel):
    """A build's embedded chunks and its report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks: tuple[EmbeddedChunk, ...]
    report: IndexReport


def build_index(
    profile: WikiProfile,
    source: ChunkSourceRepository,
    provider: EmbeddingProvider,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> IndexResult:
    """Chunk a wiki's articles and embed them. Does not write anything."""
    wiki = profile.wiki.id
    titles = TitleCanonicalizer(source.redirects(wiki))
    known = frozenset(f"{wiki}:{title}" for title in source.entity_titles(wiki))
    chunker = Chunker(wiki, profile.index, titles, known or None)

    chunks: list[Chunk] = []
    pages_seen = 0
    pages_excluded = 0
    pages_without_chunks = 0
    for page in source.articles(wiki, profile.index.namespace_id):
        if limit is not None and pages_seen >= limit:
            break
        pages_seen += 1
        if profile.index.excludes(page.title):
            pages_excluded += 1
            continue
        page_chunks = chunker.chunk_page(page)
        if not page_chunks:
            pages_without_chunks += 1
        chunks.extend(page_chunks)

    embedded = _embed(chunks, provider, progress)
    return IndexResult(
        chunks=tuple(embedded),
        report=IndexReport(
            provider=provider.name,
            dimensions=provider.dimensions,
            pages_seen=pages_seen,
            pages_excluded=pages_excluded,
            pages_without_chunks=pages_without_chunks,
            chunks=len(chunks),
            chunks_without_mentions=sum(1 for c in chunks if not c.mentioned_entity_ids),
            total_tokens=sum(estimate_tokens(c.text) for c in chunks),
        ),
    )


def _embed(
    chunks: Sequence[Chunk],
    provider: EmbeddingProvider,
    progress: Callable[[str], None] | None,
) -> list[EmbeddedChunk]:
    """Embed in batches, checking the provider keeps its own dimension promise.

    A provider that returns a different width than it advertises would otherwise fail
    at the ``INSERT``, thousands of rows later, with an error naming the column rather
    than the model.
    """
    embedded: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), _BATCH):
        batch = chunks[start : start + _BATCH]
        vectors = provider.embed_documents([chunk.text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != provider.dimensions:
                raise ValueError(
                    f"{provider.name} advertises {provider.dimensions} dimensions but "
                    f"returned {len(vector)} for chunk {chunk.chunk_id}"
                )
            embedded.append(EmbeddedChunk(chunk=chunk, embedding=vector))
        if progress:
            progress(f"embedded {len(embedded):,}/{len(chunks):,} chunks")
    return embedded
