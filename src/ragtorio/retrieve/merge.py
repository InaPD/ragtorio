"""Turning two retrievals into one labeled, budgeted context.

**Labeled, not concatenated.** A graph fact came from a template parameter somebody
typed into an infobox; a passage is prose that may be stale, hedged, or about a
different game version. Handing the answering model one undifferentiated wall of text
throws away the single most useful thing the system knows about its own evidence. The
blocks are named, and Phase 6's prompt can tell the model what each name means.

**Graph facts are never the thing that gets cut.** When the budget binds, passages are
dropped from the bottom. There are far fewer graph lines, they are far denser, and they
are the half the benchmark is meant to show matters; a truncated recipe tree is a wrong
answer, while one fewer passage is a slightly less contextualised right one.

Chunk ids come out alongside the text because Phase 6 validates every citation against
them: a claim citing an id that was never retrieved is the definition of a fabricated
source, and that check needs the exact set.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragtorio.index.chunk import estimate_tokens
from ragtorio.index.models import ChunkMatch
from ragtorio.retrieve.models import ContextBlock, GraphResult, RetrievedContext, Route

#: Total context tokens. Comfortable for an answering prompt and small enough that the
#: cached system prefix stays the dominant part of a request.
DEFAULT_TOKEN_BUDGET = 6000

#: A passage shorter than this after truncation is not worth the citation it costs.
MIN_PASSAGE_TOKENS = 40


def merge(
    route: Route,
    graph: GraphResult | None = None,
    passages: Sequence[ChunkMatch] = (),
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    latency_ms: float = 0.0,
) -> RetrievedContext:
    """Combine both halves into labeled blocks that fit the budget."""
    blocks: list[ContextBlock] = []
    remaining = token_budget

    if graph is not None and not graph.is_empty:
        text = "\n".join(graph.lines)
        tokens = estimate_tokens(text)
        blocks.append(ContextBlock(label="graph_facts", text=text, tokens=tokens))
        remaining -= tokens

    kept, chunk_ids = _fit(passages, remaining)
    if kept:
        text = "\n\n".join(kept)
        blocks.append(ContextBlock(label="passages", text=text, tokens=estimate_tokens(text)))

    return RetrievedContext(
        question=route.question,
        route=route,
        blocks=tuple(blocks),
        chunk_ids=tuple(chunk_ids),
        graph=graph,
        latency_ms=latency_ms,
    )


def _fit(passages: Sequence[ChunkMatch], budget: int) -> tuple[list[str], list[str]]:
    """Render passages in rank order until the budget runs out.

    Deduplication is by ``chunk_id``: the vector retriever can legitimately return the
    same chunk twice when a filtered and an unfiltered search are combined, and two
    copies of a passage in a prompt reads to the model as corroboration.
    """
    rendered: list[str] = []
    chunk_ids: list[str] = []
    seen: set[str] = set()

    for match in passages:
        chunk = match.chunk
        if chunk.chunk_id in seen or budget <= MIN_PASSAGE_TOKENS:
            continue
        text = f"[{chunk.chunk_id}] {chunk.heading}\n{chunk.text}"
        tokens = estimate_tokens(text)
        if tokens > budget:
            continue  # a later, shorter passage may still fit
        seen.add(chunk.chunk_id)
        rendered.append(text)
        chunk_ids.append(chunk.chunk_id)
        budget -= tokens

    return rendered, chunk_ids
