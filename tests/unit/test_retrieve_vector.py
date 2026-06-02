"""The vector half: the entity filter, and the fallback that stops it being absolute."""

from __future__ import annotations

from ragtorio.index.embed import HashingEmbedder
from ragtorio.index.models import Chunk, EmbeddedChunk
from ragtorio.index.store import InMemoryChunkStore
from ragtorio.retrieve.models import ResolvedEntity, Route
from ragtorio.retrieve.vector import VectorRetriever

PASSAGES = {
    "c1": ("Oil processing", "heavy oil cracks into light oil", ("factorio:Heavy oil",)),
    "c2": ("Pollution", "pollution attracts biters to the base", ("factorio:Pollution",)),
    "c3": ("Railway", "trains need fuel to move between stations", ("factorio:Railway",)),
    "c4": ("Fluid system", "pipes move fluid between machines", ("factorio:Pipe",)),
}


def store() -> InMemoryChunkStore:
    embedder = HashingEmbedder()
    filled = InMemoryChunkStore()
    filled.replace_all(
        "factorio",
        [
            EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=chunk_id,
                    wiki="factorio",
                    page_id=i,
                    title=title,
                    revision_id=1,
                    text=text,
                    mentioned_entity_ids=mentions,
                ),
                embedding=embedder.embed_documents([text])[0],
            )
            for i, (chunk_id, (title, text, mentions)) in enumerate(PASSAGES.items(), start=1)
        ],
    )
    return filled


def route(question: str, *entity_ids: str) -> Route:
    return Route(
        question=question,
        intent="vector",
        entities=tuple(
            ResolvedEntity(mention=entity_id, id=entity_id, title=entity_id, matched_by="title")
            for entity_id in entity_ids
        ),
    )


def test_without_entities_it_is_an_ordinary_similarity_search():
    retriever = VectorRetriever(store(), HashingEmbedder(), "factorio", top_k=2)
    matches = retriever.run(route("heavy oil cracks into light oil"))
    assert matches[0].chunk.title == "Oil processing"


def test_the_entity_filter_narrows_the_search():
    """Enough passages carry the entity, so the filter is allowed to be the answer."""
    retriever = VectorRetriever(store(), HashingEmbedder(), "factorio", top_k=4)
    matches = retriever.run(route("anything at all", "factorio:Heavy oil"))
    # Only one chunk mentions it, so widening kicks in - but it stays first.
    assert matches[0].chunk.chunk_id == "c1"


def test_a_filter_that_matches_almost_nothing_widens_rather_than_returning_one_hit():
    """A question whose entity appears in one passage is a routing success and a
    retrieval failure only if the filter gets the last word."""
    retriever = VectorRetriever(store(), HashingEmbedder(), "factorio", top_k=4)
    matches = retriever.run(route("trains and fuel", "factorio:Heavy oil"))
    assert len(matches) > 1
    assert {m.chunk.chunk_id for m in matches} > {"c1"}


def test_widening_never_duplicates_a_chunk():
    retriever = VectorRetriever(store(), HashingEmbedder(), "factorio", top_k=4)
    matches = retriever.run(route("pollution biters", "factorio:Pollution"))
    ids = [m.chunk.chunk_id for m in matches]
    assert len(ids) == len(set(ids))


def test_the_top_k_is_respected_after_widening():
    retriever = VectorRetriever(store(), HashingEmbedder(), "factorio", top_k=2)
    assert len(retriever.run(route("anything", "factorio:Heavy oil"))) == 2


def test_an_entity_no_passage_mentions_still_returns_prose():
    retriever = VectorRetriever(store(), HashingEmbedder(), "factorio", top_k=3)
    matches = retriever.run(route("pipes move fluid", "factorio:Nothing"))
    assert matches and matches[0].chunk.title == "Fluid system"
