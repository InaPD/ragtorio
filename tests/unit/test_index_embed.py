"""Embedding providers: determinism, the asymmetry, and the vendor request shape."""

from __future__ import annotations

import httpx
import pytest
import respx

from ragtorio.index.embed import (
    VOYAGE_API_URL,
    HashingEmbedder,
    VoyageEmbedder,
    build_provider,
    cosine,
)


def test_the_same_text_always_gives_the_same_vector():
    """The point of the stand-in: a recall number that is reproducible anywhere."""
    left, right = HashingEmbedder(), HashingEmbedder()
    assert left.embed_query("crude oil") == right.embed_query("crude oil")


def test_vectors_are_unit_length_so_a_dot_product_is_a_cosine():
    vector = HashingEmbedder().embed_query("oil refinery cracking")
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_an_empty_text_gives_a_zero_vector_rather_than_an_error():
    """Chunking should never produce one, but a query might, and a crash here would
    surface three layers away as a failed search."""
    vector = HashingEmbedder(dimensions=16).embed_query("")
    assert vector == (0.0,) * 16


def test_dimensions_are_honoured_and_advertised():
    embedder = HashingEmbedder(dimensions=64)
    assert embedder.dimensions == 64
    assert len(embedder.embed_documents(["text"])[0]) == 64


def test_a_provider_cannot_be_built_with_a_useless_width():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        HashingEmbedder(dimensions=0)


def test_lexical_overlap_still_ranks_the_right_passage_first():
    """Enough signal for the store's own tests to assert on an ordering."""
    embedder = HashingEmbedder()
    query = embedder.embed_query("how do I crack heavy oil")
    relevant = embedder.embed_documents(["heavy oil can be cracked into light oil"])[0]
    unrelated = embedder.embed_documents(["biters are attracted by pollution"])[0]
    assert cosine(query, relevant) > cosine(query, unrelated)


def test_cosine_rejects_a_dimension_mismatch():
    """The failure that otherwise shows up as a nonsense similarity score."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_build_provider_names_what_it_knows_when_asked_for_something_else():
    with pytest.raises(ValueError, match="unknown embedding provider"):
        build_provider("word2vec")


def test_build_provider_returns_the_stand_in_by_name():
    assert build_provider("hashing", dimensions=32).dimensions == 32


@respx.mock
def test_voyage_sends_the_asymmetric_input_type_and_the_requested_width():
    """Query and document vectors are not interchangeable; the flag is what says so."""
    route = respx.post(VOYAGE_API_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
    )
    embedder = VoyageEmbedder(model="voyage-3.5", dimensions=2, api_key="test-key")

    embedder.embed_documents(["a passage"])
    embedder.embed_query("a question")

    document_request, query_request = (call.request for call in route.calls)
    assert document_request.read().decode().count('"input_type":"document"') == 1
    assert query_request.read().decode().count('"input_type":"query"') == 1
    assert '"output_dimension":2' in query_request.read().decode()
    assert query_request.headers["Authorization"] == "Bearer test-key"


@respx.mock
def test_voyage_vectors_come_back_normalised_and_in_input_order():
    respx.post(VOYAGE_API_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 4.0]},
                ]
            },
        )
    )
    vectors = VoyageEmbedder(dimensions=2, api_key="k").embed_documents(["first", "second"])
    assert vectors == [(0.6, 0.8), (0.0, 1.0)]


@respx.mock
def test_voyage_batches_rather_than_sending_one_huge_request():
    route = respx.post(VOYAGE_API_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    VoyageEmbedder(dimensions=1, api_key="k", batch_size=1).embed_documents(["a", "b", "c"])
    assert route.call_count == 3


def test_voyage_refuses_to_start_without_a_key(monkeypatch: pytest.MonkeyPatch):
    """Failing here beats failing after an hour of chunking."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder()


def test_embedding_nothing_costs_no_request():
    assert VoyageEmbedder(api_key="k").embed_documents([]) == []
