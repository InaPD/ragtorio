"""Merging: the labels, the budget, and what gets cut when it binds."""

from __future__ import annotations

from ragtorio.index.models import Chunk, ChunkMatch
from ragtorio.retrieve.merge import merge
from ragtorio.retrieve.models import GraphResult, Route


def route(**kwargs: object) -> Route:
    return Route.model_validate({"question": "q", "intent": "both", **kwargs})


def match(chunk_id: str, text: str = "some prose about oil", title: str = "Oil") -> ChunkMatch:
    return ChunkMatch(
        chunk=Chunk(
            chunk_id=chunk_id,
            wiki="factorio",
            page_id=1,
            title=title,
            revision_id=7,
            section_path=("Overview",),
            text=text,
        ),
        score=0.9,
    )


def graph(*lines: str) -> GraphResult:
    return GraphResult(template="recipe_tree", lines=lines)


def test_the_two_halves_arrive_as_separately_labeled_blocks():
    """Phase 6 has to be able to tell the model which evidence is which."""
    context = merge(route(), graph=graph("Iron plate: 1"), passages=[match("c1")])
    assert [block.label for block in context.blocks] == ["graph_facts", "passages"]


def test_a_half_that_retrieved_nothing_produces_no_block():
    assert [b.label for b in merge(route(), passages=[match("c1")]).blocks] == ["passages"]
    assert [b.label for b in merge(route(), graph=graph("x")).blocks] == ["graph_facts"]
    assert merge(route()).is_empty


def test_an_empty_graph_result_is_not_an_empty_block():
    """A template that matched nothing should not put a labeled, blank section in the
    prompt for the model to reason about."""
    assert merge(route(), graph=GraphResult(template="recipe_tree")).blocks == ()


def test_every_passage_is_rendered_with_the_id_a_citation_will_use():
    context = merge(route(), passages=[match("factorio:1:0000")])
    assert "[factorio:1:0000]" in context.blocks[0].text
    assert context.chunk_ids == ("factorio:1:0000",)


def test_the_heading_trail_travels_with_the_passage():
    context = merge(route(), passages=[match("c1", title="Oil processing")])
    assert "Oil processing > Overview" in context.blocks[0].text


def test_a_chunk_returned_twice_is_included_once():
    """Two copies of a passage in a prompt read to the model as corroboration."""
    context = merge(route(), passages=[match("c1"), match("c1"), match("c2")])
    assert context.chunk_ids == ("c1", "c2")


def test_passages_are_dropped_when_the_budget_binds():
    long_text = "oil " * 1000
    context = merge(route(), passages=[match(f"c{i}", long_text) for i in range(10)])

    assert 0 < len(context.chunk_ids) < 10
    assert context.total_tokens <= 6000


def test_graph_facts_are_never_the_thing_that_gets_cut():
    """A truncated recipe tree is a wrong answer; one fewer passage is not."""
    facts = graph(*[f"Ingredient {i}: 1" for i in range(50)])
    context = merge(route(), graph=facts, passages=[match("c1", "oil " * 400)], token_budget=200)

    labels = [block.label for block in context.blocks]
    assert labels == ["graph_facts"]
    assert context.chunk_ids == ()


def test_a_passage_too_long_for_the_budget_is_skipped_not_truncated():
    """A later, shorter passage still gets its chance."""
    context = merge(
        route(),
        passages=[match("huge", "oil " * 4000), match("small", "a short passage about oil")],
        token_budget=300,
    )
    assert context.chunk_ids == ("small",)


def test_the_route_and_its_timing_travel_with_the_context():
    context = merge(route(template="recipe_tree"), graph=graph("x"), latency_ms=12.5)
    assert context.route.template == "recipe_tree"
    assert context.latency_ms == 12.5
    assert context.question == "q"


def test_token_counts_are_reported_per_block_and_in_total():
    context = merge(route(), graph=graph("Iron plate: 1"), passages=[match("c1")])
    assert all(block.tokens > 0 for block in context.blocks)
    assert context.total_tokens == sum(block.tokens for block in context.blocks)
