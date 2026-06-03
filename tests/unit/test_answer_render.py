"""Rendering retrieved context into a prompt.

What matters here is the shape the evidence arrives in, because that is what the model
reasons over: a recipe tree that loses its nesting is a list of numbers with no
relationship between them, and a research chain that loses its order is a set.
"""

from __future__ import annotations

from ragtorio.answer.render import citation_instruction, render_graph, render_prompt
from ragtorio.index.models import Chunk, ChunkMatch
from ragtorio.retrieve.merge import merge
from ragtorio.retrieve.models import GraphResult, Route


def route(**kwargs: object) -> Route:
    return Route.model_validate({"question": "what is it made of", "intent": "both", **kwargs})


def match(chunk_id: str = "factorio:1:0000", text: str = "Circuits are assembled.") -> ChunkMatch:
    return ChunkMatch(
        chunk=Chunk(
            chunk_id=chunk_id,
            wiki="factorio",
            page_id=1,
            title="Electronic circuit",
            revision_id=7,
            section_path=("Crafting",),
            text=text,
        ),
        score=0.9,
    )


TREE = {
    "title": "Electronic circuit",
    "amount": 1.0,
    "is_raw": False,
    "children": [
        {
            "title": "Iron plate",
            "amount": 1.0,
            "is_raw": False,
            "children": [
                {"title": "Iron ore", "amount": 1.0, "is_raw": True, "children": []},
            ],
        },
        {
            "title": "Copper cable",
            "amount": 3.0,
            "is_raw": False,
            "children": [
                {"title": "Copper ore", "amount": 1.5, "is_raw": True, "children": []},
            ],
        },
    ],
}


def tree_result() -> GraphResult:
    return GraphResult(template="recipe_tree", rows=(TREE,), lines=("Electronic circuit: 1",))


def test_a_recipe_tree_keeps_its_nesting_and_its_quantities():
    rendered = render_graph(tree_result())
    assert "- 1 Electronic circuit" in rendered
    assert "  - 1 Iron plate" in rendered
    assert "    - 1 Iron ore (raw material)" in rendered
    assert "      " not in rendered  # nothing deeper than the tree actually is


def test_a_recipe_tree_ends_with_the_total_it_exists_to_produce():
    rendered = render_graph(tree_result())
    assert "Raw material total for 1 Electronic circuit: 1.5 Copper ore, 1 Iron ore." in rendered


def test_a_branch_cut_for_looping_says_so_rather_than_looking_raw():
    """A cycle that renders as "(raw material)" would tell the model that sulfuric acid
    is mined, which is exactly the wrong conclusion to hand it."""
    looped = {
        "title": "Sulfuric acid",
        "amount": 5.0,
        "is_raw": True,
        "truncated": True,
        "children": [],
    }
    rendered = render_graph(GraphResult(template="recipe_tree", rows=(looped,)))
    assert "the chain loops here" in rendered
    assert "raw material" not in rendered


def test_a_research_chain_is_numbered_because_the_order_is_the_answer():
    rows = (
        {
            "target": "Advanced oil processing",
            "unlock": "Advanced oil processing (research)",
            "prerequisites": ["Oil processing", "Chemical science pack", "Oil processing"],
        },
    )
    rendered = render_graph(GraphResult(template="unlock_chain", rows=rows))
    assert "  1. Oil processing" in rendered
    assert "  2. Chemical science pack" in rendered
    # A diamond in the prerequisite graph must not become a repeated research step.
    assert rendered.count("Oil processing\n") <= 1


def test_a_template_without_a_structured_renderer_keeps_the_retrievers_lines():
    result = GraphResult(
        template="consumers_of",
        rows=({"recipe": "Battery"},),
        lines=("Battery consumes 1 of it.",),
    )
    assert render_graph(result) == "Battery consumes 1 of it."


def test_the_prompt_labels_the_two_kinds_of_evidence_separately():
    context = merge(route(), graph=tree_result(), passages=[match()])
    prompt = render_prompt(context)
    assert "<question>" in prompt
    assert "<graph_facts>" in prompt and "</graph_facts>" in prompt
    assert "<passages>" in prompt and "[factorio:1:0000]" in prompt


def test_a_question_that_retrieved_nothing_says_so_rather_than_arriving_empty():
    prompt = render_prompt(merge(route()))
    assert "<no_evidence>" in prompt


def test_the_citation_instruction_only_offers_what_was_actually_retrieved():
    graph_only = citation_instruction(merge(route(), graph=tree_result()))
    assert "[graph]" in graph_only
    assert "factorio:1:0000" not in graph_only

    passages_only = citation_instruction(merge(route(), passages=[match()]))
    assert "[graph]" not in passages_only
    assert "[factorio:1:0000]" in passages_only


def test_with_no_evidence_the_instruction_is_to_say_so():
    assert "do not know" in citation_instruction(merge(route()))
