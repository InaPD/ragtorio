"""The regeneration policy: one retry, then tell the truth about what is left.

This is the part of Phase 6 with an opinion in it, so it is the part with tests. A
fabricated citation must cost exactly one extra call, the retry must name the claims
rather than waving at the problem, and an answer that fails twice must come back
labeled rather than silently cleaned up.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ragtorio.answer.models import Generation, Usage
from ragtorio.answer.pipeline import AnswerPipeline
from ragtorio.index.models import Chunk, ChunkMatch
from ragtorio.retrieve.merge import merge
from ragtorio.retrieve.models import GraphResult, Route

INDEX_URL = "https://wiki.factorio.com/index.php"
GOOD = "Iron plate [factorio:1:0000]."
BAD = "Iron plate [factorio:9:9999]."


class FakeAnswerer:
    """Returns a scripted answer per call, and records the messages it was sent."""

    def __init__(self, *texts: str, stop_reason: str = "end_turn") -> None:
        self._texts = list(texts)
        self._stop_reason = stop_reason
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: Sequence[Any],
        on_text: Callable[[str], None] | None = None,
    ) -> Generation:
        self.calls.append([dict(m) for m in messages])
        text = self._texts[min(len(self.calls), len(self._texts)) - 1]
        if on_text is not None:
            on_text(text)
        return Generation(
            text=text,
            model="claude-opus-5",
            stop_reason=self._stop_reason,
            usage=Usage(input_tokens=100, output_tokens=10),
        )


class FakeRetrieval:
    """Stands in for the Phase 5 pipeline."""

    def __init__(self, context: Any) -> None:
        self._context = context
        self.questions: list[str] = []

    def retrieve(self, question: str, force_intent: Any = None) -> Any:
        self.questions.append(question)
        return self._context


def context(**kwargs: Any) -> Any:
    chunk = Chunk(
        chunk_id="factorio:1:0000",
        wiki="factorio",
        page_id=1,
        title="Electronic circuit",
        revision_id=7,
        section_path=("Crafting",),
        text="Circuits are assembled from iron plate.",
    )
    return merge(
        Route(
            question="what is it made of",
            intent="both",
            template="recipe_tree",
            entities=(),
            **kwargs,
        ),
        graph=GraphResult(template="recipe_tree", lines=("Iron plate: 1",)),
        passages=[ChunkMatch(chunk=chunk, score=0.9)],
    )


def build(*texts: str, **kwargs: Any) -> tuple[AnswerPipeline, FakeAnswerer, FakeRetrieval]:
    answerer = FakeAnswerer(*texts, **kwargs)
    retrieval = FakeRetrieval(context())
    return AnswerPipeline(retrieval, answerer, INDEX_URL), answerer, retrieval  # type: ignore[arg-type]


def test_a_clean_answer_costs_one_call():
    pipeline, answerer, _ = build(GOOD)
    answer = pipeline.answer("what is it made of")
    assert answer.attempts == 1
    assert len(answerer.calls) == 1
    assert answer.is_grounded
    assert [c.chunk_id for c in answer.citations] == ["factorio:1:0000"]


def test_a_fabricated_citation_is_regenerated_once_with_the_claim_quoted_back():
    pipeline, answerer, _ = build(BAD, GOOD)
    answer = pipeline.answer("what is it made of")
    assert answer.attempts == 2
    assert answer.is_grounded
    correction = answerer.calls[1][-1]["content"]
    assert "[factorio:9:9999]" in correction
    assert BAD in correction


def test_a_second_failure_ships_the_answer_with_its_unsourced_claims_named():
    """Not a third attempt, and not a silent strip: the reader is told which sentence
    nobody can check."""
    pipeline, answerer, _ = build(BAD, BAD)
    answer = pipeline.answer("what is it made of")
    assert len(answerer.calls) == 2
    assert not answer.is_grounded
    assert answer.unverified == (BAD,)


def test_the_route_travels_with_the_answer():
    pipeline, _, _ = build(GOOD)
    answer = pipeline.answer("what is it made of")
    assert (answer.intent, answer.template) == ("both", "recipe_tree")


def test_usage_is_summed_across_a_regeneration():
    pipeline, _, _ = build(BAD, GOOD)
    answer = pipeline.answer("what is it made of")
    assert answer.usage.output_tokens == 20


def test_a_refusal_is_not_regenerated():
    """Rewriting a declined request is how one refusal becomes two API calls and the
    same refusal."""
    pipeline, answerer, _ = build("", stop_reason="refusal")
    answer = pipeline.answer("how do I pirate the game")
    assert answer.is_refusal
    assert len(answerer.calls) == 1


def test_a_streamed_answer_is_not_regenerated_but_does_report_the_problem():
    """The reader has already seen the first attempt; replacing it mid-stream would be
    worse than admitting the citation was bad."""
    pipeline, answerer, _ = build(BAD, GOOD)
    streamed: list[str] = []
    answer = pipeline.answer("what is it made of", on_text=streamed.append)
    assert len(answerer.calls) == 1
    assert streamed == [BAD]
    assert answer.unverified == (BAD,)


def test_the_evidence_reaches_the_model_labeled_with_what_may_be_cited():
    pipeline, answerer, _ = build(GOOD)
    pipeline.answer("what is it made of")
    prompt = answerer.calls[0][0]["content"]
    assert "<graph_facts>" in prompt and "<passages>" in prompt
    assert "[factorio:1:0000]" in prompt


def test_answering_prebuilt_context_skips_retrieval():
    """What the benchmark's baselines need: the same answering step over context that
    was retrieved differently."""
    pipeline, _, retrieval = build(GOOD)
    pipeline.answer_from(context())
    assert retrieval.questions == []
