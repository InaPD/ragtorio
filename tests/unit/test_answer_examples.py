"""The example run: the exit criterion, and the documents it writes.

The run itself needs a model and two databases, so what is tested here is the part
that does not: that the summary in the written document is computed from the answers
rather than asserted, that an ungrounded answer is visible in the index rather than
buried, and that the shipped question set is actually the size the plan asks for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragtorio.answer.examples import (
    TARGET_QUESTIONS,
    ExampleQuestion,
    load_examples,
    run_examples,
    write_examples,
)
from ragtorio.answer.models import Answer, Citation, Usage


def answer(question: str, **kwargs: Any) -> Answer:
    return Answer(
        question=question,
        text="One iron plate [factorio:1:0000].",
        citations=(
            Citation(
                chunk_id="factorio:1:0000",
                title="Electronic circuit",
                section="Crafting",
                revision_id=7,
                url="https://wiki.factorio.com/index.php?title=Electronic_circuit&oldid=7",
            ),
        ),
        cited_graph=True,
        intent="graph",
        template="recipe_tree",
        usage=Usage(output_tokens=42),
        generation_ms=100.0,
        **kwargs,
    )


class FakePipeline:
    def __init__(self, *answers: Answer) -> None:
        self._answers = list(answers)
        self.asked: list[str] = []

    def answer(self, question: str, on_text: Any = None) -> Answer:
        self.asked.append(question)
        return self._answers[len(self.asked) - 1]


def questions(*texts: str) -> tuple[ExampleQuestion, ...]:
    return tuple(ExampleQuestion(question=text) for text in texts)


def run(*answers: Answer, notes: tuple[str, ...] = ()) -> Any:
    asked = questions(*[a.question for a in answers])
    if notes:
        asked = tuple(q.model_copy(update={"note": n}) for q, n in zip(asked, notes, strict=True))
    return run_examples("factorio", FakePipeline(*answers), asked)  # type: ignore[arg-type]


def test_the_summary_counts_what_the_answers_actually_contain():
    result = run(answer("q one"), answer("q two"))
    assert len(result.answers) == 2
    assert result.citations == 2
    assert result.citations_resolve
    assert result.output_tokens == 84


def test_an_answer_with_an_unverified_claim_fails_the_criterion():
    result = run(answer("q one"), answer("q two", unverified=("Copper [factorio:9:9].",)))
    assert not result.citations_resolve
    assert len(result.unverified) == 1


def test_latency_percentiles_come_from_the_run_not_from_a_note_somebody_wrote():
    fast = answer("q one").model_copy(update={"generation_ms": 100.0})
    slow = answer("q two").model_copy(update={"generation_ms": 900.0})
    result = run_examples("factorio", FakePipeline(fast, slow), questions("q one", "q two"))  # type: ignore[arg-type]
    assert result.p50_ms == 100.0
    assert result.p95_ms == 900.0


def test_each_answer_is_written_as_its_own_document_with_a_clickable_source(tmp_path: Path):
    result = run(answer("what raw ore does one electronic circuit cost"))
    written = write_examples(result, tmp_path)
    assert [p.name for p in written] == [
        "README.md",
        "01-what-raw-ore-does-one-electronic-circuit-cost.md",
    ]
    document = written[1].read_text(encoding="utf-8")
    assert "oldid=7" in document
    assert "`[factorio:1:0000]`" in document
    assert "recipe_tree" in document


def test_the_note_explaining_why_a_question_is_in_the_set_is_kept(tmp_path: Path):
    result = run(answer("q one"), notes=("Three hops.",))
    document = write_examples(result, tmp_path)[1].read_text(encoding="utf-8")
    assert "Three hops." in document


def test_an_unverified_claim_is_named_in_its_document_and_flagged_in_the_index(tmp_path: Path):
    result = run(answer("q one", unverified=("Copper [factorio:9:9].",)))
    written = write_examples(result, tmp_path)
    assert "Unverified claims" in written[1].read_text(encoding="utf-8")
    index = written[0].read_text(encoding="utf-8")
    assert "unverified claims" in index
    assert "not all" in index


def test_the_index_records_the_numbers_the_exit_criterion_is_read_off(tmp_path: Path):
    index = write_examples(run(answer("q one")), tmp_path)[0].read_text(encoding="utf-8")
    assert "| Citations resolving to retrieved chunks | 100% |" in index
    assert "p95" in index


def test_a_refusal_is_reported_rather_than_written_as_a_blank_answer(tmp_path: Path):
    refused = answer("how much is a licence").model_copy(
        update={"text": "", "stop_reason": "refusal", "citations": ()}
    )
    result = run_examples("factorio", FakePipeline(refused), questions("how much is a licence"))  # type: ignore[arg-type]
    assert result.refusals == 1
    document = write_examples(result, tmp_path)[1].read_text(encoding="utf-8")
    assert "declined" in document


def test_a_wiki_with_no_example_set_is_an_error_not_an_empty_perfect_run(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="example questions"):
        load_examples("nowhere", tmp_path)


def test_the_shipped_factorio_set_is_the_size_the_plan_asks_for():
    loaded = load_examples("factorio")
    assert len(loaded) == TARGET_QUESTIONS
    assert all(question.note for question in loaded)
