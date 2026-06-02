"""The router-accuracy harness, and the shipped labeled set it scores against."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragtorio.retrieve.evaluate import (
    LabeledQuestion,
    RouteOutcome,
    RouterReport,
    evaluate_router,
    load_questions,
)
from ragtorio.retrieve.models import Route


class StubRouter:
    """Answers with a fixed route per question."""

    def __init__(self, answers: dict[str, Route]) -> None:
        self.answers = answers

    def route(self, question: str) -> Route:
        return self.answers.get(question, Route(question=question, intent="both"))


def route(intent: str, template: str | None = None, **kwargs: object) -> Route:
    return Route.model_validate({"question": "q", "intent": intent, "template": template, **kwargs})


def outcome(intent_ok: bool, template_ok: bool | None = None) -> RouteOutcome:
    return RouteOutcome(
        question="q",
        expected_intent="graph",
        route=route("graph"),
        intent_ok=intent_ok,
        template_ok=template_ok,
    )


def test_a_specific_label_accepts_the_same_intent():
    assert LabeledQuestion(question="q", intent="graph").intent_ok("graph")


def test_a_specific_label_accepts_both_because_extra_retrieval_is_a_cost_not_an_error():
    assert LabeledQuestion(question="q", intent="graph").intent_ok("both")


def test_a_specific_label_rejects_the_opposite_intent():
    assert not LabeledQuestion(question="q", intent="graph").intent_ok("vector")


def test_a_both_label_accepts_anything():
    """The question genuinely needs both halves, so either choice can answer it."""
    labeled = LabeledQuestion(question="q", intent="both")
    assert all(labeled.intent_ok(i) for i in ("graph", "vector", "both"))  # type: ignore[arg-type]


def test_intent_and_template_are_scored_separately():
    """They fail differently, and a blended number would hide the worse one."""
    report = RouterReport(
        outcomes=(outcome(True, True), outcome(True, False), outcome(False, True))
    )
    assert report.intent_accuracy == pytest.approx(2 / 3)
    assert report.template_accuracy == pytest.approx(2 / 3)


def test_template_accuracy_only_counts_questions_whose_label_names_one():
    report = RouterReport(outcomes=(outcome(True, True), outcome(True, None)))
    assert report.template_accuracy == 1.0


def test_the_exit_criterion_needs_both_measures_at_ninety_percent():
    """Exactly 90% passes; either measure below it fails, independently."""
    nine = tuple(outcome(True, True) for _ in range(9))
    assert RouterReport(outcomes=(*nine, outcome(False, True))).meets_target

    eight = tuple(outcome(True, True) for _ in range(8))
    assert not RouterReport(
        outcomes=(*eight, outcome(False, True), outcome(False, True))
    ).meets_target
    assert not RouterReport(
        outcomes=(*eight, outcome(True, False), outcome(True, False))
    ).meets_target


def test_an_empty_report_scores_zero_rather_than_dividing_by_nothing():
    report = RouterReport()
    assert report.intent_accuracy == 0.0
    assert report.template_accuracy == 0.0
    assert report.p50_ms == 0.0


def test_evaluating_routes_every_question_and_records_what_happened():
    questions = [
        LabeledQuestion(question="a", intent="graph", template="recipe_tree"),
        LabeledQuestion(question="b", intent="vector"),
    ]
    router = StubRouter(
        {
            "a": route("graph", "recipe_tree"),
            "b": route("graph"),  # wrong: a prose question sent to the graph
        }
    )
    report = evaluate_router(router, questions)

    assert report.intent_accuracy == 0.5
    assert report.template_accuracy == 1.0
    assert [o.question for o in report.failures] == ["b"]


def test_entities_that_resolved_to_nothing_are_surfaced_separately():
    """A router that classifies perfectly but names unresolvable entities produces
    empty graph results, and the accuracy numbers alone would not show it."""
    router = StubRouter(
        {"a": route("graph", "recipe_tree", unresolved=("Flurbo engine", "Widget"))}
    )
    report = evaluate_router(
        router, [LabeledQuestion(question="a", intent="graph", template="recipe_tree")]
    )
    assert report.unresolved_entities == ("Flurbo engine", "Widget")


def test_downgrades_are_counted():
    router = StubRouter({"a": route("both", downgraded=True)})
    report = evaluate_router(router, [LabeledQuestion(question="a", intent="both")])
    assert report.downgraded == 1


def test_a_missing_labeled_set_says_so_rather_than_reporting_zero(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no labeled question set"):
        load_questions("nowhere", tmp_path)


def test_a_template_outside_the_enum_is_rejected_in_the_labels_too(tmp_path: Path):
    (tmp_path / "w.routing.yaml").write_text(
        yaml.safe_dump({"questions": [{"question": "q", "intent": "graph", "template": "nope"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_questions("w", tmp_path)


def test_the_shipped_factorio_set_is_valid_and_large_enough():
    """The plan calls for a 40-question labeled set."""
    questions = load_questions("factorio")
    assert len(questions) >= 40
    assert len({q.question for q in questions}) == len(questions)


def test_the_shipped_set_covers_every_template_and_both_non_graph_intents():
    """A set weighted to one template would score well and prove nothing."""
    questions = load_questions("factorio")
    templates = {q.template for q in questions if q.template}
    assert templates == {"recipe_tree", "unlock_chain", "consumers_of", "tier_compare"}
    assert {q.intent for q in questions} == {"graph", "vector", "both"}


def test_only_graph_labels_name_a_template():
    """A prose question has no template to get right, so labelling one is a mistake."""
    for question in load_questions("factorio"):
        if question.intent != "graph":
            assert question.template is None, question.question
