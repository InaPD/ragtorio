"""Measuring the router against a hand-labeled question set.

The router is the one model call in the pipeline, so it is the one component whose
behaviour can change without anybody touching the code - a prompt edit, a model
version, a temperature the SDK defaults differently. A labeled set is the only way to
notice.

**Intent and template are scored separately.** They fail differently: a question routed
``vector`` when it should have been ``graph`` returns prose where a number was wanted,
which a reader spots immediately; the right intent with the wrong template returns
confident, well-formed facts about a different question, which a reader does not.
Reporting one blended number would hide the worse failure inside the better one.

A ``both`` label means either specific intent would also be acceptable - it is the
routing equivalent of "don't care", and scoring it strictly would punish a router for
being appropriately cautious.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from ragtorio.config import profiles_dir
from ragtorio.retrieve.models import Intent, Route, TemplateName
from ragtorio.retrieve.router import Router

#: The plan's exit criterion, kept next to the thing that measures it.
TARGET_ACCURACY = 0.9


class LabeledQuestion(BaseModel):
    """One question and the route a human says it deserves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    intent: Intent
    template: TemplateName | None = None

    def intent_ok(self, actual: Intent) -> bool:
        """Whether the routed intent is acceptable.

        A ``both`` label accepts anything: the question genuinely needs both halves, so
        a router that picked one and got a good answer has not made a mistake worth
        counting. A specific label accepts ``both`` too - retrieving extra is a cost,
        not an error - but not the opposite specific intent.
        """
        return self.intent == "both" or actual in (self.intent, "both")

    def template_ok(self, actual: TemplateName | None) -> bool:
        """Whether the template matches. Only scored when the label names one."""
        return actual == self.template


class RouteOutcome(BaseModel):
    """What the router did with one labeled question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    expected_intent: Intent
    expected_template: TemplateName | None = None
    route: Route
    intent_ok: bool
    template_ok: bool | None = None
    latency_ms: float = 0.0


class RouterReport(BaseModel):
    """Router accuracy, split the two ways it can be wrong."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcomes: tuple[RouteOutcome, ...] = ()

    @property
    def intent_accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.intent_ok) / len(self.outcomes)

    @property
    def template_accuracy(self) -> float:
        """Over the questions whose label names a template."""
        scored = [o for o in self.outcomes if o.template_ok is not None]
        if not scored:
            return 0.0
        return sum(1 for o in scored if o.template_ok) / len(scored)

    @property
    def unresolved_entities(self) -> tuple[str, ...]:
        """Every entity the router named that the graph could not match.

        Worth surfacing separately: a router that classifies perfectly but names
        entities nobody can resolve produces empty graph results, and the accuracy
        numbers above would not show it.
        """
        seen: dict[str, None] = {}
        for outcome in self.outcomes:
            for mention in outcome.route.unresolved:
                seen.setdefault(mention, None)
        return tuple(seen)

    @property
    def failures(self) -> tuple[RouteOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.intent_ok or o.template_ok is False)

    @property
    def downgraded(self) -> int:
        return sum(1 for o in self.outcomes if o.route.downgraded)

    @property
    def p50_ms(self) -> float:
        return _percentile([o.latency_ms for o in self.outcomes], 0.50)

    @property
    def p95_ms(self) -> float:
        return _percentile([o.latency_ms for o in self.outcomes], 0.95)

    @property
    def meets_target(self) -> bool:
        """The Phase 5 exit criterion: 90% on the labeled set, both ways."""
        return self.intent_accuracy >= TARGET_ACCURACY and self.template_accuracy >= TARGET_ACCURACY


def load_questions(wiki_id: str, directory: Path | None = None) -> tuple[LabeledQuestion, ...]:
    """Load ``wikis/<wiki_id>.routing.yaml``.

    Raises:
        FileNotFoundError: if the wiki has no labeled set. An absent file is not a
            valid state - an evaluation over nothing would report a confident 0%.
    """
    base = directory or profiles_dir()
    path = base / f"{wiki_id}.routing.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no labeled question set at {path}. Router accuracy is measured against "
            "hand-labeled questions; write that file before running `ragtorio route eval`."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(LabeledQuestion.model_validate(entry) for entry in data.get("questions", []))


def evaluate_router(router: Router, questions: Sequence[LabeledQuestion]) -> RouterReport:
    """Route every labeled question and score the results."""
    outcomes: list[RouteOutcome] = []
    for labeled in questions:
        started = time.perf_counter()
        route = router.route(labeled.question)
        elapsed = (time.perf_counter() - started) * 1000
        outcomes.append(
            RouteOutcome(
                question=labeled.question,
                expected_intent=labeled.intent,
                expected_template=labeled.template,
                route=route,
                intent_ok=labeled.intent_ok(route.intent),
                template_ok=(
                    labeled.template_ok(route.template) if labeled.template is not None else None
                ),
                latency_ms=elapsed,
            )
        )
    return RouterReport(outcomes=tuple(outcomes))


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Forty measurements do not need numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]
