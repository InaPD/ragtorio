"""Answering a fixed question set and writing the answers down.

The Phase 6 exit criterion is twenty answered questions saved under ``docs/examples/``,
every citation resolving to a retrieved chunk, and a p95 latency somebody has written
down. This is the code that produces all three, so the number in the document and the
number the system produced cannot drift.

Saved as markdown, one file per question, with the citations as links. The point is
that a reader can click one and land on the exact revision the answer was written
from - an answer nobody can check is a demo, not a result.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from ragtorio.answer.models import Answer
from ragtorio.answer.pipeline import AnswerPipeline
from ragtorio.config import profiles_dir

#: The plan's exit criterion, kept next to the thing that measures it.
TARGET_QUESTIONS = 20


class ExampleQuestion(BaseModel):
    """One question to answer, and why it is in the set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    note: str | None = None


class ExampleRun(BaseModel):
    """Everything one pass over the set produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wiki: str
    questions: tuple[ExampleQuestion, ...] = ()
    answers: tuple[Answer, ...] = ()

    @property
    def citations(self) -> int:
        return sum(len(answer.citations) for answer in self.answers)

    @property
    def unverified(self) -> tuple[Answer, ...]:
        """Answers with a claim citing something that was never retrieved."""
        return tuple(answer for answer in self.answers if not answer.is_grounded)

    @property
    def regenerated(self) -> int:
        return sum(1 for answer in self.answers if answer.attempts > 1)

    @property
    def refusals(self) -> int:
        return sum(1 for answer in self.answers if answer.is_refusal)

    @property
    def citations_resolve(self) -> bool:
        """The exit criterion: no answer cites anything that was not retrieved."""
        return not self.unverified

    @property
    def p50_ms(self) -> float:
        return _percentile([answer.latency_ms for answer in self.answers], 0.50)

    @property
    def p95_ms(self) -> float:
        return _percentile([answer.latency_ms for answer in self.answers], 0.95)

    @property
    def output_tokens(self) -> int:
        return sum(answer.usage.output_tokens for answer in self.answers)

    @property
    def input_tokens(self) -> int:
        return sum(answer.usage.input_tokens for answer in self.answers)

    @property
    def cache_read_tokens(self) -> int:
        return sum(answer.usage.cache_read_tokens for answer in self.answers)


def load_examples(wiki_id: str, directory: Path | None = None) -> tuple[ExampleQuestion, ...]:
    """Load ``wikis/<wiki_id>.examples.yaml``.

    Raises:
        FileNotFoundError: if the wiki has no example set. An empty run would write a
            document claiming a perfect citation rate over nothing.
    """
    base = directory or profiles_dir()
    path = base / f"{wiki_id}.examples.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no example questions at {path}. The Phase 6 exit criterion is "
            f"{TARGET_QUESTIONS} answered questions; write that file first."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(ExampleQuestion.model_validate(entry) for entry in data.get("questions", []))


def run_examples(
    wiki_id: str,
    pipeline: AnswerPipeline,
    questions: Sequence[ExampleQuestion],
) -> ExampleRun:
    """Answer every question in the set."""
    return ExampleRun(
        wiki=wiki_id,
        questions=tuple(questions),
        answers=tuple(pipeline.answer(question.question) for question in questions),
    )


def write_examples(run: ExampleRun, directory: Path) -> list[Path]:
    """Write one markdown file per answer plus an index, and report what was written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [directory / "README.md"]
    (written[0]).write_text(_index(run), encoding="utf-8")

    paired = zip(run.questions, run.answers, strict=True)
    for position, (question, answer) in enumerate(paired, start=1):
        path = directory / f"{position:02d}-{_slug(question.question)}.md"
        path.write_text(_document(question, answer), encoding="utf-8")
        written.append(path)
    return written


def _index(run: ExampleRun) -> str:
    """The summary the exit criterion is read off."""
    lines = [
        f"# Example answers - {run.wiki}",
        "",
        f"{len(run.answers)} questions answered by `ragtorio examples run {run.wiki}`. Every",
        "citation below was checked against the chunk ids that question actually retrieved;",
        "an answer citing anything else is listed as unverified rather than quietly cleaned up.",
        "",
        "| | |",
        "|---|---|",
        f"| Questions | {len(run.answers)} |",
        f"| Citations | {run.citations} |",
        f"| Citations resolving to retrieved chunks | "
        f"{'100%' if run.citations_resolve else 'not all'} |",
        f"| Answers with unverified claims | {len(run.unverified)} |",
        f"| Answers regenerated once | {run.regenerated} |",
        f"| Refused | {run.refusals} |",
        f"| Latency | p50 {run.p50_ms:,.0f} ms, p95 {run.p95_ms:,.0f} ms |",
        f"| Tokens | {run.input_tokens:,} in ({run.cache_read_tokens:,} cached), "
        f"{run.output_tokens:,} out |",
        "",
        "## Questions",
        "",
    ]
    paired = zip(run.questions, run.answers, strict=True)
    for position, (question, answer) in enumerate(paired, start=1):
        name = f"{position:02d}-{_slug(question.question)}.md"
        flag = "" if answer.is_grounded else " (unverified claims)"
        lines.append(f"{position}. [{question.question}]({name}) - {answer.intent}{flag}")
    return "\n".join(lines) + "\n"


def _document(question: ExampleQuestion, answer: Answer) -> str:
    lines = [f"# {question.question}", ""]
    if question.note:
        lines += [f"*{question.note}*", ""]
    lines += [answer.text or "*(the model declined to answer)*", "", "---", ""]

    route = f"**Route:** {answer.intent}"
    if answer.template:
        route += f" via `{answer.template}`"
    if answer.entities:
        route += f" on {', '.join(answer.entities)}"
    lines.append(route)
    lines.append("")

    if answer.cited_graph:
        lines += ["**Graph facts** were used and cited as `[graph]`.", ""]
    if answer.citations:
        lines.append("**Sources**")
        lines.append("")
        for citation in answer.citations:
            where = citation.title
            if citation.section:
                where = f"{where} > {citation.section}"
            lines.append(f"- `[{citation.chunk_id}]` [{where}]({citation.url})")
        lines.append("")
    if answer.unverified:
        lines.append("**Unverified claims** (cited a source that was not retrieved)")
        lines.append("")
        lines += [f"- {claim}" for claim in answer.unverified]
        lines.append("")
    lines.append(
        f"*{answer.attempts} generation(s), {answer.latency_ms:,.0f} ms, "
        f"{answer.usage.output_tokens:,} output tokens, model {answer.model or 'unknown'}.*"
    )
    return "\n".join(lines) + "\n"


def _slug(question: str) -> str:
    """A filename from a question. Short, because it is read in a directory listing."""
    words = re.sub(r"[^a-z0-9]+", "-", question.casefold()).strip("-").split("-")
    return "-".join(words[:8]) or "question"


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Twenty measurements do not need numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]
