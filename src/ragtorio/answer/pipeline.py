"""Question in, checked answer out: retrieve, render, generate, validate, once more.

The regeneration loop lives here rather than in ``generate`` because it is a policy,
not a way of calling the API: one retry, the offending claims quoted back, and after
that the answer ships with its unsourced claims named rather than a third attempt or
an exception. Two calls is where the cost of another attempt stops being obviously
worth it - a model that has already been shown exactly which sentences were wrong and
cited a phantom source again is not going to be talked round on the third try.

**Streaming answers are not regenerated.** A caller streaming tokens to a browser has
already shown the reader the first attempt; replacing it with a second one mid-stream
would be worse than telling the truth about it. So a streamed answer that cites
something unretrievable comes back with those claims marked unverified in its final
event, and the caller can decide what to do. The non-streaming path - the CLI, the
benchmark, and ``POST /ask`` without ``stream`` - gets the full two attempts.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from anthropic.types.beta import BetaMessageParam

from ragtorio.answer.generate import Answerer
from ragtorio.answer.models import Answer, CitationCheck, Generation, Usage
from ragtorio.answer.render import citation_instruction, render_prompt
from ragtorio.answer.validate import regeneration_note, unverified_claims, validate
from ragtorio.retrieve.models import Intent, RetrievedContext
from ragtorio.retrieve.pipeline import RetrievalPipeline

#: One generation plus one corrective regeneration. See the module docstring.
DEFAULT_MAX_ATTEMPTS = 2


class AnswerPipeline:
    """Retrieves, answers, and checks the answer's citations."""

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        answerer: Answerer,
        index_url: str,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self._retrieval = retrieval
        self._answerer = answerer
        self._index_url = index_url
        self._max_attempts = max(1, max_attempts)

    def answer(
        self,
        question: str,
        force_intent: Intent | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """Everything one question produces, retrieval included."""
        context = self._retrieval.retrieve(question, force_intent=force_intent)
        return self.answer_from(context, on_text=on_text)

    def answer_from(
        self,
        context: RetrievedContext,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """The generation half alone, over context somebody else retrieved.

        Split out because the benchmark's baselines retrieve differently and answer
        identically, and because the API's streaming path retrieves before it opens
        the response body.
        """
        messages: list[BetaMessageParam] = [{"role": "user", "content": _user_message(context)}]
        attempts = self._max_attempts if on_text is None else 1

        usage = Usage()
        elapsed = 0.0
        generation = Generation(text="", model="")
        check = CitationCheck()

        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            generation = self._answerer.generate(messages, on_text)
            elapsed += (time.perf_counter() - started) * 1000
            usage += generation.usage
            if generation.is_refusal:
                break
            check = validate(generation.text, context, self._index_url)
            if check.ok or attempt == attempts:
                break
            messages += [
                {"role": "assistant", "content": generation.text},
                {"role": "user", "content": regeneration_note(check)},
            ]
            # Only the first attempt reaches a streaming caller, so a regeneration
            # never double-writes into somebody's response body.
            on_text = None

        return Answer(
            question=context.question,
            text=generation.text,
            citations=check.citations,
            cited_graph=check.cited_graph,
            unverified=() if check.ok else unverified_claims(check),
            intent=context.route.intent,
            template=context.route.template,
            entities=tuple(entity.title for entity in context.route.entities),
            attempts=attempt,
            model=generation.model,
            stop_reason=generation.stop_reason,
            usage=usage,
            retrieval_ms=context.latency_ms,
            generation_ms=elapsed,
        )


def _user_message(context: RetrievedContext) -> str:
    """The evidence, then what may be cited. The instruction goes last because it is
    the part that changes per question, and the model reads it against what it just
    saw rather than having to remember it."""
    return f"{render_prompt(context)}\n\n{citation_instruction(context)}"
