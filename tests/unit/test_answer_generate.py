"""The answering call, with the SDK faked.

Whether Opus writes good answers is a question for `ragtorio examples run`. What is
worth pinning here is everything around the call: that the system prompt is sent as a
cached prefix, that a refusal is visible instead of looking like an empty answer, that
usage comes back for the benchmark to price, and that a missing key fails loudly
rather than becoming a silent degradation.
"""

from __future__ import annotations

from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from ragtorio.answer.generate import (
    FALLBACK_BETA,
    AnswererUnavailableError,
    AnthropicAnswerer,
)


class FakeUsage:
    input_tokens = 1200
    output_tokens = 300
    cache_read_input_tokens = 1000
    cache_creation_input_tokens = None  # absent on a cache hit, as the API has it


class FakeBlock:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text
        self.thinking = text


class FakeMessage:
    def __init__(self, blocks: list[FakeBlock], stop_reason: str = "end_turn") -> None:
        self.content = blocks
        self.stop_reason = stop_reason
        self.model = "claude-opus-5"
        self.usage = FakeUsage()


class FakeStream:
    def __init__(self, message: FakeMessage) -> None:
        self._message = message
        self.text_stream = [b.text for b in message.content if b.type == "text"]

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> FakeMessage:
        return self._message


class FakeClient:
    """Stands in for ``client.beta.messages``, recording what it was called with."""

    def __init__(self, message: FakeMessage | None = None, error: Exception | None = None) -> None:
        self._message = message or FakeMessage([FakeBlock("text", "Iron plate. [graph]")])
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeStream(self._message)


def build(**kwargs: Any) -> tuple[AnthropicAnswerer, FakeClient]:
    client = FakeClient(kwargs.pop("message", None), kwargs.pop("error", None))
    return AnthropicAnswerer(client=client, **kwargs), client


def test_the_text_blocks_are_the_answer_and_thinking_is_not():
    message = FakeMessage([FakeBlock("thinking", "let me count"), FakeBlock("text", "1 iron.")])
    answerer, _ = build(message=message)
    assert answerer.generate([{"role": "user", "content": "q"}]).text == "1 iron."


def test_the_system_prompt_is_sent_as_a_cached_prefix():
    """Everything that varies per question is in the user message, so the prefix is
    the same bytes on every request and can actually be cached."""
    answerer, client = build()
    answerer.generate([{"role": "user", "content": "q"}])
    system = client.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "crafting game" in system[0]["text"]


def test_the_request_asks_for_opus_adaptive_thinking_and_refusal_fallbacks():
    answerer, client = build()
    answerer.generate([{"role": "user", "content": "q"}])
    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default"
    assert FALLBACK_BETA in call["betas"]


def test_streamed_text_reaches_the_callback_as_it_arrives():
    message = FakeMessage([FakeBlock("text", "one "), FakeBlock("text", "two")])
    answerer, _ = build(message=message)
    seen: list[str] = []
    answerer.generate([{"role": "user", "content": "q"}], seen.append)
    assert seen == ["one ", "two"]


def test_usage_comes_back_for_the_benchmark_to_price():
    answerer, _ = build()
    usage = answerer.generate([{"role": "user", "content": "q"}]).usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (1200, 300, 1000)
    # Absent in the API response rather than zero, and must not become None here.
    assert usage.cache_write_tokens == 0


def test_a_refusal_is_a_visible_state_not_an_empty_answer():
    answerer, _ = build(message=FakeMessage([], stop_reason="refusal"))
    generation = answerer.generate([{"role": "user", "content": "q"}])
    assert generation.is_refusal
    assert generation.text == ""


def test_missing_credentials_fail_loudly():
    """Answering has no degraded mode: unlike routing, there is nothing useful to
    return without the model, so this must not be swallowed."""
    answerer, _ = build(error=TypeError("Could not resolve authentication method"))
    with pytest.raises(AnswererUnavailableError, match="ANTHROPIC_API_KEY"):
        answerer.generate([{"role": "user", "content": "q"}])


def test_rejected_credentials_say_which_problem_it_is():
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com"))
    error = anthropic.AuthenticationError("bad key", response=response, body=None)
    answerer, _ = build(error=error)
    with pytest.raises(AnswererUnavailableError, match="rejected the credentials"):
        answerer.generate([{"role": "user", "content": "q"}])


def test_an_api_failure_is_reported_rather_than_returning_half_an_answer():
    answerer, _ = build(
        error=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    )
    with pytest.raises(AnswererUnavailableError):
        answerer.generate([{"role": "user", "content": "q"}])
