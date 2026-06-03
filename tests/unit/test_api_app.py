"""The HTTP surface, over a faked pipeline.

The endpoints themselves are thin, so what is tested here is what they are for:
validation at the boundary, a rate limit that actually refuses, a health check that
fails when a store is down rather than when the process is dead, and - the one that
matters - that no internal detail leaves the process in an error body.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ragtorio.answer.generate import AnswererUnavailableError
from ragtorio.answer.models import Answer, Citation, Usage
from ragtorio.api.app import Health, Service, create_app

ANSWER = Answer(
    question="what is it made of",
    text="One iron plate and three copper cable [factorio:1:0000].",
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
    intent="both",
    template="recipe_tree",
    entities=("Electronic circuit",),
    usage=Usage(input_tokens=10, output_tokens=5),
    generation_ms=120.0,
)


class FakePipeline:
    def __init__(self, answer: Answer | None = None, error: Exception | None = None) -> None:
        self._answer = answer or ANSWER
        self._error = error
        self.questions: list[str] = []

    def answer(self, question: str, on_text: Any = None) -> Answer:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        if on_text is not None:
            for word in self._answer.text.split(" "):
                on_text(word + " ")
        return self._answer


def client(
    pipeline: FakePipeline | None = None,
    health: Health | None = None,
    rate_limit: int = 20,
) -> TestClient:
    service = Service(
        wiki="factorio",
        pipeline=pipeline or FakePipeline(),  # type: ignore[arg-type]
        health=lambda: health or Health(postgres=True, neo4j=True),
    )
    app = create_app("factorio", lambda: (service, lambda: None), rate_limit=rate_limit)
    return TestClient(app, raise_server_exceptions=False)


def test_health_reports_each_store_and_passes_when_both_answer():
    with client() as http:
        response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "wiki": "factorio",
        "postgres": True,
        "neo4j": True,
    }


def test_health_fails_when_a_store_is_down():
    """A check that only proves the web server started is how a deployment with an
    unreachable graph looks healthy for a week."""
    with client(health=Health(postgres=True, neo4j=False)) as http:
        response = http.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_ask_returns_the_answer_its_citations_and_the_route():
    with client() as http:
        body = http.post("/ask", json={"question": "what is it made of"}).json()
    assert body["answer"].startswith("One iron plate")
    assert body["grounded"] is True
    assert body["cited_graph"] is True
    assert body["citations"][0]["url"].endswith("oldid=7")
    assert body["template"] == "recipe_tree"


def test_an_answer_with_unverified_claims_says_so_on_the_wire():
    """A client that wants to refuse to display an ungrounded answer should not have
    to parse English to find out it is one."""
    ungrounded = ANSWER.model_copy(update={"unverified": ("Copper [factorio:9:9999].",)})
    with client(FakePipeline(ungrounded)) as http:
        body = http.post("/ask", json={"question": "what is it made of"}).json()
    assert body["grounded"] is False
    assert body["unverified"] == ["Copper [factorio:9:9999]."]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": "x" * 501},
        {},
        {"question": "what is it made of", "wiki": "minecraft"},
    ],
)
def test_a_malformed_request_is_refused_at_the_boundary(payload: dict[str, Any]):
    with client() as http:
        assert http.post("/ask", json=payload).status_code == 422


def test_a_client_over_the_rate_limit_is_refused_and_told_when_to_return():
    with client(rate_limit=2) as http:
        for _ in range(2):
            assert http.post("/ask", json={"question": "what is it made of"}).status_code == 200
        response = http.post("/ask", json={"question": "what is it made of"})
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0


def test_an_unavailable_model_is_a_503_a_caller_can_retry():
    pipeline = FakePipeline(error=AnswererUnavailableError("no credentials for sk-ant-xyz"))
    with client(pipeline) as http:
        response = http.post("/ask", json={"question": "what is it made of"})
    assert response.status_code == 503
    assert "sk-ant-xyz" not in response.text


def test_an_unexpected_failure_leaks_nothing():
    """The whole point of the handler: a psycopg error in a response body tells a
    stranger the database user, the table names and the driver version."""
    pipeline = FakePipeline(error=RuntimeError('relation "chunk" does not exist'))
    with client(pipeline) as http:
        response = http.post("/ask", json={"question": "what is it made of"})
    assert response.status_code == 500
    assert "chunk" not in response.text
    assert response.json() == {"error": "the request could not be completed"}


def test_streaming_sends_text_as_it_arrives_then_one_final_event_with_the_checks():
    with (
        client() as http,
        http.stream(
            "POST", "/ask", json={"question": "what is it made of", "stream": True}
        ) as response,
    ):
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = [chunk for chunk in body.split("\n\n") if chunk.strip()]
    assert events[0].startswith("event: token")
    assert json.loads(events[0].split("data: ", 1)[1])["text"] == "One "
    assert events[-1].startswith("event: final")
    final = json.loads(events[-1].split("data: ", 1)[1])
    assert final["citations"][0]["chunk_id"] == "factorio:1:0000"
    assert final["grounded"] is True


def test_a_failure_mid_stream_arrives_as_an_error_event_rather_than_a_dropped_body():
    pipeline = FakePipeline(error=RuntimeError('relation "chunk" does not exist'))
    with (
        client(pipeline) as http,
        http.stream(
            "POST", "/ask", json={"question": "what is it made of", "stream": True}
        ) as response,
    ):
        body = "".join(response.iter_text())
    assert "event: error" in body
    assert "chunk" not in body
