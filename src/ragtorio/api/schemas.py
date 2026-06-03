"""Request and response bodies.

Validated by pydantic at the boundary, which is where untrusted input stops being
untrusted shape and becomes untrusted content. The length bounds on ``question`` are
not politeness: the question goes into a prompt, and an unbounded one is a way to make
somebody else's API bill interesting.

Responses carry the route and the citation state as data rather than prose. A client
that wants to show "answered from the recipe graph" or refuse to display an answer
with unverified claims should not have to parse English to do it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ragtorio.answer.models import Answer, Citation

#: Long enough for a real question, short enough that nobody prompts through it.
MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    """One question. ``stream`` switches the response to server-sent events."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    stream: bool = False


class CitationOut(BaseModel):
    """A citation as the wire sees it: the id, where it came from, and a pinned URL."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    title: str
    section: str
    revision_id: int
    url: str

    @classmethod
    def of(cls, citation: Citation) -> CitationOut:
        return cls(**citation.model_dump())


class AskResponse(BaseModel):
    """An answer and everything a caller needs to judge it."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    cited_graph: bool = False
    unverified: list[str] = Field(default_factory=list)
    grounded: bool = True
    intent: str = "both"
    template: str | None = None
    entities: list[str] = Field(default_factory=list)
    attempts: int = 1
    latency_ms: float = 0.0

    @classmethod
    def of(cls, answer: Answer) -> AskResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            citations=[CitationOut.of(c) for c in answer.citations],
            cited_graph=answer.cited_graph,
            unverified=list(answer.unverified),
            grounded=answer.is_grounded,
            intent=answer.intent,
            template=answer.template,
            entities=list(answer.entities),
            attempts=answer.attempts,
            latency_ms=round(answer.latency_ms, 1),
        )


class HealthResponse(BaseModel):
    """Whether the two stores an answer needs are actually reachable.

    A health check that only proves the web server started is what lets a deployment
    with an empty graph look healthy for a week.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    wiki: str
    postgres: bool
    neo4j: bool


class ErrorResponse(BaseModel):
    """What a failure looks like. One sentence, no internals."""

    model_config = ConfigDict(extra="forbid")

    error: str
