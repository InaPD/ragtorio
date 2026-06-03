"""What answering produces.

A :class:`Citation` is a chunk id that survived validation, resolved back to the wiki
revision it came from. It carries the revision-pinned URL rather than a page title
because a wiki answer that links to the live page is unfalsifiable six months later:
the page has changed, and there is no way to tell whether the answer was wrong or the
game was. ``index.php?title=X&oldid=N`` is the exact text the model was given.

:class:`Answer` records ``attempts`` and ``unverified`` because a grounded answer that
had to be regenerated once, or that still ended up with a claim nobody can check, is
not the same artifact as one that came back clean - and the difference has to reach
the caller rather than being smoothed over in a log line.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

#: What the answer cites when it is using the deterministic half. Graph facts have no
#: chunk id - they come from infobox parameters, not from prose - so they need a token
#: of their own, and giving them one is what keeps "every citation resolves to a
#: retrieved chunk" a true statement rather than a definition with a hole in it.
GRAPH_CITATION = "graph"


class Usage(BaseModel):
    """Tokens one answer cost. Phase 7 reports cost per query from these."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Summed across a regeneration: two calls, one answer, one cost."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class Citation(BaseModel):
    """One retrieved passage the answer cited, pinned to the revision it was read at."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    title: str
    section: str
    revision_id: int
    url: str


class Generation(BaseModel):
    """One model call's output. Not an answer yet: nothing has been checked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str
    stop_reason: str | None = None
    usage: Usage = Usage()

    @property
    def is_refusal(self) -> bool:
        """Whether the model declined. ``content`` is empty when it did, so the
        caller must check this before reading ``text`` as an answer."""
        return self.stop_reason == "refusal"


class CitationCheck(BaseModel):
    """What validation found: what resolved, what did not, and what claimed it.

    ``offending_claims`` is the part that does work. A regeneration prompt that only
    said "one of your citations was wrong" invites the model to strip citations
    wholesale; naming the sentences lets it fix those and leave the rest alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citations: tuple[Citation, ...] = ()
    cited_graph: bool = False
    unknown_ids: tuple[str, ...] = ()
    offending_claims: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.unknown_ids


class Answer(BaseModel):
    """A generated answer and everything known about how trustworthy it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    text: str
    citations: tuple[Citation, ...] = ()
    cited_graph: bool = False
    unverified: tuple[str, ...] = ()
    # The route, copied rather than referenced: a caller deciding whether to trust a
    # number wants to know it came from the graph, and an HTTP client should not have
    # to be handed the whole retrieval object to find that out.
    intent: str = "both"
    template: str | None = None
    entities: tuple[str, ...] = ()
    attempts: int = 1
    model: str = ""
    stop_reason: str | None = None
    usage: Usage = Usage()
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def is_grounded(self) -> bool:
        """Whether every citation in the text resolved to something retrieved."""
        return not self.unverified

    @property
    def is_refusal(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def latency_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms
