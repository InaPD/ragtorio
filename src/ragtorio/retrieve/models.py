"""What routing and retrieval pass around.

``RouteDecision`` is the model's output and nothing more: three fields, each
constrained, with no query text anywhere in it. ``Route`` is what the system believes
after that decision has been checked against the graph - the entities the model named
resolved to real node ids, or recorded as unresolved. Keeping them apart is what makes
"the model hallucinated an entity" a visible state rather than an empty result set.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ragtorio.index.models import Chunk

#: Which halves of the system to consult. ``both`` is the safe answer and the one a
#: low-confidence route falls back to.
Intent = Literal["graph", "vector", "both"]

#: The four Cypher templates. The router may name one or none; it may never write a
#: query, so this enum is the complete set of graph questions the system can ask.
TemplateName = Literal["recipe_tree", "unlock_chain", "consumers_of", "tier_compare"]

TEMPLATE_NAMES: tuple[str, ...] = (
    "recipe_tree",
    "unlock_chain",
    "consumers_of",
    "tier_compare",
)


class RouteDecision(BaseModel):
    """The router's answer, exactly as the schema constrains it.

    ``confidence`` exists so the caller can downgrade a shaky ``graph`` route to
    ``both`` rather than returning nothing: a wrong template answers the wrong
    question confidently, while ``both`` merely costs one extra query.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    template: TemplateName | None = None
    entities: list[str] = Field(default_factory=list)
    compare_by: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ResolvedEntity(BaseModel):
    """One entity the router named, matched to a node the graph actually has."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mention: str
    id: str
    title: str
    labels: tuple[str, ...] = ()
    matched_by: Literal["title", "alias", "search"]


class Route(BaseModel):
    """A decision that has been checked: resolved entities, and what did not resolve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    intent: Intent
    template: TemplateName | None = None
    compare_by: str | None = None
    entities: tuple[ResolvedEntity, ...] = ()
    unresolved: tuple[str, ...] = ()
    confidence: float = 1.0
    downgraded: bool = False

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(entity.id for entity in self.entities)

    @property
    def uses_graph(self) -> bool:
        return self.intent in ("graph", "both")

    @property
    def uses_vector(self) -> bool:
        return self.intent in ("vector", "both")


class GraphResult(BaseModel):
    """One template's output: the rows it returned and a rendering of them.

    ``lines`` is what reaches the answering model. Rows are kept alongside because a
    renderer that wants a nested tree rather than flat lines needs the structure, and
    Phase 6 has to be able to build one without re-running the query.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    template: str
    rows: tuple[dict[str, Any], ...] = ()
    lines: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.lines


class ContextBlock(BaseModel):
    """One labeled section of the context handed to the answering model.

    Labeled and kept separate rather than concatenated: a passage is prose somebody
    wrote and may be out of date, a graph fact is a template parameter that was
    extracted deterministically, and the model is told which is which so it can say so.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Literal["graph_facts", "passages"]
    text: str
    tokens: int


class RetrievedContext(BaseModel):
    """Everything one question retrieved, ready for Phase 6 to render and cite.

    The passages are carried whole rather than as the ids alone. Phase 6 has to turn a
    citation back into ``index.php?title=X&oldid=N``, and the page title and revision
    that URL needs live on the chunk; looking them up again from the id would be a
    second query against a store the answerer has no reason to hold open.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    route: Route
    blocks: tuple[ContextBlock, ...] = ()
    passages: tuple[Chunk, ...] = ()
    graph: GraphResult | None = None
    latency_ms: float = 0.0

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """The ids a citation may name, in the order they were given to the model."""
        return tuple(chunk.chunk_id for chunk in self.passages)

    @property
    def total_tokens(self) -> int:
        return sum(block.tokens for block in self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks
