"""What resolution produces: nodes and edges ready to load, and what it could not."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ResolvedNode(BaseModel):
    """One graph node. ``id`` is unique across labels; a page that is both an Item
    and has its own recipe becomes two ``ResolvedNode``\\ s (see ``resolve.py``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    labels: tuple[str, ...]
    props: dict[str, Any]


class ResolvedEdge(BaseModel):
    """One graph edge, already pointed at real node ids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_id: str
    rel_type: str
    to_id: str
    props: dict[str, Any] = {}


class UnresolvedReference(BaseModel):
    """A fact whose object named no node this resolver could find.

    Logged rather than silently dropped: a title that should resolve but doesn't
    (a typo, an uncrawled page, a redirect the harvester missed) is exactly the kind
    of gap ``ragtorio graph check`` exists to surface.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    predicate: str
    object: str


class ResolvedGraph(BaseModel):
    """A complete resolution run: every node and edge to load, and what didn't resolve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[ResolvedNode, ...]
    edges: tuple[ResolvedEdge, ...]
    unresolved: tuple[UnresolvedReference, ...]
