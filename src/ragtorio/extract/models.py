"""What extraction produces.

A ``Fact`` is deliberately generic: one subject, one predicate, one object, with
whatever extra properties the predicate needs (an ingredient amount, a probability) and
where it came from. The graph loader (Phase 3) is what turns a stream of these into
nodes and edges; this module does not know about Neo4j, node identity, or entity
resolution at all.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Provenance(BaseModel):
    """Where one fact came from, down to the revision and the field that produced it.

    Kept on every fact rather than once per batch because facts from one page's crawl
    can still end up mixed together once they reach the graph loader.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    wiki: str
    page_id: int
    revision_id: int
    field: str


class Fact(BaseModel):
    """One extracted statement: subject, predicate, object, with labels and provenance.

    ``predicate`` carries the profile's own vocabulary verbatim (``prop.stack_size``,
    ``rel.CONSUMES``) rather than a cleaned-up name, so a fact can be traced straight
    back to the field mapping that produced it. ``object`` is ``None`` for facts that
    are inherently multi-valued (a version event's meaning lives entirely in ``props``).

    Confidence is not tracked here: it is always 1.0 for a template-derived fact, and
    the graph loader is what stamps that onto the edge it builds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    subject_labels: tuple[str, ...]
    predicate: str
    object: str | float | None
    object_labels: tuple[str, ...] = ()
    props: dict[str, Any] = {}
    provenance: Provenance
