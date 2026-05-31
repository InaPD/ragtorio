"""Turns extracted facts into a graph: canonical ids, the Item/Recipe split, and
edges pointed at real nodes or logged as unresolved.

**Design decisions worth naming, since none of them are purely mechanical:**

- A page that both classifies as an ``Item`` (or ``Fluid``) and carries its own
  recipe becomes two nodes: ``factorio:Title`` (the item) and
  ``factorio:Title (recipe)`` (the recipe). A page already classified as ``Recipe``
  needs no split. ``CONSUMES``, ``PRODUCES``, ``CRAFTED_AT`` and the recipe's own
  ``crafting_time`` belong to the recipe aspect; everything else belongs to the item.
- ``rel.CONSUMED_BY`` (an item's "consumers" field, which carries no amount) is
  rewritten to a ``CONSUMES`` edge from the consumer's recipe aspect, then merged
  with any ``CONSUMES`` edge already produced by that consumer's own ``recipe``
  field. Where both exist, the one carrying real data (an amount) wins, since the
  two describe the same edge from opposite ends of the wiki.
- ``:Station`` is not a type-map label: it is added, after every edge is resolved,
  to whatever node is on the receiving end of a ``CRAFTED_AT`` edge. That is what
  keeps the profile from having to hand-list which machines happen to craft things.
- ``rel.UNLOCKS`` prefers a title's recipe aspect (the ontology's own arrow points at
  a Recipe) and falls back to the plain item id when that title has no recipe data.
- ``rel.REQUIRES`` only becomes an edge when *both* ends resolve to an ``Unlock``:
  ``required-technologies`` is present on Item and Recipe pages too, mixing in
  science-pack costs the ontology does not define this edge over. Dropping those is
  a deliberate filter, not something worth logging as unresolved.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ragtorio.extract.models import Fact
from ragtorio.harvest.models import RawRedirect
from ragtorio.ontology.models import ResolvedEdge, ResolvedGraph, ResolvedNode, UnresolvedReference

#: Facts that belong to a page's recipe aspect rather than its item aspect.
_RECIPE_ASPECT_PREDICATES = frozenset(
    {"prop.crafting_time", "rel.CONSUMES", "rel.PRODUCES", "rel.CRAFTED_AT"}
)

#: predicate -> Neo4j relationship type. CONSUMED_BY is handled separately: it
#: rewrites to CONSUMES with subject and object swapped, not a 1:1 rename.
_REL_TYPE = {
    "rel.CONSUMES": "CONSUMES",
    "rel.PRODUCES": "PRODUCES",
    "rel.CRAFTED_AT": "CRAFTED_AT",
    "rel.UNLOCKS": "UNLOCKS",
    "rel.REQUIRES": "REQUIRES",
    "rel.COSTS": "COSTS",
}

_VERSION_PART = re.compile(r"\d+")


def _recipe_id(title: str) -> str:
    return f"{title} (recipe)"


@dataclass
class _PageAspects:
    """What one wiki title resolves to: an item-or-similar id, and a recipe id only
    if that title actually carries recipe data."""

    labels: tuple[str, ...]
    item_id: str
    recipe_id: str | None = None
    props: dict[str, Any] = field(default_factory=dict)
    version_events: list[tuple[str, str]] = field(default_factory=list)


class EntityResolver:
    """Resolves one wiki's facts into a :class:`~ragtorio.ontology.models.ResolvedGraph`.

    Aliases (redirects plus community shorthand) are folded in eagerly so that any
    fact naming an alias instead of the canonical title still resolves.
    """

    def __init__(
        self,
        wiki: str,
        facts: list[Fact],
        redirects: list[RawRedirect],
        aliases: dict[str, str],
        archived_titles: frozenset[str],
    ) -> None:
        self._wiki = wiki
        self._facts = facts
        self._archived_titles = archived_titles
        # Every alias/redirect source, in its original display casing, plus the
        # case-insensitive lookup key used to actually resolve one.
        sources: dict[str, str] = {r.from_title: r.to_title for r in redirects} | dict(aliases)
        self._alias_target = {source.casefold(): target for source, target in sources.items()}
        self._aliases_of: dict[str, list[str]] = defaultdict(list)
        for source, target in sources.items():
            self._aliases_of[target].append(source)

    def resolve(self) -> ResolvedGraph:
        aspects = self._build_aspects()
        nodes = self._build_nodes(aspects)
        edges, unresolved = self._build_edges(aspects)
        edges = self._add_station_label(nodes, edges)
        return ResolvedGraph(nodes=tuple(nodes), edges=tuple(edges), unresolved=tuple(unresolved))

    # -- pass 1: what each title *is* -------------------------------------------

    def _build_aspects(self) -> dict[str, _PageAspects]:
        by_subject: dict[str, list[Fact]] = defaultdict(list)
        for fact in self._facts:
            by_subject[fact.subject].append(fact)

        aspects: dict[str, _PageAspects] = {}
        for title, page_facts in by_subject.items():
            labels = page_facts[0].subject_labels
            has_recipe_data = any(f.predicate in _RECIPE_ASPECT_PREDICATES for f in page_facts)
            needs_split = has_recipe_data and "Recipe" not in labels
            aspects[title] = _PageAspects(
                labels=labels,
                item_id=self._id(title),
                recipe_id=self._id(_recipe_id(title))
                if needs_split
                else (self._id(title) if has_recipe_data else None),
            )
            self._apply_scalar_props(aspects[title], page_facts)
        return aspects

    def _apply_scalar_props(self, aspects: _PageAspects, page_facts: list[Fact]) -> None:
        for fact in page_facts:
            if not fact.predicate.startswith("prop."):
                continue
            name = fact.predicate.removeprefix("prop.")
            if name == "version_event":
                version = str(fact.props.get("version", ""))
                text = str(fact.props.get("text", ""))
                aspects.version_events.append((version, text))
                continue
            if fact.predicate in _RECIPE_ASPECT_PREDICATES:
                continue  # crafting_time is attached when building the recipe node
            aspects.props[name] = fact.object

    # -- pass 2: nodes ------------------------------------------------------------

    def _build_nodes(self, aspects: dict[str, _PageAspects]) -> list[ResolvedNode]:
        nodes: list[ResolvedNode] = []
        item_facts = defaultdict(list)
        for fact in self._facts:
            item_facts[fact.subject].append(fact)

        for title, page in aspects.items():
            crafting_time = next(
                (f.object for f in item_facts[title] if f.predicate == "prop.crafting_time"), None
            )
            if page.recipe_id is not None and page.recipe_id != page.item_id:
                # Split: a plain item/fluid node, and a separate recipe node.
                nodes.append(self._item_node(title, page))
                nodes.append(
                    ResolvedNode(
                        id=page.recipe_id,
                        labels=("Recipe",),
                        props={"title": _recipe_id(title), "crafting_time": crafting_time},
                    )
                )
            elif page.recipe_id is not None:
                # Already Recipe-labelled: one node carries everything.
                merged = dict(page.props)
                merged["crafting_time"] = crafting_time
                merged.setdefault("title", title)
                merged["is_archived"] = title in self._archived_titles
                merged["introduced_in"] = _introduced_in(page.version_events)
                merged["aliases"] = sorted(self._aliases_of.get(title, []))
                nodes.append(ResolvedNode(id=page.item_id, labels=page.labels, props=merged))
            else:
                nodes.append(self._item_node(title, page))
        return nodes

    def _item_node(self, title: str, page: _PageAspects) -> ResolvedNode:
        props = dict(page.props)
        props.setdefault("title", title)
        props["is_archived"] = title in self._archived_titles
        props["introduced_in"] = _introduced_in(page.version_events)
        props["aliases"] = sorted(self._aliases_of.get(title, []))
        return ResolvedNode(id=page.item_id, labels=page.labels, props=props)

    # -- pass 3: edges ------------------------------------------------------------

    def _build_edges(
        self, aspects: dict[str, _PageAspects]
    ) -> tuple[list[ResolvedEdge], list[UnresolvedReference]]:
        edges: dict[tuple[str, str, str], ResolvedEdge] = {}
        unresolved: list[UnresolvedReference] = []

        def keep(edge: ResolvedEdge) -> None:
            key = (edge.from_id, edge.rel_type, edge.to_id)
            existing = edges.get(key)
            if existing is None or (not existing.props and edge.props):
                edges[key] = edge

        for fact in self._facts:
            target_title = self._canonical(str(fact.object)) if fact.object is not None else ""
            target = aspects.get(target_title)

            if fact.predicate == "rel.CONSUMED_BY":
                # "X is consumed by Y" -> Y's recipe CONSUMES X, no amount known here.
                if target is None or target.recipe_id is None:
                    unresolved.append(
                        UnresolvedReference(
                            subject=fact.subject, predicate=fact.predicate, object=target_title
                        )
                    )
                    continue
                item = aspects[fact.subject]
                edge = ResolvedEdge(
                    from_id=target.recipe_id, rel_type="CONSUMES", to_id=item.item_id
                )
                keep(edge)
                continue

            rel_type = _REL_TYPE.get(fact.predicate)
            if rel_type is None:
                continue  # a prop.* fact, already folded into node properties

            subject_page = aspects[fact.subject]
            if fact.predicate in _RECIPE_ASPECT_PREDICATES:
                # A CONSUMES/PRODUCES/CRAFTED_AT fact means this subject has recipe
                # data, so _build_aspects always set a recipe_id for it.
                assert subject_page.recipe_id is not None
                from_id = subject_page.recipe_id
            else:
                from_id = subject_page.item_id

            if rel_type == "REQUIRES":
                subject_is_unlock = "Unlock" in subject_page.labels
                target_is_unlock = target is not None and "Unlock" in target.labels
                if not (subject_is_unlock and target_is_unlock):
                    continue  # deliberately filtered, not logged: see module docstring
                assert target is not None
                keep(ResolvedEdge(from_id=from_id, rel_type=rel_type, to_id=target.item_id))
                continue

            if target is None:
                unresolved.append(
                    UnresolvedReference(
                        subject=fact.subject, predicate=fact.predicate, object=target_title
                    )
                )
                continue

            to_id = target.item_id
            if rel_type == "UNLOCKS" and target.recipe_id is not None:
                to_id = target.recipe_id  # the ontology's own arrow: Unlock -> Recipe

            keep(ResolvedEdge(from_id=from_id, rel_type=rel_type, to_id=to_id, props=fact.props))

        return list(edges.values()), unresolved

    def _add_station_label(
        self, nodes: list[ResolvedNode], edges: list[ResolvedEdge]
    ) -> list[ResolvedEdge]:
        station_ids = {e.to_id for e in edges if e.rel_type == "CRAFTED_AT"}
        if station_ids:
            for i, node in enumerate(nodes):
                if node.id in station_ids and "Station" not in node.labels:
                    nodes[i] = node.model_copy(update={"labels": (*node.labels, "Station")})
        return edges

    def _canonical(self, title: str) -> str:
        """The redirect/alias target for a title, or the title itself if it is
        already canonical (or unknown)."""
        return self._alias_target.get(title.casefold(), title)

    def _id(self, title: str) -> str:
        return f"{self._wiki}:{self._canonical(title)}"


def _introduced_in(events: list[tuple[str, str]]) -> str | None:
    """The version a page was introduced: the event whose text says so, else the
    earliest version by real (not lexicographic) ordering."""
    if not events:
        return None
    for version, text in events:
        if "introduced" in text.lower():
            return version
    return min(events, key=lambda e: _version_key(e[0]))[0]


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _VERSION_PART.findall(version)) or (0,)
