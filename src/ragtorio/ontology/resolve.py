"""Turns extracted facts into a graph: canonical ids, the Item/Recipe split, and
edges pointed at real nodes or logged as unresolved.

**Design decisions worth naming, since none of them are purely mechanical:**

- A page that both classifies as an ``Item`` (or ``Fluid``) and carries its own
  recipe becomes two nodes: ``factorio:Title`` (the item) and
  ``factorio:Title (recipe)`` (the recipe), the suffix being profile config. A page
  already classified as ``Recipe`` needs no split. ``CONSUMES``, ``PRODUCES``,
  ``CRAFTED_AT`` and the recipe's own ``crafting_time`` belong to the recipe aspect;
  everything else belongs to the item.
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
- ``rel.ALLOWS`` (the ``allows`` field) is inverted into ``REQUIRES``: "A allows B"
  is the same statement as "B requires A", written from the other end. This is where
  the technology tree actually comes from. The profile originally left ``allows``
  unmapped on the theory that it duplicated ``required-technologies``; measuring the
  crawl showed otherwise - not one of 696 ``required-technologies`` entries names a
  technology, they are all science packs and items, so without ``allows`` the graph
  had exactly one ``REQUIRES`` edge in it.
- **A technology is referenced without the suffix its page carries.** ``allows`` says
  "Flammables"; the page is "Flammables (research)". The profile's
  ``resolution.reference_suffixes`` lists what to try, so the convention stays in
  config rather than in this module.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ragtorio.extract.models import Fact
from ragtorio.harvest.models import RawRedirect
from ragtorio.ontology.canonical import TitleCanonicalizer
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

#: Categories describing the page rather than the thing. Present on hundreds of pages
#: each, so they would swamp any grouping built on this property.
_BOOKKEEPING_CATEGORIES = frozenset({"English page", "Infobox page", "Archived"})


def _recipe_id(title: str, suffix: str) -> str:
    """The second node a page gets when it is both an item and a recipe.

    The suffix is profile config rather than a literal here: it is a naming
    convention, and the same string has to be understood by ``recipe_tree`` when it
    looks for an item's own recipe.
    """
    return f"{title} {suffix}"


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
        categories: dict[str, list[str]] | None = None,
        reference_suffixes: list[str] | None = None,
        recipe_suffix: str = "(recipe)",
    ) -> None:
        self._wiki = wiki
        self._facts = facts
        self._archived_titles = archived_titles
        self._titles = TitleCanonicalizer(redirects, aliases)
        self._categories = categories or {}
        self._suffixes = list(reference_suffixes or [])
        self._recipe_suffix = recipe_suffix

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
                recipe_id=self._id(_recipe_id(title, self._recipe_suffix))
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
                        props={
                            "title": _recipe_id(title, self._recipe_suffix),
                            "crafting_time": crafting_time,
                        },
                    )
                )
            elif page.recipe_id is not None:
                # Already Recipe-labelled: one node carries everything.
                merged = dict(page.props)
                merged["crafting_time"] = crafting_time
                merged.setdefault("title", title)
                merged["is_archived"] = title in self._archived_titles
                merged["introduced_in"] = _introduced_in(page.version_events)
                merged["aliases"] = self._titles.aliases_of(title)
                merged["categories"] = self._categories_of(title)
                nodes.append(ResolvedNode(id=page.item_id, labels=page.labels, props=merged))
            else:
                nodes.append(self._item_node(title, page))
        return nodes

    def _item_node(self, title: str, page: _PageAspects) -> ResolvedNode:
        props = dict(page.props)
        props.setdefault("title", title)
        props["is_archived"] = title in self._archived_titles
        props["introduced_in"] = _introduced_in(page.version_events)
        props["aliases"] = self._titles.aliases_of(title)
        props["categories"] = self._categories_of(title)
        return ResolvedNode(id=page.item_id, labels=page.labels, props=props)

    def _categories_of(self, title: str) -> list[str]:
        """The wiki's own categories for a title, minus its bookkeeping ones.

        "English page" and "Infobox page" are on almost every page and group nothing a
        player would recognise, so leaving them in would make ``tier_compare`` return
        the whole wiki for any seed.
        """
        return [c for c in self._categories.get(title, []) if c not in _BOOKKEEPING_CATEGORIES]

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

            if fact.predicate == "rel.ALLOWS":
                # "A allows B" is "B requires A". Both ends must be technologies:
                # `allows` also names archived pages this crawl never classified.
                follow_on = self._lookup_unlock(aspects, target_title)
                subject_page = aspects[fact.subject]
                if follow_on is None or "Unlock" not in subject_page.labels:
                    if follow_on is None:
                        unresolved.append(
                            UnresolvedReference(
                                subject=fact.subject,
                                predicate=fact.predicate,
                                object=target_title,
                            )
                        )
                    continue
                keep(
                    ResolvedEdge(
                        from_id=follow_on.item_id,
                        rel_type="REQUIRES",
                        to_id=subject_page.item_id,
                    )
                )
                continue

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

    def _lookup_unlock(self, aspects: dict[str, _PageAspects], title: str) -> _PageAspects | None:
        """Find the ``Unlock`` a reference names, trying the profile's suffixes.

        Only used where an Unlock is what the reference must be, so a suffix can never
        turn an item reference into a technology by accident.
        """
        for candidate in (title, *(f"{title} {suffix}" for suffix in self._suffixes)):
            page = aspects.get(self._canonical(candidate))
            if page is not None and "Unlock" in page.labels:
                return page
        return None

    def _canonical(self, title: str) -> str:
        """The redirect/alias target for a title, or the title itself if it is
        already canonical (or unknown)."""
        return self._titles.canonical(title)

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
