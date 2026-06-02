"""Running one of four named Cypher templates, with parameters only.

**No query is ever built from a string.** Three of the templates are ``.cypher`` files
read off disk and handed to the driver unchanged; the fourth, ``recipe_tree``, is the
recursive walk Phase 3 already implemented in Python, because a probability-weighted
expected-value calculation over a tree is not something Cypher does well and the wiki's
own reasoning about uranium processing is the arithmetic that walk performs. What every
template shares is the property that matters: the model picks a name from an enum, and
everything variable about the query arrives as a bound parameter.

Rendering lives here too. A template's rows are shaped by its query, so the code that
knows what a row means is the code that should turn it into a line - and what reaches
the answering model is lines, not result objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neo4j import Driver

from ragtorio.config import RetrievalConfig
from ragtorio.ontology.recipe_tree import RecipeTreeNode, raw_totals, recipe_tree
from ragtorio.retrieve.models import GraphResult, Route

CYPHER_DIR = Path(__file__).with_name("cypher")

#: Templates that are a file. ``recipe_tree`` is deliberately absent; see the docstring.
_FILE_TEMPLATES = ("unlock_chain", "consumers_of", "tier_compare")

#: Rows per template. Enough to answer "what uses sulfuric acid" without turning the
#: context window into a directory listing.
DEFAULT_LIMIT = 25

#: Which properties ``tier_compare`` may order by is profile config (the ``retrieval``
#: section), not a constant here: they are whatever that wiki's infobox fields were
#: mapped to, and naming Factorio's three in this module put one game's vocabulary in
#: the middle of the retrieval layer. The set stays closed wherever it comes from,
#: because the property reaches Cypher as a dynamic key - bounding it means a malformed
#: router output can only ever name a column that exists.


class GraphRetriever:
    """Runs the graph half of a route. The caller owns the driver."""

    def __init__(
        self,
        driver: Driver,
        wiki: str,
        *,
        limit: int = DEFAULT_LIMIT,
        raw_items: frozenset[str] = frozenset(),
        recipe_suffix: str = "(recipe)",
        retrieval: RetrievalConfig | None = None,
    ) -> None:
        self._driver = driver
        self._wiki = wiki
        self._limit = limit
        self._raw_items = raw_items
        self._recipe_suffix = recipe_suffix
        self._retrieval = retrieval or RetrievalConfig()
        self._queries = {name: _read(name) for name in _FILE_TEMPLATES}

    def run(self, route: Route) -> GraphResult:
        """Execute the route's template, or return an empty result if it has none."""
        if route.template is None or not route.entities:
            return GraphResult(template=route.template or "none")
        entity = route.entities[0]

        if route.template == "recipe_tree":
            tree = recipe_tree(
                self._driver,
                self._wiki,
                entity.title,
                raw_items=self._raw_items,
                recipe_suffix=self._recipe_suffix,
            )
            return _render_recipe_tree(tree)

        params: dict[str, Any] = {"entity_id": entity.id, "limit": self._limit}
        if route.template == "tier_compare":
            ordered_by = self._retrieval.comparable(route.compare_by)
            if ordered_by is None:
                # A profile declaring no comparable properties cannot answer this
                # template at all, and an empty result says so better than a crash.
                return GraphResult(template="tier_compare")
            params["property"] = ordered_by
            params["prefix"] = f"{self._wiki}:"

        with self._driver.session() as session:
            rows = [dict(record) for record in session.run(self._queries[route.template], **params)]
        return _render(route.template, rows)


def _read(name: str) -> str:
    path = CYPHER_DIR / f"{name}.cypher"
    if not path.is_file():  # pragma: no cover - a packaging error, not a runtime one
        raise FileNotFoundError(f"no Cypher template at {path}")
    return path.read_text(encoding="utf-8")


def _render(template: str, rows: list[dict[str, Any]]) -> GraphResult:
    renderer = {
        "unlock_chain": _render_unlock_chain,
        "consumers_of": _render_consumers_of,
        "tier_compare": _render_tier_compare,
    }[template]
    return GraphResult(template=template, rows=tuple(rows), lines=tuple(renderer(rows)))


def _render_unlock_chain(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        # A diamond in the tree can put the same technology on the path twice; a
        # research order that lists it twice is just wrong.
        chain = " -> ".join(dict.fromkeys(str(name) for name in row.get("prerequisites") or ()))
        lines.append(f"{row['target']} is gated by the technology {row['unlock']}.")
        if chain:
            lines.append(f"Research order: {chain}.")
    return lines


def _render_consumers_of(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        amount = row.get("amount")
        quantity = f"{amount:g}" if isinstance(amount, int | float) else "an unstated amount"
        produces = ", ".join(str(p) for p in row.get("produces") or ()) or "nothing recorded"
        stations = ", ".join(str(s) for s in row.get("stations") or ())
        where = f" at {stations}" if stations else ""
        lines.append(f"{row['recipe']} consumes {quantity} of it{where} and produces {produces}.")
    return lines


def _render_tier_compare(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        value = row["value"]
        rendered = f"{value:g}" if isinstance(value, int | float) else str(value)
        shared = ", ".join(str(c) for c in row.get("categories") or ())
        lines.append(f"{row['title']}: {rendered} ({shared})")
    return lines


def _render_recipe_tree(tree: RecipeTreeNode) -> GraphResult:
    """A tree as indented lines plus a raw-material total.

    Flat lines rather than a nested structure because this is what goes into a prompt;
    the tree itself is kept on the result so Phase 6 can render it as a real list.
    """
    lines: list[str] = []
    _walk(tree, lines, depth=0)
    totals = raw_totals(tree)
    if totals:
        summed = ", ".join(f"{name} {amount:g}" for name, amount in sorted(totals.items()))
        lines.append(f"Raw materials for {tree.amount:g} {tree.title}: {summed}.")
    return GraphResult(template="recipe_tree", rows=(_as_row(tree),), lines=tuple(lines))


def _walk(node: RecipeTreeNode, lines: list[str], depth: int) -> None:
    marker = " (raw)" if node.is_raw and not node.truncated else ""
    cut = " (cycle, not followed)" if node.truncated else ""
    lines.append(f"{'  ' * depth}{node.title}: {node.amount:g}{marker}{cut}")
    for child in node.children:
        _walk(child, lines, depth + 1)


def _as_row(node: RecipeTreeNode) -> dict[str, Any]:
    return {
        "title": node.title,
        "amount": node.amount,
        "is_raw": node.is_raw,
        "truncated": node.truncated,
        "children": [_as_row(child) for child in node.children],
    }
