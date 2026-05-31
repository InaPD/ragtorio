"""``recipe_tree``: an item's ingredients, recursively, down to raw materials.

The first Cypher template (section 6, Phase 3). One small query per level; the
multiplication happens in Python, level by level, which is what keeps the recursion
readable and lets a depth limit stand in for real cycle detection (Kovarex
legitimately consumes and produces the same item - see ``check.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver, Session

#: Kovarex-shaped cycles terminate here rather than recursing forever. 15 levels is
#: far deeper than any real Factorio recipe chain, so hitting it means a cycle.
DEFAULT_MAX_DEPTH = 15

_FIND_RECIPE = """
MATCH (r:Recipe)-[p:PRODUCES]->(i {id: $item_id})
RETURN r.id AS recipe_id, p.amount AS amount, p.probability AS probability
ORDER BY CASE WHEN r.id = $preferred_id THEN 0 ELSE 1 END, r.id
LIMIT 1
"""

_INGREDIENTS = """
MATCH (r {id: $recipe_id})-[c:CONSUMES]->(i)
RETURN i.title AS title, c.amount AS amount
ORDER BY i.title
"""


@dataclass(frozen=True)
class RecipeTreeNode:
    """One item's position in the tree: how much of it is needed here, and what it
    takes to make that much - or nothing, if it is raw or the recursion was cut off."""

    title: str
    amount: float
    is_raw: bool
    truncated: bool = False
    children: tuple[RecipeTreeNode, ...] = ()


def recipe_tree(
    driver: Driver,
    wiki: str,
    item_title: str,
    quantity: float = 1.0,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> RecipeTreeNode:
    """The tree for ``quantity`` units of ``item_title``.

    When more than one recipe produces the item, the recipe named after the item
    itself wins (the common case: an item's own infobox recipe); otherwise the
    lowest id, so the choice is at least deterministic.
    """
    with driver.session() as session:
        return _tree(session, wiki, item_title, quantity, max_depth)


def _tree(
    session: Session, wiki: str, title: str, quantity: float, depth_left: int
) -> RecipeTreeNode:
    if depth_left <= 0:
        return RecipeTreeNode(title=title, amount=quantity, is_raw=True, truncated=True)

    item_id = f"{wiki}:{title}"
    preferred_id = f"{wiki}:{title} (recipe)"
    row = session.run(_FIND_RECIPE, item_id=item_id, preferred_id=preferred_id).single()
    if row is None:
        return RecipeTreeNode(title=title, amount=quantity, is_raw=True)

    # A probabilistic recipe (uranium processing) yields `amount` units only
    # `probability` of the time; the batches needed for a raw-material total is an
    # expected-value calculation, matching how the wiki itself reasons about this
    # exact recipe ("expected output of 1 uranium-235 per ~143 crafting cycles").
    expected_yield = (row["amount"] or 1.0) * (row["probability"] or 1.0)
    batches = quantity / expected_yield if expected_yield else 0.0

    ingredients = list(session.run(_INGREDIENTS, recipe_id=row["recipe_id"]))
    children = tuple(
        _tree(session, wiki, ing["title"], ing["amount"] * batches, depth_left - 1)
        for ing in ingredients
    )
    return RecipeTreeNode(title=title, amount=quantity, is_raw=False, children=children)


def raw_totals(node: RecipeTreeNode) -> dict[str, float]:
    """Sum of every raw material at the leaves, added across every branch that uses it."""
    if node.is_raw:
        return {node.title: node.amount}
    totals: dict[str, float] = {}
    for child in node.children:
        for name, amount in raw_totals(child).items():
            totals[name] = totals.get(name, 0.0) + amount
    return totals
