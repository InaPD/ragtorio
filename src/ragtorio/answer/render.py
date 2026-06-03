"""Turning what was retrieved into the text the answering model reads.

Phase 5's renderer produces flat, indented lines because they go into a log and a
terminal. An answering prompt deserves better: a recipe tree is a nested list with a
quantity on every line and a raw-material total at the end, and a research chain is an
ordered list, because the order *is* the answer and a model handed an unordered set
will confidently invent one.

**The two halves stay labeled.** Graph facts and passages arrive in separate tagged
sections, and the system prompt tells the model what each label means: one is an
infobox parameter extracted deterministically, the other is prose somebody wrote that
may be about a different version of the game. Concatenating them would throw away the
most useful thing the system knows about its own evidence.
"""

from __future__ import annotations

from typing import Any

from ragtorio.answer.models import GRAPH_CITATION
from ragtorio.retrieve.models import GraphResult, RetrievedContext

#: Deeper than this and the indentation stops carrying meaning in a prompt. Real
#: Factorio chains reach 10; the recipe walk's own bound is 15.
MAX_RENDER_DEPTH = 12


def render_prompt(context: RetrievedContext) -> str:
    """The complete user message: the question, then the labeled evidence."""
    sections = [f"<question>\n{context.question}\n</question>"]

    graph_text = render_graph_block(context)
    if graph_text:
        sections.append(f"<graph_facts>\n{graph_text}\n</graph_facts>")

    passages = _block_text(context, "passages")
    if passages:
        sections.append(f"<passages>\n{passages}\n</passages>")

    if len(sections) == 1:
        sections.append("<no_evidence>\nNothing was retrieved for this question.\n</no_evidence>")
    return "\n\n".join(sections)


def render_graph_block(context: RetrievedContext) -> str:
    """The graph half, structured where the template's rows allow it.

    Falls back to the merged block's flat text when there are no rows to work from -
    a forced ``vector`` intent still carries whatever the merge step produced, and a
    renderer that returned nothing there would silently drop evidence.
    """
    if context.graph is not None and not context.graph.is_empty:
        return render_graph(context.graph)
    return _block_text(context, "graph_facts")


def render_graph(graph: GraphResult) -> str:
    """One template's rows as the shape that template's answer wants."""
    renderer = {
        "recipe_tree": _render_recipe_tree,
        "unlock_chain": _render_unlock_chain,
    }.get(graph.template)
    if renderer is None or not graph.rows:
        return "\n".join(graph.lines)
    return renderer(list(graph.rows))


def _render_recipe_tree(rows: list[dict[str, Any]]) -> str:
    """The tree as a nested list, plus the raw-material total it exists to produce."""
    root = rows[0]
    lines: list[str] = []
    _walk(root, lines, depth=0)
    totals = _raw_totals(root)
    if totals:
        summed = ", ".join(f"{amount:g} {name}" for name, amount in sorted(totals.items()))
        lines.append("")
        lines.append(f"Raw material total for {_quantity(root)}: {summed}.")
    return "\n".join(lines)


def _walk(node: dict[str, Any], lines: list[str], depth: int) -> None:
    if depth > MAX_RENDER_DEPTH:
        return
    note = ""
    if node.get("truncated"):
        note = " (already on this branch; the chain loops here)"
    elif node.get("is_raw"):
        note = " (raw material)"
    lines.append(f"{'  ' * depth}- {_quantity(node)}{note}")
    for child in node.get("children") or ():
        _walk(child, lines, depth + 1)


def _quantity(node: dict[str, Any]) -> str:
    amount = node.get("amount")
    rendered = f"{amount:g}" if isinstance(amount, int | float) else str(amount)
    return f"{rendered} {node.get('title', '?')}"


def _raw_totals(node: dict[str, Any]) -> dict[str, float]:
    """Sum the leaves. The tree already carries the multiplied amounts."""
    totals: dict[str, float] = {}
    _collect(node, totals)
    return totals


def _collect(node: dict[str, Any], totals: dict[str, float]) -> None:
    children = node.get("children") or ()
    amount = node.get("amount")
    if not children and isinstance(amount, int | float):
        title = str(node.get("title", "?"))
        totals[title] = totals.get(title, 0.0) + float(amount)
        return
    for child in children:
        _collect(child, totals)


def _render_unlock_chain(rows: list[dict[str, Any]]) -> str:
    """The research order as an ordered list. The order is the answer."""
    lines: list[str] = []
    for row in rows:
        lines.append(f"{row.get('target')} is gated by the technology {row.get('unlock')}.")
        # A diamond in the prerequisite graph puts a technology on the path twice, and
        # a research order that lists it twice reads as two separate steps.
        chain = list(dict.fromkeys(str(name) for name in row.get("prerequisites") or ()))
        if chain:
            lines.append("Research in this order:")
            lines.extend(f"  {n}. {name}" for n, name in enumerate(chain, start=1))
        lines.append("")
    return "\n".join(lines).rstrip()


def _block_text(context: RetrievedContext, label: str) -> str:
    for block in context.blocks:
        if block.label == label:
            return block.text
    return ""


def citation_instruction(context: RetrievedContext) -> str:
    """What this particular question's evidence may be cited as.

    Built per request rather than baked into the system prompt: telling a model it may
    write ``[graph]`` on a question where nothing came from the graph is an invitation
    to cite a source that does not exist, and that is exactly the failure the validator
    is there to catch.
    """
    tokens: list[str] = []
    if context.graph is not None and not context.graph.is_empty:
        tokens.append(f"[{GRAPH_CITATION}] for anything in <graph_facts>")
    if context.chunk_ids:
        listed = ", ".join(f"[{chunk_id}]" for chunk_id in context.chunk_ids)
        tokens.append(f"the passage's own id for anything in <passages>: {listed}")
    if not tokens:
        return "There is no evidence to cite. Say that you do not know."
    return "Cite " + "; and ".join(tokens) + "."
