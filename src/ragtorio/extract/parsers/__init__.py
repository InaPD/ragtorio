"""Pure functions from a raw infobox parameter value to a parsed shape.

Every parser here takes only the string value: no page context, no profile, no
network. That is what lets them be tested against the committed wikitext fixtures
with no mocking at all, and it is what a Lua or Cargo extractor for a different wiki
would reimplement without touching anything else in this package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragtorio.extract.parsers.lists import leveled_name_list, plus_list
from ragtorio.extract.parsers.recipe import Ingredient, RecipeExpr, factorio_recipe_expr

#: Every parser a profile may name, keyed exactly as ``config.KNOWN_PARSERS`` spells it.
#: A profile can reference a name here that has no entry yet (reserved for a future
#: wiki); :class:`~ragtorio.extract.template.TemplateExtractor` fails fast on that at
#: construction time rather than silently mishandling a field at extraction time.
PARSER_REGISTRY: dict[str, Callable[[str], Any]] = {
    "factorio_recipe_expr": factorio_recipe_expr,
    "plus_list": plus_list,
    "leveled_name_list": leveled_name_list,
}

__all__ = [
    "PARSER_REGISTRY",
    "Ingredient",
    "RecipeExpr",
    "factorio_recipe_expr",
    "leveled_name_list",
    "plus_list",
]
