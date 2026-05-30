"""The recipe grammar: ``Time, N + Name, N + Name, N [= Name, N + Name, N]``.

Used for two different profile fields with two different meanings. On ``recipe`` the
left side is what a machine consumes and the right side, if present, is what it
produces; on ``cost`` (a technology's science-pack price) there is no right side at
all. Both share the same "Time plus a list of amounts" shape, which is why one parser
serves both -- ``TemplateExtractor`` decides what the result means, this module only
parses it.

An ingredient may also appear bare, with no amount (``= Archive:Iron axe``), which
defaults to 1. On the output side only, an amount strictly between 0 and 1 is not a
quantity: it is a probability of producing exactly one unit, which is how the wiki
expresses recipes like uranium processing that yield one item chosen at random.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

_PLUS_SPLIT = re.compile(r"\s*\+\s*")

#: The pseudo-ingredient every recipe and cost expression opens with.
_TIME = "Time"


class Ingredient(BaseModel):
    """One ``Name, amount`` entry. ``probability`` is 1.0 unless the amount was a
    fractional output amount, in which case ``amount`` becomes 1 and the fraction
    moves here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    amount: float
    probability: float = 1.0


class RecipeExpr(BaseModel):
    """A parsed expression. ``outputs`` is empty when the wikitext gave no ``=`` side;
    the caller supplies the implicit self-output, since this parser has no page
    context to know what that title is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: float
    inputs: tuple[Ingredient, ...]
    outputs: tuple[Ingredient, ...]


def factorio_recipe_expr(value: str) -> RecipeExpr:
    """Parse one recipe or cost expression.

    Raises:
        ValueError: the expression is empty, an ingredient has no name, an amount is
            not a number, or the expression does not open with ``Time, <n>``.
    """
    lhs, has_output, rhs = value.partition("=")
    lhs_ingredients = _parse_ingredient_list(lhs, probabilistic=False)
    if not lhs_ingredients or lhs_ingredients[0].name != _TIME:
        raise ValueError(f"recipe expression does not start with 'Time, <n>': {value!r}")
    time = lhs_ingredients[0].amount
    inputs = lhs_ingredients[1:]
    outputs = _parse_ingredient_list(rhs, probabilistic=True) if has_output else ()
    return RecipeExpr(time=time, inputs=tuple(inputs), outputs=tuple(outputs))


def _parse_ingredient_list(text: str, *, probabilistic: bool) -> tuple[Ingredient, ...]:
    ingredients = []
    for part in _PLUS_SPLIT.split(text.strip()):
        if not part:
            continue
        name, has_amount, amount_text = part.partition(",")
        name = name.strip()
        if not name:
            raise ValueError(f"empty ingredient name in {text!r}")
        amount = _parse_amount(amount_text.strip()) if has_amount else 1.0
        if probabilistic and 0 < amount < 1:
            ingredients.append(Ingredient(name=name, amount=1.0, probability=amount))
        else:
            ingredients.append(Ingredient(name=name, amount=amount))
    return tuple(ingredients)


def _parse_amount(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"not a number: {raw!r}") from exc
