"""``factorio_recipe_expr``, checked against the shapes the real wiki actually uses."""

from __future__ import annotations

import pytest

from ragtorio.extract.parsers.recipe import Ingredient, factorio_recipe_expr


def test_single_ingredient_no_output_side() -> None:
    """Iron gear wheel: the implicit output is the caller's job, not the parser's."""
    expr = factorio_recipe_expr("Time, 0.5 + Iron plate, 2")
    assert expr.time == 0.5
    assert expr.inputs == (Ingredient(name="Iron plate", amount=2.0),)
    assert expr.outputs == ()


def test_multi_output_with_larger_output_amounts() -> None:
    """Kovarex: a recipe whose output feeds its own input, net-positive uranium-235."""
    expr = factorio_recipe_expr(
        "Time, 60 + Uranium-235, 40 + Uranium-238, 5 = Uranium-235, 41 + Uranium-238, 2"
    )
    assert expr.time == 60.0
    assert expr.inputs == (
        Ingredient(name="Uranium-235", amount=40.0),
        Ingredient(name="Uranium-238", amount=5.0),
    )
    assert expr.outputs == (
        Ingredient(name="Uranium-235", amount=41.0),
        Ingredient(name="Uranium-238", amount=2.0),
    )


def test_fractional_output_amounts_become_probabilities() -> None:
    """Uranium processing: each craft yields exactly one item, chosen at random."""
    expr = factorio_recipe_expr(
        "Time, 12 + Uranium ore, 10 = Uranium-235, 0.007 + Uranium-238, 0.993"
    )
    assert expr.outputs == (
        Ingredient(name="Uranium-235", amount=1.0, probability=0.007),
        Ingredient(name="Uranium-238", amount=1.0, probability=0.993),
    )


def test_bare_output_with_no_amount_defaults_to_one() -> None:
    """Iron axe: the output side names one page, with no ``, amount`` at all."""
    expr = factorio_recipe_expr("Time, 0.5 + Iron plate, 3 + Iron stick, 2 = Archive:Iron axe")
    assert expr.outputs == (Ingredient(name="Archive:Iron axe", amount=1.0),)


def test_technology_cost_has_no_output_side() -> None:
    expr = factorio_recipe_expr("Time, 10 + Automation science pack, 1")
    assert expr.inputs == (Ingredient(name="Automation science pack", amount=1.0),)
    assert expr.outputs == ()


def test_missing_time_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not start with 'Time"):
        factorio_recipe_expr("Iron plate, 2")


def test_non_numeric_amount_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a number"):
        factorio_recipe_expr("Time, 1 + Iron plate, many")


def test_empty_ingredient_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty ingredient name"):
        factorio_recipe_expr("Time, 1 + , 2")
