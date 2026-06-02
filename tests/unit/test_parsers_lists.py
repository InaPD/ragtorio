"""``plus_list``: the ``A + B + C`` grammar, unopinionated about what the names mean."""

from __future__ import annotations

from ragtorio.extract.parsers.lists import leveled_name_list, plus_list


def test_splits_and_trims_entries() -> None:
    assert plus_list("Assembling machine + Player") == ["Assembling machine", "Player"]
    assert plus_list("A  +   B") == ["A", "B"]


def test_lowercase_entries_are_kept_verbatim() -> None:
    """Iron axe: 'manual' is not a title-cased entity name, and that's fine here."""
    assert plus_list("manual + assembling machine") == ["manual", "assembling machine"]


def test_no_plus_or_stray_plus() -> None:
    assert plus_list("Electronics") == ["Electronics"]
    assert plus_list("A + + B") == ["A", "B"]
    assert plus_list("") == []


class TestLeveledNameList:
    """The `allows` grammar: a technology name with the levels it applies to appended."""

    def test_a_trailing_level_is_dropped(self):
        assert leveled_name_list("Automation 2, 2") == ["Automation 2"]

    def test_a_trailing_level_range_is_dropped(self):
        assert leveled_name_list("Gun turret damage, 2-7") == ["Gun turret damage"]

    def test_it_still_splits_on_plus(self):
        assert leveled_name_list("Flammables + Plastics + Sulfur processing") == [
            "Flammables",
            "Plastics",
            "Sulfur processing",
        ]

    def test_levels_are_stripped_per_entry(self):
        assert leveled_name_list("Bulk inserter + Mining productivity, 1 + Modules") == [
            "Bulk inserter",
            "Mining productivity",
            "Modules",
        ]

    def test_a_number_that_is_part_of_the_name_survives(self):
        """ "Automation 2" is a technology; ", 2" is a level. Only the latter goes."""
        assert leveled_name_list("Logistics 2") == ["Logistics 2"]

    def test_an_empty_value_yields_nothing(self):
        assert leveled_name_list("") == []
