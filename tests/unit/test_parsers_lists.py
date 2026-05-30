"""``plus_list``: the ``A + B + C`` grammar, unopinionated about what the names mean."""

from __future__ import annotations

from ragtorio.extract.parsers.lists import plus_list


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
