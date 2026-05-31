"""Loading community shorthand. Optional: a wiki with no file just gets none."""

from __future__ import annotations

from ragtorio.ontology.aliases import load_aliases


def test_factorio_aliases_load() -> None:
    aliases = load_aliases("factorio")
    assert aliases["green circuit"] == "Electronic circuit"
    assert aliases["blue science"] == "Chemical science pack"


def test_missing_aliases_file_is_empty() -> None:
    assert load_aliases("nosuchwiki") == {}
