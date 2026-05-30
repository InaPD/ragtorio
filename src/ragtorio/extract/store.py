"""Where extraction's output goes.

A run replaces every fact for the wiki rather than upserting one at a time: facts are
fully recomputable from ``raw_page``, so after a parser change the old set is simply
wrong and should not linger mixed in with the new one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragtorio.extract.models import Fact


class FactStore(Protocol):
    """The persistence extraction needs, and nothing more."""

    def replace_all(self, wiki: str, facts: Sequence[Fact]) -> None:
        """Delete every fact stored for ``wiki`` and insert this run's instead."""
        ...

    def count(self, wiki: str) -> int:
        """How many facts are currently stored for ``wiki``."""
        ...


class InMemoryFactStore:
    """Facts held in a dict. Backs ``--dry-run`` and extractor-adjacent tests."""

    def __init__(self) -> None:
        self.facts: dict[str, tuple[Fact, ...]] = {}

    def replace_all(self, wiki: str, facts: Sequence[Fact]) -> None:
        self.facts[wiki] = tuple(facts)

    def count(self, wiki: str) -> int:
        return len(self.facts.get(wiki, ()))
