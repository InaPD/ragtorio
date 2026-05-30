"""The in-memory fact store: a run replaces a wiki's facts, scoped per wiki."""

from __future__ import annotations

from ragtorio.extract.models import Fact, Provenance
from ragtorio.extract.store import InMemoryFactStore


def fact() -> Fact:
    return Fact(
        subject="X",
        subject_labels=("Item",),
        predicate="prop.stack_size",
        object=100.0,
        provenance=Provenance(wiki="factorio", page_id=1, revision_id=1, field="stack-size"),
    )


def test_replace_all_wipes_what_was_there_before_and_is_scoped_per_wiki() -> None:
    store = InMemoryFactStore()
    store.replace_all("factorio", [fact(), fact(), fact()])
    store.replace_all("factorio", [fact()])
    store.replace_all("stardew", [fact()])
    assert store.count("factorio") == 1
    assert store.count("stardew") == 1
