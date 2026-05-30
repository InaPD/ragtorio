"""The in-memory page repository that backs extractor tests."""

from __future__ import annotations

from datetime import UTC, datetime

from ragtorio.extract.repository import InMemoryPageRepository
from ragtorio.harvest.models import RawPage

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def page(title: str, ns: int = 0, wiki: str = "factorio") -> RawPage:
    return RawPage(
        wiki=wiki, page_id=1, ns=ns, title=title, revision_id=1, revised_at=NOW, wikitext="x"
    )


def test_by_namespace_filters_wiki_and_namespace() -> None:
    repo = InMemoryPageRepository(
        [
            page("Iron plate", ns=0),
            page("Infobox:Iron plate", ns=3002),
            page("Parsnip", wiki="stardew"),
        ]
    )
    assert {p.title for p in repo.by_namespace("factorio", 0)} == {"Iron plate"}


def test_by_title_is_scoped_and_missing_pages_are_none() -> None:
    repo = InMemoryPageRepository([page("Iron plate")])
    assert repo.by_title("factorio", "Iron plate") is not None
    assert repo.by_title("factorio", "Nothing here") is None
    assert repo.by_title("stardew", "Iron plate") is None
