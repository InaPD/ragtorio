"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.models import RawPage

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wikitext" / "factorio"

#: scripts/fetch_fixtures.py's slugify() strips ":" and "(" ")" for the filesystem, so
#: the real title has to be recovered from that script's own PAGES list rather than
#: guessed from the filename -- "Automation_research.txt" is really
#: "Automation (research)", not "Automation research".
FACTORIO_FIXTURE_TITLES: dict[str, tuple[str, int]] = {
    "Iron_gear_wheel": ("Iron gear wheel", 0),
    "Infobox__Iron_gear_wheel": ("Infobox:Iron gear wheel", 3002),
    "Electronic_circuit": ("Electronic circuit", 0),
    "Infobox__Electronic_circuit": ("Infobox:Electronic circuit", 3002),
    "Oil_processing": ("Oil processing", 0),
    "Infobox__Uranium_processing": ("Infobox:Uranium processing", 3002),
    "Uranium_processing": ("Uranium processing", 0),
    "Kovarex_enrichment_process": ("Kovarex enrichment process", 0),
    "Infobox__Kovarex_enrichment_process": ("Infobox:Kovarex enrichment process", 3002),
    "Automation_research": ("Automation (research)", 0),
    "Infobox__Automation_research": ("Infobox:Automation (research)", 3002),
    "Assembling_machine_2": ("Assembling machine 2", 0),
    "Infobox__Assembling_machine_2": ("Infobox:Assembling machine 2", 3002),
    "Infobox__Iron_axe": ("Infobox:Iron axe", 3002),
    "Infobox__Yumako_tree": ("Infobox:Yumako tree", 3002),
}


@pytest.fixture
def wikitext() -> dict[str, str]:
    """Every committed Factorio fixture, keyed by filename stem."""
    return {p.stem: p.read_text(encoding="utf-8") for p in FIXTURE_DIR.glob("*.txt")}


@pytest.fixture
def factorio_pages() -> list[RawPage]:
    """Every committed Factorio fixture as a ``RawPage``, with its real wiki title.

    This is the closest thing to a real crawl the test suite has: the extractor tests
    run against it rather than against synthetic wikitext wherever a real fixture
    already covers the case.
    """
    revised_at = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        RawPage(
            wiki="factorio",
            page_id=page_id,
            ns=ns,
            title=title,
            revision_id=1,
            revised_at=revised_at,
            wikitext=(FIXTURE_DIR / f"{stem}.txt").read_text(encoding="utf-8"),
        )
        for page_id, (stem, (title, ns)) in enumerate(FACTORIO_FIXTURE_TITLES.items(), start=1)
    ]


@pytest.fixture
def minimal_profile_data() -> dict[str, Any]:
    """The smallest profile dict that validates. Tests mutate copies of this."""
    return {
        "wiki": {
            "id": "testwiki",
            "api": "https://example.test/api.php",
            "license": "CC BY-SA 4.0",
            "attribution": "Test Wiki",
            "language_filter": {"strategy": "none"},
        },
        "infobox": {
            "location": "inline",
            "template": "Infobox",
            "type_field": "kind",
            "type_map": {"widget": ["Item"]},
        },
    }


@pytest.fixture
def fast_client() -> Callable[..., MediaWikiClient]:
    """A drop-in for ``MediaWikiClient`` that ignores the profile's 1 req/s.

    Patch it over ``ragtorio.cli.MediaWikiClient`` so a CLI test does not really wait.
    """

    def build(api_url: str, user_agent: str, *, rps: float) -> MediaWikiClient:
        return MediaWikiClient(api_url, user_agent, rps=1000.0, sleep=lambda _: None)

    return build


@pytest.fixture
def factorio_responder() -> Callable[[httpx.Request], httpx.Response]:
    """Answers the crawl of the real ``factorio`` profile with one page and one redirect."""
    identity = {"pageid": 1, "ns": 0, "title": "Iron plate"}
    timestamp = "2026-01-01T00:00:00Z"

    def respond(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("meta") == "siteinfo":
            return httpx.Response(200, json={"query": {"languages": [{"code": "de"}]}})
        if params.get("prop") == "redirects":
            pages = [{"title": "Iron plate", "redirects": [{"title": "Green circuit"}]}]
            return httpx.Response(200, json={"query": {"pages": pages}})
        if params.get("generator") == "allpages":
            if params["gapnamespace"] != "0":
                return httpx.Response(200, json={"query": {"pages": []}})
            stub = {**identity, "revisions": [{"revid": 7, "timestamp": timestamp}]}
            return httpx.Response(200, json={"query": {"pages": [stub]}})
        full = {
            **identity,
            "revisions": [
                {"revid": 7, "timestamp": timestamp, "slots": {"main": {"content": "body"}}}
            ],
            "categories": [{"title": "Category:Items"}],
        }
        return httpx.Response(200, json={"query": {"pages": [full]}})

    return respond
