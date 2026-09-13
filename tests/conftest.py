"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragtorio.harvest.client import MediaWikiClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wikitext" / "factorio"


@pytest.fixture
def wikitext() -> dict[str, str]:
    """Every committed Factorio fixture, keyed by filename stem."""
    return {p.stem: p.read_text(encoding="utf-8") for p in FIXTURE_DIR.glob("*.txt")}


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
        if params.get("gapfilterredir") == "redirects":
            pairs = [{"from": "Green circuit", "to": "Iron plate"}]
            return httpx.Response(200, json={"query": {"redirects": pairs}})
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
