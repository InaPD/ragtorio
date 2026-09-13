"""The crawl.

The behaviours worth pinning down are the ones that cost real money in requests: that
translations never reach the fetch pass, that an unchanged wiki is not refetched, and
that content is asked for 50 titles at a time rather than one.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from ragtorio.config import WikiProfile
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.crawl import Crawler
from ragtorio.harvest.language import TranslationFilter
from ragtorio.harvest.store import InMemoryHarvestStore

API = "https://example.test/api.php"


class Page:
    """One page on the fake wiki."""

    def __init__(
        self,
        page_id: int,
        ns: int,
        title: str,
        revid: int = 1,
        text: str = "body",
        categories: tuple[str, ...] = (),
    ) -> None:
        self.page_id = page_id
        self.ns = ns
        self.title = title
        self.revid = revid
        self.text = text
        self.categories = categories

    def stub(self) -> dict[str, Any]:
        return {
            "pageid": self.page_id,
            "ns": self.ns,
            "title": self.title,
            "revisions": [{"revid": self.revid, "timestamp": "2026-01-01T00:00:00Z"}],
        }

    def full(self) -> dict[str, Any]:
        return {
            "pageid": self.page_id,
            "ns": self.ns,
            "title": self.title,
            "revisions": [
                {
                    "revid": self.revid,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "slots": {"main": {"content": self.text}},
                }
            ],
            "categories": [{"title": c} for c in self.categories],
        }


class FakeWiki:
    """Answers the four query shapes the crawler issues, keyed on parameters.

    Dispatching on parameters rather than call order keeps these tests readable and
    stops them breaking every time the crawler reorders a pass.
    """

    def __init__(
        self,
        pages: list[Page],
        redirects: dict[int, list[tuple[str, str]]] | None = None,
        languages: tuple[str, ...] = ("de", "fr", "pt-br"),
    ) -> None:
        self.pages = pages
        self.redirects = redirects or {}
        self.languages = languages
        self.content_requests: list[list[str]] = []
        self.catalogue_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("meta") == "siteinfo":
            return self._json({"languages": [{"code": c} for c in self.languages]})
        if params.get("generator") == "allpages":
            namespace = int(params["gapnamespace"])
            if params.get("gapfilterredir") == "redirects":
                pairs = self.redirects.get(namespace, [])
                return self._json({"redirects": [{"from": f, "to": t} for f, t in pairs]})
            self.catalogue_requests += 1
            listed = [p.stub() for p in self.pages if p.ns == namespace]
            return self._json({"pages": listed})
        if "titles" in params:
            titles = params["titles"].split("|")
            self.content_requests.append(titles)
            wanted = [p.full() for p in self.pages if p.title in titles]
            return self._json({"pages": wanted})
        raise AssertionError(f"unexpected query: {dict(params)}")

    @staticmethod
    def _json(query: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"query": query})


@pytest.fixture
def profile(minimal_profile_data: dict[str, Any]) -> WikiProfile:
    """A profile shaped like Factorio's: three namespaces, translations filtered."""
    data = {
        **minimal_profile_data,
        "wiki": {
            **minimal_profile_data["wiki"],
            "id": "factorio",
            "api": API,
            "language_filter": {"strategy": "subpage_suffix", "keep": "en"},
            "archived": {"namespace_id": 3004},
            "extra_namespaces": [3002],
        },
    }
    return WikiProfile.model_validate(data)


def make_crawler(
    profile: WikiProfile,
    store: InMemoryHarvestStore,
    *,
    batch_size: int = 50,
) -> Crawler:
    client = MediaWikiClient(API, "ua", rps=1000.0, sleep=lambda _: None)
    return Crawler(
        client,
        profile,
        store,
        batch_size=batch_size,
        translations=TranslationFilter("en", frozenset({"de", "fr", "pt-br"})),
    )


@respx.mock
def test_crawls_every_namespace_the_profile_asks_for(profile: WikiProfile) -> None:
    wiki = FakeWiki(
        [
            Page(1, 0, "Iron plate", categories=("Category:Items",)),
            Page(2, 3002, "Infobox:Iron plate"),
            Page(3, 3004, "Archive:Burner lab"),
        ]
    )
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    stats = make_crawler(profile, store).run()

    assert stats.fetched == 3
    assert stats.by_namespace == {0: 1, 3002: 1, 3004: 1}
    assert {p.title for p in store.pages.values()} == {
        "Iron plate",
        "Infobox:Iron plate",
        "Archive:Burner lab",
    }


@respx.mock
def test_stores_wikitext_and_categories(profile: WikiProfile) -> None:
    wiki = FakeWiki(
        [Page(1, 0, "Iron plate", text="== Uses ==", categories=("Category:Items", "Category:A"))]
    )
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    make_crawler(profile, store).run()

    page = store.pages[("factorio", 1)]
    assert page.wikitext == "== Uses =="
    assert page.categories == ("Category:Items", "Category:A")
    assert page.revision_id == 1


@respx.mock
def test_translations_are_dropped_before_any_content_is_fetched(profile: WikiProfile) -> None:
    wiki = FakeWiki(
        [
            Page(1, 0, "Iron plate"),
            Page(2, 0, "Iron plate/de"),
            Page(3, 0, "Iron plate/fr"),
            Page(4, 0, "Blueprint/tips"),
        ]
    )
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    stats = make_crawler(profile, store).run()

    assert stats.translations == 2
    assert stats.fetched == 2
    fetched_titles = [t for batch in wiki.content_requests for t in batch]
    assert "Iron plate/de" not in fetched_titles
    assert "Blueprint/tips" in fetched_titles


@respx.mock
def test_second_run_fetches_nothing(profile: WikiProfile) -> None:
    """The Phase 1 exit criterion: re-crawling an unchanged wiki costs no content."""
    wiki = FakeWiki([Page(n, 0, f"Page {n}") for n in range(1, 6)])
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()

    first = make_crawler(profile, store).run()
    assert first.fetched == 5

    wiki.content_requests.clear()
    second = make_crawler(profile, store).run()

    assert second.fetched == 0
    assert second.unchanged == 5
    assert wiki.content_requests == []


@respx.mock
def test_changed_revision_is_refetched(profile: WikiProfile) -> None:
    pages = [Page(1, 0, "Iron plate", revid=1), Page(2, 0, "Copper plate", revid=1)]
    wiki = FakeWiki(pages)
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    make_crawler(profile, store).run()

    pages[0].revid = 2
    pages[0].text = "edited"
    wiki.content_requests.clear()
    stats = make_crawler(profile, store).run()

    assert stats.fetched == 1
    assert stats.unchanged == 1
    assert wiki.content_requests == [["Iron plate"]]
    assert store.pages[("factorio", 1)].wikitext == "edited"


@respx.mock
def test_content_is_requested_in_batches(profile: WikiProfile) -> None:
    wiki = FakeWiki([Page(n, 0, f"Page {n}") for n in range(1, 8)])
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    make_crawler(profile, store, batch_size=3).run()

    assert [len(batch) for batch in wiki.content_requests] == [3, 3, 1]


@respx.mock
def test_redirects_are_stored_and_translations_excluded(profile: WikiProfile) -> None:
    wiki = FakeWiki(
        [Page(1, 0, "Electronic circuit")],
        redirects={
            0: [
                ("Green circuit", "Electronic circuit"),
                ("Grüner Schaltkreis/de", "Electronic circuit"),
            ]
        },
    )
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    stats = make_crawler(profile, store).run()

    assert stats.redirects == 1
    assert store.redirects[("factorio", "Green circuit")].to_title == "Electronic circuit"


@respx.mock
def test_limit_caps_pages_per_namespace(profile: WikiProfile) -> None:
    wiki = FakeWiki([Page(n, 0, f"Page {n}") for n in range(1, 21)])
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    stats = make_crawler(profile, store).run(limit=4)

    assert stats.by_namespace[0] == 4
    assert stats.fetched == 4


@respx.mock
def test_run_is_recorded(profile: WikiProfile) -> None:
    respx.get(API).mock(side_effect=FakeWiki([Page(1, 0, "Iron plate")]))
    store = InMemoryHarvestStore()
    stats = make_crawler(profile, store).run()

    assert len(store.runs) == 1
    assert store.runs[0]["wiki"] == "factorio"
    assert store.runs[0]["stats"] == stats
    assert store.runs[0]["error"] is None


@respx.mock
def test_failure_closes_the_run_with_an_error_and_propagates(profile: WikiProfile) -> None:
    respx.get(API).mock(return_value=httpx.Response(200, json={"query": {"languages": []}}))
    store = InMemoryHarvestStore()
    crawler = make_crawler(profile, store)
    respx.get(API).mock(
        return_value=httpx.Response(200, json={"error": {"code": "badvalue", "info": "nope"}})
    )

    with pytest.raises(Exception, match="badvalue"):
        crawler.run()

    assert store.runs[0]["error"] is not None
    assert "badvalue" in str(store.runs[0]["error"])


@respx.mock
def test_progress_is_reported_per_namespace(profile: WikiProfile) -> None:
    respx.get(API).mock(side_effect=FakeWiki([Page(1, 0, "Iron plate")]))
    messages: list[str] = []
    client = MediaWikiClient(API, "ua", rps=1000.0, sleep=lambda _: None)
    Crawler(
        client,
        profile,
        InMemoryHarvestStore(),
        progress=messages.append,
        translations=TranslationFilter("en", frozenset({"de"})),
    ).run()

    assert any("ns 0" in m for m in messages)
    assert any("redirects" in m for m in messages)


@pytest.fixture
def mainspace_only(minimal_profile_data: dict[str, Any]) -> WikiProfile:
    """One namespace, so a response sequence is deterministic enough to assert on."""
    data = {
        **minimal_profile_data,
        "wiki": {
            **minimal_profile_data["wiki"],
            "id": "factorio",
            "api": API,
            "language_filter": {"strategy": "none"},
        },
    }
    return WikiProfile.model_validate(data)


def sequence(profile: WikiProfile, responses: list[httpx.Response]) -> InMemoryHarvestStore:
    """Run a crawl against a fixed response sequence and return the store."""
    respx.get(API).mock(side_effect=responses)
    store = InMemoryHarvestStore()
    make_crawler(profile, store).run()
    return store


@respx.mock
def test_pages_without_a_revision_are_skipped(mainspace_only: WikiProfile) -> None:
    store = sequence(
        mainspace_only,
        [
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {"pageid": 1, "ns": 0, "title": "Ghost"},
                            {"ns": 0, "title": "Nonexistent", "missing": True},
                        ]
                    }
                },
            ),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    assert store.pages == {}


@respx.mock
def test_revision_without_a_text_slot_is_skipped(mainspace_only: WikiProfile) -> None:
    store = sequence(
        mainspace_only,
        [
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "pageid": 1,
                                "ns": 0,
                                "title": "Binary",
                                "revisions": [{"revid": 5, "timestamp": "2026-01-01T00:00:00Z"}],
                            }
                        ]
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "pageid": 1,
                                "ns": 0,
                                "title": "Binary",
                                "revisions": [
                                    {
                                        "revid": 5,
                                        "timestamp": "2026-01-01T00:00:00Z",
                                        "slots": {"main": {"badcontentformat": True}},
                                    }
                                ],
                            }
                        ]
                    }
                },
            ),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    assert store.pages == {}


@respx.mock
def test_categories_are_merged_across_continuation(mainspace_only: WikiProfile) -> None:
    """MediaWiki splits long category lists over several responses for the same page."""
    identity = {"pageid": 1, "ns": 0, "title": "Iron plate"}
    revision = [
        {"revid": 7, "timestamp": "2026-01-01T00:00:00Z", "slots": {"main": {"content": "body"}}}
    ]
    store = sequence(
        mainspace_only,
        [
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                **identity,
                                "revisions": [{"revid": 7, "timestamp": "2026-01-01T00:00:00Z"}],
                            }
                        ]
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                **identity,
                                "revisions": revision,
                                "categories": [{"title": "Category:Items"}],
                            }
                        ]
                    },
                    "continue": {"clcontinue": "1|Category:Smelting"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                **identity,
                                "categories": [
                                    {"title": "Category:Smelting"},
                                    {"title": "Category:Items"},
                                ],
                            }
                        ]
                    }
                },
            ),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    page = store.pages[("factorio", 1)]
    assert page.wikitext == "body"
    assert page.categories == ("Category:Items", "Category:Smelting")


IDENTITY = {"pageid": 1, "ns": 0, "title": "Iron plate"}
STUB_REVISION = [{"revid": 7, "timestamp": "2026-01-01T00:00:00Z"}]
FULL_REVISION = [
    {"revid": 7, "timestamp": "2026-01-01T00:00:00Z", "slots": {"main": {"content": "body"}}}
]


def catalogue_of(*pages: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"query": {"pages": list(pages)}})


@respx.mock
def test_catalogue_entry_without_a_revision_id_is_skipped(mainspace_only: WikiProfile) -> None:
    store = sequence(
        mainspace_only,
        [
            catalogue_of({**IDENTITY, "revisions": [{"timestamp": "2026-01-01T00:00:00Z"}]}),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    assert store.pages == {}


@respx.mock
def test_missing_page_in_a_content_response_is_skipped(mainspace_only: WikiProfile) -> None:
    store = sequence(
        mainspace_only,
        [
            catalogue_of({**IDENTITY, "revisions": STUB_REVISION}),
            catalogue_of({"ns": 0, "title": "Iron plate", "missing": True}),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    assert store.pages == {}


@respx.mock
def test_page_whose_content_never_arrives_is_skipped(mainspace_only: WikiProfile) -> None:
    """Identity and categories came back, but no revision ever did."""
    store = sequence(
        mainspace_only,
        [
            catalogue_of({**IDENTITY, "revisions": STUB_REVISION}),
            catalogue_of({**IDENTITY, "categories": [{"title": "Category:Items"}]}),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    assert store.pages == {}


@respx.mock
def test_revision_arriving_in_a_later_chunk_is_still_used(mainspace_only: WikiProfile) -> None:
    """Categories first, text second: the body must not be lost."""
    store = sequence(
        mainspace_only,
        [
            catalogue_of({**IDENTITY, "revisions": STUB_REVISION}),
            httpx.Response(
                200,
                json={
                    "query": {"pages": [{**IDENTITY, "categories": [{"title": "Category:A"}]}]},
                    "continue": {"rvcontinue": "next"},
                },
            ),
            catalogue_of({**IDENTITY, "revisions": FULL_REVISION}),
            httpx.Response(200, json={"query": {}}),
        ],
    )
    page = store.pages[("factorio", 1)]
    assert page.wikitext == "body"
    assert page.categories == ("Category:A",)


@respx.mock
def test_malformed_redirect_entries_are_skipped(mainspace_only: WikiProfile) -> None:
    respx.get(API).mock(
        side_effect=[
            catalogue_of(),
            httpx.Response(
                200,
                json={
                    "query": {
                        "redirects": [
                            {"from": "Green circuit"},
                            {"to": "Iron plate"},
                            {"from": "Red belt", "to": "Fast transport belt"},
                        ]
                    }
                },
            ),
        ]
    )
    store = InMemoryHarvestStore()
    stats = make_crawler(mainspace_only, store).run()

    assert stats.redirects == 1
    assert ("factorio", "Red belt") in store.redirects


@respx.mock
def test_limit_caps_redirects_too(profile: WikiProfile) -> None:
    wiki = FakeWiki(
        [],
        redirects={0: [(f"Alias {n}", "Iron plate") for n in range(10)]},
    )
    respx.get(API).mock(side_effect=wiki)
    store = InMemoryHarvestStore()
    stats = make_crawler(profile, store).run(limit=3)

    assert stats.redirects == 3
