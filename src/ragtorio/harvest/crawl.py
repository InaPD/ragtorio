"""The crawl: enumerate a wiki's pages, fetch what changed, store it verbatim.

Two passes, for one reason. Listing pages without their text is cheap (500 per request)
and tells us every page's current revision id; fetching text is expensive (50 per
request, and the text is the bulk of the payload). Comparing revision ids between the
passes means a re-crawl of an unchanged wiki issues a handful of listing requests and
fetches no content at all, which is what makes re-running this safe and quick.

Redirects are collected separately because ``redirects=1`` makes the API resolve them
and report the ``from``/``to`` pairs directly, saving us from parsing ``#REDIRECT``
out of wikitext.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from typing import Any

from ragtorio.config import WikiProfile
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.language import TranslationFilter
from ragtorio.harvest.language import for_profile as translation_filter_for
from ragtorio.harvest.models import CrawlStats, PageStub, RawPage, RawRedirect
from ragtorio.harvest.store import HarvestStore

#: MediaWiki refuses content for more than 50 pages in one query.
CONTENT_BATCH = 50

#: Listing without content is cheap, so ask for the largest page the API will give.
LIST_LIMIT = "max"


def _noop(message: str) -> None:
    """Default progress sink."""


class Crawler:
    """Crawls one wiki according to its profile.

    The client, store and progress sink are injected, so the whole crawl runs in tests
    against mocked HTTP and an in-memory store.
    """

    def __init__(
        self,
        client: MediaWikiClient,
        profile: WikiProfile,
        store: HarvestStore,
        *,
        batch_size: int = CONTENT_BATCH,
        progress: Callable[[str], None] = _noop,
        translations: TranslationFilter | None = None,
    ) -> None:
        self._client = client
        self._profile = profile
        self._store = store
        self._batch_size = batch_size
        self._progress = progress
        self._translations = translations or translation_filter_for(profile, client)

    @property
    def wiki(self) -> str:
        return self._profile.wiki.id

    def run(self, *, limit: int | None = None) -> CrawlStats:
        """Crawl every namespace the profile asks for, then redirects.

        ``limit`` caps the pages kept per namespace; it exists so a smoke run costs
        seconds instead of forty minutes.
        """
        run_id = self._store.start_run(self.wiki)
        stats = CrawlStats()
        known = self._store.known_revisions(self.wiki)
        try:
            for namespace in self._profile.crawl_namespaces:
                stats = stats.merge(self._crawl_namespace(namespace, known, limit))
            stats = stats.merge(CrawlStats(redirects=self._crawl_redirects(limit)))
        except Exception as exc:
            self._store.finish_run(run_id, stats, error=f"{type(exc).__name__}: {exc}")
            raise
        self._store.finish_run(run_id, stats)
        return stats

    def _crawl_namespace(
        self, namespace: int, known: dict[int, int], limit: int | None
    ) -> CrawlStats:
        listed = kept = translations = unchanged = fetched = 0
        pending: list[PageStub] = []

        for stub in self._catalogue(namespace):
            listed += 1
            if self._translations.is_translation(stub.title):
                translations += 1
                continue
            kept += 1
            if known.get(stub.page_id) == stub.revision_id:
                unchanged += 1
            else:
                pending.append(stub)
                if len(pending) >= self._batch_size:
                    fetched += self._fetch_and_save(pending)
                    pending = []
            if limit is not None and kept >= limit:
                break

        if pending:
            fetched += self._fetch_and_save(pending)

        self._progress(
            f"ns {namespace}: listed {listed:,}, kept {kept:,} "
            f"(dropped {translations:,} translations), "
            f"fetched {fetched:,}, unchanged {unchanged:,}"
        )
        return CrawlStats(
            listed=listed,
            translations=translations,
            unchanged=unchanged,
            fetched=fetched,
            by_namespace={namespace: kept},
        )

    def _catalogue(self, namespace: int) -> Iterator[PageStub]:
        """Every non-redirect page in a namespace, with its current revision id."""
        pages = self._client.query_paged(
            generator="allpages",
            gapnamespace=namespace,
            gapfilterredir="nonredirects",
            gaplimit=LIST_LIMIT,
            prop="revisions",
            rvprop="ids|timestamp",
        )
        for chunk in pages:
            for page in chunk.get("pages", []):
                stub = _to_stub(page)
                if stub is not None:
                    yield stub

    def _fetch_and_save(self, stubs: Sequence[PageStub]) -> int:
        pages = self._fetch(stubs)
        self._store.save_pages(pages)
        return len(pages)

    def _fetch(self, stubs: Sequence[PageStub]) -> list[RawPage]:
        """Fetch wikitext and categories for one batch of titles."""
        merged: dict[int, dict[str, Any]] = {}
        chunks = self._client.query_paged(
            titles="|".join(stub.title for stub in stubs),
            prop="revisions|categories",
            rvprop="content|ids|timestamp",
            rvslots="main",
            cllimit=LIST_LIMIT,
        )
        for chunk in chunks:
            for page in chunk.get("pages", []):
                if page.get("missing") or "pageid" not in page:
                    continue
                entry = merged.setdefault(page["pageid"], {"page": page, "categories": []})
                # Continuation pages repeat the identity but carry only further
                # categories, so keep the first body that actually had a revision.
                if page.get("revisions") and not entry["page"].get("revisions"):
                    entry["page"] = page
                entry["categories"].extend(
                    category["title"]
                    for category in page.get("categories", [])
                    if "title" in category
                )

        return [
            raw
            for entry in merged.values()
            if (raw := _to_raw_page(self.wiki, entry["page"], entry["categories"])) is not None
        ]

    def _crawl_redirects(self, limit: int | None) -> int:
        """Collect redirect pairs, which Phase 3 turns into node aliases."""
        total = 0
        for namespace in self._profile.crawl_namespaces:
            # islice stops mid-response, not merely at the next one: one listing
            # holds up to 500 redirects, so --limit has to cut inside a chunk.
            found = list(islice(self._redirect_pairs(namespace), limit))
            self._store.save_redirects(found)
            total += len(found)
        self._progress(f"redirects: {total:,}")
        return total

    def _redirect_pairs(self, namespace: int) -> Iterator[RawRedirect]:
        """Every redirect in a namespace whose two ends are both real English pages.

        Walked from the *target* side: ``prop=redirects`` hangs the pages pointing at
        each page off that page, which beats both parsing ``#REDIRECT [[...]]`` out of
        wikitext and ``list=allredirects``, whose entries name the target by title but
        the source only by page id. ``redirects=1``, which would resolve the pairs
        directly, is refused outright by the API alongside an ``allpages`` generator.

        Both ends are filtered: a redirect *from* a translated subpage is noise, and a
        redirect *to* one points at a page this crawl never stored.
        """
        chunks = self._client.query_paged(
            generator="allpages",
            gapnamespace=namespace,
            gapfilterredir="nonredirects",
            gaplimit=LIST_LIMIT,
            prop="redirects",
            rdnamespace=namespace,
            rdlimit="max",
        )
        for chunk in chunks:
            for page in chunk.get("pages", []):
                target = page.get("title")
                if not target or self._translations.is_translation(target):
                    continue
                for item in page.get("redirects", []):
                    source = item.get("title")
                    if not source or self._translations.is_translation(source):
                        continue
                    yield RawRedirect(wiki=self.wiki, from_title=source, to_title=target)


def _to_stub(page: dict[str, Any]) -> PageStub | None:
    """A catalogue entry, or ``None`` if the API gave us nothing usable."""
    revisions = page.get("revisions") or []
    if page.get("missing") or not revisions or "pageid" not in page:
        return None
    revision = revisions[0]
    if "revid" not in revision or "timestamp" not in revision:
        return None
    return PageStub(
        page_id=page["pageid"],
        ns=page["ns"],
        title=page["title"],
        revision_id=revision["revid"],
        revised_at=revision["timestamp"],
    )


def _to_raw_page(wiki: str, page: dict[str, Any], categories: list[str]) -> RawPage | None:
    """Build a storable page, or ``None`` when the revision carries no text slot.

    Category names keep their namespace prefix (``Category:Archived``) because the
    prefix is localised and throwing it away here would lose information that only the
    wiki can supply.
    """
    revisions = page.get("revisions") or []
    if not revisions:
        return None
    revision = revisions[0]
    content = revision.get("slots", {}).get("main", {}).get("content")
    if content is None:
        return None
    return RawPage(
        wiki=wiki,
        page_id=page["pageid"],
        ns=page["ns"],
        title=page["title"],
        revision_id=revision["revid"],
        revised_at=revision["timestamp"],
        wikitext=content,
        categories=tuple(dict.fromkeys(categories)),
    )
