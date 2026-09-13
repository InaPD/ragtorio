"""Dropping translated subpages before they multiply the crawl twentyfold.

The Factorio wiki carries roughly twenty translations of most articles as subpages
(``Iron plate/de``). Crawling them costs twenty times the requests and adds nothing:
the graph comes from infobox parameters, which are language independent, and the prose
index is English.

Deciding from the title shape alone is not safe. ``Rail/de`` is German, but a page like
``Achievements/list`` is not a translation and neither is ``Blueprint/tips``. So the
shape test is intersected with the language codes the wiki itself declares through
``siteinfo``, which makes the filter correct on any wiki rather than tuned to this one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragtorio.config import WikiProfile
from ragtorio.harvest.client import MediaWikiClient, MediaWikiError

#: A trailing ``/de``, ``/zh``, ``/pt-br`` marks a translated subpage.
LANGUAGE_SUFFIX = re.compile(r"/([a-z]{2,3}(?:-[a-z]{2,4})?)$")


@dataclass(frozen=True)
class TranslationFilter:
    """Decides whether a title is a translation of another page.

    ``keep`` of ``None`` means the wiki has no translated subpages, so nothing is
    filtered. An empty ``codes`` means siteinfo did not answer; the filter then falls
    back to the shape test alone, which over-filters slightly rather than silently
    letting twenty languages into the corpus.
    """

    keep: str | None
    codes: frozenset[str] = frozenset()

    def is_translation(self, title: str) -> bool:
        if self.keep is None:
            return False
        match = LANGUAGE_SUFFIX.search(title)
        if match is None:
            return False
        code = match.group(1)
        if code == self.keep:
            return False
        return code in self.codes if self.codes else True


def fetch_language_codes(client: MediaWikiClient) -> frozenset[str]:
    """Every language code the wiki knows. Empty if the wiki will not say."""
    try:
        result = client.query(meta="siteinfo", siprop="languages")
    except (MediaWikiError, RuntimeError):
        return frozenset()
    languages = result.get("languages", [])
    return frozenset(
        str(entry["code"]) for entry in languages if isinstance(entry, dict) and "code" in entry
    )


def for_profile(profile: WikiProfile, client: MediaWikiClient) -> TranslationFilter:
    """Build the filter a profile asks for, consulting the wiki when needed."""
    language_filter = profile.wiki.language_filter
    if language_filter.strategy == "none":
        return TranslationFilter(keep=None)
    return TranslationFilter(keep=language_filter.keep, codes=fetch_language_codes(client))
