"""Turning a title as written into the one canonical title a node id is built from.

Two consumers need exactly this and nothing else: entity resolution, which has to
decide whether a fact's object names a node it already built, and the chunker, which
has to decide which entities a passage mentions. Both start from a title someone typed
- a redirect, community shorthand, or a wikilink with MediaWiki's own spelling rules -
and both must arrive at the same answer, or a chunk's ``mentioned_entity_ids`` would
not join to the graph it is supposed to complement.

Link normalisation is deliberately separate from alias lookup. ``[[crude oil]]`` and
``[[Crude_oil]]`` are the same page because of how MediaWiki stores titles, which is a
fact about wiki syntax; "green circuit" is ``Electronic circuit`` because a human said
so. Only the second involves the alias table, so only text that really came from
wiki markup goes through :meth:`TitleCanonicalizer.normalize_link`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

from ragtorio.harvest.models import RawRedirect

_WHITESPACE = re.compile(r"[\s_]+")


class TitleCanonicalizer:
    """Resolves titles through redirects and community shorthand.

    Both mappings are folded into one lookup because a caller never cares which of
    the two answered: a redirect and a hand-written alias are the same statement
    about the same title, differing only in who wrote it down.
    """

    def __init__(
        self,
        redirects: Iterable[RawRedirect] = (),
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        sources: dict[str, str] = {r.from_title: r.to_title for r in redirects}
        sources |= dict(aliases or {})
        self._target = {source.casefold(): target for source, target in sources.items()}
        self._aliases_of: dict[str, list[str]] = defaultdict(list)
        for source, target in sources.items():
            self._aliases_of[target].append(source)

    def canonical(self, title: str) -> str:
        """The redirect or alias target for a title, or the title itself.

        An unknown title is returned unchanged rather than rejected: a reference the
        crawl never saw is still worth carrying forward so that whoever consumes it
        can log exactly what failed to resolve.
        """
        return self._target.get(title.casefold(), title)

    def normalize_link(self, target: str) -> str:
        """A wikilink or template argument as MediaWiki itself would store it.

        Underscores are spaces, runs of whitespace collapse, any section anchor is
        dropped, and the first letter is capitalised - ``[[crude oil#Uses]]`` and
        ``[[Crude_oil]]`` both name the page ``Crude oil``.
        """
        without_anchor = target.split("#", 1)[0]
        collapsed = _WHITESPACE.sub(" ", without_anchor).strip()
        return collapsed[:1].upper() + collapsed[1:]

    def resolve_link(self, target: str) -> str:
        """Normalise a title from wiki markup, then follow redirects and aliases."""
        return self.canonical(self.normalize_link(target))

    def aliases_of(self, title: str) -> list[str]:
        """Every redirect source and shorthand pointing at this title, sorted."""
        return sorted(self._aliases_of.get(title, []))
