"""Article wikitext to retrievable passages.

Sections, not sliding windows. A wiki article is already segmented by its author into
units that answer one question each, and a `== heading ==` is a far better boundary
than any fixed stride: it never cuts a sentence, it comes with a title worth showing to
the answering model, and it keeps the two halves of a "first do X, then do Y" apart
only when the author meant them apart. Sections longer than the budget are packed
greedily from whole paragraphs, so the fallback degrades to something still readable.

**Three decisions worth naming:**

- **Tables are dropped from the text but not from the mentions.** A wikitext table is
  the graph's territory - the recipe table on ``Oil processing`` is exactly what
  Phase 2 already extracted, in a form that survives being queried. Flattened into
  prose it is noise that would dilute the section's embedding. Its ``{{Icon}}`` calls
  still say, correctly, that this section is about crude oil, so mentions are collected
  before the table is removed.
- **Mentions come from templates as much as from links.** This wiki writes
  ``{{Icon|Crude oil|100}}`` far more often than it links crude oil, so a links-only
  reading of "what does this section mention" would miss most of it. Which templates
  name an entity is profile configuration, not a constant here.
- **A mention must resolve to something the graph knows.** ``{{icon|time|5}}`` names no
  entity; neither does a red link. Filtering against the set of titles that actually
  produced facts is what keeps ``mentioned_entity_ids`` joinable to node ids instead of
  being a second, subtly different vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence

import mwparserfromhell

from ragtorio.config import IndexConfig
from ragtorio.harvest.models import RawPage
from ragtorio.index.models import Chunk
from ragtorio.ontology.canonical import TitleCanonicalizer

#: A section heading: two or more ``=`` of matching length around a title.
_HEADING = re.compile(r"^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.MULTILINE)

#: A wikitext table. Non-greedy, so consecutive tables are removed one at a time;
#: nested tables (rare, and never in article prose here) leave their closing ``|}``.
_TABLE = re.compile(r"^\{\|.*?^\|\}[ \t]*$", re.DOTALL | re.MULTILINE)

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF = re.compile(r"<ref[^>]*?/>|<ref.*?</ref>", re.DOTALL | re.IGNORECASE)
_GALLERY = re.compile(r"<gallery.*?</gallery>", re.DOTALL | re.IGNORECASE)

#: ``[[Target]]`` or ``[[Target#Anchor|shown text]]``.
_WIKILINK = re.compile(r"\[\[\s*([^\[\]|]+?)\s*(?:\|[^\[\]]*)?\]\]")

#: Leading list, indent and definition markers, which carry no meaning once the
#: surrounding markup is gone.
_LIST_MARKER = re.compile(r"^[*#:;]+\s*")

_BLANK_RUN = re.compile(r"\n{3,}")

#: A link target naming a file, a category or a maintenance page rather than an article.
#: ``Category:`` is assumed English, as it already is in ``ontology/repository.py``.
#: A leading colon is how MediaWiki writes an interwiki or category *link* rather than
#: a transclusion or a membership - ``[[:Wikipedia:Uranium-235]]`` names a page on
#: another wiki entirely, and naming it here would invent an entity id for it.
_NON_ARTICLE_PREFIX = re.compile(r"^:|^(?:file|image|media|category|template|help|special):", re.I)

#: Sentence end used only when a single paragraph exceeds the whole budget, which on
#: this wiki means a wall-of-text section with no blank lines.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: Average characters per token, measured across GPT-style and BERT-style tokenizers on
#: English prose. Used instead of a real tokenizer because the budget it feeds is
#: itself a soft one: being 10% out changes how a long section is split, not whether
#: the result is correct, and the alternative is a heavyweight dependency in the one
#: part of the pipeline that is otherwise pure text handling.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Roughly how many tokens a passage costs. Deliberately an estimate; see above."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


class Chunker:
    """Splits one wiki's articles into chunks, according to its profile.

    ``known_entity_ids`` is the set of node ids the graph actually has. Mentions are
    filtered against it, so an id on a chunk is always an id a Cypher query can be
    run with. Passing ``None`` keeps every resolvable mention, which is what the
    chunker's own tests want and what a run before ``ragtorio graph load`` gets.
    """

    def __init__(
        self,
        wiki: str,
        config: IndexConfig,
        titles: TitleCanonicalizer | None = None,
        known_entity_ids: frozenset[str] | None = None,
    ) -> None:
        self._wiki = wiki
        self._config = config
        self._titles = titles or TitleCanonicalizer()
        self._known = known_entity_ids

    def chunk_page(self, page: RawPage) -> list[Chunk]:
        """Every chunk one article yields, in document order."""
        chunks: list[Chunk] = []
        for path, body in _sections(_strip_noise(page.wikitext)):
            if self._is_dropped(path):
                continue
            cleaned = _clean_text(body)
            # The floor applies to the section, not to each piece of one. A section
            # that is a single "See below." line is navigation; the short tail of a
            # section that was long enough to split is still that section's prose,
            # and dropping it would silently lose text the author wrote.
            if len(cleaned) < self._config.min_chars:
                continue
            mentions = self._mentions(body)
            for text in _split_to_budget(cleaned, self._config.max_tokens):
                chunks.append(
                    Chunk(
                        chunk_id=f"{page.wiki}:{page.page_id}:{len(chunks):04d}",
                        wiki=page.wiki,
                        page_id=page.page_id,
                        title=page.title,
                        revision_id=page.revision_id,
                        section_path=path,
                        text=text,
                        mentioned_entity_ids=mentions,
                    )
                )
        return chunks

    def chunk_pages(self, pages: Iterable[RawPage]) -> list[Chunk]:
        """Every chunk a set of articles yields."""
        return [chunk for page in pages for chunk in self.chunk_page(page)]

    def _is_dropped(self, path: tuple[str, ...]) -> bool:
        """Whether a section, or any section it sits under, is configured away.

        Checking the whole path matters: "See also" subsections are dropped with their
        parent, and nothing under a dropped heading was ever prose worth retrieving.
        """
        dropped = self._config.dropped_section_names
        return any(heading.casefold() in dropped for heading in path)

    def _mentions(self, wikitext: str) -> tuple[str, ...]:
        """Entity ids named by this section's links and mention templates.

        Order is preserved and duplicates removed, so the first thing a section names
        stays first - which makes the column readable when debugging a bad retrieval.
        """
        seen: dict[str, None] = {}
        for target in _link_targets(wikitext, self._config.mention_template_names):
            if _NON_ARTICLE_PREFIX.match(target):
                continue
            entity_id = f"{self._wiki}:{self._titles.resolve_link(target)}"
            if self._known is not None and entity_id not in self._known:
                continue
            seen.setdefault(entity_id, None)
        return tuple(seen)


def _strip_noise(wikitext: str) -> str:
    """Remove markup that is never prose: comments, footnotes and galleries."""
    without_comments = _COMMENT.sub("", wikitext)
    without_refs = _REF.sub("", without_comments)
    return _GALLERY.sub("", without_refs)


def _sections(wikitext: str) -> Iterator[tuple[tuple[str, ...], str]]:
    """Walk an article as ``(heading path, body wikitext)`` pairs, in document order.

    The lead section comes first with an empty path. A heading's level decides where
    it sits in the path, so ``=== Tips ===`` under ``== Setting up ==`` yields
    ``("Setting up", "Tips")`` and a following ``== Transporting ==`` correctly pops
    both.
    """
    matches = list(_HEADING.finditer(wikitext))
    lead = wikitext[: matches[0].start()] if matches else wikitext
    if lead.strip():
        yield (), lead

    path: list[str] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = _clean_heading(match.group(2))
        del path[level - 2 :]
        path.append(heading)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        body = wikitext[match.end() : end]
        if body.strip():
            yield tuple(path), body


def _clean_heading(raw: str) -> str:
    """A heading as a human reads it: no markup, no ``{{SA}}`` expansion markers."""
    return _clean_text(raw).strip() or raw.strip()


def _clean_text(wikitext: str) -> str:
    """Wikitext to plain prose: no tables, no templates, links reduced to their text."""
    without_tables = _TABLE.sub("", wikitext)
    stripped = mwparserfromhell.parse(without_tables).strip_code(normalize=True, collapse=True)
    lines = [_LIST_MARKER.sub("", line).strip() for line in str(stripped).splitlines()]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def _link_targets(wikitext: str, mention_templates: frozenset[str]) -> Iterator[str]:
    """Every title this wikitext names, whether by link or by mention template."""
    for match in _WIKILINK.finditer(wikitext):
        yield match.group(1)
    if not mention_templates:
        return
    for template in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
        if str(template.name).strip().casefold() not in mention_templates:
            continue
        if template.params and not template.params[0].showkey:
            yield str(template.params[0].value).strip()


def _split_to_budget(text: str, max_tokens: int) -> list[str]:
    """One section's text as passages, each within the token budget.

    Paragraphs are packed greedily rather than split evenly: an even split would move
    every boundary in a section just because its last paragraph was long, and a
    boundary that follows the author's own is worth more than one that balances.
    """
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]
    return _pack(_paragraphs(text, max_tokens), max_tokens, separator="\n\n")


def _paragraphs(text: str, max_tokens: int) -> list[str]:
    """The text's paragraphs, with any single paragraph over budget split by sentence."""
    out: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if estimate_tokens(paragraph) <= max_tokens:
            out.append(paragraph)
        else:
            out.extend(_pack(_SENTENCE.split(paragraph), max_tokens, separator=" "))
    return out


def _pack(parts: Sequence[str], max_tokens: int, separator: str) -> list[str]:
    """Greedily join parts into groups that each fit the budget.

    A part that is over budget on its own becomes its own group rather than being cut:
    truncating mid-sentence to hit a soft limit costs more than the limit saves.
    """
    packed: list[str] = []
    current: list[str] = []
    size = 0
    for part in parts:
        cost = estimate_tokens(part)
        if current and size + cost > max_tokens:
            packed.append(separator.join(current))
            current, size = [], 0
        current.append(part)
        size += cost
    if current:
        packed.append(separator.join(current))
    return packed
