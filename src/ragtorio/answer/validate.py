"""Checking that an answer cites only what was actually retrieved.

This is the phase's whole point. A model handed labeled evidence and asked to cite it
will, occasionally, cite something that looks exactly like a chunk id and was never in
the prompt - a page it remembers, an id one digit off, a plausible section of an
article that does exist. Nothing downstream can tell that apart from a real citation
by reading it, which is why it is checked mechanically against the retrieved set
rather than judged.

**A failure names the sentences, not just the ids.** An unknown id is regenerated once
with the offending claims quoted back, because a regeneration prompt that only says
"a citation was wrong" gets an answer with the citations quietly removed - which is
the same text, now unfalsifiable. Told which claims, the model fixes those.

**A second failure is recorded, not hidden.** The claims that still cite nothing
retrievable come back on the answer as ``unverified``. An answer that admits which of
its sentences are unsourced is worth more than one that silently drops them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from urllib.parse import quote

from ragtorio.answer.models import GRAPH_CITATION, Citation, CitationCheck
from ragtorio.index.models import Chunk
from ragtorio.retrieve.models import RetrievedContext

#: Anything in square brackets that is not obviously prose. Deliberately loose: the
#: point is to catch a fabricated citation, and one that does not match the chunk-id
#: grammar is still a fabricated citation. Brackets around a real phrase ("[sic]") are
#: filtered out by the length and whitespace rules below, not by the pattern.
_BRACKETED = re.compile(r"\[([^\[\]\n]{1,200})\]")

#: A chunk id is ``{wiki}:{page_id}:{sequence}``. Used to tell a citation attempt from
#: an ordinary bracketed aside, not to decide whether the id is real.
_ID_SHAPE = re.compile(r"^[A-Za-z0-9_.\-]+:\d+:\d+$")

#: Sentence split good enough to quote a claim back. Not a tokenizer: it only has to
#: cut at somewhere a reader would agree is a sentence boundary.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def validate(text: str, context: RetrievedContext, index_url: str) -> CitationCheck:
    """Check every citation in ``text`` against what ``context`` actually retrieved."""
    by_id = {chunk.chunk_id: chunk for chunk in context.passages}
    graph_available = context.graph is not None and not context.graph.is_empty

    cited: list[str] = []
    unknown: list[str] = []
    cited_graph = False

    for token in _citation_tokens(text):
        if token == GRAPH_CITATION:
            # Citing the graph when no graph facts were retrieved is as fabricated as
            # citing a chunk id that was never in the prompt, and is treated as such.
            if graph_available:
                cited_graph = True
            else:
                unknown.append(token)
        elif token in by_id:
            if token not in cited:
                cited.append(token)
        elif _ID_SHAPE.match(token):
            unknown.append(token)

    unique_unknown = tuple(dict.fromkeys(unknown))
    return CitationCheck(
        citations=tuple(_citation(by_id[chunk_id], index_url) for chunk_id in cited),
        cited_graph=cited_graph,
        unknown_ids=unique_unknown,
        offending_claims=claims_citing(text, unique_unknown),
    )


def _citation_tokens(text: str) -> list[str]:
    """Every citation attempt in the text, in order.

    One bracket may hold several ids (``[a, b]``): the prompt asks for one per bracket,
    and an answer that comma-separates them is making a citation either way.
    """
    tokens: list[str] = []
    for match in _BRACKETED.finditer(text):
        for part in match.group(1).split(","):
            token = part.strip()
            if token and " " not in token:
                tokens.append(token)
    return tokens


def claims_citing(text: str, ids: Sequence[str]) -> tuple[str, ...]:
    """The sentences that cite one of ``ids``, to quote back in a regeneration."""
    if not ids:
        return ()
    wanted = set(ids)
    claims: list[str] = []
    for sentence in _sentences(text):
        if wanted & set(_citation_tokens(sentence)) and sentence not in claims:
            claims.append(sentence)
    return tuple(claims)


def _sentences(text: str) -> list[str]:
    """Sentences, treating each line as its own boundary so a bulleted list of facts
    does not come back as one enormous claim."""
    sentences: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            sentences.extend(part.strip() for part in _SENTENCE.split(stripped) if part.strip())
    return sentences


def _citation(chunk: Chunk, index_url: str) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        section=" > ".join(chunk.section_path),
        revision_id=chunk.revision_id,
        url=citation_url(index_url, chunk.title, chunk.revision_id),
    )


def citation_url(index_url: str, title: str, revision_id: int) -> str:
    """The revision-pinned URL for a page, which is what a citation links to.

    Pinned rather than current on purpose: a wiki answer that links to the live page
    cannot be checked later, because the page has moved on and there is no way to tell
    a wrong answer from a changed game. ``oldid`` is the revision the chunk was
    embedded from.
    """
    return f"{index_url}?title={quote(title.replace(' ', '_'), safe='/:()')}&oldid={revision_id}"


def regeneration_note(check: CitationCheck) -> str:
    """The one corrective message a failed answer gets, naming what went wrong."""
    ids = ", ".join(f"[{unknown}]" for unknown in check.unknown_ids)
    claims = "\n".join(f"- {claim}" for claim in check.offending_claims)
    body = (
        f"Your previous answer cited sources that were not in the context: {ids}. "
        "Those ids do not exist. Rewrite the answer using only the ids listed above, "
        "and drop any statement you cannot support from the evidence you were given - "
        "do not keep the statement and remove its citation."
    )
    return f"{body}\n\nThe statements that cited them:\n{claims}" if claims else body


def unverified_claims(check: CitationCheck) -> tuple[str, ...]:
    """What is still unsourced after the last attempt, for the answer to admit to."""
    return check.offending_claims or tuple(f"[{unknown}]" for unknown in check.unknown_ids)


def index_url_for(api_url: str) -> str:
    """A wiki's article endpoint, derived from the ``api.php`` URL in its profile.

    Every MediaWiki install serves both from the same directory, so this is a rename
    rather than a guess - and it keeps the profile from carrying a second URL that can
    drift out of step with the first.
    """
    return re.sub(r"api\.php$", "index.php", api_url)


def format_citations(citations: Iterable[Citation]) -> list[str]:
    """Citations as lines a reader can follow. Used by the CLI and the examples run."""
    lines = []
    for citation in citations:
        where = f"{citation.title} > {citation.section}" if citation.section else citation.title
        lines.append(f"[{citation.chunk_id}] {where} - {citation.url}")
    return lines
