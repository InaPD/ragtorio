"""What chunking and search pass around.

A :class:`Chunk` carries its provenance rather than a foreign key to it. The citation
validator in Phase 6 has to turn a chunk id back into
``index.php?title=X&oldid=N`` without a second query, and a retrieved chunk that has
been merged with graph facts has long since lost any connection it had to the row it
came from.

The embedding is a separate model rather than a nullable field on ``Chunk`` because
the two have different lifetimes: chunking is deterministic and offline, embedding
costs money or a GPU. Keeping them apart is what lets the chunker be tested against
committed wikitext with no model present at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """One retrievable passage of article prose.

    ``section_path`` is the heading trail that leads to the text (``("Setting up oil
    processing", "Tips")``), empty for a page's lead section. It is shown to the
    answering model, which needs to know that a passage came from a section titled
    "Known issues" and not from the article's own definition of the thing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    wiki: str
    page_id: int
    title: str
    revision_id: int
    section_path: tuple[str, ...] = ()
    text: str
    mentioned_entity_ids: tuple[str, ...] = ()

    @property
    def heading(self) -> str:
        """A one-line label: the page title, plus the section trail when there is one."""
        return " > ".join((self.title, *self.section_path))


class EmbeddedChunk(BaseModel):
    """A chunk and its vector, ready to store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    embedding: tuple[float, ...]


class ChunkMatch(BaseModel):
    """One search result. ``score`` is cosine similarity, so 1.0 is identical."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    score: float
