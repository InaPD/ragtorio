"""The pgvector implementation of :class:`~ragtorio.index.store.ChunkStore`.

Vectors cross the boundary as pgvector's own text form, ``'[0.1,0.2,...]'``, with an
explicit ``::vector`` cast, rather than through the ``pgvector`` Python package's type
adapters. One dependency fewer for a format that is three lines to write and cannot
drift: the alternative buys automatic round-tripping of a type this module reads back
exactly once, and never as a vector.

Distance is cosine (``<=>``) throughout, matching the HNSW index's ``vector_cosine_ops``
and the normalised vectors every provider hands over. Score is reported as similarity
(``1 - distance``) because a number that goes up when a result is better is worth the
one subtraction.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg
from psycopg import sql

from ragtorio.index.models import Chunk, ChunkMatch, EmbeddedChunk

_DELETE = "DELETE FROM chunk WHERE wiki = %s"

_INSERT = """
INSERT INTO chunk (
    chunk_id, wiki, page_id, title, revision_id, section_path, text,
    embedding, mentioned_entity_ids
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
"""

_SELECT = """
SELECT chunk_id, page_id, title, revision_id, section_path, text, mentioned_entity_ids,
       1 - (embedding <=> %(query)s::vector) AS score
  FROM chunk
 WHERE wiki = %(wiki)s
   AND (%(entities)s::text[] IS NULL OR mentioned_entity_ids && %(entities)s::text[])
 ORDER BY embedding <=> %(query)s::vector
 LIMIT %(limit)s
"""

_COLUMN_DIMENSION = """
SELECT a.atttypmod
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
 WHERE c.relname = 'chunk' AND a.attname = 'embedding' AND a.attnum > 0
"""

type Conn = psycopg.Connection[tuple[object, ...]]


class PostgresChunkStore:
    """Persists and searches chunks. The caller owns the connection and its lifetime."""

    def __init__(self, conn: Conn) -> None:
        self._conn = conn

    def replace_all(self, wiki: str, chunks: Sequence[EmbeddedChunk]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_DELETE, (wiki,))
            if chunks:
                cur.executemany(_INSERT, [_row(wiki, stored) for stored in chunks])
        self._conn.commit()

    def count(self, wiki: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunk WHERE wiki = %s", (wiki,))
            row = cur.fetchone()
        assert row is not None  # pragma: no cover - count() always returns one row
        value = row[0]
        if not isinstance(value, int):
            raise TypeError(f"expected an integer column, got {type(value).__name__}")
        return value

    def titles(self, wiki: str) -> frozenset[str]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT title FROM chunk WHERE wiki = %s", (wiki,))
            return frozenset(str(row[0]) for row in cur.fetchall())

    def search(
        self,
        wiki: str,
        embedding: Sequence[float],
        limit: int = 10,
        entity_ids: Sequence[str] = (),
        ef_search: int | None = None,
    ) -> list[ChunkMatch]:
        with self._conn.cursor() as cur:
            if ef_search is not None:
                # SET LOCAL, so the setting dies with the transaction and one tuned
                # query never silently changes the next caller's recall. SET takes no
                # bind parameters, so the value is composed with psycopg's own quoting
                # rather than interpolated by hand.
                cur.execute(
                    sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(int(ef_search)))
                )
            cur.execute(
                _SELECT,
                {
                    "wiki": wiki,
                    "query": _vector_literal(embedding),
                    "entities": list(entity_ids) or None,
                    "limit": limit,
                },
            )
            rows = cur.fetchall()
        self._conn.rollback()  # end the transaction SET LOCAL opened; nothing was written
        return [_match(wiki, row) for row in rows]


def ensure_embedding_dimension(conn: Conn, dimensions: int) -> bool:
    """Make ``chunk.embedding`` the given width, returning whether it had to change.

    The column is declared ``vector(768)`` for the default local model, but vector
    width is a property of the provider and not of the schema: Voyage's models do not
    offer 768 at all. Rather than making a provider switch a hand-written migration,
    the build checks the column first and widens or narrows it - refusing when rows
    exist, because a stored vector cannot be reinterpreted at another width and
    silently dropping someone's index is not this function's decision to make.
    """
    current = _column_dimension(conn)
    if current == dimensions:
        return False
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunk")
        row = cur.fetchone()
        stored = _as_int(row[0]) if row else 0
        if stored:
            raise RuntimeError(
                f"chunk.embedding is vector({current}) but this provider emits "
                f"{dimensions} dimensions, and {stored:,} chunks are already stored. "
                "Re-index every wiki from scratch: TRUNCATE chunk, then run "
                "`ragtorio index build` for each."
            )
        cur.execute("DROP INDEX IF EXISTS chunk_embedding_idx")
        cur.execute(f"ALTER TABLE chunk ALTER COLUMN embedding TYPE vector({int(dimensions)})")
        cur.execute(
            "CREATE INDEX chunk_embedding_idx ON chunk USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()
    return True


def _column_dimension(conn: Conn) -> int | None:
    """The declared width of ``chunk.embedding``, or ``None`` if it is unconstrained."""
    with conn.cursor() as cur:
        cur.execute(_COLUMN_DIMENSION)
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("no chunk.embedding column; run `ragtorio init-db` first")
    typmod = _as_int(row[0])
    return typmod if typmod > 0 else None


def _as_int(value: object) -> int:
    """Narrow a column psycopg hands back untyped, rather than assuming."""
    if not isinstance(value, int):
        raise TypeError(f"expected an integer column, got {type(value).__name__}")
    return value


def _row(wiki: str, stored: EmbeddedChunk) -> tuple[object, ...]:
    chunk = stored.chunk
    return (
        chunk.chunk_id,
        wiki,
        chunk.page_id,
        chunk.title,
        chunk.revision_id,
        list(chunk.section_path),
        chunk.text,
        _vector_literal(stored.embedding),
        list(chunk.mentioned_entity_ids),
    )


def _match(wiki: str, row: tuple[object, ...]) -> ChunkMatch:
    chunk_id, page_id, title, revision_id, section_path, text, mentions, score = row
    return ChunkMatch(
        chunk=Chunk.model_validate(
            {
                "chunk_id": chunk_id,
                "wiki": wiki,
                "page_id": page_id,
                "title": title,
                "revision_id": revision_id,
                "section_path": tuple(section_path or ()),  # type: ignore[arg-type]
                "text": text,
                "mentioned_entity_ids": tuple(mentions or ()),  # type: ignore[arg-type]
            }
        ),
        score=float(score),  # type: ignore[arg-type]
    )


def _vector_literal(embedding: Sequence[float]) -> str:
    """pgvector's input format: a bracketed, comma-separated list."""
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"
