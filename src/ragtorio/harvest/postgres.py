"""The Postgres implementation of :class:`~ragtorio.harvest.store.HarvestStore`.

Writes are upserts keyed on ``(wiki, page_id)`` so a crawl can be interrupted and
resumed without producing duplicates or a half-written page.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg

from ragtorio.harvest.models import CrawlStats, RawPage, RawRedirect

_INSERT_RUN = """
INSERT INTO crawl_run (wiki) VALUES (%s) RETURNING run_id
"""

_FINISH_RUN = """
UPDATE crawl_run
   SET finished_at = now(), listed = %s, translations = %s,
       unchanged = %s, fetched = %s, redirects = %s, error = %s
 WHERE run_id = %s
"""

# A page that was moved keeps its page_id but changes title, which would collide with
# the unique title index while the stale row is still there. Clear it first.
_CLEAR_MOVED_TITLE = """
DELETE FROM raw_page WHERE wiki = %s AND title = %s AND page_id <> %s
"""

_UPSERT_PAGE = """
INSERT INTO raw_page (wiki, page_id, ns, title, revision_id, revised_at, wikitext)
     VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (wiki, page_id) DO UPDATE
        SET ns = EXCLUDED.ns,
            title = EXCLUDED.title,
            revision_id = EXCLUDED.revision_id,
            revised_at = EXCLUDED.revised_at,
            wikitext = EXCLUDED.wikitext,
            fetched_at = now()
"""

_DELETE_CATEGORIES = """
DELETE FROM raw_category WHERE wiki = %s AND page_id = ANY(%s)
"""

_INSERT_CATEGORY = """
INSERT INTO raw_category (wiki, page_id, category) VALUES (%s, %s, %s)
ON CONFLICT DO NOTHING
"""

_UPSERT_REDIRECT = """
INSERT INTO raw_redirect (wiki, from_title, to_title) VALUES (%s, %s, %s)
ON CONFLICT (wiki, from_title) DO UPDATE SET to_title = EXCLUDED.to_title
"""


def _as_int(value: object) -> int:
    """Postgres hands back ``object`` under strict typing; assert what the column is.

    A column that is suddenly not an integer means the schema drifted, which is worth a
    loud failure rather than a silent cast.
    """
    if isinstance(value, int):
        return value
    raise TypeError(f"expected an integer column, got {type(value).__name__}")


class PostgresHarvestStore:
    """Persists a crawl. The caller owns the connection and its lifetime."""

    def __init__(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        self._conn = conn

    def start_run(self, wiki: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(_INSERT_RUN, (wiki,))
            row = cur.fetchone()
        if row is None:  # pragma: no cover - RETURNING always yields a row
            raise RuntimeError("crawl_run insert returned nothing")
        self._conn.commit()
        return _as_int(row[0])

    def finish_run(self, run_id: int, stats: CrawlStats, error: str | None = None) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                _FINISH_RUN,
                (
                    stats.listed,
                    stats.translations,
                    stats.unchanged,
                    stats.fetched,
                    stats.redirects,
                    error,
                    run_id,
                ),
            )
        self._conn.commit()

    def known_revisions(self, wiki: str) -> dict[int, int]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT page_id, revision_id FROM raw_page WHERE wiki = %s", (wiki,))
            return {_as_int(page_id): _as_int(rev) for page_id, rev in cur.fetchall()}

    def save_pages(self, pages: Sequence[RawPage]) -> None:
        if not pages:
            return
        with self._conn.cursor() as cur:
            cur.executemany(_CLEAR_MOVED_TITLE, [(p.wiki, p.title, p.page_id) for p in pages])
            cur.executemany(
                _UPSERT_PAGE,
                [
                    (p.wiki, p.page_id, p.ns, p.title, p.revision_id, p.revised_at, p.wikitext)
                    for p in pages
                ],
            )
            cur.execute(_DELETE_CATEGORIES, (pages[0].wiki, [p.page_id for p in pages]))
            rows = [(p.wiki, p.page_id, name) for p in pages for name in p.categories]
            if rows:
                cur.executemany(_INSERT_CATEGORY, rows)
        self._conn.commit()

    def save_redirects(self, redirects: Sequence[RawRedirect]) -> None:
        if not redirects:
            return
        with self._conn.cursor() as cur:
            cur.executemany(
                _UPSERT_REDIRECT, [(r.wiki, r.from_title, r.to_title) for r in redirects]
            )
        self._conn.commit()

    def namespace_counts(self, wiki: str) -> dict[int, int]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT ns, count(*) FROM raw_page WHERE wiki = %s GROUP BY ns ORDER BY ns",
                (wiki,),
            )
            return {_as_int(ns): _as_int(count) for ns, count in cur.fetchall()}
