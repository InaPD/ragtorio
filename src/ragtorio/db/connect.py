"""Opening a connection and applying the schema.

The schema lives in ``schema.sql`` rather than in Python so it can be read, diffed and
applied with ``psql`` when something goes wrong at 2am.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(dsn: str) -> psycopg.Connection[tuple[object, ...]]:
    """Open a connection. The caller owns it and should use it as a context manager."""
    return psycopg.connect(dsn)


def apply_schema(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    """Create every table and index that does not exist yet. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
