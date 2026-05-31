"""Opening a driver and applying the graph schema.

Mirrors ``db/connect.py``'s split between connection and schema: the schema lives in
its own ``.cypher`` file so it can be read and applied outside Python, and applying it
is idempotent (``IF NOT EXISTS`` throughout).
"""

from __future__ import annotations

from pathlib import Path

from neo4j import Driver, GraphDatabase

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "ontology" / "schema.cypher"


def connect(uri: str, user: str, password: str) -> Driver:
    """Open a driver. The caller owns it and should use it as a context manager."""
    return GraphDatabase.driver(uri, auth=(user, password))


def apply_schema(driver: Driver) -> None:
    """Create every constraint and index that does not exist yet. Idempotent."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in _statements(text):
        with driver.session() as session:
            session.run(statement)


def _statements(text: str) -> list[str]:
    """Split a ``.cypher`` file into individual statements.

    The driver runs one statement per call, unlike ``psql``, so ``//`` comment lines
    are stripped before splitting on ``;`` rather than left for the server to skip.
    """
    without_comments = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("//")
    )
    return [s.strip() for s in without_comments.split(";") if s.strip()]
