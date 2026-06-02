"""Turning the names a router said into node ids the graph actually has.

**Every entity is resolved before any query runs.** A template parameter is a node id,
never a string a model produced, so a hallucinated entity fails here - visibly, as an
unresolved mention - instead of silently matching nothing inside a Cypher query and
returning an empty result that looks like "the wiki does not say".

Three passes, cheapest and most certain first: exact title, then the ``aliases``
property that Phase 3 filled from redirects and community shorthand, then the
full-text index as a last resort. The order matters: full-text search will happily
return *something* for any input, so it must never pre-empt an exact match, and its
result is labeled ``search`` so a caller can tell a confident match from a guess.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from neo4j import Driver, Session

from ragtorio.retrieve.models import ResolvedEntity

#: Characters Lucene treats as syntax. Stripped rather than escaped: none of them
#: carries meaning in a wiki title, and the index's own analyser splits "Uranium-235"
#: into "uranium" and "235" either way. Escaping them left bare AND/OR/NOT keywords
#: behind, which Lucene then read as operators and rejected as a malformed query -
#: a model that wraps an entity in quotes should miss, not raise.
_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/&|]')

_BY_TITLE = """
MATCH (n) WHERE n.id STARTS WITH $prefix AND toLower(n.title) = toLower($name)
RETURN n.id AS id, n.title AS title, labels(n) AS labels
ORDER BY size(labels(n)) DESC, n.id
LIMIT 1
"""

_BY_ALIAS = """
MATCH (n) WHERE n.id STARTS WITH $prefix
  AND any(a IN coalesce(n.aliases, []) WHERE toLower(a) = toLower($name))
RETURN n.id AS id, n.title AS title, labels(n) AS labels
ORDER BY size(labels(n)) DESC, n.id
LIMIT 1
"""

_BY_SEARCH = """
CALL db.index.fulltext.queryNodes('entity_aliases', $search) YIELD node, score
WITH node, score WHERE node.id STARTS WITH $prefix
RETURN node.id AS id, node.title AS title, labels(node) AS labels, score
ORDER BY score DESC
LIMIT 1
"""

#: Below this the full-text hit is noise. Lucene scores are unbounded and corpus
#: dependent, so this is a floor against nonsense rather than a calibrated threshold:
#: the index holds under a thousand short titles, and a real match scores well clear
#: of it while an invented one does not.
MIN_SEARCH_SCORE = 1.0


class GraphEntityResolver:
    """Resolves entity mentions against a loaded graph. The caller owns the driver."""

    def __init__(self, driver: Driver, wiki: str, *, min_search_score: float = MIN_SEARCH_SCORE):
        self._driver = driver
        self._prefix = f"{wiki}:"
        self._min_score = min_search_score

    def resolve_all(self, mentions: list[str]) -> tuple[list[ResolvedEntity], list[str]]:
        """Resolve every mention, returning what matched and what did not.

        Duplicates collapse: a question naming the same thing twice ("iron plate or
        iron plates") should not weight a query twice or log two entities.
        """
        resolved: dict[str, ResolvedEntity] = {}
        unresolved: list[str] = []
        with self._driver.session() as session:
            for mention in mentions:
                cleaned = mention.strip()
                if not cleaned:
                    continue
                entity = self._resolve_one(session, cleaned)
                if entity is None:
                    if cleaned not in unresolved:
                        unresolved.append(cleaned)
                elif entity.id not in resolved:
                    resolved[entity.id] = entity
        return list(resolved.values()), unresolved

    def _resolve_one(self, session: Session, name: str) -> ResolvedEntity | None:
        for query, matched_by in ((_BY_TITLE, "title"), (_BY_ALIAS, "alias")):
            row = session.run(query, prefix=self._prefix, name=name).single()
            if row is not None:
                return _entity(name, row, matched_by)

        search = _lucene(name)
        if not search:
            return None
        row = session.run(_BY_SEARCH, prefix=self._prefix, search=search).single()
        if row is not None and float(row["score"]) >= self._min_score:
            return _entity(name, row, "search")
        return None


def _entity(mention: str, row: Mapping[str, Any], matched_by: str) -> ResolvedEntity:
    return ResolvedEntity.model_validate(
        {
            "mention": mention,
            "id": row["id"],
            "title": row["title"],
            "labels": tuple(row["labels"] or ()),
            "matched_by": matched_by,
        }
    )


def _lucene(name: str) -> str:
    """A full-text query for a name: every term required, each quoted as a literal.

    ``AND``, not ``OR``. Joining loosely meant any shared word was enough, and
    "Flurbo engine" - a thing that does not exist - came back as "Engine unit" with a
    passing score. That is the exact failure this module exists to prevent: an
    invented entity that resolves anyway is worse than one that does not resolve,
    because the first produces a confident answer about the wrong thing.

    Quoting each term keeps a leftover keyword literal rather than operative.
    """
    terms = [term for term in _LUCENE_SPECIAL.sub(" ", name).split() if term]
    return " AND ".join(f'"{term}"' for term in terms)
