"""The ``A + B + C`` grammar used for producers, consumers and technology lists.

Whether each name resolves to a real graph node is entity resolution's problem
(Phase 3): "manual" and "Player" show up here as ordinary entries, exactly as the wiki
wrote them.
"""

from __future__ import annotations

import re

_PLUS_SPLIT = re.compile(r"\s*\+\s*")

#: A trailing level or level range: "Automation 2, 2" and "Gun turret damage, 2-7"
#: both name one technology, with the levels it applies to appended. The levels are
#: dropped rather than modelled: the ontology has no notion of a technology level, and
#: keeping them would make every entry fail to resolve to the page it names.
_TRAILING_LEVELS = re.compile(r",\s*\d+(?:\s*-\s*\d+)?\s*$")


def plus_list(value: str) -> list[str]:
    """Split on ``+``, trimming whitespace and dropping stray empty entries."""
    return [part for part in (p.strip() for p in _PLUS_SPLIT.split(value.strip())) if part]


def leveled_name_list(value: str) -> list[str]:
    """``plus_list``, with a trailing level or level range stripped from each entry.

    The ``allows`` field's grammar. Without this, "Automation 2, 2" is a name nothing
    resolves to, and the technology tree loses the edge.
    """
    return [
        name for name in (_TRAILING_LEVELS.sub("", p).strip() for p in plus_list(value)) if name
    ]
