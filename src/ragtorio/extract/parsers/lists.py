"""The ``A + B + C`` grammar used for producers, consumers and technology lists.

Whether each name resolves to a real graph node is entity resolution's problem
(Phase 3): "manual" and "Player" show up here as ordinary entries, exactly as the wiki
wrote them.
"""

from __future__ import annotations

import re

_PLUS_SPLIT = re.compile(r"\s*\+\s*")


def plus_list(value: str) -> list[str]:
    """Split on ``+``, trimming whitespace and dropping stray empty entries."""
    return [part for part in (p.strip() for p in _PLUS_SPLIT.split(value.strip())) if part]
