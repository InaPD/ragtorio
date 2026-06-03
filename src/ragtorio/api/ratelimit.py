"""A per-IP request cap, in memory.

Every ``/ask`` spends money: one routing call, one or two answering calls on the
largest model in the lineup. An open endpoint with no cap is not a rate-limiting
oversight, it is a funding model for whoever finds it first.

In-process and per-instance, which is honest about what it is: enough for the single
container this ships as, and wrong the moment there are two of them behind a load
balancer. That is a note in the README rather than a Redis dependency for a research
project - but it is a note, because a cap that silently stops capping is worse than
none.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

#: Generous for a person, tight for a script.
DEFAULT_LIMIT = 20
DEFAULT_WINDOW_SECONDS = 60.0

#: Stop tracking clients that have gone quiet, so a long-lived process does not hold a
#: timestamp deque per IP that ever touched it.
_IDLE_EVICTION_FACTOR = 2


class RateLimiter:
    """A sliding window of request times per client key. Thread-safe."""

    def __init__(
        self,
        limit: int = DEFAULT_LIMIT,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, deque[float]] = {}

    def check(self, key: str) -> float | None:
        """Record a request, or return the seconds until the client may retry."""
        now = self._clock()
        with self._lock:
            self._evict(now)
            hits = self._seen.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return max(0.0, self._window - (now - hits[0]))
            hits.append(now)
            return None

    def _evict(self, now: float) -> None:
        cutoff = self._window * _IDLE_EVICTION_FACTOR
        stale = [key for key, hits in self._seen.items() if not hits or now - hits[-1] > cutoff]
        for key in stale:
            del self._seen[key]
