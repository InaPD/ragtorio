"""A polite, retrying MediaWiki API client.

Every wiki this project touches is someone else's server, usually behind Cloudflare.
The rules encoded here are the ones wiki operators actually ask for: identify yourself,
stay under one request per second by default, pass ``maxlag`` so you back off when the
site's replicas fall behind, and honour ``Retry-After`` instead of hammering.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Self

import httpx

#: Ask the server to reject our request rather than add load when replication lags.
DEFAULT_MAXLAG = 5

#: Status codes worth retrying. Everything else is a bug on our side.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class MediaWikiError(RuntimeError):
    """The API returned a structured ``error`` object we cannot retry past."""

    def __init__(self, code: str, info: str) -> None:
        super().__init__(f"MediaWiki API error [{code}]: {info}")
        self.code = code
        self.info = info


class RateLimiter:
    """Spaces calls at least ``1/rps`` seconds apart.

    The clock and sleep function are injectable so tests can run instantly.
    """

    def __init__(
        self,
        rps: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._min_interval = 1.0 / rps
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_allowed: float | None = None

    def acquire(self) -> None:
        now = self._monotonic()
        if self._next_allowed is not None and now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._next_allowed
        self._next_allowed = now + self._min_interval


class MediaWikiClient:
    """Talks to one wiki's ``api.php``.

    Always requests ``formatversion=2``, which returns pages as a list rather than a
    dict keyed by stringified page id.
    """

    def __init__(
        self,
        api_url: str,
        user_agent: str,
        *,
        rps: float = 1.0,
        max_retries: int = 5,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_url = api_url
        self.max_retries = max_retries
        self._sleep = sleep
        self._limiter = RateLimiter(rps, sleep=sleep)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def request(self, **params: Any) -> dict[str, Any]:
        """One API call, rate limited and retried. Returns the decoded JSON body."""
        payload: dict[str, Any] = {
            "format": "json",
            "formatversion": 2,
            "maxlag": DEFAULT_MAXLAG,
            **params,
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            try:
                response = self._client.get(self.api_url, params=payload)
            except httpx.HTTPError as exc:  # network-level failure
                last_error = exc
                self._backoff(attempt)
                continue

            if response.status_code in RETRY_STATUS:
                self._backoff(attempt, retry_after=response.headers.get("Retry-After"))
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
                continue

            response.raise_for_status()
            body: dict[str, Any] = response.json()

            error = body.get("error")
            if error:
                code = str(error.get("code", "unknown"))
                if code == "maxlag":
                    self._backoff(attempt, retry_after=response.headers.get("Retry-After"))
                    last_error = MediaWikiError(code, str(error.get("info", "")))
                    continue
                raise MediaWikiError(code, str(error.get("info", "")))

            return body

        raise RuntimeError(
            f"giving up on {self.api_url} after {self.max_retries + 1} attempts"
        ) from last_error

    def query(self, **params: Any) -> dict[str, Any]:
        """``action=query``, returning the ``query`` sub-object (empty dict if absent)."""
        body = self.request(action="query", **params)
        result: dict[str, Any] = body.get("query", {})
        return result

    def query_paged(self, **params: Any) -> Iterator[dict[str, Any]]:
        """``action=query`` following continuation, yielding each page's ``query`` object."""
        continuation: dict[str, Any] = {}
        while True:
            body = self.request(action="query", **params, **continuation)
            chunk = body.get("query")
            if chunk:
                yield chunk
            next_continue = body.get("continue")
            if not next_continue:
                return
            continuation = dict(next_continue)

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        """Honour ``Retry-After`` when the server sends one, else exponential backoff."""
        if retry_after:
            try:
                self._sleep(min(float(retry_after), 60.0))
                return
            except ValueError:  # HTTP-date form, not seconds
                pass
        self._sleep(min(2.0**attempt, 30.0))
