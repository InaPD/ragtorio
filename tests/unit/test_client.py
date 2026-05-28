"""The MediaWiki client's politeness and retry behaviour.

Wiki operators ask for rate limiting, maxlag and Retry-After. Those are the
behaviours worth testing; the happy path is the easy part.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from field_assistant.harvest.client import (
    DEFAULT_MAXLAG,
    MediaWikiClient,
    MediaWikiError,
    RateLimiter,
)

API = "https://example.test/api.php"


class FakeClock:
    """A monotonic clock that only advances when someone sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestRateLimiter:
    def test_first_call_does_not_sleep(self) -> None:
        clock = FakeClock()
        RateLimiter(1.0, monotonic=clock.monotonic, sleep=clock.sleep).acquire()
        assert clock.slept == []

    def test_subsequent_calls_are_spaced(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(2.0, monotonic=clock.monotonic, sleep=clock.sleep)
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        assert clock.slept == [0.5, 0.5]

    def test_no_sleep_when_enough_time_already_passed(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(1.0, monotonic=clock.monotonic, sleep=clock.sleep)
        limiter.acquire()
        clock.now += 10.0
        limiter.acquire()
        assert clock.slept == []

    def test_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(0)


def make_client(sleep: object = None) -> MediaWikiClient:
    return MediaWikiClient(API, "test-agent/1.0", rps=1000.0, sleep=sleep or (lambda _: None))


def backoff_sleeps(recorded: list[float]) -> list[float]:
    """Drop the rate limiter's sub-millisecond spacing sleeps, keeping real backoffs.

    Both share one injected sleep function, so tests about backoff must filter.
    """
    return [s for s in recorded if s >= 0.01]


@respx.mock
def test_request_sends_formatversion_and_maxlag() -> None:
    route = respx.get(API).mock(return_value=httpx.Response(200, json={"query": {}}))
    with make_client() as client:
        client.query(meta="siteinfo")
    params = route.calls[0].request.url.params
    assert params["formatversion"] == "2"
    assert params["maxlag"] == str(DEFAULT_MAXLAG)
    assert params["action"] == "query"


@respx.mock
def test_user_agent_is_sent() -> None:
    route = respx.get(API).mock(return_value=httpx.Response(200, json={"query": {}}))
    with make_client() as client:
        client.query()
    assert route.calls[0].request.headers["User-Agent"] == "test-agent/1.0"


@respx.mock
def test_retries_429_and_honours_retry_after() -> None:
    slept: list[float] = []
    respx.get(API).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(200, json={"query": {"ok": True}}),
        ]
    )
    with make_client(sleep=slept.append) as client:
        assert client.query() == {"ok": True}
    assert backoff_sleeps(slept) == [3.0]


@respx.mock
def test_retries_server_errors_with_backoff() -> None:
    slept: list[float] = []
    respx.get(API).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(502),
            httpx.Response(200, json={"query": {"ok": True}}),
        ]
    )
    with make_client(sleep=slept.append) as client:
        assert client.query() == {"ok": True}
    assert backoff_sleeps(slept) == [1.0, 2.0]


@respx.mock
def test_maxlag_error_is_retried_not_raised() -> None:
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json={"error": {"code": "maxlag", "info": "lagged"}}),
            httpx.Response(200, json={"query": {"ok": True}}),
        ]
    )
    with make_client() as client:
        assert client.query() == {"ok": True}


@respx.mock
def test_other_api_errors_raise_immediately() -> None:
    respx.get(API).mock(
        return_value=httpx.Response(200, json={"error": {"code": "badvalue", "info": "nope"}})
    )
    with make_client() as client, pytest.raises(MediaWikiError) as exc:
        client.query()
    assert exc.value.code == "badvalue"


@respx.mock
def test_network_errors_are_retried() -> None:
    respx.get(API).mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={"query": {"ok": 1}})]
    )
    with make_client() as client:
        assert client.query() == {"ok": 1}


@respx.mock
def test_gives_up_after_max_retries() -> None:
    respx.get(API).mock(return_value=httpx.Response(503))
    client = MediaWikiClient(API, "ua", rps=1000.0, max_retries=2, sleep=lambda _: None)
    with client, pytest.raises(RuntimeError, match="giving up"):
        client.query()


@respx.mock
def test_malformed_retry_after_falls_back_to_backoff() -> None:
    slept: list[float] = []
    respx.get(API).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            httpx.Response(200, json={"query": {}}),
        ]
    )
    with make_client(sleep=slept.append) as client:
        client.query()
    assert backoff_sleeps(slept) == [1.0]


@respx.mock
def test_query_returns_empty_dict_when_absent() -> None:
    respx.get(API).mock(return_value=httpx.Response(200, json={"batchcomplete": True}))
    with make_client() as client:
        assert client.query() == {}


@respx.mock
def test_query_paged_follows_continuation() -> None:
    respx.get(API).mock(
        side_effect=[
            httpx.Response(
                200,
                json={"query": {"allpages": [{"title": "A"}]}, "continue": {"apcontinue": "B"}},
            ),
            httpx.Response(200, json={"query": {"allpages": [{"title": "B"}]}}),
        ]
    )
    with make_client() as client:
        chunks = list(client.query_paged(list="allpages"))
    assert [c["allpages"][0]["title"] for c in chunks] == ["A", "B"]


@respx.mock
def test_query_paged_passes_continuation_params_back() -> None:
    route = respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json={"query": {"x": 1}, "continue": {"apcontinue": "Next"}}),
            httpx.Response(200, json={"query": {"x": 2}}),
        ]
    )
    with make_client() as client:
        list(client.query_paged(list="allpages"))
    assert route.calls[1].request.url.params["apcontinue"] == "Next"


@respx.mock
def test_injected_http_client_is_not_closed_by_us() -> None:
    respx.get(API).mock(return_value=httpx.Response(200, json={"query": {}}))
    inner = httpx.Client()
    with MediaWikiClient(API, "ua", rps=1000.0, client=inner) as client:
        client.query()
    assert not inner.is_closed
    inner.close()
