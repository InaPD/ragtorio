"""The per-IP cap. A clock is injected so the window can be tested without waiting."""

from __future__ import annotations

from ragtorio.api.ratelimit import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_requests_under_the_limit_pass():
    limiter = RateLimiter(limit=3, window_seconds=60, clock=Clock())
    assert [limiter.check("1.2.3.4") for _ in range(3)] == [None, None, None]


def test_the_next_request_is_refused_with_a_wait_a_client_can_act_on():
    clock = Clock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    limiter.check("1.2.3.4")
    clock.now += 10
    limiter.check("1.2.3.4")
    retry_after = limiter.check("1.2.3.4")
    assert retry_after is not None
    assert 49 <= retry_after <= 50  # until the oldest of the two leaves the window


def test_clients_are_counted_separately():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=Clock())
    assert limiter.check("1.2.3.4") is None
    assert limiter.check("5.6.7.8") is None


def test_the_window_slides_rather_than_resetting_on_a_boundary():
    clock = Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    limiter.check("1.2.3.4")
    clock.now += 59
    assert limiter.check("1.2.3.4") is not None
    clock.now += 2
    assert limiter.check("1.2.3.4") is None


def test_clients_that_go_quiet_stop_being_tracked():
    """Otherwise a long-lived process keeps a deque per address that ever touched it."""
    clock = Clock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    limiter.check("1.2.3.4")
    clock.now += 600
    limiter.check("5.6.7.8")
    assert "1.2.3.4" not in limiter._seen
