"""The inbound limiter on its own: fake clock, fake Redis, a broken Redis."""

import logging

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.rate_limit import (
    Decision,
    InboundLimiter,
    MemoryRateLimiter,
    OutageLog,
    Rate,
    RedisRateLimiter,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class DownRedis:
    """Every command fails as if Redis were unreachable."""

    def register_script(self, script):
        async def run(*args, **kwargs):
            raise RedisConnectionError("Connection refused")

        return run


@pytest.fixture
def fake_clock():
    return FakeClock()


@pytest.fixture(params=["redis", "memory"])
def limiter(request, fake_clock):
    """Both implementations must behave the same."""
    if request.param == "redis":
        return RedisRateLimiter(fakeredis.aioredis.FakeRedis(), now=fake_clock)
    return MemoryRateLimiter(now=fake_clock)


def test_parse_rates():
    assert Rate.parse("10/minute") == Rate(10, 60)
    assert Rate.parse("5 / hours") == Rate(5, 3600)
    assert Rate.parse("1/second") == Rate(1, 1)
    assert str(Rate.parse("120/minute")) == "120/minute"
    for bad in ["10", "0/minute", "ten/minute", "10/week", "-1/minute"]:
        with pytest.raises(ValueError):
            Rate.parse(bad)


async def test_blocks_the_n_plus_first_request(limiter, fake_clock):
    rate = Rate(3, 60)
    decisions = []
    for _ in range(3):
        decisions.append(await limiter.hit("k", rate))
        fake_clock.advance(1)
    assert [d.allowed for d in decisions] == [True, True, True]
    assert [d.remaining for d in decisions] == [2, 1, 0]

    blocked = await limiter.hit("k", rate)
    assert not blocked.allowed
    assert blocked.remaining == 0
    # The first request was at t=0, now is t=3: it leaves the window at t=60.
    assert blocked.retry_after == 57
    assert blocked.headers()["Retry-After"] == "57"
    assert blocked.headers()["RateLimit-Limit"] == "3"
    assert blocked.headers()["RateLimit-Remaining"] == "0"


async def test_resets_after_the_window(limiter, fake_clock):
    rate = Rate(2, 60)
    await limiter.hit("k", rate)
    await limiter.hit("k", rate)
    assert not (await limiter.hit("k", rate)).allowed
    fake_clock.advance(59.9)
    assert not (await limiter.hit("k", rate)).allowed
    fake_clock.advance(0.1)  # exactly 60 s after the first two
    assert (await limiter.hit("k", rate)).allowed


async def test_window_slides(limiter, fake_clock):
    """Not a fixed window: requests leave one by one, 60 s after each was made."""
    rate = Rate(2, 60)
    await limiter.hit("k", rate)  # t=0
    fake_clock.advance(30)
    await limiter.hit("k", rate)  # t=30
    fake_clock.advance(31)  # t=61: the first has left, the second hasn't
    assert (await limiter.hit("k", rate)).allowed
    assert not (await limiter.hit("k", rate)).allowed


async def test_refused_requests_do_not_extend_the_block(limiter, fake_clock):
    rate = Rate(1, 60)
    await limiter.hit("k", rate)
    for _ in range(20):
        fake_clock.advance(2)
        assert not (await limiter.hit("k", rate)).allowed
    fake_clock.advance(20)  # t=60
    assert (await limiter.hit("k", rate)).allowed


async def test_keys_are_independent(limiter):
    rate = Rate(1, 60)
    assert (await limiter.hit("user-a", rate)).allowed
    assert not (await limiter.hit("user-a", rate)).allowed
    assert (await limiter.hit("user-b", rate)).allowed


async def test_redis_keys_expire_on_their_own(fake_clock):
    redis = fakeredis.aioredis.FakeRedis()
    await RedisRateLimiter(redis, now=fake_clock).hit("documind:rl:test", Rate(5, 60))
    ttl = await redis.pttl("documind:rl:test")
    assert 0 < ttl <= 60_000


async def test_memory_limiter_is_bounded(fake_clock):
    limiter = MemoryRateLimiter(max_keys=3, now=fake_clock)
    for key in ["a", "b", "c", "d"]:
        await limiter.hit(key, Rate(1, 60))
    # "a" (least recently used) was dropped, so it starts fresh.
    assert len(limiter._logs) == 3
    assert (await limiter.hit("a", Rate(1, 60))).allowed
    assert not (await limiter.hit("d", Rate(1, 60))).allowed


async def test_redis_down_fails_open_with_a_loud_log(fake_clock, caplog):
    limiter = InboundLimiter(DownRedis(), MemoryRateLimiter(now=fake_clock), OutageLog())
    with caplog.at_level(logging.ERROR, logger="app.core.rate_limit"):
        for _ in range(50):
            assert await limiter.hit("k", Rate(1, 60)) is None  # allowed, unlimited
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1  # throttled, not 50
    assert "fail open" in errors[0].getMessage()


async def test_redis_down_login_uses_the_fallback(fake_clock):
    limiter = InboundLimiter(DownRedis(), MemoryRateLimiter(now=fake_clock), OutageLog())
    rate = Rate(2, 60)
    results = [await limiter.hit("login", rate, fallback=True) for _ in range(3)]
    assert all(isinstance(r, Decision) for r in results)
    assert [r.allowed for r in results] == [True, True, False]


def test_outage_log_repeats_after_the_interval(caplog):
    times = iter([0.0, 10.0, 31.0])
    outage = OutageLog(every=30, now=lambda: next(times))
    with caplog.at_level(logging.ERROR, logger="app.core.rate_limit"):
        for _ in range(3):
            outage.report("x", RedisConnectionError("down"))
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "1 similar messages suppressed" in messages[1]
