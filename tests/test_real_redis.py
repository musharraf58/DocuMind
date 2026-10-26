"""The Lua scripts against a REAL Redis server.

Every other test uses fakeredis, which runs our Lua scripts in its own Lua
interpreter. That is fast and needs no server, but it is an emulation: a
script that relies on something fakeredis does differently would pass there
and break in production. These tests run the same scripts on a real server.

They use database 15 of the Redis in REDIS_URL (never the app's own data,
which lives in database 0) and delete their keys afterwards. Without a
reachable Redis they are skipped, unless REQUIRE_REAL_SERVICES=1 (set in CI),
where a missing Redis is a failure: CI must not pass by skipping them.
"""

import os
import uuid

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.rate_limit import Rate, RedisRateLimiter
from app.services.embeddings.errors import EmbeddingBusyError
from app.services.embeddings.redis_rate_limit import RedisRateLimiter as OutboundLimiter
from app.services.login_guard import LoginGuard, LoginPolicy
from app.services.quotas import QUESTIONS, Quotas

TEST_DB = 15


@pytest.fixture
async def real_redis():
    url = get_settings().REDIS_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
    redis = Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
    try:
        await redis.ping()
    except (RedisError, OSError):
        await redis.aclose()
        if os.environ.get("REQUIRE_REAL_SERVICES") == "1":
            pytest.fail(f"REQUIRE_REAL_SERVICES=1 but no Redis answers at {url}")
        pytest.skip("no real Redis reachable")
    await redis.flushdb()  # database 15 belongs to these tests alone
    yield redis
    await redis.flushdb()
    await redis.aclose()


async def test_inbound_sliding_window(real_redis):
    now = [1000.0]
    limiter = RedisRateLimiter(real_redis, now=lambda: now[0])
    key = f"documind:rl:test:u:{uuid.uuid4()}"
    rate = Rate(2, 60)
    assert (await limiter.hit(key, rate)).allowed
    assert (await limiter.hit(key, rate)).allowed
    refused = await limiter.hit(key, rate)
    assert not refused.allowed and refused.retry_after == 60
    now[0] += 61  # both requests leave the window
    assert (await limiter.hit(key, rate)).allowed


async def test_quota_take_stops_at_the_limit(real_redis):
    limit = 3
    quotas = Quotas(real_redis, get_settings().model_copy(update={"DAILY_QUESTIONS_PER_USER": 3}))
    user_id = uuid.uuid4()
    for _ in range(limit):
        assert (await quotas.take(user_id, QUESTIONS)).allowed
    taken = await quotas.take(user_id, QUESTIONS)
    assert not taken.allowed and taken.used == limit
    assert await real_redis.ttl(quotas._user_key(user_id, QUESTIONS)) > 0  # expires by itself


async def test_login_delay_grows_after_free_attempts(real_redis):
    now = [1000.0]
    policy = LoginPolicy(free_attempts=1, base_delay=2, max_delay=60, window=900)
    guard = LoginGuard(real_redis, policy, now=lambda: now[0])
    assert await guard.failed("a@example.com", "1.2.3.4") == 0
    assert await guard.failed("a@example.com", "1.2.3.4") == 2
    assert await guard.wait_needed("a@example.com", "1.2.3.4") == 2
    await guard.succeeded("a@example.com", "1.2.3.4")
    assert await guard.wait_needed("a@example.com", "1.2.3.4") == 0


async def test_outbound_limiter_shares_one_window(real_redis):
    now = [1000.0]
    prefix = f"documind:test-limit:{uuid.uuid4()}"
    # Two limiters on one prefix = two processes sharing the provider account.
    first, second = (
        OutboundLimiter(real_redis, 1000, 2, key_prefix=prefix, clock=lambda: now[0])
        for _ in range(2)
    )
    await first.acquire(100, "query", max_wait=0)
    await second.acquire(100, "query", max_wait=0)
    with pytest.raises(EmbeddingBusyError):
        await first.acquire(100, "query", max_wait=0)
    assert await second.usage() == (2, 200)
