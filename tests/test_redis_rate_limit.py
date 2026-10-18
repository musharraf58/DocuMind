"""The shared limiter across two processes (the API and the worker), each
with its own Redis client on one shared (fake) Redis server."""

import asyncio
import random

import fakeredis
import pytest

from app.services.embeddings.errors import RetryableEmbeddingError
from app.services.embeddings.redis_rate_limit import DEFAULT_KEY_PREFIX, RedisRateLimiter
from tests.embedding_fakes import FakeClock

# The defaults for a 10,000 tokens / 3 requests per minute account:
TOKENS, REQUESTS = 5000, 3  # token budget (x0.5 safety), the request limit
SEARCH_TOKENS, SEARCH_REQUESTS = 250, 1  # kept for searches: ingestion gets 4,750 and 2


def process(server: fakeredis.FakeServer, clock: FakeClock) -> RedisRateLimiter:
    """One process's limiter: its own client, the shared server."""
    return RedisRateLimiter(
        fakeredis.aioredis.FakeRedis(server=server),
        TOKENS,
        REQUESTS,
        SEARCH_TOKENS,
        SEARCH_REQUESTS,
        clock=clock.clock,
        sleep=clock.sleep,
    )


def windows(sent):
    """For each request, everything sent in the 60 s ending at it."""
    for at, _, _ in sent:
        yield [(tokens, kind) for t, tokens, kind in sent if at - 60 < t <= at]


@pytest.mark.parametrize("seed", range(8))
async def test_two_processes_together_never_exceed_the_budget(seed):
    rng = random.Random(seed)
    server, clock = fakeredis.FakeServer(), FakeClock()
    api, worker = process(server, clock), process(server, clock)
    sent: list[tuple[float, int, str]] = []

    async def ingest():  # the worker: document batches, as fast as allowed
        for _ in range(25):
            tokens = rng.randint(500, worker.max_tokens("document"))
            await worker.acquire(tokens, "document")
            sent.append((clock.now, tokens, "document"))

    async def search():  # the API: small queries now and then
        for _ in range(25):
            await clock.sleep(rng.choice([0, 5, 20, 45]))
            tokens = rng.randint(5, 40)
            await api.acquire(tokens, "query")
            sent.append((clock.now, tokens, "query"))

    await asyncio.gather(ingest(), search())

    assert len(sent) == 50
    for window in windows(sent):
        # Both processes together stay within the account's budget...
        assert len(window) <= REQUESTS
        assert sum(tokens for tokens, _ in window) <= TOKENS
        # ...and ingestion never dips into what's kept for searches.
        documents = [tokens for tokens, kind in window if kind == "document"]
        assert len(documents) <= REQUESTS - SEARCH_REQUESTS
        assert sum(documents) <= TOKENS - SEARCH_TOKENS


async def test_search_gets_through_while_ingestion_uses_its_whole_share():
    server, clock = fakeredis.FakeServer(), FakeClock()
    api, worker = process(server, clock), process(server, clock)
    await worker.acquire(2000, "document")
    await worker.acquire(2700, "document")  # 2 requests, 4,700 tokens: ingestion is full

    assert await worker.time_until_available(10, "document") > 0  # the worker must wait
    reservation = await api.acquire(30, "query", max_wait=20)  # a search doesn't
    assert reservation.waited == 0


async def test_a_429_seen_by_one_process_holds_the_other():
    server, clock = fakeredis.FakeServer(), FakeClock()
    api, worker = process(server, clock), process(server, clock)
    await worker.penalize(30)
    assert await api.time_until_available(10, "query") == pytest.approx(30)


async def test_a_corrected_token_count_is_seen_by_the_other_process():
    server, clock = fakeredis.FakeServer(), FakeClock()
    api, worker = process(server, clock), process(server, clock)
    reservation = await worker.acquire(1000, "document")
    await reservation.correct(1600)
    assert await api.usage() == (1, 1600)


async def test_old_entries_are_removed_from_redis():
    server, clock = fakeredis.FakeServer(), FakeClock()
    limiter = process(server, clock)
    client = fakeredis.aioredis.FakeRedis(server=server)
    for _ in range(40):
        await limiter.acquire(100, "query")
        clock.now += 61  # each request has left the window before the next
    assert await client.zcard(f"{DEFAULT_KEY_PREFIX}:log") == 1
    assert await client.hlen(f"{DEFAULT_KEY_PREFIX}:tokens") == 1
    assert await client.pttl(f"{DEFAULT_KEY_PREFIX}:log") > 0  # and idle keys expire


async def test_redis_down_is_a_transient_embedding_error():
    server, clock = fakeredis.FakeServer(), FakeClock()
    limiter = process(server, clock)
    server.connected = False
    with pytest.raises(RetryableEmbeddingError, match="rate limiter is unavailable"):
        await limiter.acquire(10, "query")
