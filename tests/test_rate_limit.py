import asyncio
import random

import pytest

from app.services.embeddings.errors import EmbeddingBusyError, RequestTooLargeError
from app.services.embeddings.rate_limit import (
    WAKE_PADDING_SECONDS,
    SlidingWindowRateLimiter,
    estimate_seconds,
)
from tests.embedding_fakes import FakeClock

PAD = WAKE_PADDING_SECONDS


def make(clock: FakeClock, tokens=8000, requests=2, reserve_tokens=0, reserve_requests=0):
    return SlidingWindowRateLimiter(
        token_budget=tokens,
        request_budget=requests,
        query_reserve_tokens=reserve_tokens,
        query_reserve_requests=reserve_requests,
        clock=clock.clock,
        sleep=clock.sleep,
    )


async def test_requests_within_budget_go_through_without_waiting():
    clock = FakeClock()
    limiter = make(clock)
    assert (await limiter.acquire(3000)).waited == 0
    assert (await limiter.acquire(3000)).waited == 0
    assert clock.sleeps == []
    assert (await limiter.usage()) == (2, 6000)


async def test_request_budget_makes_the_third_request_wait_for_the_first_to_expire():
    clock = FakeClock()
    limiter = make(clock, requests=2)
    start = clock.now
    await limiter.acquire(10)
    clock.now += 5
    await limiter.acquire(10)

    waited = (await limiter.acquire(10)).waited  # must wait until the first leaves (t = 60)
    assert clock.now == pytest.approx(start + 60 + PAD)
    assert waited == pytest.approx(55 + PAD)
    assert (await limiter.usage()) == (2, 20)


async def test_token_budget_waits_only_until_enough_old_tokens_expire():
    clock = FakeClock()
    limiter = make(clock, tokens=8000, requests=10)
    start = clock.now
    await limiter.acquire(3000)  # t=0
    clock.now = start + 10
    await limiter.acquire(3000)  # t=10
    clock.now = start + 20
    await limiter.acquire(2000)  # t=20, window now holds 8000

    # 4000 more needs 4000 freed: the first 3000 alone isn't enough, so it
    # must wait for the second (t=10) to expire too, i.e. until t=70.
    await limiter.acquire(4000)
    assert clock.now == pytest.approx(start + 70 + PAD)


async def test_a_request_that_could_never_fit_fails_at_once():
    clock = FakeClock()
    limiter = make(clock, tokens=8000)
    with pytest.raises(RequestTooLargeError, match="9000 tokens; at most 8000"):
        await limiter.acquire(9000)
    assert clock.sleeps == []
    assert (await limiter.usage()) == (0, 0)


async def test_documents_leave_room_for_queries():
    clock = FakeClock()
    limiter = make(clock, tokens=8000, requests=2, reserve_tokens=400, reserve_requests=1)
    assert limiter.max_tokens("document") == 7600
    assert limiter.max_tokens("query") == 8000

    await limiter.acquire(7600)  # a full document batch
    # The second request slot and the last 400 tokens belong to queries.
    assert (await limiter.acquire(50, kind="query")).waited == 0
    assert clock.sleeps == []

    start = clock.now
    await limiter.acquire(100, kind="document")  # has to wait for the batch to expire
    assert clock.now >= start + 60


async def test_document_larger_than_its_share_is_too_large_but_a_query_is_not():
    limiter = make(FakeClock(), tokens=8000, reserve_tokens=400)
    with pytest.raises(RequestTooLargeError):
        await limiter.acquire(7700, kind="document")
    await limiter.acquire(7700, kind="query")


async def test_query_waits_when_the_wait_is_short_enough():
    clock = FakeClock()
    limiter = make(clock, requests=1)
    await limiter.acquire(10)
    clock.now += 45  # the slot frees up in 15 s

    waited = (await limiter.acquire(10, kind="query", max_wait=20)).waited
    assert waited == pytest.approx(15 + PAD)


async def test_query_fails_fast_when_the_wait_is_too_long():
    clock = FakeClock()
    limiter = make(clock, requests=1)
    await limiter.acquire(10)
    clock.now += 5  # 55 s to go, but the query only waits 20

    before = clock.now
    with pytest.raises(EmbeddingBusyError) as info:
        await limiter.acquire(10, kind="query", max_wait=20)
    assert info.value.retry_after == pytest.approx(55)
    assert "retry in about 55 seconds" in str(info.value)
    assert clock.now == before and clock.sleeps == []  # didn't wait
    assert (await limiter.usage()) == (1, 10)  # and didn't use up a slot


async def test_penalty_holds_everyone_until_it_passes():
    clock = FakeClock()
    limiter = make(clock)
    start = clock.now
    await limiter.penalize(30)

    with pytest.raises(EmbeddingBusyError) as info:
        await limiter.acquire(10, kind="query", max_wait=10)
    assert info.value.retry_after == pytest.approx(30)

    await limiter.acquire(10)
    assert clock.now == pytest.approx(start + 30 + PAD)


async def test_waits_in_the_log_do_not_exceed_what_is_needed():
    clock = FakeClock()
    limiter = make(clock, tokens=1000, requests=5)
    await limiter.acquire(1000)
    await limiter.acquire(1)
    assert sum(clock.sleeps) == pytest.approx(60 + PAD)


def windows_ok(sent: list[tuple[float, int]], token_budget: int, request_budget: int) -> None:
    """For every request, the window of the 60 s ending at it must be in budget."""
    for at, _ in sent:
        in_window = [tokens for t, tokens in sent if at - 60 < t <= at]
        assert len(in_window) <= request_budget, f"{len(in_window)} requests in 60s ending {at}"
        assert sum(in_window) <= token_budget, f"{sum(in_window)} tokens in 60s ending {at}"


@pytest.mark.parametrize("seed", range(15))
async def test_no_window_ever_exceeds_the_budget(seed):
    rng = random.Random(seed)
    token_budget, request_budget = rng.choice([(8000, 2), (500, 3), (100, 1), (3000, 5)])
    clock = FakeClock()
    limiter = make(
        clock,
        tokens=token_budget,
        requests=request_budget,
        reserve_tokens=token_budget // 20,
        reserve_requests=1 if request_budget > 1 else 0,
    )
    sent: list[tuple[float, int]] = []
    for _ in range(80):
        clock.now += rng.choice([0, 0, 0.5, 3, 20, 90])  # bursts, gaps, long pauses
        kind = rng.choice(["document", "document", "query"])
        tokens = rng.randint(1, limiter.max_tokens(kind))
        await limiter.acquire(tokens, kind=kind)
        sent.append((clock.now, tokens))
    windows_ok(sent, token_budget, request_budget)


async def test_concurrent_callers_together_stay_within_budget():
    clock = FakeClock()
    limiter = make(clock, tokens=5000, requests=3)
    sent: list[tuple[float, int]] = []

    async def worker(tokens: int) -> None:
        for _ in range(6):
            await limiter.acquire(tokens)
            sent.append((clock.now, tokens))

    await asyncio.gather(*(worker(t) for t in (1000, 2500, 4000, 700)))
    assert len(sent) == 24
    windows_ok(sent, 5000, 3)


async def test_time_until_available_matches_acquire_and_reserves_nothing():
    clock = FakeClock()
    limiter = make(clock, requests=1)
    await limiter.acquire(10)
    clock.now += 15

    assert await limiter.time_until_available(10) == pytest.approx(45)
    assert (await limiter.usage()) == (1, 10)  # only looked
    waited = (await limiter.acquire(10)).waited
    assert waited == pytest.approx(45 + PAD)


async def test_correcting_a_reservation_raises_but_never_lowers_its_tokens():
    clock = FakeClock()
    limiter = make(clock, tokens=1000, requests=10)
    reservation = await limiter.acquire(400)
    await reservation.correct(700)  # the provider counted more than we estimated
    assert (await limiter.usage()) == (1, 700)
    await reservation.correct(0)  # no count reported
    await reservation.correct(500)  # or a smaller one
    assert (await limiter.usage()) == (1, 700)


async def test_corrected_count_makes_the_next_request_wait():
    clock = FakeClock()
    limiter = make(clock, tokens=1000, requests=10)
    await (await limiter.acquire(500)).correct(900)
    # By our estimate 500 + 400 would fit; by the provider's count it doesn't.
    start = clock.now
    await limiter.acquire(400)
    assert clock.now - start == pytest.approx(60 + PAD)


def test_estimate_seconds():
    assert estimate_seconds(0, 8000) == 0
    assert estimate_seconds(8000, 8000) == 0  # fits in the first window
    assert estimate_seconds(48_000, 8000) == 300  # six windows' worth: 5 waits of 60 s


async def test_works_with_the_real_clock_and_sleep():
    """Everything above runs on a fake clock. This uses the real one, with a
    window of 0.3 s instead of 60 s so it takes about a second."""
    import time

    limiter = SlidingWindowRateLimiter(token_budget=1000, request_budget=2, window=0.3)
    sent: list[tuple[float, int]] = []
    start = time.monotonic()
    for _ in range(6):
        await limiter.acquire(10)
        sent.append((time.monotonic(), 10))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.6 - 0.05  # 6 requests at 2 per 0.3 s: two full windows of waiting
    for at, _ in sent:
        assert sum(1 for t, _ in sent if at - 0.3 < t <= at) <= 2
