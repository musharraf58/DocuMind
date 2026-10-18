import asyncio
import logging
import math
import random

import pytest

from app.services.embeddings import (
    EmbeddingBusyError,
    EmbeddingError,
    FakeEmbeddingProvider,
    RequestTooLargeError,
    RetryableEmbeddingError,
)
from app.services.embeddings.base import BatchResult
from tests.embedding_fakes import DIMENSIONS, FakeClock, ScriptedProvider, limited_provider


def texts(n: int) -> list[str]:
    return [f"chunk number {i}" for i in range(n)]


# --- Fake provider ---


async def test_fake_vectors_are_deterministic_and_sized():
    provider = FakeEmbeddingProvider(model="fake", dimensions=DIMENSIONS)
    first = await provider.embed_documents(["alpha", "beta"])
    again = await provider.embed_documents(["alpha", "beta"])
    assert first == again
    assert first[0] != first[1]
    assert all(len(v) == DIMENSIONS for v in first)
    assert math.isclose(sum(x * x for x in first[0]), 1.0)


async def test_fake_respects_configured_dimension():
    provider = FakeEmbeddingProvider(model="fake", dimensions=8)
    assert len(await provider.embed_query("hello")) == 8


# --- Batching ---


async def test_ten_texts_with_batch_size_three_make_four_calls():
    provider = ScriptedProvider(batch_size=3)
    vectors = await provider.embed_documents(texts(10))

    assert [len(batch) for batch, _ in provider.calls] == [3, 3, 3, 1]
    assert [t for batch, _ in provider.calls for t in batch] == texts(10)  # order kept
    assert len(vectors) == 10
    assert all(input_type == "document" for _, input_type in provider.calls)


async def test_batch_size_is_capped_by_provider_limit():
    class SmallLimit(ScriptedProvider):
        max_batch_texts = 4

    provider = SmallLimit(batch_size=64)
    await provider.embed_documents(texts(10))
    assert [len(batch) for batch, _ in provider.calls] == [4, 4, 2]


async def test_batches_stay_within_token_limit():
    class TokenLimited(ScriptedProvider):
        max_batch_tokens = 40  # we use half as a safety margin: 20 tokens per request

    provider = TokenLimited(batch_size=64)
    long_texts = [" ".join(["word"] * 9)] * 6  # ~9 tokens each
    await provider.embed_documents(long_texts)
    assert [len(batch) for batch, _ in provider.calls] == [2, 2, 2]


# --- Token-aware batching ---


def sizes(provider: ScriptedProvider) -> list[int]:
    return [len(batch) for batch, _ in provider.calls]


async def test_batch_ends_when_next_chunk_would_exceed_token_budget():
    provider = ScriptedProvider(batch_size=64, token_budget=1000)
    await provider.embed_documents(texts(5), token_counts=[400] * 5)
    # 400 + 400 fits; a third would make 1200 > 1000.
    assert sizes(provider) == [2, 2, 1]


async def test_batch_may_fill_the_budget_exactly():
    provider = ScriptedProvider(token_budget=1000)
    await provider.embed_documents(texts(4), token_counts=[500, 500, 1000, 1])
    assert sizes(provider) == [2, 1, 1]  # 500+500 = budget; 1000 alone; then 1


async def test_batch_size_still_applies_when_tokens_are_small():
    provider = ScriptedProvider(batch_size=3, token_budget=1000)
    await provider.embed_documents(texts(7), token_counts=[10] * 7)
    assert sizes(provider) == [3, 3, 1]


async def test_given_token_counts_are_used_instead_of_recounting():
    # "tiny" is one token to tiktoken, but the caller says each is 900.
    provider = ScriptedProvider(token_budget=1000)
    await provider.embed_documents(["tiny", "tiny", "tiny"], token_counts=[900, 900, 900])
    assert sizes(provider) == [1, 1, 1]


async def test_tokens_are_counted_when_the_caller_gives_no_counts():
    provider = ScriptedProvider(token_budget=20)
    nine_tokens = " ".join(["word"] * 9)
    await provider.embed_documents([nine_tokens] * 6)
    assert sizes(provider) == [2, 2, 2]


async def test_chunk_larger_than_budget_is_refused_before_any_request():
    provider = ScriptedProvider(token_budget=1000)
    with pytest.raises(RequestTooLargeError, match=r"1500 tokens; at most 1000") as info:
        await provider.embed_documents(texts(3), token_counts=[300, 300, 1500])
    assert provider.calls == []  # not even the chunks before it were sent
    assert "CHUNK_TOKENS" in str(info.value)


async def test_without_a_token_budget_nothing_is_counted():
    provider = ScriptedProvider(batch_size=2)
    assert provider.token_budget is None
    await provider.embed_documents(texts(5), token_counts=[10**9] * 5)
    assert sizes(provider) == [2, 2, 1]


async def test_mismatched_token_counts_are_a_bug():
    provider = ScriptedProvider(token_budget=1000)
    with pytest.raises(ValueError, match="one entry per text"):
        await provider.embed_documents(texts(3), token_counts=[1, 2])


@pytest.mark.parametrize("seed", range(10))
async def test_no_batch_ever_exceeds_the_budget(seed):
    rng = random.Random(seed)
    budget = rng.randint(200, 2000)
    counts = [rng.randint(1, budget) for _ in range(rng.randint(1, 60))]
    provider = ScriptedProvider(batch_size=rng.randint(1, 20), token_budget=budget)
    await provider.embed_documents(texts(len(counts)), token_counts=counts)

    position = 0
    for batch, _ in provider.calls:
        batch_counts = counts[position : position + len(batch)]
        assert sum(batch_counts) <= budget
        assert len(batch) <= provider.batch_size
        position += len(batch)
    assert position == len(counts)  # everything sent exactly once, in order


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
async def test_blank_text_is_never_sent(bad):
    provider = ScriptedProvider()
    with pytest.raises(ValueError):
        await provider.embed_documents(["fine", bad])
    with pytest.raises(ValueError):
        await provider.embed_query(bad)
    assert provider.calls == []


async def test_query_uses_query_input_type():
    provider = ScriptedProvider()
    vector = await provider.embed_query("what is pgvector?")
    assert provider.calls == [(["what is pgvector?"], "query")]
    assert len(vector) == DIMENSIONS


# --- Retries ---


async def test_retry_succeeds_after_two_retryable_failures():
    provider = ScriptedProvider(
        failures=[RetryableEmbeddingError("rate limited"), RetryableEmbeddingError("503")]
    )
    vectors = await provider.embed_documents(texts(2))

    assert len(vectors) == 2
    assert len(provider.calls) == 3
    delays = provider._sleep.delays
    assert len(delays) == 2
    assert 1.0 <= delays[0] <= 2.0  # base 2 s, scaled by jitter to 50-100%
    assert 2.0 <= delays[1] <= 4.0  # doubled


async def test_gives_up_after_max_retries():
    failures = [RetryableEmbeddingError("Embedding service is unavailable")] * 10
    provider = ScriptedProvider(failures=failures, max_retries=3)
    with pytest.raises(RetryableEmbeddingError, match="unavailable"):
        await provider.embed_documents(texts(1))
    assert len(provider.calls) == 4  # first try + 3 retries
    assert len(provider._sleep.delays) == 3


async def test_non_retryable_error_fails_at_once():
    provider = ScriptedProvider(failures=[EmbeddingError("rejected the API key")])
    with pytest.raises(EmbeddingError, match="API key"):
        await provider.embed_documents(texts(5))
    assert len(provider.calls) == 1
    assert provider._sleep.delays == []


async def test_retry_after_hint_is_respected():
    provider = ScriptedProvider(failures=[RetryableEmbeddingError("429", retry_after=7)])
    await provider.embed_documents(texts(1))
    assert provider._sleep.delays == [7]


async def test_backoff_grows_exponentially_with_jitter_and_is_capped():
    provider = ScriptedProvider(retry_base_seconds=2, retry_max_seconds=10)
    for _ in range(50):  # jitter is random, so look at many draws
        assert 1 <= provider._retry_delay(0, None) <= 2
        assert 2 <= provider._retry_delay(1, None) <= 4
        assert 4 <= provider._retry_delay(2, None) <= 8
        assert 5 <= provider._retry_delay(3, None) <= 10  # 16 would exceed the cap of 10
        assert 5 <= provider._retry_delay(30, None) <= 10
    assert len({provider._retry_delay(0, None) for _ in range(50)}) > 1  # really random


async def test_429_waits_grow_from_the_minimum_with_jitter_on_top():
    provider = ScriptedProvider(min_retry_wait=20, retry_max_seconds=60)
    for _ in range(50):
        assert 20 <= provider._retry_delay(0, None, rate_limited=True) <= 25
        assert 40 <= provider._retry_delay(1, None, rate_limited=True) <= 50
        assert 60 <= provider._retry_delay(2, None, rate_limited=True) <= 75  # capped at 60
        assert 60 <= provider._retry_delay(9, None, rate_limited=True) <= 75
        # A Retry-After below the minimum doesn't shorten the wait...
        assert 20 <= provider._retry_delay(0, 3, rate_limited=True) <= 25
        # ...and one above it is honoured.
        assert 45 <= provider._retry_delay(0, 45, rate_limited=True) <= 56.25
    # Other errors keep the short backoff: they say nothing about our quota.
    assert provider._retry_delay(0, None, rate_limited=False) <= 2


async def test_minimum_wait_wins_even_above_the_backoff_cap():
    provider = ScriptedProvider(min_retry_wait=90, retry_max_seconds=60)
    assert provider._retry_delay(0, None, rate_limited=True) >= 90


async def test_retries_after_429s_never_exceed_the_real_limits():
    """The reported case: 3 requests and 10,000 tokens per minute, a 33-chunk
    document, and the provider answering 429 several times. Every attempt,
    including each failed one, must count, so that no 60-second window ever
    holds more than our budget (2 requests, 5,000 tokens), let alone the real limit."""
    clock = FakeClock()
    rate_limited = RetryableEmbeddingError("429", rate_limited=True)
    provider = limited_provider(
        clock,
        tokens=5000,
        requests=2,
        reserve_tokens=250,
        reserve_requests=1,
        failures=[rate_limited] * 4,
        min_retry_wait=20,
    )
    await provider.embed_documents(texts(33), token_counts=[374] * 33)

    assert len(provider.calls) == 4 + 3  # four failed attempts, then three batches
    tokens_per_attempt = [len(batch) * 374 for batch, _ in provider.calls]
    window_ok(provider.times, tokens_per_attempt, 5000, 2)
    gaps = [b - a for a, b in zip(provider.times, provider.times[1:], strict=False)]
    assert min(gaps) >= 20  # nothing ever fired within seconds of a failure


async def test_query_after_a_429_reports_busy_instead_of_retrying_quickly():
    clock = FakeClock()
    provider = limited_provider(
        clock,
        requests=10,
        failures=[RetryableEmbeddingError("429", rate_limited=True)],
        min_retry_wait=20,
        query_max_wait=20,
    )
    with pytest.raises(EmbeddingBusyError) as info:
        await provider.embed_query("anything")
    assert 20 <= info.value.retry_after <= 25
    assert len(provider.calls) == 1  # no second attempt
    assert clock.now == 1000.0  # and no waiting


async def test_retry_after_beyond_ceiling_fails_with_a_retry_time():
    provider = ScriptedProvider(failures=[RetryableEmbeddingError("429", retry_after=300)])
    with pytest.raises(EmbeddingBusyError, match="retry in about 300 seconds"):
        await provider.embed_documents(texts(1))
    assert provider._sleep.delays == []


async def test_each_batch_gets_its_own_retries():
    # Three failures in each of two batches: more than 3 retries in total, but
    # never more than 3 for any one batch.
    failure = RetryableEmbeddingError("503")
    provider = ScriptedProvider(batch_size=1, max_retries=3)
    provider.failures = [failure] * 3
    await provider.embed_documents(texts(1))
    provider.failures = [failure] * 3
    await provider.embed_documents(texts(1))
    assert len(provider.calls) == 8


# --- Rate limiting ---


def window_ok(times, token_counts, token_budget, request_budget):
    sent = list(zip(times, token_counts, strict=True))
    for at, _ in sent:
        in_window = [tokens for t, tokens in sent if at - 60 < t <= at]
        assert len(in_window) <= request_budget
        assert sum(in_window) <= token_budget


async def test_a_long_run_is_spread_over_minutes_without_exceeding_the_limits():
    # 96 chunks of 500 tokens = 48,000 tokens against 8,000 per minute.
    clock = FakeClock()
    provider = limited_provider(
        clock, tokens=8000, requests=2, reserve_tokens=400, reserve_requests=1
    )
    start = clock.now
    vectors = await provider.embed_documents(texts(96), token_counts=[500] * 96)

    assert len(vectors) == 96
    batch_tokens = [len(batch) * 500 for batch, _ in provider.calls]
    assert max(batch_tokens) <= 7600  # the document share of the budget
    assert len(provider.calls) == 7  # 15 chunks each, then 6
    window_ok(provider.times, batch_tokens, 8000, 2)
    # One batch per minute: six waits of a minute between the seven requests.
    assert clock.now - start == pytest.approx(6 * 60, abs=1)


async def test_every_attempt_counts_against_the_limit_including_retries():
    clock = FakeClock()
    provider = limited_provider(
        clock,
        requests=2,
        failures=[RetryableEmbeddingError("503"), RetryableEmbeddingError("503")],
    )
    start = clock.now
    await provider.embed_documents(texts(1), token_counts=[10])

    assert len(provider.times) == 3
    # Two requests per minute: the third attempt had to wait for the first to expire.
    assert provider.times[2] >= start + 60
    window_ok(provider.times, [10] * 3, 8000, 2)


async def test_retry_after_on_429_is_honoured_through_the_limiter():
    clock = FakeClock()
    provider = limited_provider(
        clock, failures=[RetryableEmbeddingError("429", retry_after=30, rate_limited=True)]
    )
    await provider.embed_documents(texts(1), token_counts=[10])
    first, second = provider.times
    assert 30 <= second - first <= 30 * 1.25 + 0.1  # Retry-After, plus jitter on top


async def test_429_without_retry_after_waits_at_least_the_minimum():
    clock = FakeClock()
    provider = limited_provider(
        clock,
        requests=10,
        failures=[RetryableEmbeddingError("429", rate_limited=True)],
        retry_base_seconds=2,  # only for other errors; a 429 never retries this fast
        min_retry_wait=20,
    )
    await provider.embed_documents(texts(1), token_counts=[10])
    first, second = provider.times
    assert 20 <= second - first <= 25.1


async def test_a_429_holds_other_callers_too():
    clock = FakeClock()
    provider = limited_provider(
        clock,
        requests=10,
        failures=[RetryableEmbeddingError("429", retry_after=40, rate_limited=True)],
    )
    await provider.embed_documents(texts(1), token_counts=[10])
    # The provider said "come back in 40 s": a different caller sharing the
    # limiter now (at the moment of the 429) would have to wait for that too.
    # Right at the moment of the 429, any caller (even a query, with the whole
    # budget free) would have to wait out the provider's 40 s, plus jitter.
    clock.now = provider.times[0]
    held_for = await provider.limiter.time_until_available(1, "query")
    assert 40 <= held_for <= 50


async def test_server_errors_do_not_hold_the_limiter():
    clock = FakeClock()
    provider = limited_provider(clock, requests=10, failures=[RetryableEmbeddingError("503")])
    await provider.embed_documents(texts(1), token_counts=[10])
    # A provider fault says nothing about our quota: nobody is held.
    assert await provider.limiter.time_until_available(1, "query") == 0
    assert len(provider.times) == 2  # but the retry was still sent, and counted


async def test_limiter_budget_caps_the_batch_size():
    clock = FakeClock()
    provider = limited_provider(clock, tokens=1000, requests=10, reserve_tokens=100)
    assert provider.token_budget == 900
    await provider.embed_documents(texts(4), token_counts=[400] * 4)
    assert sizes(provider) == [2, 2]


async def test_chunk_over_the_limiters_document_share_is_refused():
    clock = FakeClock()
    provider = limited_provider(clock, tokens=1000, requests=10, reserve_tokens=100)
    with pytest.raises(RequestTooLargeError):
        await provider.embed_documents(texts(1), token_counts=[950])
    assert provider.calls == []


# --- Logging ---


async def test_progress_is_logged_per_batch(caplog):
    caplog.set_level(logging.INFO, logger="app")
    clock = FakeClock()
    provider = limited_provider(
        clock, tokens=8000, requests=2, reserve_tokens=400, reserve_requests=1
    )
    await provider.embed_documents(texts(96), token_counts=[500] * 96)

    lines = [r.getMessage() for r in caplog.records if r.name == "app.services.embeddings.base"]
    assert lines[0].startswith("Embedding 96 chunks (~48000 tokens) in 7 batches")
    assert "about 6 min" in lines[0]
    progress = [line for line in lines if line.startswith("embedded ")]
    assert len(progress) == 7
    assert progress[0].startswith("embedded 15/96 chunks (batch 1/7: ~7500 tokens estimated")
    assert "15 reported by the provider" in progress[0]  # ScriptedProvider reports 1 per text
    assert "waited 0.0s" in progress[0]  # the first batch goes straight out
    assert "waited 60.0s" in progress[1]  # later ones wait for the minute to pass
    assert progress[-1].startswith("embedded 96/96 chunks (batch 7/7")
    assert lines[-1].startswith("Embedding finished: 96 chunks, 7 requests")
    assert "tokens ~48000 estimated" in lines[-1]


async def test_estimated_and_actual_tokens_are_compared_per_batch(caplog):
    caplog.set_level(logging.INFO, logger="app")
    provider = ScriptedProvider(
        token_budget=1000,
        batch_size=2,
        reported_tokens=lambda batch: 550 * len(batch),  # 10% more than our 500 each
    )
    await provider.embed_documents(texts(4), token_counts=[500] * 4)
    assert "~1000 tokens estimated, 1100 reported by the provider (+10%)" in caplog.text
    assert "tokens ~2000 estimated, 2200 reported by the provider (+10%)" in caplog.text


async def test_warns_when_the_provider_counts_more_than_the_batch_budget(caplog):
    caplog.set_level(logging.INFO, logger="app")
    provider = ScriptedProvider(token_budget=1000, reported_tokens=lambda batch: 1300)
    await provider.embed_documents(texts(2), token_counts=[450, 450])
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "counted 1300 tokens for a batch estimated at 900" in w and "EMBEDDING_RATE_SAFETY" in w
        for w in warnings
    )


async def test_missing_provider_count_is_said_plainly(caplog):
    caplog.set_level(logging.INFO, logger="app")
    provider = ScriptedProvider(token_budget=1000, reported_tokens=lambda batch: 0)
    await provider.embed_documents(texts(1), token_counts=[100])
    assert "no count reported by the provider" in caplog.text


async def test_provider_count_corrects_the_limiter_and_delays_the_next_batch():
    clock = FakeClock()
    counts = {"first": 900, "second": 400}  # what the provider says each used
    provider = limited_provider(
        clock,
        tokens=1000,
        requests=10,
        batch_size=1,
        reported_tokens=lambda batch: sum(counts[t] for t in batch),
    )
    await provider.embed_documents(["first", "second"], token_counts=[500, 400])
    # Estimated 500 + 400 would fit in the minute; the provider's 900 + 400 doesn't.
    first, second = provider.times
    assert second - first == pytest.approx(60, abs=0.1)


async def test_retry_warnings_are_logged_without_secrets(caplog):
    caplog.set_level(logging.INFO, logger="app")
    provider = ScriptedProvider(failures=[RetryableEmbeddingError("Embedding service timed out")])
    await provider.embed_documents(texts(1))
    assert "retry 1/6" in caplog.text
    assert "Embedding service timed out" in caplog.text


async def test_retry_log_reports_the_real_wait_not_just_the_backoff(caplog):
    caplog.set_level(logging.INFO, logger="app")
    clock = FakeClock()
    # Documents may send one request per minute. After a 429 the backoff alone
    # might say "2 s", but the next attempt really goes out a minute later.
    provider = limited_provider(
        clock,
        requests=2,
        reserve_requests=1,
        failures=[RetryableEmbeddingError("rate limited", rate_limited=True)],
        retry_base_seconds=2,
    )
    await provider.embed_documents(texts(1), token_counts=[10])

    first, second = provider.times
    assert second - first == pytest.approx(60, abs=0.1)
    assert "retry 1/6 in about 60s (rate-limit window)" in caplog.text


async def test_heartbeat_is_awaited_before_every_attempt_including_retries():
    beats = []

    async def heartbeat():
        beats.append(len(provider.calls))  # how many requests had been sent by then

    provider = ScriptedProvider(batch_size=1, failures=[RetryableEmbeddingError("503")])
    await provider.embed_documents(texts(2), heartbeat=heartbeat)
    # attempt 1 (fails), attempt 2 (retry, ok), attempt 3 (second batch)
    assert beats == [0, 1, 2]


# --- Queries never wait for long ---


async def test_query_waits_a_short_while_for_capacity():
    clock = FakeClock()
    provider = limited_provider(clock, requests=1, query_max_wait=20)
    await provider.embed_documents(texts(1), token_counts=[10])
    clock.now += 45  # the one request slot frees up in 15 s

    before = clock.now
    vector = await provider.embed_query("where is it?")
    assert len(vector) == DIMENSIONS
    assert clock.now - before == pytest.approx(15, abs=0.1)


async def test_query_fails_fast_instead_of_waiting_too_long():
    clock = FakeClock()
    provider = limited_provider(clock, requests=1, query_max_wait=20)
    await provider.embed_documents(texts(1), token_counts=[10])
    clock.now += 5  # 55 s until a slot frees up

    before = clock.now
    with pytest.raises(EmbeddingBusyError, match="retry in about 55 seconds") as info:
        await provider.embed_query("where is it?")
    assert info.value.retry_after == pytest.approx(55, abs=0.1)
    assert clock.now == before  # it didn't wait
    assert len(provider.calls) == 1  # and sent nothing


async def test_query_gets_through_while_a_document_uses_its_full_share():
    clock = FakeClock()
    provider = limited_provider(
        clock, tokens=8000, requests=2, reserve_tokens=400, reserve_requests=1, query_max_wait=20
    )
    await provider.embed_documents(texts(1), token_counts=[7600])  # the whole document share

    before = clock.now
    await provider.embed_query("still works")
    assert clock.now == before  # the reserve kept room, so no waiting at all


async def test_query_is_held_up_while_the_provider_says_to_back_off():
    clock = FakeClock()
    provider = limited_provider(
        clock,
        requests=10,
        failures=[RetryableEmbeddingError("429", retry_after=40, rate_limited=True)],
        query_max_wait=20,
    )
    document = asyncio.create_task(provider.embed_documents(texts(1), token_counts=[10]))
    await asyncio.sleep(0)  # the document gets its 429 and starts waiting out the 40 s

    with pytest.raises(EmbeddingBusyError) as info:
        await provider.embed_query("anyone there?")
    assert 39 < info.value.retry_after <= 50  # 40 s plus up to 25% jitter
    await document  # and the document itself does finish once the time has passed


async def test_query_retries_stop_at_the_deadline():
    clock = FakeClock()
    provider = limited_provider(
        clock,
        requests=100,
        failures=[RetryableEmbeddingError("503")] * 5,
        retry_base_seconds=8,
        query_max_wait=10,
    )
    # First retry waits 4-8 s (fits); the second waits at least 8 s more, which
    # would pass the 10 s deadline.
    with pytest.raises(EmbeddingBusyError):
        await provider.embed_query("flaky")
    assert len(provider.calls) == 2
    assert clock.now - 1000.0 <= 10


async def test_documents_have_no_deadline():
    clock = FakeClock()
    provider = limited_provider(clock, requests=1, query_max_wait=1)
    await provider.embed_documents(texts(1), token_counts=[10])
    await provider.embed_documents(texts(1), token_counts=[10])  # waits a full minute: fine
    assert len(provider.calls) == 2


# --- Response checks ---


class FixedOutput(ScriptedProvider):
    """Returns whatever vectors it's given, regardless of its settings."""

    def __init__(self, vectors, **kwargs):
        super().__init__(**kwargs)
        self.vectors = vectors

    async def _embed_batch(self, texts, input_type):
        return BatchResult(self.vectors, 0)


async def test_wrong_dimension_from_provider_is_an_error():
    provider = FixedOutput([[0.1] * 1024], dimensions=512)
    with pytest.raises(EmbeddingError, match="wrong size"):
        await provider.embed_documents(texts(1))


async def test_wrong_vector_count_from_provider_is_an_error():
    provider = FixedOutput([[0.1] * DIMENSIONS], dimensions=DIMENSIONS)
    with pytest.raises(EmbeddingError, match="unexpected response"):
        await provider.embed_documents(texts(2))
