"""The LLM provider base class: rate limiting, retries, failing fast."""

import pytest

from app.services.embeddings.rate_limit import SlidingWindowRateLimiter
from app.services.llm import (
    FakeLLMProvider,
    LLMBusyError,
    LLMError,
    RetryableLLMError,
)
from tests.embedding_fakes import FakeClock

TIMEOUT = RetryableLLMError("The answer service timed out")


def rate_limited(retry_after=None):
    return RetryableLLMError("rate limited", retry_after=retry_after, rate_limited=True)


def limited(clock: FakeClock, requests: int = 10, **kwargs) -> FakeLLMProvider:
    limiter = SlidingWindowRateLimiter(10**9, requests, clock=clock.clock, sleep=clock.sleep)
    return FakeLLMProvider(limiter=limiter, sleep=clock.sleep, **kwargs)


async def test_fake_is_deterministic_and_records_calls():
    llm = FakeLLMProvider(answer="Paris [1].")
    result = await llm.generate("system rules", "the question", 600)
    assert result.text == "Paris [1]."
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    assert llm.calls == [("system rules", "the question", 600)]


async def test_answer_can_depend_on_the_prompt():
    llm = FakeLLMProvider(answer=lambda system, user: f"saw {len(user)} chars")
    assert (await llm.generate("s", "hello", 10)).text == "saw 5 chars"


async def test_transient_failure_is_retried_with_backoff():
    clock = FakeClock()
    llm = FakeLLMProvider(failures=[TIMEOUT], sleep=clock.sleep)
    result = await llm.generate("s", "u", 10)
    assert result.text
    assert len(llm.calls) == 2
    assert 0.5 <= clock.sleeps[0] <= 1.0  # base 1 s, with jitter


async def test_permanent_failure_is_not_retried():
    clock = FakeClock()
    llm = FakeLLMProvider(failures=[LLMError("The answer service rejected the API key")])
    with pytest.raises(LLMError, match="API key"):
        await llm.generate("s", "u", 10)
    assert len(llm.calls) == 1
    assert clock.sleeps == []


async def test_gives_up_after_max_retries():
    llm = FakeLLMProvider(failures=[TIMEOUT] * 5, max_retries=2, sleep=FakeClock().sleep)
    with pytest.raises(RetryableLLMError, match="timed out"):
        await llm.generate("s", "u", 10)
    assert len(llm.calls) == 3


async def test_retry_after_is_honoured_through_the_limiter():
    clock = FakeClock()
    llm = limited(clock, failures=[rate_limited(retry_after=5)])
    start = clock.now
    await llm.generate("s", "u", 10)
    assert len(llm.calls) == 2
    assert clock.now - start >= 5


async def test_long_retry_after_fails_fast_with_a_retry_hint():
    clock = FakeClock()
    llm = limited(clock, failures=[rate_limited(retry_after=45)])
    with pytest.raises(LLMBusyError, match="retry in about 45 seconds") as info:
        await llm.generate("s", "u", 10)
    assert info.value.retry_after == 45
    assert len(llm.calls) == 1  # no second request
    assert clock.sleeps == []  # and no waiting


async def test_rate_limit_used_up_fails_fast_without_calling_the_provider():
    clock = FakeClock()
    llm = limited(clock, requests=1)
    await llm.generate("s", "u", 10)
    clock.now += 5  # 55 s until the next request is allowed

    with pytest.raises(LLMBusyError) as info:
        await llm.generate("s", "u", 10)
    assert info.value.retry_after == pytest.approx(55, abs=0.1)
    assert len(llm.calls) == 1


async def test_repeated_rate_limiting_ends_as_busy_not_as_an_outage():
    clock = FakeClock()
    llm = limited(clock, failures=[rate_limited()] * 5, max_retries=2)
    with pytest.raises(LLMBusyError):
        await llm.generate("s", "u", 10)
    assert len(llm.calls) == 3


def test_backoff_is_capped_and_honours_retry_after():
    llm = FakeLLMProvider(retry_base_seconds=1, retry_max_seconds=10)
    assert all(llm._delay(attempt, None) <= 10 for attempt in range(20))
    assert llm._delay(0, 7) == 7
