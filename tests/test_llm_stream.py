"""LLMProvider.stream(): pieces then usage, rate limiting, retries only before
the first piece, and cancellation."""

import asyncio

import pytest

from app.services.embeddings.rate_limit import SlidingWindowRateLimiter
from app.services.llm import FakeLLMProvider, LLMBusyError, LLMError, LLMUsage, RetryableLLMError
from tests.embedding_fakes import FakeClock

TIMEOUT = RetryableLLMError("The answer service timed out")


async def collect(stream) -> tuple[str, LLMUsage | None]:
    text, usage = "", None
    async for piece in stream:
        if isinstance(piece, LLMUsage):
            usage = piece
        else:
            text += piece
    return text, usage


async def test_streams_pieces_then_usage():
    llm = FakeLLMProvider(chunks=["Cats ", "purr ", "[1]."])
    pieces = [p async for p in llm.stream("rules", "question", 600)]
    assert pieces[:3] == ["Cats ", "purr ", "[1]."]
    assert isinstance(pieces[-1], LLMUsage) and len(pieces) == 4
    assert pieces[-1].prompt_tokens > 0 and pieces[-1].completion_tokens > 0
    assert llm.calls == [("rules", "question", 600)]
    assert llm.streams_completed == 1


async def test_default_chunks_are_the_answer_word_by_word():
    llm = FakeLLMProvider(answer="One two three [1].")
    text, usage = await collect(llm.stream("s", "u", 10))
    assert text == "One two three [1]."
    assert usage is not None


async def test_chunk_delays_use_the_injected_clock():
    clock = FakeClock()
    llm = FakeLLMProvider(chunks=["a", "b", "c"], chunk_delay=5, chunk_sleep=clock.sleep)
    await collect(llm.stream("s", "u", 10))
    assert clock.sleeps == [5, 5, 5]


async def test_streaming_goes_through_the_rate_limiter_and_fails_fast():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(10**9, 1, clock=clock.clock, sleep=clock.sleep)
    llm = FakeLLMProvider(limiter=limiter, sleep=clock.sleep)
    await collect(llm.stream("s", "u", 10))
    clock.now += 10  # the only request slot frees up in 50 s

    with pytest.raises(LLMBusyError) as info:
        await collect(llm.stream("s", "u", 10))
    assert info.value.retry_after == pytest.approx(50, abs=0.1)
    assert len(llm.calls) == 1  # the provider was never called the second time


async def test_failure_before_the_first_piece_is_retried():
    clock = FakeClock()
    llm = FakeLLMProvider(chunks=["ok"], failures=[TIMEOUT], sleep=clock.sleep)
    text, _ = await collect(llm.stream("s", "u", 10))
    assert text == "ok"
    assert len(llm.calls) == 2


async def test_failure_after_text_was_sent_is_not_retried():
    llm = FakeLLMProvider(chunks=["Cats ", "purr"], fail_after=1, fail_with=TIMEOUT)
    received = []
    with pytest.raises(RetryableLLMError):
        async for piece in llm.stream("s", "u", 10):
            received.append(piece)
    assert received == ["Cats "]
    assert len(llm.calls) == 1  # one request: the text already sent can't be taken back


async def test_rate_limited_before_first_piece_ends_as_busy():
    llm = FakeLLMProvider(
        failures=[RetryableLLMError("rate limited", retry_after=60, rate_limited=True)]
    )
    with pytest.raises(LLMBusyError):
        await collect(llm.stream("s", "u", 10))


async def test_permanent_error_is_not_retried():
    llm = FakeLLMProvider(failures=[LLMError("The answer service rejected the API key")])
    with pytest.raises(LLMError, match="API key"):
        await collect(llm.stream("s", "u", 10))
    assert len(llm.calls) == 1


async def test_stopping_early_closes_the_provider_stream():
    llm = FakeLLMProvider(chunks=["a", "b", "c", "d"])
    stream = llm.stream("s", "u", 10)
    assert await anext(stream) == "a"
    await stream.aclose()
    assert llm.streams_cancelled == 1 and llm.streams_completed == 0


async def test_cancelling_the_consuming_task_cancels_the_provider_stream():
    llm = FakeLLMProvider(chunks=["a", "b", "c"], pause_after=1)

    async def consume():
        return await collect(llm.stream("s", "u", 10))

    task = asyncio.create_task(consume())
    await asyncio.wait_for(llm.paused.wait(), 5)  # mid-stream, after "a"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.streams_cancelled == 1 and llm.streams_completed == 0


async def test_generate_still_works_alongside_stream():
    llm = FakeLLMProvider(answer="Plain [1].")
    assert (await llm.generate("s", "u", 10)).text == "Plain [1]."
