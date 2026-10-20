"""Gemini streaming (streamGenerateContent?alt=sse) against a simulated server."""

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from app.services.llm import (
    GeminiLLMProvider,
    LLMBusyError,
    LLMError,
    LLMRefusalError,
    LLMUsage,
    RetryableLLMError,
)
from tests.embedding_fakes import RecordingSleep

KEY = "AIza-stream-SECRET-key"


def gemini(handler, **kwargs) -> GeminiLLMProvider:
    kwargs.setdefault("sleep", RecordingSleep())
    return GeminiLLMProvider(
        api_key=SecretStr(kwargs.pop("api_key", KEY)),
        transport=httpx.MockTransport(handler),
        model="gemini-3.5-flash-lite",
        **kwargs,
    )


def event(text=None, reason=None, usage=None, thought=None, **extra) -> dict:
    parts = []
    if thought:
        parts.append({"text": thought, "thought": True})
    if text is not None:
        parts.append({"text": text})
    data = {"candidates": [{"content": {"role": "model", "parts": parts}}]}
    if reason:
        data["candidates"][0]["finishReason"] = reason
    if usage:
        data["usageMetadata"] = usage
    data.update(extra)
    return data


def sse(*events) -> bytes:
    return "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events).encode()


def ok(*events) -> httpx.Response:
    return httpx.Response(200, content=sse(*events), headers={"content-type": "text/event-stream"})


USAGE = {"promptTokenCount": 100, "candidatesTokenCount": 7, "thoughtsTokenCount": 3}
THREE_EVENTS = (
    event("Cats ", thought="hmm"),
    event("purr "),
    event("[1].", reason="STOP", usage=USAGE),
)


async def collect(stream):
    pieces = [p async for p in stream]
    return pieces[:-1], pieces[-1]


async def test_streams_text_with_the_documented_request():
    seen = []

    def handler(request):
        seen.append(request)
        return ok(*THREE_EVENTS)

    texts, usage = await collect(gemini(handler).stream("Rules.", "Question", 600))

    [request] = seen
    assert request.url.path == "/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent"
    assert request.url.params["alt"] == "sse"
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)
    body = json.loads(request.content)
    assert body["systemInstruction"] == {"parts": [{"text": "Rules."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Question"}]}]
    assert texts == ["Cats ", "purr ", "[1]."]  # thought parts left out
    assert usage == LLMUsage(100, 10)  # thinking counts as completion


async def test_usage_on_every_event_keeps_the_last():
    early = {"promptTokenCount": 100, "candidatesTokenCount": 1}
    final = {"promptTokenCount": 100, "candidatesTokenCount": 9}
    _, usage = await collect(
        gemini(
            lambda r: ok(event("a", usage=early), event("b", reason="STOP", usage=final))
        ).stream("s", "u", 10)
    )
    assert usage == LLMUsage(100, 9)


async def test_error_before_the_stream_is_retried():
    responses = iter([httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}})])
    llm = gemini(lambda r: next(responses, None) or ok(*THREE_EVENTS))
    texts, _ = await collect(llm.stream("s", "u", 10))
    assert "".join(texts) == "Cats purr [1]."
    assert len(llm._sleep.delays) == 1


async def test_daily_quota_before_the_stream_fails_fast():
    quota = [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]
    response = httpx.Response(
        429, json={"error": {"status": "RESOURCE_EXHAUSTED", "details": quota}}
    )
    with pytest.raises(LLMBusyError, match="daily limit"):
        await collect(gemini(lambda r: response).stream("s", "u", 10))


async def test_permanent_error_has_a_safe_message():
    response = httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}})
    with pytest.raises(LLMError) as info:
        await collect(gemini(lambda r: response).stream("s", "u", 10))
    assert str(info.value) == "The answer service rejected the API key"
    assert KEY not in str(info.value)


async def test_blocked_prompt_is_a_refusal():
    blocked = ok({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMRefusalError):
        await collect(gemini(lambda r: blocked).stream("s", "u", 10))


async def test_safety_stop_without_text_is_a_refusal():
    with pytest.raises(LLMRefusalError):
        await collect(gemini(lambda r: ok(event(reason="SAFETY"))).stream("s", "u", 10))


async def test_malformed_event_is_a_safe_error():
    bad = httpx.Response(200, content=b"data: {not json\r\n\r\n")
    with pytest.raises(LLMError, match="unexpected response"):
        await collect(gemini(lambda r: bad).stream("s", "u", 10))


async def test_missing_key_fails_without_a_request():
    calls = []
    with pytest.raises(LLMError, match="API key is not configured"):
        await collect(gemini(lambda r: calls.append(r) or ok(), api_key="").stream("s", "u", 10))
    assert calls == []


class Upstream(httpx.AsyncByteStream):
    """A streaming body that pauses after its first event, and records
    whether the client closed the connection."""

    def __init__(self, fail_midway: bool = False) -> None:
        self.closed = False
        self.fail_midway = fail_midway
        self.release = asyncio.Event()

    async def __aiter__(self):
        yield sse(event("First "))
        if self.fail_midway:
            raise httpx.ReadError("connection reset")
        await self.release.wait()
        yield sse(event("second.", reason="STOP", usage=USAGE))

    async def aclose(self) -> None:
        self.closed = True


async def test_stopping_early_closes_the_connection_to_gemini():
    upstream = Upstream()
    llm = gemini(lambda r: httpx.Response(200, stream=upstream))
    stream = llm.stream("s", "u", 10)

    assert await anext(stream) == "First "
    await stream.aclose()  # e.g. the browser went away
    assert upstream.closed  # we hung up on Google: it stops generating


async def test_cancelled_task_closes_the_connection_to_gemini():
    upstream = Upstream()
    llm = gemini(lambda r: httpx.Response(200, stream=upstream))
    first_piece = asyncio.Event()

    async def consume():
        async for _ in llm.stream("s", "u", 10):
            first_piece.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_piece.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert upstream.closed


async def test_connection_lost_mid_stream_is_not_retried():
    upstream = Upstream(fail_midway=True)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=upstream)

    received = []
    with pytest.raises(RetryableLLMError):
        async for piece in gemini(handler).stream("s", "u", 10):
            received.append(piece)
    assert received == ["First "]
    assert len(calls) == 1
