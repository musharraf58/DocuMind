"""The Gemini provider against a simulated Gemini server (no network)."""

import json
import logging
from datetime import datetime

import fakeredis
import httpx
import pytest
from pydantic import SecretStr

from app.core.config import get_settings
from app.services.llm import (
    FakeLLMProvider,
    GeminiLLMProvider,
    LLMBusyError,
    LLMError,
    LLMRefusalError,
    RetryableLLMError,
    get_llm_provider,
)
from app.services.llm.gemini import PACIFIC, _seconds_until_pacific_midnight
from tests.embedding_fakes import RecordingSleep

KEY = "AIza-test-SECRET-key-123"


def gemini(handler, api_key: str = KEY, **kwargs) -> GeminiLLMProvider:
    kwargs.setdefault("sleep", RecordingSleep())
    return GeminiLLMProvider(
        api_key=SecretStr(api_key),
        transport=httpx.MockTransport(handler),
        model="gemini-3.5-flash-lite",
        **kwargs,
    )


def answer(text="The answer [1].", **extra) -> httpx.Response:
    body = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "thinking...", "thought": True}, {"text": text}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 12,
            "totalTokenCount": 162,
        },
    }
    body.update(extra)
    return httpx.Response(200, json=body)


def error(status: int, google_status: str, message: str = "", details=None, headers=None):
    payload = {"error": {"code": status, "status": google_status, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return httpx.Response(status, json=payload, headers=headers or {})


async def test_sends_the_documented_request_with_the_key_in_a_header():
    seen = []

    def handler(request):
        seen.append(request)
        return answer()

    result = await gemini(handler, temperature=1.0).generate(
        "Rules here.", "Sources + question", 600
    )

    [request] = seen
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash-lite:generateContent"
    )
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)  # never in the URL, which gets logged
    assert json.loads(request.content) == {
        "systemInstruction": {"parts": [{"text": "Rules here."}]},
        "contents": [{"role": "user", "parts": [{"text": "Sources + question"}]}],
        "generationConfig": {"maxOutputTokens": 600, "temperature": 1.0},
    }
    assert result.text == "The answer [1]."  # thought parts left out
    assert (result.prompt_tokens, result.completion_tokens) == (120, 42)  # thinking counts


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_server_errors_are_retried(status):
    responses = iter([error(status, "UNAVAILABLE")])
    llm = gemini(lambda r: next(responses, None) or answer())
    assert (await llm.generate("s", "u", 10)).text == "The answer [1]."
    assert len(llm._sleep.delays) == 1


async def test_timeouts_are_retried():
    errors = iter([httpx.ReadTimeout("slow")])

    def handler(request):
        exc = next(errors, None)
        if exc:
            raise exc
        return answer()

    llm = gemini(handler)
    assert (await llm.generate("s", "u", 10)).text
    assert len(llm._sleep.delays) == 1


async def test_429_uses_retry_after_header():
    responses = iter([error(429, "RESOURCE_EXHAUSTED", headers={"Retry-After": "3"})])
    llm = gemini(lambda r: next(responses, None) or answer())
    await llm.generate("s", "u", 10)
    assert llm._sleep.delays == [3]


async def test_429_uses_googles_retry_info_detail():
    retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "4s"}]
    responses = iter([error(429, "RESOURCE_EXHAUSTED", details=retry_info)])
    llm = gemini(lambda r: next(responses, None) or answer())
    await llm.generate("s", "u", 10)
    assert llm._sleep.delays == [4]


async def test_429_with_a_long_wait_fails_fast_as_busy():
    retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "40s"}]
    calls = []

    def handler(request):
        calls.append(request)
        return error(429, "RESOURCE_EXHAUSTED", details=retry_info)

    with pytest.raises(LLMBusyError) as info:
        await gemini(handler).generate("s", "u", 10)
    assert info.value.retry_after == 40
    assert len(calls) == 1


async def test_daily_quota_exhausted_says_when_it_resets():
    quota = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
        }
    ]
    calls = []

    def handler(request):
        calls.append(request)
        return error(429, "RESOURCE_EXHAUSTED", details=quota)

    with pytest.raises(LLMBusyError, match="daily limit is used up; it resets at midnight") as e:
        await gemini(handler).generate("s", "u", 10)
    assert e.value.retry_after > 0
    assert len(calls) == 1  # retrying before midnight would be pointless


def test_seconds_until_pacific_midnight():
    eleven_pm = datetime(2026, 10, 19, 23, 0, tzinfo=PACIFIC)
    assert _seconds_until_pacific_midnight(eleven_pm) == 3600


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "The answer service rejected the request"),
        (401, "The answer service rejected the API key"),
        (403, "The answer service rejected the API key"),
        (404, "The answer model was not found; check LLM_MODEL"),
        (418, "Answer request failed (HTTP 418)"),
    ],
)
async def test_permanent_errors_fail_at_once_with_safe_messages(status, message, caplog):
    caplog.set_level(logging.DEBUG)
    calls = []

    def handler(request):
        calls.append(request)
        return error(status, "INVALID_ARGUMENT", message=f"bad key {KEY}; prompt: secret text")

    llm = gemini(handler)
    with pytest.raises(LLMError) as info:
        await llm.generate("s", "secret text", 10)
    assert not isinstance(info.value, RetryableLLMError)
    assert str(info.value) == message
    assert len(calls) == 1
    for leak in (KEY, "secret text"):
        assert leak not in str(info.value) and leak not in caplog.text


async def test_missing_key_fails_without_a_request():
    calls = []
    with pytest.raises(LLMError, match="API key is not configured"):
        await gemini(lambda r: calls.append(r) or answer(), api_key="").generate("s", "u", 10)
    assert calls == []


async def test_blocked_prompt_is_a_refusal():
    blocked = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMRefusalError):
        await gemini(lambda r: blocked).generate("s", "u", 10)


async def test_safety_stop_with_no_text_is_a_refusal():
    body = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
    with pytest.raises(LLMRefusalError):
        await gemini(lambda r: httpx.Response(200, json=body)).generate("s", "u", 10)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"candidates": []}),
        httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "  "}]}}]}),
    ],
)
async def test_unusable_responses_are_safe_errors(response):
    with pytest.raises(LLMError, match="unexpected response|empty answer"):
        await gemini(lambda r: response).generate("s", "u", 10)


async def test_truncated_answer_is_still_returned(caplog):
    cut = answer("Partial answer [1]")
    body = json.loads(cut.content)
    body["candidates"][0]["finishReason"] = "MAX_TOKENS"
    result = await gemini(lambda r: httpx.Response(200, json=body)).generate("s", "u", 10)
    assert result.text == "Partial answer [1]"
    assert "cut off" in caplog.text


def test_factory_selects_provider(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    provider = get_llm_provider(fakeredis.aioredis.FakeRedis())
    assert isinstance(provider, GeminiLLMProvider)
    assert provider.model == "gemini-3.5-flash-lite"
    assert provider.limiter.request_budget == settings.LLM_RPM_LIMIT
    assert KEY not in repr(provider._api_key)

    monkeypatch.setattr(settings, "LLM_PROVIDER", "fake")
    assert isinstance(get_llm_provider(), FakeLLMProvider)
