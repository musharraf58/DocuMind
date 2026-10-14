import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import get_settings
from app.services.embeddings import (
    EmbeddingError,
    FakeEmbeddingProvider,
    RetryableEmbeddingError,
    VoyageEmbeddingProvider,
    get_embedding_provider,
)
from tests.embedding_fakes import DIMENSIONS, RecordingSleep

API_KEY = "pa-test-SECRET-key-123"


def voyage(handler, api_key: str = API_KEY, **kwargs) -> VoyageEmbeddingProvider:
    kwargs.setdefault("max_retries", 2)
    return VoyageEmbeddingProvider(
        api_key=SecretStr(api_key),
        transport=httpx.MockTransport(handler),
        model="voyage-4-lite",
        dimensions=DIMENSIONS,
        sleep=RecordingSleep(),
        **kwargs,
    )


def success(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    # Voyage gives each embedding an index; return them out of order to prove
    # we sort by it rather than trusting the order.
    data = [
        {"object": "embedding", "embedding": [i / 10] * DIMENSIONS, "index": i}
        for i in range(len(texts))
    ]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": data[::-1],
            "model": "voyage-4-lite",
            "usage": {"total_tokens": 5 * len(texts)},
        },
    )


async def test_sends_documented_request_and_parses_response():
    seen = []

    def handler(request):
        seen.append(request)
        return success(request)

    vectors = await voyage(handler).embed_documents(["a", "b", "c"])

    [request] = seen
    assert request.url == "https://api.voyageai.com/v1/embeddings"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(request.content) == {
        "input": ["a", "b", "c"],
        "model": "voyage-4-lite",
        "input_type": "document",
        "output_dimension": DIMENSIONS,
        "truncation": False,
    }
    assert [v[0] for v in vectors] == [0.0, 0.1, 0.2]  # sorted by index


async def test_query_is_sent_with_query_input_type():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return success(request)

    await voyage(handler).embed_query("how do I deploy?")
    assert seen[0]["input_type"] == "query"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_retryable_statuses_are_retried(status):
    responses = iter([httpx.Response(status), httpx.Response(status)])

    def handler(request):
        return next(responses, None) or success(request)

    provider = voyage(handler)
    assert len(await provider.embed_documents(["x"])) == 1
    assert len(provider._sleep.delays) == 2


async def test_timeouts_and_connection_errors_are_retried():
    errors = iter([httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])

    def handler(request):
        error = next(errors, None)
        if error:
            raise error
        return success(request)

    provider = voyage(handler)
    assert len(await provider.embed_documents(["x"])) == 1
    assert len(provider._sleep.delays) == 2


async def test_gives_up_with_safe_message_after_retries():
    provider = voyage(lambda request: httpx.Response(503, text="upstream exploded"))
    with pytest.raises(RetryableEmbeddingError) as info:
        await provider.embed_documents(["x"])
    assert str(info.value) == "Embedding service is unavailable; try again later"


async def test_retry_after_header_is_used():
    responses = iter([httpx.Response(429, headers={"Retry-After": "3"})])
    provider = voyage(lambda request: next(responses, None) or success(request))
    await provider.embed_documents(["x"])
    assert provider._sleep.delays == [3]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "Embedding service rejected the input"),
        (401, "Embedding service rejected the API key"),
        (403, "Embedding service refused the request"),
        (418, "Embedding request failed (HTTP 418)"),
    ],
)
async def test_non_retryable_statuses_fail_at_once_with_safe_message(status, message):
    calls = []

    def handler(request):
        calls.append(request)
        # A hostile or chatty body: must not leak into our message.
        return httpx.Response(status, json={"detail": f"bad key {API_KEY}; input was: secret"})

    provider = voyage(handler)
    with pytest.raises(EmbeddingError) as info:
        await provider.embed_documents(["x"])
    assert not isinstance(info.value, RetryableEmbeddingError)
    assert str(info.value) == message
    assert API_KEY not in str(info.value)
    assert len(calls) == 1
    assert provider._sleep.delays == []


async def test_missing_api_key_fails_without_a_request():
    calls = []
    provider = voyage(lambda request: calls.append(request) or success(request), api_key="")
    with pytest.raises(EmbeddingError, match="API key is not configured"):
        await provider.embed_documents(["x"])
    assert calls == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"data": [{"no_embedding": True}]}),
        httpx.Response(200, json={"something": "else"}),
    ],
)
async def test_malformed_response_is_a_safe_error(response):
    provider = voyage(lambda request: response)
    with pytest.raises(EmbeddingError, match="unexpected response"):
        await provider.embed_documents(["x"])


def test_token_limit_follows_model():
    lite = VoyageEmbeddingProvider(api_key=SecretStr("k"), model="voyage-4-lite", dimensions=8)
    large = VoyageEmbeddingProvider(api_key=SecretStr("k"), model="voyage-4-large", dimensions=8)
    assert lite.max_batch_tokens == 1_000_000
    assert large.max_batch_tokens == 120_000


def test_api_key_is_hidden_when_printed():
    provider = voyage(success)
    assert API_KEY not in repr(provider._api_key)
    assert API_KEY not in str(get_settings().EMBEDDING_API_KEY)


def test_factory_selects_provider_from_settings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 7)
    provider = get_embedding_provider()
    assert isinstance(provider, VoyageEmbeddingProvider)
    assert (provider.model, provider.dimensions, provider.batch_size) == ("voyage-4-lite", 1024, 7)

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "fake")
    fake = get_embedding_provider()
    assert isinstance(fake, FakeEmbeddingProvider)
    # Never labelled with the real model's name, or fake vectors stored during
    # development would be searched alongside real ones.
    assert fake.model == "fake-sha256" != settings.EMBEDDING_MODEL


def test_dimension_setting_must_match_database_column(monkeypatch):
    monkeypatch.setattr(get_settings(), "EMBEDDING_DIMENSIONS", 512)
    with pytest.raises(EmbeddingError, match="don't match the database"):
        get_embedding_provider()
