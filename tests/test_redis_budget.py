"""How many Redis commands each kind of request costs (docs/HARDENING.md,
"Redis command budget"). Upstash's free tier allows 500,000 commands a
month, so these numbers decide how much traffic the free tier carries.

The providers here are the REAL Voyage and Gemini clients, with their real
Redis rate limiters and the real global LLM budget, talking to simulated
HTTP servers. Every command reaching the (fake) Redis is counted. The exact
counts are asserted, so a change that adds Redis traffic to a request shows
up here and in review, instead of on the bill.
"""

import json

import httpx
import pytest
from pydantic import SecretStr

from app.api.deps import embedding_provider, llm_provider
from app.core.config import get_settings
from app.core.redis import get_redis
from app.main import app
from app.services.embeddings import VoyageEmbeddingProvider, make_rate_limiter
from app.services.embeddings.redis_rate_limit import RedisRateLimiter
from app.services.llm import (
    LLM_RATE_LIMIT_KEY,
    UNLIMITED_TOKENS,
    GeminiLLMProvider,
    GlobalLLMBudget,
)
from tests.embedding_fakes import RecordingSleep
from tests.helpers import PASSWORD, create_collection, register
from tests.redis_counting import CountingRedis

DIMENSIONS = 1024


def voyage_server(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [
        {"object": "embedding", "embedding": [(i + 1) / 10] * DIMENSIONS, "index": i}
        for i in range(len(texts))
    ]
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 5 * len(texts)}})


ANSWER = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Cats sleep a lot [1]."}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10},
}


def gemini_server(request: httpx.Request) -> httpx.Response:
    if "streamGenerateContent" in request.url.path:
        body = f"data: {json.dumps(ANSWER)}\r\n\r\n".encode()
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=ANSWER)


@pytest.fixture
async def redis(client, monkeypatch):
    """Real providers over simulated HTTP, everything on one counting Redis."""
    counting = CountingRedis()
    settings = get_settings()
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSIONS", DIMENSIONS)
    # Counting commands, not testing throttling: let every request through.
    monkeypatch.setattr(settings, "EMBEDDING_RPM_LIMIT", 1000)
    monkeypatch.setattr(settings, "EMBEDDING_INGEST_RPM_SHARE", 1)
    voyage = VoyageEmbeddingProvider(
        api_key=SecretStr("voyage-test-key"),
        transport=httpx.MockTransport(voyage_server),
        model="voyage-4-lite",
        dimensions=DIMENSIONS,
        limiter=make_rate_limiter(counting),
        sleep=RecordingSleep(),
    )
    gemini = GeminiLLMProvider(
        api_key=SecretStr("gemini-test-key"),
        transport=httpx.MockTransport(gemini_server),
        model="gemini-3.5-flash-lite",
        limiter=RedisRateLimiter(
            counting, UNLIMITED_TOKENS, settings.LLM_RPM_LIMIT, key_prefix=LLM_RATE_LIMIT_KEY
        ),
        budget=GlobalLLMBudget(counting),
        sleep=RecordingSleep(),
    )
    app.dependency_overrides[get_redis] = lambda: counting
    app.dependency_overrides[embedding_provider] = lambda: voyage
    app.dependency_overrides[llm_provider] = lambda: gemini
    counting.voyage = voyage
    return counting


# Steady state (CountingRedis.steady): scripts are loaded once per Redis
# server, not per request.
EXPECTED = {
    "GET /health/live": [],
    "GET /health/ready": ["PING", "EXISTS"],  # EXISTS: Celery mode's worker heartbeat
    "POST /auth/register": ["EVALSHA:rate_limit.hit"],
    "POST /auth/login": ["EVALSHA:rate_limit.hit", "EVALSHA:rate_limit.hit", "HMGET"],
    "GET /auth/me": ["EVALSHA:rate_limit.hit"],
    "GET /usage": ["EVALSHA:rate_limit.hit", "MGET"],
    "POST /collections/{id}/documents": [
        "EVALSHA:rate_limit.hit",
        "GET",  # embedding tokens left today
        "EVALSHA:quota.take",  # daily uploads
    ],
    # Per document: the GET; per embedding batch: the try and the add.
    "job (1 document, 1 batch)": ["GET", "EVALSHA:outbound.try", "EVALSHA:quota.add"],
    "POST /collections/{id}/search": [
        "EVALSHA:rate_limit.hit",
        "GET",  # embedding tokens left today
        "EVALSHA:outbound.try",
        # Only when the provider counted more tokens than we estimated (it
        # usually does: a different tokenizer, plus its own prompt).
        "EVALSHA:outbound.correct",
        "EVALSHA:quota.add",
    ],
    "POST /collections/{id}/ask": [
        "EVALSHA:rate_limit.hit",
        "EVALSHA:quota.admit",
        "EVALSHA:outbound.try",  # the question's embedding
        "EVALSHA:outbound.correct",  # as for search
        "EVALSHA:quota.add",  # its tokens, for the daily quota
        "EVALSHA:quota.take",  # the global LLM budget
        "EVALSHA:outbound.try",  # the LLM rate limit
    ],
}
EXPECTED["POST /collections/{id}/ask/stream"] = EXPECTED["POST /collections/{id}/ask"]


async def measure_all(client, redis, job_queue) -> dict[str, list[str]]:
    rows = {}

    async def run(name, request, ok=True):
        redis.reset()
        response = await request
        if ok:
            assert response.status_code < 300, response.text
        rows[name] = redis.steady

    await run("GET /health/live", client.get("/health/live"))
    await run("GET /health/ready", client.get("/health/ready"), ok=False)
    email = "budget@example.com"
    await run(
        "POST /auth/register",
        client.post("/auth/register", json={"email": email, "password": PASSWORD}),
    )
    login = client.post("/auth/login", data={"username": email, "password": PASSWORD})
    redis.reset()
    response = await login
    rows["POST /auth/login"] = redis.steady
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    await run("GET /auth/me", client.get("/auth/me", headers=headers))
    await run("GET /usage", client.get("/usage", headers=headers))

    collection_id = await create_collection(client, headers)
    base = f"/collections/{collection_id}"
    text = b"Cats sleep a lot. They nap sixteen hours a day. " * 20
    await run(
        "POST /collections/{id}/documents",
        client.post(f"{base}/documents", files={"file": ("cats.txt", text)}, headers=headers),
    )
    redis.reset()
    await job_queue.run_all(redis.voyage)
    rows["job (1 document, 1 batch)"] = redis.steady

    await run(
        "POST /collections/{id}/search",
        client.post(f"{base}/search", json={"query": "cats"}, headers=headers),
    )
    question = {"question": "Do cats sleep?"}
    await run(
        "POST /collections/{id}/ask", client.post(f"{base}/ask", json=question, headers=headers)
    )
    redis.reset()
    async with client.stream("POST", f"{base}/ask/stream", json=question, headers=headers) as r:
        body = (await r.aread()).decode()
    assert "Cats sleep a lot" in body, body
    rows["POST /collections/{id}/ask/stream"] = redis.steady
    return rows


async def test_redis_commands_per_request(client, redis, job_queue, capsys):
    rows = await measure_all(client, redis, job_queue)
    with capsys.disabled():  # visible with -s: the table in docs/HARDENING.md
        for name, commands in rows.items():
            print(f"{name:36} {len(commands):2}  {', '.join(commands)}")
    assert rows == EXPECTED


async def test_failed_login_still_clears_its_record_on_success(client, redis):
    """Skipping the DEL is only for logins with no failures on record."""
    await register(client, "slip@example.com")
    await client.post("/auth/login", data={"username": "slip@example.com", "password": "wrong-pw"})
    redis.reset()
    response = await client.post(
        "/auth/login", data={"username": "slip@example.com", "password": PASSWORD}
    )
    assert response.status_code == 200
    assert "DEL" in redis.commands


def test_per_user_limit_classes_match_the_api():
    """Deleting a user's keys by name relies on this list being complete."""
    from app.api import limits
    from app.services.quotas import PER_USER_LIMIT_CLASSES

    per_user = {
        dependency.dependency.__name__.removeprefix("rate_limit_")
        for dependency in (
            limits.limit_default,
            limits.limit_ask,
            limits.limit_upload,
            limits.limit_account,
        )
    }
    assert set(PER_USER_LIMIT_CLASSES) == per_user
    by_ip = {limits.LimitClass.login, limits.LimitClass.register}
    assert set(limits.LimitClass) - by_ip == per_user


async def test_deleting_an_account_clears_its_keys_in_one_command(client, redis):
    from app.core import redis_keys

    await register(client, "bye@example.com")
    login = await client.post(
        "/auth/login", data={"username": "bye@example.com", "password": PASSWORD}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    me = (await client.get("/auth/me", headers=headers)).json()["id"]
    await client.get("/usage", headers=headers)
    redis.reset()
    response = await client.request(
        "DELETE", "/auth/me", json={"password": PASSWORD}, headers=headers
    )
    assert response.status_code == 204
    assert redis.steady == ["EVALSHA:rate_limit.hit", "DEL"]
    for pattern in redis_keys.user_key_patterns(me):
        assert [key async for key in redis.scan_iter(match=pattern)] == []
