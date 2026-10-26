"""Lite mode never puts a secret in a log line or an error body.

One run through lite mode (inline jobs, S3 storage, invite-only
registration, the REAL Voyage and Gemini clients against simulated servers)
with a distinctive value in every secret, through the paths most likely to
leak one: wrong invite codes, providers rejecting their keys, the storage
bucket failing, Redis and Postgres unreachable. Every log record (at DEBUG,
from every logger) and every response body is then searched for every
secret.
"""

import asyncio
import json
import logging
from functools import partial

import boto3
import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.database import make_engine
from app.core.rate_limit import outage_log
from app.core.redis import get_redis
from app.main import app
from app.services import background, embeddings, llm
from app.services.background import InlineRuntime
from app.services.queue import get_queue
from tests.s3_fakes import ACCESS_KEY, BUCKET
from tests.s3_fakes import SECRET_KEY as S3_SECRET

SECRETS = {
    "SECRET_KEY": "jwt-signing-SECRET-0123456789-abcdefghijklmnop",
    "INVITE_CODE": "INVITE-secret-code-42",
    "EMBEDDING_API_KEY": "voyage-SECRET-key-987654",
    "LLM_API_KEY": "gemini-SECRET-key-123456",
    "S3_SECRET_ACCESS_KEY": S3_SECRET,
    "S3_ACCESS_KEY_ID": ACCESS_KEY,
    "DB_PASSWORD": "db-SECRET-password-555",
    "REDIS_PASSWORD": "redis-SECRET-password-777",
}


class Everything(logging.Handler):
    """Every record from every logger, formatted with its arguments."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        line = record.getMessage()
        if record.exc_info:
            line += logging.Formatter().formatException(record.exc_info)
        self.lines.append(line)


@pytest.fixture
def logs():
    handler = Everything()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    outage_log._last = None  # don't let an earlier test's outage message be throttled
    yield handler
    root.removeHandler(handler)
    root.setLevel(old_level)


def voyage_server(state, request: httpx.Request) -> httpx.Response:
    if state["reject"]:
        return httpx.Response(401, json={"detail": "Provided API key is invalid."})
    texts = json.loads(request.content)["input"]
    dims = get_settings().EMBEDDING_DIMENSIONS
    data = [{"embedding": [(i + 1) / 10] * dims, "index": i} for i in range(len(texts))]
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 5 * len(texts)}})


def gemini_server(state, request: httpx.Request) -> httpx.Response:
    if state["reject"]:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "API key not valid.",
                }
            },
        )
    answer = {
        "candidates": [
            {"content": {"parts": [{"text": "Cats sleep a lot [1]."}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }
    if "streamGenerateContent" in request.url.path:
        body = f"data: {json.dumps(answer)}\r\n\r\n".encode()
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=answer)


@pytest.fixture
async def lite(client, test_engine, s3, monkeypatch):
    """The app in lite mode with a secret in every slot."""
    settings = get_settings()
    for name, value in {
        "INGESTION_MODE": "inline",
        "REGISTRATION_MODE": "invite",
        "INVITE_CODE": SecretStr(SECRETS["INVITE_CODE"]),
        "SECRET_KEY": SECRETS["SECRET_KEY"],
        "EMBEDDING_PROVIDER": "voyage",
        "EMBEDDING_API_KEY": SecretStr(SECRETS["EMBEDDING_API_KEY"]),
        "EMBEDDING_RPM_LIMIT": 1000,
        "EMBEDDING_INGEST_RPM_SHARE": 1,
        "LLM_PROVIDER": "gemini",
        "LLM_API_KEY": SecretStr(SECRETS["LLM_API_KEY"]),
        "INLINE_SHUTDOWN_GRACE_SECONDS": 0.2,
    }.items():
        monkeypatch.setattr(settings, name, value)
    redis = app.dependency_overrides[get_redis]()
    state = {"reject": False}
    # The real clients; only their HTTP goes to the simulated servers.
    monkeypatch.setattr(
        embeddings,
        "VoyageEmbeddingProvider",
        partial(
            embeddings.VoyageEmbeddingProvider,
            transport=httpx.MockTransport(partial(voyage_server, state)),
        ),
    )
    monkeypatch.setattr(
        llm,
        "GeminiLLMProvider",
        partial(
            llm.GeminiLLMProvider, transport=httpx.MockTransport(partial(gemini_server, state))
        ),
    )
    monkeypatch.setattr(embeddings, "get_redis", lambda: redis)
    monkeypatch.setattr(llm, "get_redis", lambda: redis)
    from app.api.deps import embedding_provider, llm_provider

    app.dependency_overrides.pop(embedding_provider, None)
    app.dependency_overrides.pop(llm_provider, None)

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    runtime = InlineRuntime(settings, sessions, redis)
    background._runtime = runtime
    app.dependency_overrides[get_queue] = lambda: runtime.queue
    runtime.start()
    yield state
    await runtime.stop()
    background._runtime = None


async def wait_for_status(call, document_id, status, headers) -> None:
    async def current() -> str:
        return (await call("GET", f"/documents/{document_id}", headers=headers)).json()["status"]

    async with asyncio.timeout(10):
        while await current() != status:  # noqa: ASYNC110 (polls the API)
            await asyncio.sleep(0.05)


def assert_no_secret(where: str, texts) -> None:
    blob = "\n".join(texts)
    for name, value in SECRETS.items():
        assert value not in blob, f"{name} leaked into {where}"


async def test_no_secret_in_any_log_or_response(client, lite, logs, db):
    bodies: list[str] = []

    async def call(method, url, **kwargs):
        response = await client.request(method, url, **kwargs)
        bodies.append(response.text)
        return response

    # Registration: a wrong invite code, then the right one.
    wrong = await call(
        "POST",
        "/auth/register",
        json={"email": "lite@example.com", "password": "lite-password-123", "invite_code": "nope"},
    )
    assert wrong.status_code == 403
    ok = await call(
        "POST",
        "/auth/register",
        json={
            "email": "lite@example.com",
            "password": "lite-password-123",
            "invite_code": SECRETS["INVITE_CODE"],
        },
    )
    assert ok.status_code == 201
    # A rejected request echoing a secret back must not happen either.
    invalid = await call(
        "POST",
        "/auth/register",
        json={"email": "not-an-email", "password": SECRETS["SECRET_KEY"]},
    )
    assert invalid.status_code == 422
    login = await call(
        "POST",
        "/auth/login",
        data={"username": "lite@example.com", "password": "lite-password-123"},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    collection_id = (
        await call("POST", "/collections", json={"name": "c"}, headers=headers)
    ).json()["id"]
    base = f"/collections/{collection_id}"
    upload = await call(
        "POST",
        f"{base}/documents",
        files={"file": ("cats.txt", b"Cats sleep a lot. " * 80)},
        headers=headers,
    )
    assert upload.status_code == 202
    document_id = upload.json()["id"]
    await wait_for_status(call, document_id, "ready", headers)

    question = {"question": "Do cats sleep?"}
    assert (await call("POST", f"{base}/ask", json=question, headers=headers)).status_code == 200

    # The providers reject their keys.
    lite["reject"] = True
    rejected_llm = await call("POST", f"{base}/ask", json={"question": "Again?"}, headers=headers)
    assert rejected_llm.status_code == 503
    rejected_embed = await call("POST", f"{base}/search", json={"query": "new"}, headers=headers)
    assert rejected_embed.status_code == 503
    second = await call(
        "POST",
        f"{base}/documents",
        files={"file": ("dogs.txt", b"Dogs bark. " * 80)},
        headers=headers,
    )
    second_id = second.json()["id"]
    await wait_for_status(call, second_id, "failed", headers)
    lite["reject"] = False

    # The bucket fails.
    s3 = boto3.client("s3", region_name="us-east-1")
    for item in s3.list_objects_v2(Bucket=BUCKET).get("Contents", []):
        s3.delete_object(Bucket=BUCKET, Key=item["Key"])
    s3.delete_bucket(Bucket=BUCKET)
    outage = await call(
        "POST", f"{base}/documents", files={"file": ("x.txt", b"text " * 80)}, headers=headers
    )
    assert outage.status_code == 503
    download = await call("GET", f"/documents/{document_id}/download", headers=headers)
    assert download.status_code in (404, 503)

    # Redis unreachable, with a password in its URL.
    from redis.asyncio import Redis

    down = Redis.from_url(
        f"redis://:{SECRETS['REDIS_PASSWORD']}@127.0.0.1:1/0",
        socket_timeout=0.2,
        socket_connect_timeout=0.2,
    )
    app.dependency_overrides[get_redis] = lambda: down
    assert (await call("GET", "/auth/me", headers=headers)).status_code == 200  # fails open
    assert (await call("GET", "/health/ready")).status_code == 503

    # Postgres unreachable, with a password in its URL.
    engine = make_engine(
        f"postgresql://documind:{SECRETS['DB_PASSWORD']}@127.0.0.1:1/x?sslmode=disable",
        poolclass=NullPool,
    )
    with pytest.raises(Exception) as caught:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    await engine.dispose()
    logging.getLogger("test").error("database failed", exc_info=caught.value)

    assert len(logs.lines) > 20  # the run really logged
    assert_no_secret("the logs", logs.lines)
    assert_no_secret("a response body", bodies)


def test_settings_never_show_secret_values():
    settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+asyncpg://u:p@h/db",
        REDIS_URL="redis://h",
        SECRET_KEY=SECRETS["SECRET_KEY"],
        INVITE_CODE=SECRETS["INVITE_CODE"],
        EMBEDDING_API_KEY=SECRETS["EMBEDDING_API_KEY"],
        LLM_API_KEY=SECRETS["LLM_API_KEY"],
        S3_ACCESS_KEY_ID=ACCESS_KEY,
        S3_SECRET_ACCESS_KEY=S3_SECRET,
        STORAGE_BACKEND="s3",
        S3_BUCKET="b",
    )
    shown = repr(settings) + str(settings)
    for name in ("INVITE_CODE", "EMBEDDING_API_KEY", "LLM_API_KEY", "S3_ACCESS_KEY_ID"):
        assert SECRETS[name] not in shown, name
    assert S3_SECRET not in shown
