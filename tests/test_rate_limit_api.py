"""Inbound rate limits through the real endpoints (fake Redis, fake clock)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core import clock
from app.core.config import get_settings
from app.core.redis import get_redis
from app.main import app
from tests.helpers import PASSWORD, auth_headers, create_collection, login, register, upload


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def fake_clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(clock, "utcnow", fake)
    return fake


def set_limit(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setattr(get_settings(), name, value)


class DownRedis:
    """A Redis that is unreachable: every command fails."""

    def register_script(self, script):
        async def run(*args, **kwargs):
            raise RedisConnectionError("Connection refused")

        return run

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise RedisConnectionError("Connection refused")

        return fail


# --- per-user limits ------------------------------------------------------------


async def test_default_limit_blocks_and_resets(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_DEFAULT", "3/minute")
    headers = await auth_headers(client, "limits@example.com")
    responses = [await client.get("/collections", headers=headers) for _ in range(3)]
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert [r.headers["RateLimit-Remaining"] for r in responses] == ["2", "1", "0"]
    assert responses[0].headers["RateLimit-Limit"] == "3"

    fake_clock.advance(10)
    blocked = await client.get("/collections", headers=headers)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "50"
    assert blocked.headers["RateLimit-Remaining"] == "0"
    body = blocked.json()
    assert body["code"] == "rate_limited"
    assert body["retry_after"] == 50
    assert "50 seconds" in body["detail"]

    fake_clock.advance(50)
    assert (await client.get("/collections", headers=headers)).status_code == 200


async def test_one_users_limit_never_affects_another(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_DEFAULT", "2/minute")
    alice = await auth_headers(client, "alice-rl@example.com")
    bob = await auth_headers(client, "bob-rl@example.com")
    for _ in range(2):
        assert (await client.get("/collections", headers=alice)).status_code == 200
    assert (await client.get("/collections", headers=alice)).status_code == 429
    assert (await client.get("/collections", headers=bob)).status_code == 200


async def test_ask_has_its_own_limit(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_ASK", "2/minute")
    headers = await auth_headers(client, "asker@example.com")
    collection_id = await create_collection(client, headers)
    url, body = f"/collections/{collection_id}/ask", {"question": "What is this?"}
    for _ in range(2):
        assert (await client.post(url, json=body, headers=headers)).status_code == 200
    blocked = await client.post(url, json=body, headers=headers)
    assert blocked.status_code == 429
    # The streaming endpoint shares the ask budget.
    stream = await client.post(
        f"/collections/{collection_id}/ask/stream", json=body, headers=headers
    )
    assert stream.status_code == 429
    # Other endpoints still work: one class doesn't use up another.
    assert (await client.get("/collections", headers=headers)).status_code == 200


async def test_upload_has_its_own_limit(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_UPLOAD", "2/hour")
    headers = await auth_headers(client, "uploader@example.com")
    collection_id = await create_collection(client, headers)
    for i in range(2):
        response = await upload(client, headers, collection_id, f"n{i}.txt", f"text {i}".encode())
        assert response.status_code == 202
    blocked = await upload(client, headers, collection_id, "n3.txt", b"text 3")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) == 3600
    fake_clock.advance(3600)
    assert (await upload(client, headers, collection_id, "n4.txt", b"hi there")).status_code == 202


async def test_unauthenticated_requests_are_401_not_counted(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_DEFAULT", "1/minute")
    for _ in range(3):
        assert (await client.get("/collections")).status_code == 401


async def test_health_is_not_limited(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_DEFAULT", "1/minute")
    for _ in range(5):
        assert "RateLimit-Limit" not in (await client.get("/health")).headers


async def test_disabled_switch(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_DEFAULT", "1/minute")
    monkeypatch.setattr(get_settings(), "RATE_LIMIT_ENABLED", False)
    headers = await auth_headers(client, "nolimit@example.com")
    for _ in range(3):
        assert (await client.get("/collections", headers=headers)).status_code == 200


# --- per-IP limits (before login) --------------------------------------------------


async def test_register_is_limited_per_ip(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_REGISTER_PER_IP", "2/hour")
    assert (await register(client, "r1@example.com")).status_code == 201
    assert (await register(client, "r2@example.com")).status_code == 201
    blocked = await register(client, "r3@example.com")
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "3600"


async def test_login_is_limited_per_ip(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_LOGIN_PER_IP", "3/minute")
    for i in range(3):
        await register(client, f"ip{i}@example.com")
    for i in range(3):
        assert (await login(client, f"ip{i}@example.com")).status_code == 200
    # A fourth account from the same IP: still blocked (it's the IP that's limited).
    assert (await login(client, "ip0@example.com")).status_code == 429


async def test_login_is_limited_per_email_across_ips(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_LOGIN_PER_EMAIL", "2/minute")
    monkeypatch.setattr(get_settings(), "TRUSTED_PROXY_COUNT", 1)
    await register(client, "target@example.com")

    async def attempt(ip: str):
        return await client.post(
            "/auth/login",
            data={"username": "Target@Example.com", "password": "wrong-password-1"},
            headers={"X-Forwarded-For": ip},
        )

    assert (await attempt("203.0.113.1")).status_code == 401
    assert (await attempt("203.0.113.2")).status_code == 401
    blocked = await attempt("203.0.113.3")  # a new IP doesn't help
    assert blocked.status_code == 429
    assert "login attempts" in blocked.json()["detail"]


async def test_spoofed_forwarded_for_cannot_evade_the_limit(client, monkeypatch, fake_clock):
    """No trusted proxy configured: a different made-up X-Forwarded-For on
    every request must still count against the one real connection address."""
    set_limit(monkeypatch, "RATE_LIMIT_REGISTER_PER_IP", "2/hour")
    statuses = []
    for i in range(4):
        response = await client.post(
            "/auth/register",
            json={"email": f"spoof{i}@example.com", "password": PASSWORD},
            headers={"X-Forwarded-For": f"198.51.100.{i}"},
        )
        statuses.append(response.status_code)
    assert statuses == [201, 201, 429, 429]


async def test_spoofed_entry_left_of_a_trusted_proxy_is_ignored(client, monkeypatch, fake_clock):
    """One trusted proxy: the client can prepend anything, but the address
    our proxy appended (the rightmost) is what counts."""
    set_limit(monkeypatch, "RATE_LIMIT_REGISTER_PER_IP", "2/hour")
    monkeypatch.setattr(get_settings(), "TRUSTED_PROXY_COUNT", 1)
    statuses = []
    for i in range(3):
        response = await client.post(
            "/auth/register",
            json={"email": f"behind{i}@example.com", "password": PASSWORD},
            headers={"X-Forwarded-For": f"198.51.100.{i}, 203.0.113.50"},
        )
        statuses.append(response.status_code)
    assert statuses == [201, 201, 429]


async def test_trusted_proxy_separates_real_clients(client, monkeypatch, fake_clock):
    set_limit(monkeypatch, "RATE_LIMIT_REGISTER_PER_IP", "1/hour")
    monkeypatch.setattr(get_settings(), "TRUSTED_PROXY_COUNT", 1)
    for i in range(3):
        response = await client.post(
            "/auth/register",
            json={"email": f"client{i}@example.com", "password": PASSWORD},
            headers={"X-Forwarded-For": f"203.0.113.{i}"},
        )
        assert response.status_code == 201


# --- Redis down ------------------------------------------------------------------------


@pytest.fixture
def redis_down(client):
    app.dependency_overrides[get_redis] = lambda: DownRedis()


async def test_redis_down_general_endpoints_fail_open(client, monkeypatch, fake_clock, caplog):
    set_limit(monkeypatch, "RATE_LIMIT_DEFAULT", "1/minute")
    headers = await auth_headers(client, "open@example.com")
    app.dependency_overrides[get_redis] = lambda: DownRedis()
    for _ in range(5):
        response = await client.get("/collections", headers=headers)
        assert response.status_code == 200
        assert "RateLimit-Limit" not in response.headers


async def test_redis_down_login_still_limited_by_fallback(
    client, redis_down, monkeypatch, fake_clock
):
    set_limit(monkeypatch, "RATE_LIMIT_LOGIN_PER_IP", "3/minute")
    statuses = [(await login(client, "nobody@example.com")).status_code for _ in range(4)]
    assert statuses == [401, 401, 401, 429]
    fake_clock.advance(60)
    assert (await login(client, "nobody@example.com")).status_code == 401


async def test_redis_down_login_email_limit_uses_fallback_too(
    client, redis_down, monkeypatch, fake_clock
):
    set_limit(monkeypatch, "RATE_LIMIT_LOGIN_PER_EMAIL", "2/minute")
    statuses = [(await login(client, "victim@example.com")).status_code for _ in range(3)]
    assert statuses == [401, 401, 429]


def test_an_unknown_user_id_is_just_a_key():
    from app.core import redis_keys

    user_id = uuid.uuid4()
    assert redis_keys.rate_user("ask", user_id) == f"documind:rl:ask:u:{user_id}"
    assert "victim" not in redis_keys.rate_email("login", "victim@example.com")
