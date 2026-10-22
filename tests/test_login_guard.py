from datetime import UTC, datetime, timedelta

import pytest

from app.core import clock
from app.core.config import get_settings
from tests.helpers import PASSWORD, error_body, login, register

WRONG = "not-the-password-1"


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


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "LOGIN_FREE_ATTEMPTS", 3)
    monkeypatch.setattr(settings, "LOGIN_DELAY_BASE_SECONDS", 2)
    monkeypatch.setattr(settings, "LOGIN_DELAY_MAX_SECONDS", 10)
    monkeypatch.setattr(settings, "LOGIN_FAILURE_WINDOW_MINUTES", 15)


async def test_first_failures_are_free_then_delays_grow(client, fake_clock):
    await register(client, "alice@example.com")
    for _ in range(3):
        assert (await login(client, "alice@example.com", WRONG)).status_code == 401
    # 4th failure: the next attempt must wait 2 s.
    assert (await login(client, "alice@example.com", WRONG)).status_code == 401
    blocked = await login(client, "alice@example.com", WRONG)
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "2"
    assert blocked.json()["code"] == "login_delayed"

    fake_clock.advance(2)
    assert (await login(client, "alice@example.com", WRONG)).status_code == 401  # 5th
    assert (await login(client, "alice@example.com", WRONG)).headers["Retry-After"] == "4"
    fake_clock.advance(4)
    assert (await login(client, "alice@example.com", WRONG)).status_code == 401  # 6th
    assert (await login(client, "alice@example.com", WRONG)).headers["Retry-After"] == "8"
    fake_clock.advance(8)
    assert (await login(client, "alice@example.com", WRONG)).status_code == 401  # 7th
    # Capped at the maximum (10, not 16).
    assert (await login(client, "alice@example.com", WRONG)).headers["Retry-After"] == "10"


async def test_even_the_right_password_waits_while_blocked(client, fake_clock):
    await register(client, "bob@example.com")
    for _ in range(4):
        await login(client, "bob@example.com", WRONG)
    assert (await login(client, "bob@example.com", PASSWORD)).status_code == 429
    fake_clock.advance(2)
    assert (await login(client, "bob@example.com", PASSWORD)).status_code == 200


async def test_success_resets_the_count(client, fake_clock):
    await register(client, "carol@example.com")
    for _ in range(3):
        await login(client, "carol@example.com", WRONG)
    assert (await login(client, "carol@example.com", PASSWORD)).status_code == 200
    # Three free failures again, no delay.
    for _ in range(3):
        assert (await login(client, "carol@example.com", WRONG)).status_code == 401
    assert (await login(client, "carol@example.com", PASSWORD)).status_code == 200


async def test_failures_expire_after_the_window(client, fake_clock):
    await register(client, "dave@example.com")
    for _ in range(3):
        await login(client, "dave@example.com", WRONG)
    # Redis forgets the count by itself: the key expires after the window.
    from app.core import redis_keys
    from app.core.redis import get_redis
    from app.main import app

    redis = app.dependency_overrides[get_redis]()
    ttl = await redis.ttl(redis_keys.login_failures("dave@example.com", "127.0.0.1"))
    assert 0 < ttl <= 15 * 60


async def test_unknown_email_and_wrong_password_look_the_same(client, fake_clock):
    """Same 401 body, and the same delay sequence: neither reveals whether
    the account exists."""
    await register(client, "real@example.com")
    real, ghost = [], []
    for _ in range(5):
        real.append(await login(client, "real@example.com", WRONG))
        ghost.append(await login(client, "ghost@example.com", WRONG))
    assert (
        [r.status_code for r in real]
        == [r.status_code for r in ghost]
        == [
            401,
            401,
            401,
            401,
            429,
        ]
    )
    assert error_body(real[0]) == error_body(ghost[0])
    assert error_body(real[0])["detail"] == "Incorrect email or password"
    assert error_body(real[4]) == error_body(ghost[4])
    assert real[4].headers["Retry-After"] == ghost[4].headers["Retry-After"]


async def test_another_ip_is_not_blocked(client, fake_clock, monkeypatch):
    """An attacker failing from their IP can't lock the user out of theirs."""
    monkeypatch.setattr(get_settings(), "TRUSTED_PROXY_COUNT", 1)
    await register(client, "erin@example.com")

    async def attempt(ip: str, password: str):
        return await client.post(
            "/auth/login",
            data={"username": "erin@example.com", "password": password},
            headers={"X-Forwarded-For": ip},
        )

    for _ in range(4):
        await attempt("198.51.100.66", WRONG)
    assert (await attempt("198.51.100.66", PASSWORD)).status_code == 429
    assert (await attempt("203.0.113.10", PASSWORD)).status_code == 200


async def test_email_case_does_not_give_a_fresh_count(client, fake_clock):
    await register(client, "frank@example.com")
    for email in [
        "frank@example.com",
        "Frank@Example.com",
        "FRANK@example.com",
        " frank@example.com",
    ]:
        await login(client, email, WRONG)
    assert (await login(client, "frank@EXAMPLE.com", PASSWORD)).status_code == 429


async def test_redis_down_does_not_block_logins(client, fake_clock):
    from app.core.redis import get_redis
    from app.main import app
    from tests.test_rate_limit_api import DownRedis

    await register(client, "gina@example.com")
    app.dependency_overrides[get_redis] = lambda: DownRedis()
    for _ in range(5):
        assert (await login(client, "gina@example.com", WRONG)).status_code == 401
    assert (await login(client, "gina@example.com", PASSWORD)).status_code == 200
