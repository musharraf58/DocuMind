import threading

import fakeredis
import pytest

from app.core.database import get_db
from app.core.redis import get_redis
from app.core.redis_keys import WORKER_HEARTBEAT
from app.main import app
from app.workers import heartbeat
from tests.test_rate_limit_api import DownRedis


class DownSession:
    """A database session whose server is unreachable."""

    async def execute(self, *args, **kwargs):
        raise ConnectionRefusedError("connection refused")


async def down_db():
    yield DownSession()


@pytest.fixture
async def worker_alive(client):
    redis = app.dependency_overrides[get_redis]()
    await redis.set(WORKER_HEARTBEAT, "1", ex=heartbeat.HEARTBEAT_TTL)


async def test_ready_when_everything_is_up(client, worker_alive):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "redis": "ok", "worker": "ok"}


async def test_not_ready_without_a_worker_heartbeat(client):
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["worker"] == "unavailable"
    assert (await client.get("/health/live")).status_code == 200


async def test_redis_down(client, worker_alive):
    app.dependency_overrides[get_redis] = lambda: DownRedis()
    ready = await client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["redis"] == "unavailable"
    assert ready.json()["worker"] == "unknown"
    # Liveness is about the process only: still up.
    live = await client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}


async def test_postgres_down(client, worker_alive):
    app.dependency_overrides[get_db] = down_db
    ready = await client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["database"] == "unavailable"
    assert ready.json()["redis"] == "ok"
    assert (await client.get("/health/live")).status_code == 200


async def test_old_health_endpoint_still_works(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "redis": "ok"}


def test_heartbeat_sets_an_expiring_key():
    redis = fakeredis.FakeRedis()
    heartbeat.beat(redis, now=1234.5)
    assert float(redis.get(WORKER_HEARTBEAT)) == 1234.5
    assert 0 < redis.ttl(WORKER_HEARTBEAT) <= heartbeat.HEARTBEAT_TTL


def test_heartbeat_thread_beats_and_cleans_up(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        heartbeat.Redis, "from_url", lambda *a, **kw: fakeredis.FakeRedis(server=server)
    )
    check = fakeredis.FakeRedis(server=server)
    stop = threading.Event()
    thread = heartbeat.start_heartbeat("redis://unused", stop)
    for _ in range(100):
        if check.exists(WORKER_HEARTBEAT):
            break
        threading.Event().wait(0.01)
    assert check.exists(WORKER_HEARTBEAT)
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not check.exists(WORKER_HEARTBEAT)  # a clean stop removes it at once


def test_heartbeat_is_started_when_the_worker_is_ready():
    from celery.signals import worker_ready

    from app.workers.celery_app import start_worker_heartbeat

    assert any(
        receiver() is start_worker_heartbeat
        for _, receiver in worker_ready.receivers
        if callable(receiver)
    )


async def test_health_endpoints_are_not_rate_limited(client, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "RATE_LIMIT_DEFAULT", "1/minute")
    for _ in range(3):
        assert (await client.get("/health/live")).status_code == 200
