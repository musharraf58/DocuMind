"""INGESTION_MODE=inline: documents processed inside the API process, one at
a time, from a bounded in-memory queue, with the sweeper running in-process."""

import asyncio
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.redis import get_redis
from app.main import app
from app.models import Document, DocumentStatus
from app.services import background, jobs
from app.services.background import InlineRuntime, run_maintenance
from app.services.inline_worker import InlineJobQueue, InlineWorker
from app.services.queue import get_queue
from tests.helpers import auth_headers, create_collection, files_on_disk, upload
from tests.redis_counting import CountingRedis


async def wait_until(predicate) -> None:
    async with asyncio.timeout(10):
        while not await predicate():  # noqa: ASYNC110 (polls the database)
            await asyncio.sleep(0.02)


async def no_maintenance(sessions, queue, redis) -> None:
    pass


@pytest.fixture
def inline_settings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "INGESTION_MODE", "inline")
    monkeypatch.setattr(settings, "INLINE_QUEUE_CAPACITY", 20)
    monkeypatch.setattr(settings, "INLINE_SHUTDOWN_GRACE_SECONDS", 0.2)
    return settings


@pytest.fixture
async def runtime(client, test_engine, inline_settings):
    """A running inline runtime wired to the test database and the test
    client's fake Redis, and the API enqueueing into it (as in production)."""
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    redis = app.dependency_overrides[get_redis]()
    rt = InlineRuntime(inline_settings, sessions, redis, maintenance=no_maintenance)
    background._runtime = rt
    app.dependency_overrides[get_queue] = lambda: rt.queue
    rt.start()
    yield rt
    await rt.stop()
    background._runtime = None


async def status_of(db, document_id) -> Document:
    return await db.get(Document, uuid.UUID(str(document_id)), populate_existing=True)


def block_pipeline(monkeypatch) -> asyncio.Event:
    """Make every job hang inside the pipeline (after its claim) until released."""
    release = asyncio.Event()
    started = []

    async def blocked(db, storage, embedder, document, run_id, settings):
        started.append(document.id)
        await release.wait()
        raise RuntimeError("released")

    monkeypatch.setattr(jobs, "run_pipeline", blocked)
    release.started = started
    return release


# --- the queue ------------------------------------------------------------------


async def test_queue_is_bounded_and_deduplicated():
    queue = InlineJobQueue(capacity=2)
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    assert await queue.enqueue(a)
    assert await queue.enqueue(a)  # already waiting: accepted, not added twice
    assert len(queue) == 1
    assert await queue.enqueue(b)
    assert not queue.has_room()
    assert not await queue.enqueue(c)  # full: the sweeper will try again later
    assert await queue.get() == a
    assert queue.has_room()
    assert await queue.enqueue(a)  # taken, so it may be queued again (reprocess)


async def test_retries_wait_outside_the_queue_then_join_it():
    queue = InlineJobQueue(capacity=1)
    retry = uuid.uuid4()
    assert await queue.enqueue(retry, countdown=0.05)
    assert len(queue) == 0  # a scheduled retry doesn't hold a place
    await asyncio.sleep(0.1)
    assert len(queue) == 1 and await queue.get() == retry


async def test_closed_queue_refuses_and_cancels_scheduled_retries():
    queue = InlineJobQueue(capacity=5)
    await queue.enqueue(uuid.uuid4(), countdown=0.05)
    queue.close()
    assert not queue.has_room()
    assert not await queue.enqueue(uuid.uuid4())
    await asyncio.sleep(0.1)
    assert len(queue) == 0


# --- the worker -----------------------------------------------------------------


async def test_worker_runs_strictly_one_job_at_a_time():
    queue = InlineJobQueue(capacity=10)
    running, peak, order = 0, 0, []

    async def run(document_id):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        order.append(document_id)
        running -= 1

    worker = InlineWorker(queue, run)
    ids = [uuid.uuid4() for _ in range(5)]
    for document_id in ids:
        await queue.enqueue(document_id)
    worker.start()
    for _ in range(100):
        if len(order) == 5:
            break
        await asyncio.sleep(0.02)
    await worker.stop(grace_seconds=1)
    assert peak == 1
    assert order == ids  # first in, first out


async def test_a_crashing_job_does_not_stop_the_worker():
    queue = InlineJobQueue(capacity=10)
    done = []

    async def run(document_id):
        if not done:
            done.append("crash")
            raise RuntimeError("bug")
        done.append(document_id)

    worker = InlineWorker(queue, run)
    second = uuid.uuid4()
    await queue.enqueue(uuid.uuid4())
    await queue.enqueue(second)
    worker.start()
    for _ in range(100):
        if second in done:
            break
        await asyncio.sleep(0.01)
    assert worker.alive and done == ["crash", second]
    await worker.stop(grace_seconds=0)


# --- end to end -----------------------------------------------------------------


async def test_upload_goes_from_pending_to_ready(client, runtime, db):
    headers = await auth_headers(client, "inline@example.com")
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, "a.txt", b"Cats sleep a lot. " * 60)
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    document_id = response.json()["id"]

    async def ready():
        return (await status_of(db, document_id)).status == DocumentStatus.ready

    await wait_until(ready)
    # Searchable right away: the chunks and vectors are in place.
    search = await client.post(
        f"/collections/{collection_id}/search", json={"query": "cats"}, headers=headers
    )
    assert search.status_code == 200 and search.json()


async def test_reprocess_runs_inline_too(client, runtime, db):
    headers = await auth_headers(client, "again@example.com")
    collection_id = await create_collection(client, headers)
    document_id = (await upload(client, headers, collection_id, "a.txt", b"text " * 80)).json()[
        "id"
    ]

    async def ready():
        return (await status_of(db, document_id)).status == DocumentStatus.ready

    await wait_until(ready)
    response = await client.post(f"/documents/{document_id}/reprocess", headers=headers)
    assert response.status_code == 202
    await wait_until(ready)


async def test_full_queue_refuses_uploads_before_storing_anything(
    client, runtime, db, storage_dir, monkeypatch
):
    monkeypatch.setattr(runtime.queue, "capacity", 2)
    release = block_pipeline(monkeypatch)
    headers = await auth_headers(client, "busy@example.com")
    collection_id = await create_collection(client, headers)

    first = await upload(client, headers, collection_id, "1.txt", b"one " * 50)

    async def worker_holds_it():
        return bool(release.started)

    await wait_until(worker_holds_it)
    for n in (2, 3):  # these two wait in the queue
        assert (
            await upload(client, headers, collection_id, f"{n}.txt", f"{n} ".encode() * 50)
        ).status_code == 202

    refused = await upload(client, headers, collection_id, "4.txt", b"four " * 50)
    assert refused.status_code == 503
    body = refused.json()
    assert body["code"] == "ingestion_queue_full"
    assert "try again" in body["detail"]
    assert refused.headers["Retry-After"]
    count = await db.scalar(select(func.count()).select_from(Document))
    assert count == 3 and len(files_on_disk(storage_dir)) == 3  # nothing stored for #4
    usage = (await client.get("/usage", headers=headers)).json()
    assert usage["quotas"]["uploads"]["used"] == 3  # the refused one doesn't count

    reprocess = await client.post(f"/documents/{first.json()['id']}/reprocess", headers=headers)
    assert reprocess.status_code == 503
    release.set()


async def test_shutdown_hands_the_running_document_back_without_using_an_attempt(
    client, runtime, db, monkeypatch
):
    release = block_pipeline(monkeypatch)
    headers = await auth_headers(client, "restart@example.com")
    collection_id = await create_collection(client, headers)
    document_id = (await upload(client, headers, collection_id, "a.txt", b"text " * 80)).json()[
        "id"
    ]

    async def processing():
        return (await status_of(db, document_id)).status == DocumentStatus.processing

    await wait_until(processing)
    await runtime.stop()  # grace 0.2 s, then the job is cancelled

    document = await status_of(db, document_id)
    assert document.status == DocumentStatus.pending
    assert document.attempts == 0
    assert document.processing_run_id is None
    assert document.error_message == jobs.INTERRUPTED
    release.set()


async def test_in_process_sweeper_recovers_a_stale_processing_document(client, runtime, db):
    headers = await auth_headers(client, "stale@example.com")
    collection_id = await create_collection(client, headers)
    document_id = (await upload(client, headers, collection_id, "a.txt", b"text " * 80)).json()[
        "id"
    ]

    async def ready():
        return (await status_of(db, document_id)).status == DocumentStatus.ready

    await wait_until(ready)
    # As if the process had been killed mid-job an hour ago (no clean release).
    long_ago = datetime.now(UTC) - timedelta(hours=1)
    await db.execute(
        update(Document)
        .where(Document.id == uuid.UUID(document_id))
        .values(
            status=DocumentStatus.processing,
            processing_run_id=uuid.uuid4(),
            heartbeat_at=long_ago,
            attempts=1,
        )
    )
    await db.commit()

    await run_maintenance(runtime.sessions, runtime.queue, None)  # what the periodic task runs
    await wait_until(ready)
    assert (await status_of(db, document_id)).attempts == 2


async def test_maintenance_runs_periodically_and_at_start(client, test_engine, inline_settings):
    rounds = []

    async def count_round(sessions, queue, redis):
        rounds.append(1)

    settings = inline_settings.model_copy(update={"MAINTENANCE_INTERVAL_SECONDS": 0.05})
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    rt = InlineRuntime(settings, sessions, None, maintenance=count_round)
    rt.start()
    await asyncio.sleep(0.01)
    assert rounds == [1]  # at once: catches up on anything left from before a restart
    await asyncio.sleep(0.2)
    await rt.stop()
    assert len(rounds) >= 3


# --- no Redis polling -------------------------------------------------------------


async def test_an_idle_inline_app_sends_no_redis_commands(test_engine, inline_settings):
    """The worker waits on an in-memory queue and maintenance uses Postgres:
    while nothing happens, Redis (billed per command on Upstash) hears nothing."""
    redis = CountingRedis()
    settings = inline_settings.model_copy(update={"MAINTENANCE_INTERVAL_SECONDS": 0.02})
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    rt = InlineRuntime(settings, sessions, redis)  # the real maintenance: the sweeper
    rt.start()
    await asyncio.sleep(0.3)  # many maintenance rounds
    await rt.stop()
    assert redis.commands == []


def test_inline_mode_never_loads_celery_or_its_heartbeat():
    """Celery (and the worker heartbeat that writes to Redis every 30 s) is
    not even imported by the API in inline mode."""
    code = (
        "import sys, app.main;"
        "print('celery' in sys.modules, 'app.workers.heartbeat' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "INGESTION_MODE": "inline"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


async def test_readiness_in_inline_mode_needs_no_worker_heartbeat(client, runtime):
    redis = CountingRedis()
    app.dependency_overrides[get_redis] = lambda: redis
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["worker"] == "ok"
    assert "EXISTS" not in redis.commands  # no heartbeat key looked up

    await runtime.stop()
    response = await client.get("/health/ready")
    assert response.status_code == 503 and response.json()["worker"] == "unavailable"


async def test_celery_mode_readiness_still_uses_the_heartbeat(client):
    assert get_settings().INGESTION_MODE == "celery"
    response = await client.get("/health/ready")
    assert response.json()["worker"] == "unavailable"  # no worker heartbeat in tests


# --- the lifespan -----------------------------------------------------------------


async def test_lifespan_starts_and_stops_the_runtime(test_engine, inline_settings, monkeypatch):
    import app.main as main

    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(background, "job_sessions", lambda: sessions)
    monkeypatch.setattr(sessions, "kw", {**sessions.kw, "bind": _NoDispose()})
    monkeypatch.setattr(main, "report_missing_embeddings", _noop)
    monkeypatch.setattr(main, "get_redis", lambda: CountingRedis())

    async with main.lifespan(app):
        rt = background.current()
        assert rt is not None and rt.worker.alive
        assert get_queue() is rt.queue
    assert background.current() is None
    assert not rt.worker.alive


async def test_get_queue_in_inline_mode_without_a_runtime_is_a_503(inline_settings):
    from app.core.errors import ApiError

    with pytest.raises(ApiError) as caught:
        get_queue()
    assert caught.value.status_code == 503


class _NoDispose:
    async def dispose(self) -> None:  # the test engine is the session's, not ours
        pass


async def _noop(*args) -> None:
    pass
