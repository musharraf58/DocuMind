"""The Celery wiring: configuration, the task wrappers, and the startup sweep."""

import uuid

import anyio
import fakeredis
import pytest

from app.core.config import get_settings
from app.models import Document, DocumentStatus
from app.services.embeddings import FakeEmbeddingProvider
from app.services.jobs import Outcome
from app.services.queue import CeleryJobQueue
from app.workers import tasks
from app.workers.celery_app import PROCESS_DOCUMENT, SWEEP, celery_app, sweep_at_startup
from tests.conftest import TEST_DB_URL
from tests.embedding_fakes import DIMENSIONS, ScriptedProvider
from tests.helpers import auth_headers, create_collection, upload


@pytest.fixture(autouse=True)
def worker_uses_test_services(monkeypatch):
    """Tasks build their own engine and Redis client from settings: point them
    at the test database and a fake Redis, never the real ones."""
    monkeypatch.setattr(
        get_settings(), "DATABASE_URL", TEST_DB_URL.render_as_string(hide_password=False)
    )
    monkeypatch.setattr(tasks, "make_redis", lambda: fakeredis.aioredis.FakeRedis())


@pytest.fixture
def sent(monkeypatch):
    """Tasks the worker sends (instead of a real broker)."""
    calls = []

    async def record(self, document_id, countdown=None):
        calls.append((document_id, countdown))
        return True

    monkeypatch.setattr(CeleryJobQueue, "enqueue", record)
    return calls


async def pending_document(client, job_queue) -> uuid.UUID:
    headers = await auth_headers(client, f"{uuid.uuid4().hex[:8]}@example.com")
    collection_id = await create_collection(client, headers)
    text = b"This sentence is long enough to pass the minimum text check. " * 50
    response = await upload(client, headers, collection_id, "notes.txt", text)
    job_queue.enqueued.clear()
    return uuid.UUID(response.json()["id"])


def test_worker_configuration():
    conf = celery_app.conf
    assert conf.worker_concurrency == 1  # one job at a time
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_acks_late and conf.task_reject_on_worker_lost  # redelivered on crash
    job_seconds = get_settings().JOB_TIMEOUT_MINUTES * 60
    assert conf.task_time_limit > job_seconds  # a backstop beyond the job's own timeout
    # A long job must not be handed to the worker a second time while it runs.
    assert conf.broker_transport_options["visibility_timeout"] > conf.task_time_limit
    assert conf.beat_schedule["sweep"] == {"task": SWEEP, "schedule": 60.0}
    assert {PROCESS_DOCUMENT, SWEEP} <= set(celery_app.tasks)


async def test_process_task_runs_the_job_in_its_own_event_loop(client, db, job_queue, sent):
    doc_id = await pending_document(client, job_queue)
    # Celery runs tasks synchronously; run it in a thread as the worker would.
    outcome = await anyio.to_thread.run_sync(tasks.process_document.run, str(doc_id))

    assert outcome == Outcome.ready
    document = await db.get(Document, doc_id, populate_existing=True)
    assert document.status == DocumentStatus.ready
    assert sent == []


async def test_transient_failure_is_requeued_with_a_delay(client, db, job_queue, sent, monkeypatch):
    failing = ScriptedProvider(
        failures=[tasks.jobs.RetryableEmbeddingError("Embedding service is unavailable")] * 9,
        max_retries=0,
    )
    monkeypatch.setattr(tasks, "get_embedding_provider", lambda redis: failing)
    doc_id = await pending_document(client, job_queue)

    result = await tasks.process(doc_id)

    assert result.outcome == Outcome.retry
    assert sent == [(doc_id, 30)]  # enqueued again, to run in 30 s
    assert (await db.get(Document, doc_id, populate_existing=True)).status == DocumentStatus.pending


async def test_misconfigured_embedding_fails_the_document_with_the_reason(
    client, db, job_queue, sent, monkeypatch
):
    monkeypatch.setattr(get_settings(), "EMBEDDING_DIMENSIONS", 512)
    doc_id = await pending_document(client, job_queue)

    result = await tasks.process(doc_id)

    assert result.outcome == Outcome.failed
    document = await db.get(Document, doc_id, populate_existing=True)
    assert document.status == DocumentStatus.failed
    assert "don't match the database" in document.error_message


async def test_sweep_task_uses_the_real_queue(client, db, job_queue, sent):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update

    doc_id = await pending_document(client, job_queue)
    await db.execute(
        update(Document)
        .where(Document.id == doc_id)
        .values(queued_at=datetime.now(UTC) - timedelta(minutes=10))
    )
    await db.commit()

    report = await tasks.run_sweep()
    assert report.requeued_pending == [doc_id]
    assert sent == [(doc_id, None)]


def test_worker_start_triggers_a_sweep(monkeypatch):
    sent_tasks = []
    monkeypatch.setattr(celery_app, "send_task", lambda name, **kw: sent_tasks.append(name))
    sweep_at_startup()
    assert sent_tasks == [SWEEP]


def test_fake_provider_is_used_in_tests():
    # Guard: the worker tests above must not be talking to Voyage.
    assert isinstance(tasks.get_embedding_provider(None), FakeEmbeddingProvider)
    assert DIMENSIONS == 1024


async def test_worker_records_embedding_usage_for_the_owner(client, db, job_queue, sent):
    from sqlalchemy import select

    from app.models import UsageEvent, UsageKind

    doc_id = await pending_document(client, job_queue)
    document = await db.get(Document, doc_id)
    assert (await tasks.process(doc_id)).outcome == Outcome.ready

    events = (await db.scalars(select(UsageEvent))).all()
    assert len(events) >= 1
    assert {e.kind for e in events} == {UsageKind.embed_document}
    assert {e.user_id for e in events} == {document.owner_id}
    assert {e.document_id for e in events} == {doc_id}
    assert all(e.tokens_in > 0 and e.tokens_out == 0 for e in events)
