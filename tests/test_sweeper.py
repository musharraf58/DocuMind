"""The sweeper puts back documents the queue lost track of."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.jobs import Outcome
from app.services.sweeper import sweep
from tests.helpers import auth_headers, create_collection, upload


def ago(minutes: float) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes)


async def make_documents(client, job_queue, count: int) -> list[uuid.UUID]:
    headers = await auth_headers(client, f"{uuid.uuid4().hex[:8]}@example.com")
    collection_id = await create_collection(client, headers)
    ids = []
    for n in range(count):
        response = await upload(client, headers, collection_id, f"{n}.txt", f"text {n} " * 50)
        ids.append(uuid.UUID(response.json()["id"]))
    job_queue.enqueued.clear()  # start each test with an empty queue
    return ids


async def set_state(db, document_id, **values):
    await db.execute(update(Document).where(Document.id == document_id).values(**values))
    await db.commit()


async def state(db, document_id) -> Document:
    return await db.get(Document, document_id, populate_existing=True)


async def test_old_pending_is_requeued_and_fresh_pending_is_left_alone(client, db, job_queue):
    old, fresh = await make_documents(client, job_queue, 2)
    await set_state(db, old, queued_at=ago(4))  # past SWEEP_PENDING_MINUTES (3)
    await set_state(db, fresh, queued_at=ago(1))

    report = await sweep(db, job_queue)

    assert report.requeued_pending == [old]
    assert job_queue.ids == [old]
    assert (await state(db, old)).queued_at > ago(0.5)  # the clock restarts...
    job_queue.enqueued.clear()
    assert (await sweep(db, job_queue)).requeued_pending == []  # ...so no repeat next minute
    assert job_queue.ids == []


async def test_stale_processing_is_reset_and_requeued(client, db, job_queue):
    stale, alive = await make_documents(client, job_queue, 2)
    old_run = uuid.uuid4()
    await set_state(
        db,
        stale,
        status=DocumentStatus.processing,
        attempts=1,
        processing_run_id=old_run,
        heartbeat_at=ago(11),
    )
    await set_state(
        db,
        alive,
        status=DocumentStatus.processing,
        attempts=1,
        processing_run_id=uuid.uuid4(),
        heartbeat_at=ago(2),
    )

    report = await sweep(db, job_queue)

    assert report.reset_stale == [stale]
    assert job_queue.ids == [stale]
    document = await state(db, stale)
    assert document.status == DocumentStatus.pending
    assert document.processing_run_id is None
    assert document.error_message == "Processing was interrupted; it will be retried"
    assert (await state(db, alive)).status == DocumentStatus.processing  # heartbeat is fresh

    # If the old run was somehow still alive, its next heartbeat stops it.
    with pytest.raises(processing.ClaimLostError):
        await processing.heartbeat(db, stale, old_run)


async def test_reset_document_is_then_processed_by_the_queue(client, db, job_queue):
    [doc_id] = await make_documents(client, job_queue, 1)
    await set_state(
        db,
        doc_id,
        status=DocumentStatus.processing,
        attempts=1,
        processing_run_id=uuid.uuid4(),
        heartbeat_at=ago(30),
    )
    await sweep(db, job_queue)
    [result] = await job_queue.run_all()
    assert result.outcome == Outcome.ready
    assert (await state(db, doc_id)).attempts == 2


async def test_stale_processing_at_the_attempts_cap_is_failed_not_requeued(client, db, job_queue):
    [doc_id] = await make_documents(client, job_queue, 1)
    await set_state(
        db,
        doc_id,
        status=DocumentStatus.processing,
        attempts=4,
        processing_run_id=uuid.uuid4(),
        heartbeat_at=ago(11),
    )

    report = await sweep(db, job_queue)

    assert report.failed_stale == [doc_id]
    assert job_queue.ids == []
    document = await state(db, doc_id)
    assert document.status == DocumentStatus.failed
    assert "stopped responding" in document.error_message
    assert document.finished_at is not None


async def test_processing_without_any_heartbeat_falls_back_to_start_time(client, db, job_queue):
    [doc_id] = await make_documents(client, job_queue, 1)
    await set_state(
        db,
        doc_id,
        status=DocumentStatus.processing,
        attempts=1,
        heartbeat_at=None,
        processing_started_at=ago(60),
    )
    assert (await sweep(db, job_queue)).reset_stale == [doc_id]


async def test_ready_and_failed_documents_are_never_touched(client, db, job_queue):
    ready, failed = await make_documents(client, job_queue, 2)
    await set_state(db, ready, status=DocumentStatus.ready, queued_at=ago(60))
    await set_state(db, failed, status=DocumentStatus.failed, queued_at=ago(60))

    report = await sweep(db, job_queue)
    assert (report.requeued_pending, report.reset_stale, report.failed_stale) == ([], [], [])
    assert (await state(db, ready)).status == DocumentStatus.ready
    assert (await state(db, failed)).status == DocumentStatus.failed


async def test_when_the_queue_is_down_the_document_waits_for_the_next_sweep(client, db, job_queue):
    [doc_id] = await make_documents(client, job_queue, 1)
    await set_state(db, doc_id, queued_at=ago(4))
    job_queue.down = True

    report = await sweep(db, job_queue)
    assert report.enqueue_failed == [doc_id]
    assert (await state(db, doc_id)).status == DocumentStatus.pending

    job_queue.down = False
    await set_state(db, doc_id, queued_at=ago(4))  # three minutes later
    assert (await sweep(db, job_queue)).requeued_pending == [doc_id]
    assert job_queue.ids == [doc_id]


async def test_sweeper_logs_what_it_does(client, db, job_queue, caplog):
    caplog.set_level(logging.INFO, logger="app")
    pending, stuck = await make_documents(client, job_queue, 2)
    await set_state(db, pending, queued_at=ago(4))
    await set_state(db, stuck, status=DocumentStatus.processing, attempts=1, heartbeat_at=ago(11))
    await sweep(db, job_queue)
    assert f"document {pending} pending too long, re-enqueued" in caplog.text
    assert f"document {stuck} had no heartbeat" in caplog.text
    assert "re-enqueued 1 pending, reset 1 stuck, failed 0 stuck" in caplog.text


async def test_sweeper_warns_about_ready_documents_missing_embeddings(
    client, db, job_queue, caplog
):
    caplog.set_level(logging.INFO, logger="app")
    [doc_id] = await make_documents(client, job_queue, 1)
    job_queue.enqueued.append((doc_id, None))
    await job_queue.run_all()
    await db.execute(update(Chunk).where(Chunk.document_id == doc_id).values(embedding=None))
    await db.commit()

    report = await sweep(db, job_queue)
    assert report.missing_embeddings == 1
    assert f"ready document {doc_id} has" in caplog.text
    assert "without embeddings" in caplog.text
