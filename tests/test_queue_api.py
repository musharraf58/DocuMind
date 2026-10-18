"""Upload and reprocess put work on the queue, only after it is committed."""

import uuid

from sqlalchemy import select

from app.models import Document, DocumentStatus
from app.services.queue import CeleryJobQueue
from app.workers import celery_app as celery_module
from tests.helpers import auth_headers, create_collection, upload


async def owner(client):
    headers = await auth_headers(client, "owner@example.com")
    return headers, await create_collection(client, headers)


async def test_upload_returns_202_and_enqueues_once_after_commit(client, db, job_queue):
    headers, collection_id = await owner(client)
    response = await upload(client, headers, collection_id, "a.txt", b"some text here")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending" and body["queued_at"] is not None
    doc_id = uuid.UUID(body["id"])
    assert job_queue.ids == [doc_id]
    # At the moment of enqueueing, another session (a worker) could already
    # see the document as pending: it had been committed.
    assert job_queue.not_yet_committed == []


async def test_upload_still_succeeds_when_the_queue_is_down(client, db, job_queue):
    job_queue.down = True
    headers, collection_id = await owner(client)
    response = await upload(client, headers, collection_id, "a.txt", b"some text here")

    assert response.status_code == 202
    document = await db.get(Document, uuid.UUID(response.json()["id"]))
    assert document.status == DocumentStatus.pending  # the sweeper will enqueue it
    assert job_queue.ids == []


async def test_rejected_uploads_enqueue_nothing(client, db, job_queue):
    headers, collection_id = await owner(client)
    await upload(client, headers, collection_id, "a.txt", b"same bytes")
    duplicate = await upload(client, headers, collection_id, "b.txt", b"same bytes")
    wrong_type = await upload(client, headers, collection_id, "x.exe", b"MZ")

    assert (duplicate.status_code, wrong_type.status_code) == (409, 415)
    assert len(job_queue.ids) == 1
    assert len((await db.scalars(select(Document))).all()) == 1


async def test_celery_queue_sends_the_task_by_name(monkeypatch):
    sent = []
    monkeypatch.setattr(
        celery_module.celery_app, "send_task", lambda name, **kwargs: sent.append((name, kwargs))
    )
    document_id = uuid.uuid4()
    assert await CeleryJobQueue().enqueue(document_id, countdown=30) is True
    [(name, kwargs)] = sent
    assert name == "documind.process_document"
    assert kwargs["args"] == [str(document_id)] and kwargs["countdown"] == 30


async def test_celery_queue_reports_failure_instead_of_raising(monkeypatch, caplog):
    def broker_down(*args, **kwargs):
        raise ConnectionError("Error 111 connecting to redis:6379")

    monkeypatch.setattr(celery_module.celery_app, "send_task", broker_down)
    assert await CeleryJobQueue().enqueue(uuid.uuid4()) is False
    assert "the sweeper will retry" in caplog.text
