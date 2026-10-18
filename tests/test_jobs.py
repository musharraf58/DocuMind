"""The background job: claim, process, classify failures, retry or fail."""

import asyncio
import uuid

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.chunking import ChunkData
from app.services.embeddings import (
    EmbeddingBusyError,
    EmbeddingError,
    FakeEmbeddingProvider,
    RetryableEmbeddingError,
    VoyageEmbeddingProvider,
)
from app.services.jobs import GENERIC_ERROR, Outcome, classify, run_job
from app.services.storage import get_storage
from tests.embedding_fakes import DIMENSIONS, RecordingSleep, ScriptedProvider
from tests.fixtures import build_pdf
from tests.helpers import auth_headers, create_collection, files_on_disk, upload

TEXT = ("This sentence is long enough to pass the minimum text check. " * 300).encode()


def fake() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(model="fake-model", dimensions=DIMENSIONS)


def failing(times: int = 100, error: Exception | None = None) -> ScriptedProvider:
    """Fails `times` requests, with no quick client-side retries."""
    error = error or RetryableEmbeddingError("Embedding service is unavailable; try again later")
    return ScriptedProvider(failures=[error] * times, max_retries=0)


async def new_document(client, content: bytes = TEXT, name: str = "notes.txt"):
    headers = await auth_headers(client, f"{uuid.uuid4().hex[:8]}@example.com")
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, name, content)
    assert response.status_code in (201, 202), response.text
    return headers, collection_id, uuid.UUID(response.json()["id"])


async def job(db, document_id, embedder=None):
    return await run_job(db, document_id, get_storage(), embedder or fake())


async def state(db, document_id) -> Document | None:
    return await db.get(Document, document_id, populate_existing=True)


async def chunk_count(db, document_id) -> int:
    return await db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )


async def test_pending_document_becomes_ready(client, db):
    _, _, doc_id = await new_document(client)
    result = await job(db, doc_id)

    assert result.outcome == Outcome.ready
    document = await state(db, doc_id)
    assert document.status == DocumentStatus.ready
    assert document.attempts == 1
    assert document.finished_at is not None
    assert await chunk_count(db, doc_id) > 1


async def test_a_duplicate_delivery_does_nothing(client, db):
    _, _, doc_id = await new_document(client)
    await job(db, doc_id)
    chunks_before = await chunk_count(db, doc_id)

    second = await job(db, doc_id)  # the queue delivered the same job again

    assert second.outcome == Outcome.skipped
    document = await state(db, doc_id)
    assert (document.status, document.attempts) == (DocumentStatus.ready, 1)
    assert await chunk_count(db, doc_id) == chunks_before  # no duplicates


async def test_two_deliveries_at_once_process_the_document_only_once(client, db, test_engine):
    _, _, doc_id = await new_document(client)
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)

    async def deliver():
        async with sessions() as session:
            return await run_job(session, doc_id, get_storage(), fake())

    outcomes = sorted(r.outcome for r in await asyncio.gather(deliver(), deliver()))
    assert outcomes == [Outcome.ready, Outcome.skipped]
    assert (await state(db, doc_id)).attempts == 1


async def test_transient_failure_retries_then_succeeds(client, db):
    _, _, doc_id = await new_document(client)

    first = await job(db, doc_id, failing())
    assert first.outcome == Outcome.retry
    document = await state(db, doc_id)
    assert document.status == DocumentStatus.pending
    assert document.queued_at is not None
    assert document.error_message.startswith("Attempt 1 of 4 failed:")

    second = await job(db, doc_id)
    assert second.outcome == Outcome.ready
    document = await state(db, doc_id)
    assert (document.status, document.attempts, document.error_message) == (
        DocumentStatus.ready,
        2,
        None,
    )


async def test_retries_back_off_then_the_attempts_cap_fails_the_document(client, db):
    _, _, doc_id = await new_document(client)
    results = [await job(db, doc_id, failing()) for _ in range(4)]

    assert [r.outcome for r in results] == [Outcome.retry] * 3 + [Outcome.failed]
    assert [r.retry_in for r in results[:3]] == [30, 60, 120]
    document = await state(db, doc_id)
    assert document.status == DocumentStatus.failed
    assert document.attempts == 4
    assert document.error_message == (
        "Gave up after 4 attempts: Embedding service is unavailable; try again later"
    )
    assert (await job(db, doc_id)).outcome == Outcome.skipped  # no fifth attempt


async def test_provider_asking_for_a_long_wait_delays_the_retry(client, db):
    _, _, doc_id = await new_document(client)
    result = await job(db, doc_id, failing(error=EmbeddingBusyError(retry_after=300)))
    assert result.outcome == Outcome.retry
    assert result.retry_in == 300  # not the default 30 s


async def test_scanned_pdf_fails_at_once_without_retry(client, db):
    _, _, doc_id = await new_document(client, build_pdf([[], []]), "scan.pdf")
    result = await job(db, doc_id)

    assert result.outcome == Outcome.failed
    document = await state(db, doc_id)
    assert (document.status, document.attempts) == (DocumentStatus.failed, 1)
    assert "scanned" in document.error_message
    assert (await job(db, doc_id)).outcome == Outcome.skipped


async def test_invalid_api_key_fails_at_once_without_retry(client, db):
    _, _, doc_id = await new_document(client)
    key = "pa-very-secret-key"
    voyage = VoyageEmbeddingProvider(
        api_key=SecretStr(key),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": key})),
        model="voyage-4-lite",
        dimensions=DIMENSIONS,
        sleep=RecordingSleep(),
    )
    result = await job(db, doc_id, voyage)

    assert result.outcome == Outcome.failed
    document = await state(db, doc_id)
    assert document.error_message == "Embedding service rejected the API key"
    assert key not in document.error_message


async def test_job_timeout_is_transient(client, db, monkeypatch):
    monkeypatch.setattr(get_settings(), "JOB_TIMEOUT_MINUTES", 0.002)  # ~0.1 s

    class Slow(FakeEmbeddingProvider):
        async def _embed_batch(self, texts, input_type):
            await asyncio.sleep(5)

    _, _, doc_id = await new_document(client)
    result = await job(db, doc_id, Slow(model="slow", dimensions=DIMENSIONS))

    assert result.outcome == Outcome.retry
    assert "took too long" in (await state(db, doc_id)).error_message


async def test_database_blip_is_transient(client, db, monkeypatch):
    def blip(*args):
        raise OperationalError("SELECT 1", {}, ConnectionResetError("connection reset"))

    monkeypatch.setattr(processing, "extract_and_chunk", blip)
    _, _, doc_id = await new_document(client)
    assert (await job(db, doc_id)).outcome == Outcome.retry


async def test_unexpected_error_fails_with_a_generic_message(client, db, monkeypatch, caplog):
    def crash(*args):
        raise KeyError("/srv/internal/detail")

    monkeypatch.setattr(processing, "extract_and_chunk", crash)
    _, _, doc_id = await new_document(client)
    result = await job(db, doc_id)

    assert result.outcome == Outcome.failed
    assert (await state(db, doc_id)).error_message == GENERIC_ERROR
    assert "/srv/internal/detail" in caplog.text  # details in the log only


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (RetryableEmbeddingError("x"), True),
        (EmbeddingBusyError(retry_after=10), True),
        (TimeoutError(), True),
        (ConnectionError(), True),
        (ValueError("bug"), False),
    ],
)
def test_classification(error, transient):
    assert classify(error).transient is transient


async def test_missing_stored_file_fails_with_a_clear_message(client, db, storage_dir):
    _, _, doc_id = await new_document(client)
    document = await state(db, doc_id)
    (storage_dir / document.storage_key).unlink()

    result = await job(db, doc_id)
    assert result.outcome == Outcome.failed
    assert (await state(db, doc_id)).error_message == "The stored file is missing"


async def test_document_deleted_before_the_job_is_skipped_quietly(client, db, storage_dir):
    headers, _, doc_id = await new_document(client)
    assert (await client.delete(f"/documents/{doc_id}", headers=headers)).status_code == 204

    result = await job(db, doc_id)
    assert result.outcome == Outcome.skipped
    assert await chunk_count(db, doc_id) == 0
    assert files_on_disk(storage_dir) == []


def three_chunks(*args):
    return [ChunkData(i, f"Chunk {i} text.", None, None, 4) for i in range(3)]


@pytest.mark.parametrize("batch_size", [1, 64], ids=["between-batches", "before-final-write"])
async def test_document_deleted_during_the_job_stops_it_and_leaves_nothing(
    client, db, storage_dir, monkeypatch, batch_size
):
    monkeypatch.setattr(processing, "chunk_segments", three_chunks)
    headers, _, doc_id = await new_document(client)

    class DeleteMidway(ScriptedProvider):
        async def _embed_batch(self, texts, input_type):
            if len(self.calls) == 0:  # while the first request is in flight
                response = await client.delete(f"/documents/{doc_id}", headers=headers)
                assert response.status_code == 204
            return await super()._embed_batch(texts, input_type)

    provider = DeleteMidway(batch_size=batch_size)
    result = await job(db, doc_id, provider)

    assert result.outcome == Outcome.lost
    assert len(provider.calls) == 1  # stopped before spending more of the rate limit
    assert await state(db, doc_id) is None
    assert await chunk_count(db, doc_id) == 0  # no orphan chunks
    assert files_on_disk(storage_dir) == []  # no orphan file


async def test_job_never_touches_a_ready_or_failed_document(client, db):
    _, _, doc_id = await new_document(client)
    for status in (DocumentStatus.ready, DocumentStatus.failed):
        await db.execute(update(Document).where(Document.id == doc_id).values(status=status))
        await db.commit()
        assert (await job(db, doc_id)).outcome == Outcome.skipped
        assert (await state(db, doc_id)).status == status


async def test_a_ready_document_never_has_chunks_without_embeddings(client, db, monkeypatch):
    """Whatever happens along the way, every chunk of every ready document has
    a vector and a model name."""
    from app.services.integrity import ready_documents_missing_embeddings

    ids = [(await new_document(client, TEXT + f" {n}".encode()))[2] for n in range(4)]
    a, b, c, d = ids

    await job(db, a)  # plain success

    await job(db, b)  # success, then a reprocess that fails mid-insert
    await db.execute(update(Document).where(Document.id == b).values(status=DocumentStatus.pending))
    await db.commit()
    bad = [ChunkData(0, "fine", None, None, 1), ChunkData(1, "null \x00 byte", None, None, 2)]
    with monkeypatch.context() as patch:
        patch.setattr(processing, "chunk_segments", lambda *args: bad)
        assert (await job(db, b)).outcome == Outcome.failed

    await job(db, c, failing(1))  # a transient failure, then success
    await job(db, c)

    await job(db, d)  # success, then a reprocess that fails permanently
    await db.execute(update(Document).where(Document.id == d).values(status=DocumentStatus.pending))
    await db.commit()
    result = await job(db, d, failing(error=EmbeddingError("Embedding service rejected the input")))
    assert result.outcome == Outcome.failed

    assert await ready_documents_missing_embeddings(db) == []
    ready = (
        await db.scalars(select(Document.id).where(Document.status == DocumentStatus.ready))
    ).all()
    assert set(ready) >= {a, c}
    for document_id in ready:
        chunks = (await db.scalars(select(Chunk).where(Chunk.document_id == document_id))).all()
        assert chunks and all(ch.embedding is not None and ch.embedding_model for ch in chunks)
