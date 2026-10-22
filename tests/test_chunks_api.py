import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update

from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.chunking import ChunkData
from app.services.embeddings import EmbeddingError, FakeEmbeddingProvider
from tests.fixtures import build_pdf
from tests.helpers import auth_headers, create_collection, error_body, upload

TEXT = ("This sentence is long enough to pass the minimum text check. " * 300).encode()


async def setup_document(client, email="owner@example.com", content=TEXT, name="notes.txt"):
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, name, content)
    assert response.status_code == 202
    return headers, collection_id, response.json()["id"]


async def chunk_count(db, document_id: str) -> int:
    return await db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
    )


async def test_upload_then_worker_then_list_chunks(client, job_queue):
    headers, _, doc_id = await setup_document(client)
    [result] = await job_queue.run_all()  # what the worker does
    assert result.outcome == "ready"

    response = await client.get(f"/documents/{doc_id}/chunks", headers=headers)
    assert response.status_code == 200
    chunks = response.json()
    assert chunks
    assert set(chunks[0]) == {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "page_number",
        "section_title",
        "token_count",
        "created_at",
    }
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


async def test_chunks_are_paginated_in_order(client, job_queue):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()
    url = f"/documents/{doc_id}/chunks"

    everything = (await client.get(url, params={"limit": 100}, headers=headers)).json()
    assert len(everything) >= 3
    page = (await client.get(url, params={"limit": 2, "offset": 1}, headers=headers)).json()
    assert [c["chunk_index"] for c in page] == [1, 2]
    assert (await client.get(url, params={"limit": 101}, headers=headers)).status_code == 422


async def test_unprocessed_document_has_no_chunks(client):
    headers, _, doc_id = await setup_document(client)
    response = await client.get(f"/documents/{doc_id}/chunks", headers=headers)
    assert response.status_code == 200
    assert response.json() == []


async def test_unreadable_file_shows_failed_status_and_reason(client, job_queue):
    headers, _, doc_id = await setup_document(client, content=build_pdf([[]]), name="scan.pdf")
    await job_queue.run_all()
    body = (await client.get(f"/documents/{doc_id}", headers=headers)).json()
    assert body["status"] == "failed"
    assert body["attempts"] == 1
    assert "scanned" in body["error_message"]
    assert "storage_key" not in body


async def test_reprocess_is_409_while_pending_or_processing(client, db, job_queue):
    headers, _, doc_id = await setup_document(client)  # pending, in the queue
    url = f"/documents/{doc_id}/reprocess"
    assert (await client.post(url, headers=headers)).status_code == 409

    await db.execute(
        update(Document)
        .where(Document.id == uuid.UUID(doc_id))
        .values(status=DocumentStatus.processing, processing_run_id=uuid.uuid4())
    )
    await db.commit()
    response = await client.post(url, headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"] == "Document is already queued or being processed"
    assert job_queue.ids == [uuid.UUID(doc_id)]  # only the upload's job, no extra


async def test_reprocess_requeues_a_finished_document(client, job_queue):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()

    response = await client.post(f"/documents/{doc_id}/reprocess", headers=headers)
    assert response.status_code == 202
    body = response.json()
    assert (body["status"], body["attempts"], body["error_message"]) == ("pending", 0, None)
    assert job_queue.ids == [uuid.UUID(doc_id)]
    assert job_queue.not_yet_committed == []
    [result] = await job_queue.run_all()
    assert result.outcome == "ready"


async def test_two_simultaneous_reprocess_calls_queue_it_once(client, job_queue):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()
    url = f"/documents/{doc_id}/reprocess"
    responses = await asyncio.gather(*(client.post(url, headers=headers) for _ in range(2)))
    assert sorted(r.status_code for r in responses) == [202, 409]
    assert len(job_queue.ids) == 1


async def test_other_users_documents_are_404(client, job_queue):
    owner_headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()
    intruder = await auth_headers(client, "intruder@example.com")

    reprocess = await client.post(f"/documents/{doc_id}/reprocess", headers=intruder)
    status = await client.get(f"/documents/{doc_id}", headers=intruder)
    chunks = await client.get(f"/documents/{doc_id}/chunks", headers=intruder)
    missing = await client.get(f"/documents/{uuid.uuid4()}/chunks", headers=intruder)

    assert reprocess.status_code == status.status_code == chunks.status_code == 404
    assert missing.status_code == 404
    assert error_body(reprocess) == error_body(chunks) == error_body(missing)
    # The intruder's attempt didn't touch the owner's document or the queue.
    owner_view = await client.get(f"/documents/{doc_id}", headers=owner_headers)
    assert owner_view.json()["status"] == "ready"
    assert job_queue.ids == []


async def test_endpoints_require_authentication(client):
    headers, _, doc_id = await setup_document(client)
    assert (await client.post(f"/documents/{doc_id}/reprocess")).status_code == 401
    assert (await client.get(f"/documents/{doc_id}/chunks")).status_code == 401


async def test_chunks_are_removed_when_document_is_deleted(client, db, job_queue):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()
    assert await chunk_count(db, doc_id) > 0

    assert (await client.delete(f"/documents/{doc_id}", headers=headers)).status_code == 204
    assert await chunk_count(db, doc_id) == 0


async def test_chunks_are_removed_when_collection_is_deleted(client, db, job_queue):
    headers, collection_id, doc_id = await setup_document(client)
    await job_queue.run_all()
    assert await chunk_count(db, doc_id) > 0

    response = await client.delete(f"/collections/{collection_id}", headers=headers)
    assert response.status_code == 204
    assert await chunk_count(db, doc_id) == 0


# --- One run per document ---


class Gate(FakeEmbeddingProvider):
    """Pauses inside each embedding request until released, so a test can act
    while a run is in the middle of embedding."""

    def __init__(self, fail_after_release: bool = False) -> None:
        super().__init__(model="gate-model", dimensions=1024, batch_size=1)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.fail_after_release = fail_after_release

    async def _embed_batch(self, texts, input_type):
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        if self.fail_after_release:
            raise EmbeddingError("Embedding service rejected the input")
        return await super()._embed_batch(texts, input_type)


def three_chunks(*args):
    return [ChunkData(i, f"Chunk {i} text.", None, None, 4) for i in range(3)]


async def start_gated_run(job_queue, gate):
    """Start the worker on the queued job; return once it is mid-embedding."""
    run = asyncio.create_task(job_queue.run_all(gate))
    await asyncio.wait_for(gate.entered.wait(), timeout=10)
    return run


async def requeue(client, headers, doc_id):
    response = await client.post(f"/documents/{doc_id}/reprocess", headers=headers)
    assert response.status_code == 202


async def take_over(db, doc_id: str) -> uuid.UUID:
    """What a second run does when it decides the first one was abandoned."""
    other_run = uuid.uuid4()
    await db.execute(
        update(Document)
        .where(Document.id == uuid.UUID(doc_id))
        .values(processing_run_id=other_run, processing_started_at=datetime.now(UTC))
    )
    await db.commit()
    return other_run


async def chunk_ids(db, doc_id: str) -> list[uuid.UUID]:
    rows = await db.scalars(
        select(Chunk.id).where(Chunk.document_id == uuid.UUID(doc_id)).order_by(Chunk.chunk_index)
    )
    return list(rows)


async def test_reprocess_while_the_worker_is_embedding_gets_409(client, job_queue):
    headers, _, doc_id = await setup_document(client)
    gate = Gate()
    run = await start_gated_run(job_queue, gate)

    second = await client.post(f"/documents/{doc_id}/reprocess", headers=headers)
    assert second.status_code == 409

    gate.release.set()
    [result] = await run
    assert result.outcome == "ready"


async def test_run_that_lost_its_claim_stops_before_the_next_request(
    client, db, monkeypatch, job_queue
):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()  # some existing chunks
    before = await chunk_ids(db, doc_id)

    monkeypatch.setattr(processing, "chunk_segments", three_chunks)
    await requeue(client, headers, doc_id)
    gate = Gate()
    run = await start_gated_run(job_queue, gate)
    other_run = await take_over(db, doc_id)
    gate.release.set()
    [result] = await run

    assert result.outcome == "lost"
    assert gate.calls == 1  # stopped before sending batches 2 and 3
    assert await chunk_ids(db, doc_id) == before  # wrote nothing
    db.expire_all()
    document = await db.get(Document, uuid.UUID(doc_id))
    assert (document.status, document.processing_run_id) == (DocumentStatus.processing, other_run)


async def test_takeover_after_the_last_batch_still_blocks_the_final_write(client, db, job_queue):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()
    before = await chunk_ids(db, doc_id)

    await requeue(client, headers, doc_id)
    gate = Gate()
    gate.batch_size = 64  # the whole document in one batch: no heartbeat after it
    run = await start_gated_run(job_queue, gate)
    await take_over(db, doc_id)
    gate.release.set()
    [result] = await run

    assert result.outcome == "lost"
    assert await chunk_ids(db, doc_id) == before


async def test_run_that_fails_after_losing_its_claim_leaves_the_document_alone(
    client, db, job_queue
):
    headers, _, doc_id = await setup_document(client)
    gate = Gate(fail_after_release=True)
    gate.batch_size = 64
    run = await start_gated_run(job_queue, gate)
    other_run = await take_over(db, doc_id)
    gate.release.set()
    await run

    db.expire_all()
    document = await db.get(Document, uuid.UUID(doc_id))
    # Not marked "failed" by the old run: the document belongs to the new one.
    assert document.status == DocumentStatus.processing
    assert document.processing_run_id == other_run
    assert document.error_message is None


async def test_finished_run_releases_its_claim(client, db, job_queue):
    headers, _, doc_id = await setup_document(client)
    await job_queue.run_all()
    db.expire_all()
    document = await db.get(Document, uuid.UUID(doc_id))
    assert document.processing_run_id is None
    assert document.heartbeat_at is None
    assert document.finished_at is not None
