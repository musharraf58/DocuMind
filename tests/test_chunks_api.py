import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update

from app.api.deps import embedding_provider
from app.core.config import get_settings
from app.main import app
from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.chunking import ChunkData
from app.services.embeddings import EmbeddingError, FakeEmbeddingProvider
from tests.fixtures import build_pdf
from tests.helpers import auth_headers, create_collection, upload

TEXT = ("This sentence is long enough to pass the minimum text check. " * 300).encode()


async def setup_document(client, email="owner@example.com", content=TEXT, name="notes.txt"):
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, name, content)
    assert response.status_code == 201
    return headers, collection_id, response.json()["id"]


async def chunk_count(db, document_id: str) -> int:
    return await db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
    )


async def test_process_then_list_chunks(client):
    headers, _, doc_id = await setup_document(client)

    response = await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["error_message"] is None

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


async def test_chunks_are_paginated_in_order(client):
    headers, _, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=headers)
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


async def test_unreadable_file_returns_200_with_failed_status(client):
    headers, _, doc_id = await setup_document(client, content=build_pdf([[]]), name="scan.pdf")
    response = await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "scanned" in body["error_message"]
    assert "storage_key" not in body


async def test_process_returns_409_while_already_processing(client, db):
    headers, _, doc_id = await setup_document(client)
    await db.execute(
        update(Document)
        .where(Document.id == uuid.UUID(doc_id))
        .values(status=DocumentStatus.processing, processing_started_at=datetime.now(UTC))
    )
    await db.commit()

    response = await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert response.status_code == 409
    # The running job's claim is left alone.
    assert (await client.get(f"/documents/{doc_id}", headers=headers)).json()["status"] == (
        "processing"
    )


async def test_other_users_documents_are_404(client):
    owner_headers, _, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=owner_headers)
    intruder = await auth_headers(client, "intruder@example.com")

    process = await client.post(f"/documents/{doc_id}/process", headers=intruder)
    chunks = await client.get(f"/documents/{doc_id}/chunks", headers=intruder)
    missing = await client.get(f"/documents/{uuid.uuid4()}/chunks", headers=intruder)

    assert process.status_code == chunks.status_code == missing.status_code == 404
    assert process.json() == chunks.json() == missing.json()
    # The intruder's attempt didn't touch the owner's document.
    owner_view = await client.get(f"/documents/{doc_id}", headers=owner_headers)
    assert owner_view.json()["status"] == "ready"


async def test_endpoints_require_authentication(client):
    headers, _, doc_id = await setup_document(client)
    assert (await client.post(f"/documents/{doc_id}/process")).status_code == 401
    assert (await client.get(f"/documents/{doc_id}/chunks")).status_code == 401


async def test_chunks_are_removed_when_document_is_deleted(client, db):
    headers, _, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert await chunk_count(db, doc_id) > 0

    assert (await client.delete(f"/documents/{doc_id}", headers=headers)).status_code == 204
    assert await chunk_count(db, doc_id) == 0


async def test_chunks_are_removed_when_collection_is_deleted(client, db):
    headers, collection_id, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert await chunk_count(db, doc_id) > 0

    response = await client.delete(f"/collections/{collection_id}", headers=headers)
    assert response.status_code == 204
    assert await chunk_count(db, doc_id) == 0


async def test_process_returns_503_when_embedding_is_misconfigured(client, monkeypatch):
    headers, _, doc_id = await setup_document(client)
    monkeypatch.setattr(get_settings(), "EMBEDDING_DIMENSIONS", 512)
    response = await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert response.status_code == 503
    assert "don't match the database" in response.json()["detail"]


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


async def start_gated_run(client, headers, doc_id, gate):
    app.dependency_overrides[embedding_provider] = lambda: gate
    run = asyncio.create_task(client.post(f"/documents/{doc_id}/process", headers=headers))
    await asyncio.wait_for(gate.entered.wait(), timeout=10)  # now mid-embedding
    return run


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


async def test_second_call_while_a_run_is_embedding_gets_409(client):
    headers, _, doc_id = await setup_document(client)
    gate = Gate()
    first = await start_gated_run(client, headers, doc_id, gate)

    second = await client.post(f"/documents/{doc_id}/process", headers=headers)
    assert second.status_code == 409
    assert second.json()["detail"] == "Document is already being processed"

    gate.release.set()
    done = await first
    assert done.status_code == 200 and done.json()["status"] == "ready"


async def test_run_that_lost_its_claim_stops_before_the_next_request(client, db, monkeypatch):
    headers, _, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=headers)  # some existing chunks
    before = await chunk_ids(db, doc_id)

    monkeypatch.setattr(processing, "chunk_segments", three_chunks)
    gate = Gate()
    run = await start_gated_run(client, headers, doc_id, gate)
    other_run = await take_over(db, doc_id)
    gate.release.set()
    response = await run

    assert response.status_code == 409
    assert response.json()["detail"] == "Processing was taken over by another run"
    assert gate.calls == 1  # stopped before sending batches 2 and 3
    assert await chunk_ids(db, doc_id) == before  # wrote nothing
    db.expire_all()
    document = await db.get(Document, uuid.UUID(doc_id))
    assert (document.status, document.processing_run_id) == (DocumentStatus.processing, other_run)


async def test_takeover_after_the_last_batch_still_blocks_the_final_write(client, db):
    headers, _, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=headers)
    before = await chunk_ids(db, doc_id)

    gate = Gate()
    gate.batch_size = 64  # the whole document in one batch: no heartbeat after it
    run = await start_gated_run(client, headers, doc_id, gate)
    await take_over(db, doc_id)
    gate.release.set()
    response = await run

    assert response.status_code == 409
    assert await chunk_ids(db, doc_id) == before


async def test_run_that_fails_after_losing_its_claim_leaves_the_document_alone(client, db):
    headers, _, doc_id = await setup_document(client)
    gate = Gate(fail_after_release=True)
    gate.batch_size = 64
    run = await start_gated_run(client, headers, doc_id, gate)
    other_run = await take_over(db, doc_id)
    gate.release.set()
    await run

    db.expire_all()
    document = await db.get(Document, uuid.UUID(doc_id))
    # Not marked "failed" by the old run: the document belongs to the new one.
    assert document.status == DocumentStatus.processing
    assert document.processing_run_id == other_run
    assert document.error_message is None


async def test_finished_run_releases_its_claim(client, db):
    headers, _, doc_id = await setup_document(client)
    await client.post(f"/documents/{doc_id}/process", headers=headers)
    db.expire_all()
    document = await db.get(Document, uuid.UUID(doc_id))
    assert document.processing_run_id is None
    assert document.processing_started_at is None
