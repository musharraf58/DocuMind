import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select, update

from app.models import Chunk, Document, DocumentStatus
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
