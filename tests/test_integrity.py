import logging
import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import Chunk, Document, DocumentStatus
from app.services.integrity import ready_documents_missing_embeddings, report_missing_embeddings
from tests.helpers import auth_headers, create_collection, upload

TEXT = ("This sentence is long enough to pass the minimum text check. " * 300).encode()


async def processed_document(client, name="notes.txt", email="owner@example.com") -> uuid.UUID:
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    doc_id = (await upload(client, headers, collection_id, name, TEXT)).json()["id"]
    assert (await client.post(f"/documents/{doc_id}/process", headers=headers)).status_code == 200
    return uuid.UUID(doc_id)


async def test_healthy_documents_are_not_reported(client, db):
    await processed_document(client)
    assert await ready_documents_missing_embeddings(db) == []


async def test_ready_document_with_null_vectors_is_reported(client, db):
    doc_id = await processed_document(client)
    first_two = Chunk.chunk_index < 2
    await db.execute(
        update(Chunk)
        .where(Chunk.document_id == doc_id, first_two)
        .values(embedding=None, embedding_model=None)
    )
    await db.commit()

    [problem] = await ready_documents_missing_embeddings(db)
    assert problem.document_id == doc_id
    assert problem.filename == "notes.txt"
    assert problem.chunks_without_vector == 2
    assert problem.total_chunks > 2


async def test_a_vector_without_a_model_name_is_reported_too(client, db):
    doc_id = await processed_document(client)
    await db.execute(
        update(Chunk)
        .where(Chunk.document_id == doc_id, Chunk.chunk_index == 0)
        .values(embedding_model=None)
    )
    await db.commit()
    [problem] = await ready_documents_missing_embeddings(db)
    assert problem.chunks_without_vector == 1


async def test_only_ready_documents_count(client, db):
    doc_id = await processed_document(client)
    await db.execute(update(Chunk).where(Chunk.document_id == doc_id).values(embedding=None))
    await db.execute(
        update(Document).where(Document.id == doc_id).values(status=DocumentStatus.failed)
    )
    await db.commit()
    assert await ready_documents_missing_embeddings(db) == []


async def test_startup_check_warns_and_names_the_document(client, db, test_engine, caplog):
    caplog.set_level(logging.INFO, logger="app")
    doc_id = await processed_document(client)
    await db.execute(update(Chunk).where(Chunk.document_id == doc_id).values(embedding=None))
    await db.commit()

    await report_missing_embeddings(async_sessionmaker(test_engine))

    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert str(doc_id) in warning.getMessage()
    assert "process them again" in warning.getMessage()


async def test_startup_check_is_silent_when_all_is_well(client, test_engine, caplog):
    caplog.set_level(logging.INFO, logger="app")
    await processed_document(client)
    await report_missing_embeddings(async_sessionmaker(test_engine))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_startup_check_never_stops_the_app_starting(caplog):
    def broken_session():
        raise ConnectionError("database is down")

    await report_missing_embeddings(broken_session)  # must not raise
    assert "Could not check documents for missing embeddings" in caplog.text
