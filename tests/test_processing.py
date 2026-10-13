import threading
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.chunking import ChunkData
from app.services.processing import GENERIC_ERROR, process_document
from app.services.storage import get_storage
from tests.fixtures import build_docx, build_pdf
from tests.helpers import auth_headers, create_collection, upload

FILLER = "This sentence is long enough to pass the minimum text check. "


async def uploaded(client, db: AsyncSession, filename: str, content: bytes) -> Document:
    headers = await auth_headers(client, f"{uuid.uuid4().hex[:8]}@example.com")
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, filename, content)
    assert response.status_code == 201, response.text
    return await db.get(Document, uuid.UUID(response.json()["id"]))


async def stored_chunks(db: AsyncSession, document: Document) -> list[Chunk]:
    """What is in the database right now (populate_existing overwrites objects
    the session already holds, so stale in-memory values can't hide changes)."""
    result = await db.scalars(
        select(Chunk)
        .where(Chunk.document_id == document.id)
        .order_by(Chunk.chunk_index)
        .execution_options(populate_existing=True)
    )
    return list(result.all())


async def test_txt_becomes_ready_with_chunks(client, db):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.ready
    assert document.error_message is None
    assert document.processing_started_at is None
    chunks = await stored_chunks(db, document)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.owner_id == document.owner_id for c in chunks)
    assert all(c.collection_id == document.collection_id for c in chunks)
    assert all(0 < c.token_count <= 500 for c in chunks)


async def test_multi_page_pdf_chunks_keep_page_numbers(client, db):
    pdf = build_pdf([[f"Page {n}: {FILLER}"] for n in (1, 2, 3)])
    document = await uploaded(client, db, "report.pdf", pdf)
    await process_document(db, get_storage(), document)

    chunks = await stored_chunks(db, document)
    assert [(c.page_number, c.content.split(":")[0]) for c in chunks] == [
        (1, "Page 1"),
        (2, "Page 2"),
        (3, "Page 3"),
    ]


async def test_docx_chunks_keep_section_titles(client, db):
    data = build_docx([("heading", "Methods"), ("paragraph", FILLER * 3)])
    document = await uploaded(client, db, "paper.docx", data)
    await process_document(db, get_storage(), document)

    [chunk] = await stored_chunks(db, document)
    assert chunk.section_title == "Methods"
    assert chunk.page_number is None


async def test_markdown_chunks_keep_section_titles(client, db):
    document = await uploaded(client, db, "readme.md", f"# Install\n\n{FILLER * 3}".encode())
    await process_document(db, get_storage(), document)
    [chunk] = await stored_chunks(db, document)
    assert chunk.section_title == "Install"


async def test_null_characters_are_stripped_before_saving(client, db):
    # Postgres would reject the insert if a \x00 got through.
    document = await uploaded(client, db, "nulls.txt", f"a\x00b {FILLER}".encode())
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.ready
    [chunk] = await stored_chunks(db, document)
    assert "\x00" not in chunk.content
    assert chunk.content.startswith("ab ")


async def test_scanned_pdf_fails_with_readable_message(client, db):
    document = await uploaded(client, db, "scan.pdf", build_pdf([[], []]))
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.failed
    assert "scanned" in document.error_message
    assert await stored_chunks(db, document) == []


async def test_corrupt_pdf_fails_without_crashing(client, db):
    # Passes the upload check (starts with %PDF) but isn't a real PDF.
    document = await uploaded(client, db, "broken.pdf", b"%PDF-1.7\n\x00\x01garbage")
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == "File is not a readable PDF"


async def test_unexpected_error_is_logged_but_not_shown_to_user(client, db, monkeypatch, caplog):
    def explode(*args):
        raise RuntimeError("secret detail at /srv/internal/path.py line 42")

    monkeypatch.setattr(processing, "extract_and_chunk", explode)
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == GENERIC_ERROR
    assert "secret detail" in caplog.text  # the server log keeps the details


async def test_missing_stored_file_fails_with_readable_message(client, db, storage_dir):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    (storage_dir / document.storage_key).unlink()
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == "The stored file is missing"


async def test_processing_twice_gives_same_chunks_without_duplicates(client, db):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await process_document(db, get_storage(), document)
    first = [(c.chunk_index, c.content) for c in await stored_chunks(db, document)]

    await process_document(db, get_storage(), document)
    second = [(c.chunk_index, c.content) for c in await stored_chunks(db, document)]

    assert document.status == DocumentStatus.ready
    assert second == first


async def test_failure_mid_insert_keeps_previous_chunks(client, db, monkeypatch):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await process_document(db, get_storage(), document)
    before = [(c.id, c.content) for c in await stored_chunks(db, document)]
    assert before

    # Two chunks with the same index: the old chunks are deleted, then the
    # insert violates the unique constraint halfway through the transaction.
    duplicate = ChunkData(0, "new text", None, None, 2)
    monkeypatch.setattr(processing, "chunk_segments", lambda *args: [duplicate, duplicate])
    await process_document(db, get_storage(), document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == GENERIC_ERROR
    after = [(c.id, c.content) for c in await stored_chunks(db, document)]
    assert after == before


async def test_failed_document_can_be_processed_again(client, db, monkeypatch):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    with monkeypatch.context() as patch:
        patch.setattr(processing, "extract_and_chunk", lambda *args: 1 / 0)
        await process_document(db, get_storage(), document)
    assert document.status == DocumentStatus.failed

    await process_document(db, get_storage(), document)
    assert document.status == DocumentStatus.ready
    assert document.error_message is None


async def test_already_processing_is_rejected_with_409(client, db):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await db.execute(
        update(Document)
        .where(Document.id == document.id)
        .values(status=DocumentStatus.processing, processing_started_at=datetime.now(UTC))
    )
    await db.commit()

    with pytest.raises(HTTPException) as info:
        await process_document(db, get_storage(), document)
    assert info.value.status_code == 409


async def test_abandoned_processing_claim_can_be_taken_over(client, db):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    long_ago = datetime.now(UTC) - timedelta(hours=1)
    await db.execute(
        update(Document)
        .where(Document.id == document.id)
        .values(status=DocumentStatus.processing, processing_started_at=long_ago)
    )
    await db.commit()

    await process_document(db, get_storage(), document)
    assert document.status == DocumentStatus.ready


async def test_extraction_runs_off_the_event_loop_thread(client, db, monkeypatch):
    seen = {}
    real = processing.extract_and_chunk

    def spy(*args):
        seen["thread"] = threading.current_thread()
        return real(*args)

    monkeypatch.setattr(processing, "extract_and_chunk", spy)
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await process_document(db, get_storage(), document)

    assert seen["thread"] is not threading.current_thread()
