import threading
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.chunking import ChunkData
from app.services.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    RetryableEmbeddingError,
    VoyageEmbeddingProvider,
)
from app.services.embeddings.fake import fake_vector
from app.services.processing import GENERIC_ERROR, process_document
from app.services.storage import get_storage
from tests.embedding_fakes import DIMENSIONS, RecordingSleep, ScriptedProvider
from tests.fixtures import build_docx, build_pdf
from tests.helpers import auth_headers, create_collection, upload

FILLER = "This sentence is long enough to pass the minimum text check. "


async def uploaded(client, db: AsyncSession, filename: str, content: bytes) -> Document:
    headers = await auth_headers(client, f"{uuid.uuid4().hex[:8]}@example.com")
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, filename, content)
    assert response.status_code == 201, response.text
    return await db.get(Document, uuid.UUID(response.json()["id"]))


async def run(db: AsyncSession, document: Document, embedder: EmbeddingProvider | None = None):
    embedder = embedder or FakeEmbeddingProvider(model="fake-model", dimensions=DIMENSIONS)
    return await process_document(db, get_storage(), embedder, document)


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
    await run(db, document)

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
    await run(db, document)

    chunks = await stored_chunks(db, document)
    assert [(c.page_number, c.content.split(":")[0]) for c in chunks] == [
        (1, "Page 1"),
        (2, "Page 2"),
        (3, "Page 3"),
    ]


async def test_docx_chunks_keep_section_titles(client, db):
    data = build_docx([("heading", "Methods"), ("paragraph", FILLER * 3)])
    document = await uploaded(client, db, "paper.docx", data)
    await run(db, document)

    [chunk] = await stored_chunks(db, document)
    assert chunk.section_title == "Methods"
    assert chunk.page_number is None


async def test_markdown_chunks_keep_section_titles(client, db):
    document = await uploaded(client, db, "readme.md", f"# Install\n\n{FILLER * 3}".encode())
    await run(db, document)
    [chunk] = await stored_chunks(db, document)
    assert chunk.section_title == "Install"


async def test_null_characters_are_stripped_before_saving(client, db):
    # Postgres would reject the insert if a \x00 got through.
    document = await uploaded(client, db, "nulls.txt", f"a\x00b {FILLER}".encode())
    await run(db, document)

    assert document.status == DocumentStatus.ready
    [chunk] = await stored_chunks(db, document)
    assert "\x00" not in chunk.content
    assert chunk.content.startswith("ab ")


async def test_scanned_pdf_fails_with_readable_message(client, db):
    document = await uploaded(client, db, "scan.pdf", build_pdf([[], []]))
    await run(db, document)

    assert document.status == DocumentStatus.failed
    assert "scanned" in document.error_message
    assert await stored_chunks(db, document) == []


async def test_corrupt_pdf_fails_without_crashing(client, db):
    # Passes the upload check (starts with %PDF) but isn't a real PDF.
    document = await uploaded(client, db, "broken.pdf", b"%PDF-1.7\n\x00\x01garbage")
    await run(db, document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == "File is not a readable PDF"


async def test_unexpected_error_is_logged_but_not_shown_to_user(client, db, monkeypatch, caplog):
    def explode(*args):
        raise RuntimeError("secret detail at /srv/internal/path.py line 42")

    monkeypatch.setattr(processing, "extract_and_chunk", explode)
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == GENERIC_ERROR
    assert "secret detail" in caplog.text  # the server log keeps the details


async def test_missing_stored_file_fails_with_readable_message(client, db, storage_dir):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    (storage_dir / document.storage_key).unlink()
    await run(db, document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == "The stored file is missing"


async def test_processing_twice_gives_same_chunks_without_duplicates(client, db):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)
    first = [(c.chunk_index, c.content) for c in await stored_chunks(db, document)]

    await run(db, document)
    second = [(c.chunk_index, c.content) for c in await stored_chunks(db, document)]

    assert document.status == DocumentStatus.ready
    assert second == first


async def test_failure_mid_insert_keeps_previous_chunks(client, db, monkeypatch):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)
    before = [(c.id, c.content) for c in await stored_chunks(db, document)]
    assert before

    # Cleaning normally strips \x00, which Postgres refuses to store. Bypass
    # it: the old chunks are deleted, then the insert fails inside the
    # transaction, a genuine database error.
    bad = [ChunkData(0, "new text", None, None, 2), ChunkData(1, "bad \x00 text", None, None, 3)]
    monkeypatch.setattr(processing, "chunk_segments", lambda *args: bad)
    await run(db, document)

    assert document.status == DocumentStatus.failed
    assert document.error_message == GENERIC_ERROR
    after = [(c.id, c.content) for c in await stored_chunks(db, document)]
    assert after == before


async def test_failed_document_can_be_processed_again(client, db, monkeypatch):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    with monkeypatch.context() as patch:
        patch.setattr(processing, "extract_and_chunk", lambda *args: 1 / 0)
        await run(db, document)
    assert document.status == DocumentStatus.failed

    await run(db, document)
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
        await run(db, document)
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

    await run(db, document)
    assert document.status == DocumentStatus.ready


async def test_extraction_runs_off_the_event_loop_thread(client, db, monkeypatch):
    seen = {}
    real = processing.extract_and_chunk

    def spy(*args):
        seen["thread"] = threading.current_thread()
        return real(*args)

    monkeypatch.setattr(processing, "extract_and_chunk", spy)
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document)

    assert seen["thread"] is not threading.current_thread()


# --- Embeddings ---


def ten_chunks(*args) -> list[ChunkData]:
    return [ChunkData(i, f"Chunk {i}: {FILLER}", None, None, 15) for i in range(10)]


async def test_every_chunk_gets_a_vector_and_model_name(client, db):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)

    chunks = await stored_chunks(db, document)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.embedding) == DIMENSIONS
        assert chunk.embedding_model == "fake-model"
    # Each vector belongs to its own chunk's text.
    expected = fake_vector(chunks[0].content, DIMENSIONS)
    assert [round(x, 5) for x in chunks[0].embedding] == [round(x, 5) for x in expected]


async def test_batch_size_three_with_ten_chunks_makes_four_calls(client, db, monkeypatch):
    monkeypatch.setattr(processing, "chunk_segments", ten_chunks)
    embedder = ScriptedProvider(batch_size=3)
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document, embedder)

    assert [len(batch) for batch, _ in embedder.calls] == [3, 3, 3, 1]
    chunks = await stored_chunks(db, document)
    assert len(chunks) == 10
    assert all(c.embedding is not None for c in chunks)


async def test_blank_chunks_are_never_sent_or_stored(client, db, monkeypatch):
    chunks_with_blank = [
        ChunkData(0, "first chunk", None, None, 2),
        ChunkData(1, "   \n ", None, None, 1),
        ChunkData(2, "third chunk", None, None, 2),
    ]
    monkeypatch.setattr(processing, "chunk_segments", lambda *args: chunks_with_blank)
    embedder = ScriptedProvider()
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document, embedder)

    assert [batch for batch, _ in embedder.calls] == [["first chunk", "third chunk"]]
    stored = await stored_chunks(db, document)
    assert [(c.chunk_index, c.content) for c in stored] == [(0, "first chunk"), (1, "third chunk")]


async def test_no_transaction_is_open_while_embedding(client, db):
    seen = []

    class Watching(ScriptedProvider):
        async def _embed_batch(self, texts, input_type):
            seen.append(db.in_transaction())
            return await super()._embed_batch(texts, input_type)

    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document, Watching())
    assert seen == [False]
    assert document.status == DocumentStatus.ready


async def test_retryable_failures_then_success_is_ready(client, db):
    embedder = ScriptedProvider(
        failures=[RetryableEmbeddingError("429"), RetryableEmbeddingError("503")]
    )
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document, embedder)
    assert document.status == DocumentStatus.ready
    assert len(embedder.calls) == 3


async def test_provider_failure_keeps_old_chunks_and_vectors_and_hides_key(client, db):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)
    before = [(c.id, c.content, list(c.embedding)) for c in await stored_chunks(db, document)]

    api_key = "pa-SUPER-secret-key"

    def reject(request):
        # Voyage's reply even echoes the key back; none of it may leak.
        return httpx.Response(401, json={"detail": f"Invalid key {api_key}"})

    voyage = VoyageEmbeddingProvider(
        api_key=SecretStr(api_key),
        transport=httpx.MockTransport(reject),
        model="voyage-4-lite",
        dimensions=DIMENSIONS,
        sleep=RecordingSleep(),
    )
    await run(db, document, voyage)

    assert document.status == DocumentStatus.failed
    assert document.error_message == "Embedding service rejected the API key"
    assert api_key not in document.error_message
    after = [(c.id, c.content, list(c.embedding)) for c in await stored_chunks(db, document)]
    assert after == before
    assert all(c.embedding_model == "fake-model" for c in await stored_chunks(db, document))


async def test_exhausted_retries_fail_with_safe_message(client, db):
    embedder = ScriptedProvider(
        failures=[RetryableEmbeddingError("Embedding service is unavailable; try again later")] * 5,
        max_retries=2,
    )
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    await run(db, document, embedder)
    assert document.status == DocumentStatus.failed
    assert document.error_message == "Embedding service is unavailable; try again later"


async def test_reprocessing_replaces_vectors_without_duplicates(client, db):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)
    first = await stored_chunks(db, document)
    first_ids = {c.id for c in first}

    await run(db, document, FakeEmbeddingProvider(model="fake-model-v2", dimensions=DIMENSIONS))
    second = await stored_chunks(db, document)

    assert len(second) == len(first)
    assert [c.chunk_index for c in second] == list(range(len(second)))
    assert not first_ids & {c.id for c in second}  # old rows are gone, not kept alongside
    assert {c.embedding_model for c in second} == {"fake-model-v2"}
    total = await db.scalar(select(func.count()).select_from(Chunk))
    assert total == len(second)
