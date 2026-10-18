import threading
import uuid
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import clock
from app.core.config import get_settings
from app.models import Chunk, Document, DocumentStatus
from app.services import processing
from app.services.chunking import ChunkData
from app.services.documents import mark_pending
from app.services.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    RetryableEmbeddingError,
    VoyageEmbeddingProvider,
)
from app.services.embeddings.fake import fake_vector
from app.services.jobs import GENERIC_ERROR, Outcome, run_job
from app.services.storage import get_storage
from tests.embedding_fakes import (
    DIMENSIONS,
    FakeClock,
    RecordingSleep,
    ScriptedProvider,
    limited_provider,
)
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
    """Queue the document (as reprocess does) and run the job on it, as the worker would."""
    embedder = embedder or FakeEmbeddingProvider(model="fake-model", dimensions=DIMENSIONS)
    await db.refresh(document)
    if document.status != DocumentStatus.pending:
        await mark_pending(db, document)
    result = await run_job(db, document.id, get_storage(), embedder)
    await db.refresh(document)
    return result


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
    assert document.attempts == 1
    assert document.processing_started_at is not None
    assert document.finished_at >= document.processing_started_at
    assert document.heartbeat_at is None and document.processing_run_id is None
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


async def test_job_leaves_a_document_another_worker_is_processing_alone(client, db):
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    other_run = uuid.uuid4()
    await db.execute(
        update(Document)
        .where(Document.id == document.id)
        .values(status=DocumentStatus.processing, processing_run_id=other_run)
    )
    await db.commit()

    result = await run_job(
        db, document.id, get_storage(), FakeEmbeddingProvider(model="m", dimensions=DIMENSIONS)
    )
    assert result.outcome == Outcome.skipped
    await db.refresh(document)
    assert (document.status, document.processing_run_id) == (DocumentStatus.processing, other_run)
    assert document.attempts == 0


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


# --- Long, rate-limited runs ---


def fake_chunks(count: int, tokens: int) -> list[ChunkData]:
    return [ChunkData(i, f"Chunk number {i}.", None, None, tokens) for i in range(count)]


async def test_long_rate_limited_run_keeps_its_heartbeat_fresh(
    client, db, test_engine, monkeypatch
):
    # 200 chunks x 500 tokens = 100,000 tokens, at 7,600 per minute: about 13
    # minutes of simulated time, longer than the STALE_PROCESSING_MINUTES (10)
    # after which the sweeper would treat the document as abandoned.
    monkeypatch.setattr(processing, "chunk_segments", lambda *args: fake_chunks(200, tokens=500))
    fake = FakeClock()
    real_start = datetime.now(UTC)
    monkeypatch.setattr(clock, "utcnow", lambda: real_start + timedelta(seconds=fake.now - 1000))
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    other_sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    heartbeat_ages: list[timedelta] = []

    class Probe(ScriptedProvider):
        async def _embed_batch(self, texts, input_type):
            # What the sweeper would see right now, mid-run, from its own session.
            async with other_sessions() as other:
                row = await other.get(Document, document.id)
                heartbeat_ages.append(clock.utcnow() - row.heartbeat_at)
            return await super()._embed_batch(texts, input_type)

    provider = limited_provider(
        fake, reserve_tokens=400, reserve_requests=1, provider_cls=Probe, model="long-run-model"
    )
    await run(db, document, provider)

    assert document.status == DocumentStatus.ready
    assert len(provider.calls) == 14
    assert fake.now - 1000 > 12 * 60  # really did take longer than the stale limit...
    stale = timedelta(minutes=get_settings().STALE_PROCESSING_MINUTES)
    assert max(heartbeat_ages) < min(stale, timedelta(minutes=2))  # ...but never looked it
    stored = await stored_chunks(db, document)
    assert len(stored) == 200
    assert all(c.embedding is not None and c.embedding_model == "long-run-model" for c in stored)


async def test_nothing_is_written_while_a_long_run_waits(client, db, test_engine, monkeypatch):
    """Until every batch has succeeded, the database still shows the old state."""
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)
    before = [(c.id, c.embedding_model) for c in await stored_chunks(db, document)]

    monkeypatch.setattr(processing, "chunk_segments", lambda *args: fake_chunks(40, tokens=500))
    clock = FakeClock()
    other_sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    seen_during_run: list[list[tuple]] = []

    class Peek(ScriptedProvider):
        async def _embed_batch(self, texts, input_type):
            async with other_sessions() as other:
                rows = await other.scalars(
                    select(Chunk)
                    .where(Chunk.document_id == document.id)
                    .order_by(Chunk.chunk_index)
                )
                seen_during_run.append([(c.id, c.embedding_model) for c in rows])
            return await super()._embed_batch(texts, input_type)

    provider = limited_provider(clock, reserve_tokens=400, reserve_requests=1, provider_cls=Peek)
    await run(db, document, provider)

    assert len(seen_during_run) == len(provider.calls) > 2
    assert all(snapshot == before for snapshot in seen_during_run)  # old chunks until the end
    after = await stored_chunks(db, document)
    assert len(after) == 40 and {c.embedding_model for c in after} == {"test-model"}


async def test_failure_late_in_a_long_run_keeps_the_old_chunks(client, db, monkeypatch):
    document = await uploaded(client, db, "notes.txt", (FILLER * 300).encode())
    await run(db, document)
    before = [(c.id, list(c.embedding)) for c in await stored_chunks(db, document)]

    monkeypatch.setattr(processing, "chunk_segments", lambda *args: fake_chunks(60, tokens=500))
    clock = FakeClock()
    # Batches 1-3 succeed (minutes pass); then the provider rejects everything.
    provider = limited_provider(clock, reserve_tokens=400, reserve_requests=1, max_retries=0)
    real = provider._embed_batch
    sent = 0

    async def flaky(texts, input_type):
        nonlocal sent
        sent += 1
        if sent > 3:
            raise EmbeddingError("Embedding service rejected the API key")
        return await real(texts, input_type)

    provider._embed_batch = flaky
    await run(db, document, provider)

    assert sent == 4
    assert document.status == DocumentStatus.failed
    assert document.error_message == "Embedding service rejected the API key"
    after = [(c.id, list(c.embedding)) for c in await stored_chunks(db, document)]
    assert after == before


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


async def test_provider_outage_sends_the_document_back_for_a_later_retry(client, db):
    # The client's own quick retries run out: a transient failure for the job.
    embedder = ScriptedProvider(
        failures=[RetryableEmbeddingError("Embedding service is unavailable; try again later")] * 5,
        max_retries=2,
    )
    document = await uploaded(client, db, "notes.txt", FILLER.encode())
    result = await run(db, document, embedder)

    assert result.outcome == Outcome.retry
    assert result.retry_in == 30  # JOB_RETRY_BASE_SECONDS after the first attempt
    assert document.status == DocumentStatus.pending
    assert document.attempts == 1
    assert document.error_message == (
        "Attempt 1 of 4 failed: Embedding service is unavailable; try again later"
    )


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
