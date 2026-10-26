"""The processing pipeline for one document: read the stored file, extract
and chunk the text, embed the chunks, and save them.

Run by the background job (app/services/jobs.py), which claims the document
first and decides what to do when this raises.
"""

import logging
import uuid
from dataclasses import replace

import anyio
from sqlalchemy import delete, insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.core.config import Settings, get_settings
from app.models import Chunk, Document, DocumentStatus
from app.services import usage
from app.services.chunking import ChunkData, chunk_segments
from app.services.embeddings import EmbeddingProvider, Vector
from app.services.embeddings.errors import DailyEmbeddingQuotaError
from app.services.extraction import ExtractionError, ExtractionLimits, extract
from app.services.quotas import EMBED_TOKENS
from app.services.storage import StorageBackend

logger = logging.getLogger(__name__)


class ClaimLostError(Exception):
    """This run no longer holds the document: it was deleted, or the sweeper
    decided this run had died and handed the document to a new one. We must
    stop without writing anything."""


def ours(document_id: uuid.UUID, run_id: uuid.UUID):
    """WHERE clause: this document, and the claim is still this run's."""
    return (Document.id == document_id) & (Document.processing_run_id == run_id)


def extract_and_chunk(data: bytes, content_type: str, settings: Settings) -> list[ChunkData]:
    """The CPU-heavy part. Synchronous; run it in a worker thread."""
    segments = extract(data, content_type, ExtractionLimits.from_settings(settings))
    return chunk_segments(segments, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)


async def _read_file(storage: StorageBackend, key: str) -> bytes:
    try:
        stream = await storage.open(key)
    except FileNotFoundError as exc:
        raise ExtractionError("The stored file is missing") from exc
    return b"".join([part async for part in stream])


def _drop_blank(chunks: list[ChunkData]) -> list[ChunkData]:
    """Defensive: the chunker never emits blank chunks, but a blank text must
    never reach the embedding provider. Renumbers so indexes stay 0..n-1."""
    kept = [chunk for chunk in chunks if chunk.content.strip()]
    return [replace(chunk, chunk_index=i) for i, chunk in enumerate(kept)]


async def heartbeat(db: AsyncSession, document_id: uuid.UUID, run_id: uuid.UUID) -> None:
    """Show the run is alive by refreshing heartbeat_at. The sweeper treats a
    "processing" document whose heartbeat is old as abandoned. One quick
    statement, committed immediately: nothing stays open while we then wait.

    Raises ClaimLostError if the claim is no longer ours. This runs before
    every request to the embedding provider, so a run that lost its claim (or
    whose document was deleted) stops before spending more of the rate limit."""
    alive = await db.scalar(
        update(Document)
        .where(ours(document_id, run_id))
        .values(heartbeat_at=clock.utcnow())
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    if alive is None:
        raise ClaimLostError


async def replace_chunks(
    db: AsyncSession,
    document: Document,
    run_id: uuid.UUID,
    chunks: list[ChunkData],
    vectors: list[Vector],
    embedding_model: str,
) -> None:
    """Swap the old chunks (and their vectors) for the new ones and mark the
    document ready, all in one transaction: the session opens it at the first
    statement and nothing is visible to anyone else until commit(). If any
    step fails, the caller rolls back and the old chunks are exactly as they
    were.

    The first statement both checks that the claim is still ours and locks
    the document row until commit, so no other run can take over between the
    check and the write."""
    still_ours = await db.scalar(
        update(Document)
        .where(ours(document.id, run_id))
        .values(
            status=DocumentStatus.ready,
            error_message=None,
            processing_run_id=None,
            heartbeat_at=None,
            finished_at=clock.utcnow(),
        )
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    if still_ours is None:
        raise ClaimLostError
    await db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    if chunks:
        await db.execute(
            insert(Chunk),
            [
                {
                    "document_id": document.id,
                    "owner_id": document.owner_id,
                    "collection_id": document.collection_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "token_count": chunk.token_count,
                    "embedding": vector,
                    "embedding_model": embedding_model,
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
    await db.commit()
    await db.refresh(document)


async def _check_daily_embedding_quota(needed: int) -> None:
    """Fail before spending anything if the owner's remaining embedding
    tokens for today can't cover this document (our estimate; the provider's
    count is what gets added afterwards)."""
    current = usage.current_quotas()
    if current is None:
        return
    user_id, quotas = current
    used = await quotas.used(user_id, EMBED_TOKENS)
    limit = quotas.limit(EMBED_TOKENS)
    if used is not None and used + needed > limit:
        raise DailyEmbeddingQuotaError(
            f"Daily embedding limit reached: this document needs about {needed:,} tokens "
            f"and {max(0, limit - used):,} of {limit:,} are left today. Reprocess it after "
            f"{quotas.resets_at():%H:%M} UTC."
        )


async def run_pipeline(
    db: AsyncSession,
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    document: Document,
    run_id: uuid.UUID,
    settings: Settings | None = None,
) -> None:
    """Process a document this run has claimed. Ends with the document "ready"
    (chunks and vectors replaced in one transaction). Raises on any failure,
    leaving the previous chunks and vectors untouched; the caller decides
    whether that is permanent or worth retrying."""
    settings = settings or get_settings()
    document_id = document.id
    data = await _read_file(storage, document.storage_key)
    # Parsing PDFs and counting tokens is CPU work that would freeze the
    # event loop if run here directly.
    chunks = await anyio.to_thread.run_sync(
        extract_and_chunk, data, document.content_type, settings
    )
    chunks = _drop_blank(chunks)
    await _check_daily_embedding_quota(sum(chunk.token_count for chunk in chunks))
    # Network calls, rate-limit waits and retries: possibly many minutes.
    # No database transaction is open meanwhile (the claim was committed and
    # each heartbeat commits at once), so no locks or connection are held.
    vectors = await embedder.embed_documents(
        [chunk.content for chunk in chunks],
        token_counts=[chunk.token_count for chunk in chunks],
        heartbeat=lambda: heartbeat(db, document_id, run_id),
    )
    logger.info(
        "Document %s: embedded %d chunks (%d tokens, model %s)",
        document_id,
        len(chunks),
        sum(chunk.token_count for chunk in chunks),
        embedder.model,
    )
    await replace_chunks(db, document, run_id, chunks, vectors, embedder.model)
