"""Turn a stored document into searchable chunks: extract text, chunk it,
embed the chunks, save them.

Temporary: called directly from the request (POST /documents/{id}/process).
Phase 5 moves this to a background queue.
"""

import logging
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import anyio
from fastapi import HTTPException
from sqlalchemy import delete, insert, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models import Chunk, Document, DocumentStatus
from app.services.chunking import ChunkData, chunk_segments
from app.services.embeddings import EmbeddingError, EmbeddingProvider, Vector
from app.services.extraction import ExtractionError, ExtractionLimits, extract
from app.services.storage import StorageBackend

logger = logging.getLogger(__name__)

# A "processing" claim older than this is treated as abandoned (the server
# restarted or crashed mid-run) and may be claimed again.
STALE_PROCESSING_AFTER = timedelta(minutes=10)
GENERIC_ERROR = "Could not read this file"
MAX_ERROR_LENGTH = 500


async def claim_for_processing(db: AsyncSession, document: Document) -> None:
    """Mark the document "processing", or raise 409 if someone else already is.

    Check and update happen in a single UPDATE ... WHERE, so two simultaneous
    requests can't both win: the database lets exactly one of them match.
    """
    now = datetime.now(UTC)
    claimed = await db.scalar(
        update(Document)
        .where(
            Document.id == document.id,
            or_(
                Document.status != DocumentStatus.processing,
                Document.processing_started_at.is_(None),
                Document.processing_started_at < now - STALE_PROCESSING_AFTER,
            ),
        )
        .values(status=DocumentStatus.processing, error_message=None, processing_started_at=now)
        .returning(Document.id)
    )
    await db.commit()
    if claimed is None:
        raise HTTPException(status_code=409, detail="Document is already being processed")


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


async def _replace_chunks(
    db: AsyncSession,
    document: Document,
    chunks: list[ChunkData],
    vectors: list[Vector],
    embedding_model: str,
) -> None:
    """Swap the old chunks (and their vectors) for the new ones and mark the
    document ready, all in
    one transaction: the session opens it at the first statement and nothing is
    visible to anyone else until commit(). If any step fails, the caller rolls
    back and the old chunks are exactly as they were."""
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
    document.status = DocumentStatus.ready
    document.error_message = None
    document.processing_started_at = None
    await db.commit()


async def _mark_failed(
    db: AsyncSession, document: Document, document_id: uuid.UUID, message: str
) -> None:
    # The rollback before this expired the document object, and reading an
    # expired attribute in async code raises, so update by the saved id and
    # reload explicitly.
    await db.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            status=DocumentStatus.failed,
            error_message=message[:MAX_ERROR_LENGTH],
            processing_started_at=None,
        )
    )
    await db.commit()
    await db.refresh(document)


async def process_document(
    db: AsyncSession, storage: StorageBackend, embedder: EmbeddingProvider, document: Document
) -> Document:
    """Extract, chunk and embed a document the caller has already proven they own.

    Ends with the document "ready" (chunks and vectors replaced) or "failed"
    (with a short message for the user, previous chunks and vectors
    untouched). Raises 409 if it is already being processed.
    """
    document_id = document.id
    settings = get_settings()
    await claim_for_processing(db, document)
    try:
        data = await _read_file(storage, document.storage_key)
        # Parsing PDFs and counting tokens is CPU work that would freeze the
        # event loop (and every other request) if run here directly.
        chunks = await anyio.to_thread.run_sync(
            extract_and_chunk, data, document.content_type, settings
        )
        chunks = _drop_blank(chunks)
        # Network call, possibly with retries and backoff: seconds or more.
        # No database transaction is open here (the claim was committed), so
        # no locks or connection are held while we wait on the provider.
        vectors = await embedder.embed_documents([chunk.content for chunk in chunks])
        logger.info(
            "Document %s: embedded %d chunks (%d tokens, model %s)",
            document_id,
            len(chunks),
            sum(chunk.token_count for chunk in chunks),
            embedder.model,
        )
        await _replace_chunks(db, document, chunks, vectors, embedder.model)
    except Exception as exc:
        await db.rollback()
        if isinstance(exc, ExtractionError | EmbeddingError):
            # Both carry messages written to be shown to users.
            message = str(exc)
            logger.info("Document %s failed processing: %s", document_id, message)
        else:
            # Full details go to the server log only, never to the user.
            message = GENERIC_ERROR
            logger.exception("Unexpected error processing document %s", document_id)
        await _mark_failed(db, document, document_id, message)
    return document
