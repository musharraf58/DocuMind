"""Consistency checks on stored data."""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissingEmbeddings:
    document_id: uuid.UUID
    filename: str
    chunks_without_vector: int
    total_chunks: int


async def ready_documents_missing_embeddings(db: AsyncSession) -> list[MissingEmbeddings]:
    """Documents marked "ready" that have chunks with no vector or no model name.

    Such chunks can never be found by search. Chunks made before embeddings
    existed (Phase 3) look like this too until their document is processed again.
    """
    missing = (Chunk.embedding.is_(None)) | (Chunk.embedding_model.is_(None))
    rows = await db.execute(
        select(
            Document.id,
            Document.original_filename,
            func.count().filter(missing),
            func.count(),
        )
        .join(Chunk, Chunk.document_id == Document.id)
        .where(Document.status == DocumentStatus.ready)
        .group_by(Document.id)
        .having(func.count().filter(missing) > 0)
        .order_by(Document.created_at)
    )
    return [MissingEmbeddings(*row) for row in rows]


async def report_missing_embeddings(session_factory: Callable[[], AsyncSession]) -> None:
    """Log a warning if any ready document has chunks without vectors. Meant for
    startup; never raises, so a database hiccup can't stop the app starting."""
    try:
        async with session_factory() as db:
            problems = await ready_documents_missing_embeddings(db)
    except Exception:
        logger.exception("Could not check documents for missing embeddings")
        return
    if not problems:
        return
    logger.warning(
        "%d ready document(s) have chunks without embeddings and cannot be found by search; "
        "process them again: %s",
        len(problems),
        "; ".join(
            f"{p.document_id} ({p.filename}: {p.chunks_without_vector}/{p.total_chunks} chunks)"
            for p in problems[:10]
        )
        + (" ..." if len(problems) > 10 else ""),
    )
