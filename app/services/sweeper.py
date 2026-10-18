"""The sweeper: finds documents the queue lost track of and puts them back.

A queue alone can't guarantee every document gets processed. The upload may
commit while Redis is down, a worker may crash mid-job (or be killed during a
deploy), or a message may simply be lost. The database knows the truth, so
every minute (and when the worker starts) the sweeper:

- re-enqueues documents that have been "pending" for SWEEP_PENDING_MINUTES;
- resets "processing" documents whose heartbeat is older than
  STALE_PROCESSING_MINUTES (their worker died) back to "pending" and
  re-enqueues them, or marks them "failed" once they've used all attempts;
- warns about "ready" documents with chunks that have no embedding.

Each fix is one conditional UPDATE that re-checks the condition, so a worker
whose heartbeat lands between our look and our write is never reset.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.core.config import Settings, get_settings
from app.models import Document, DocumentStatus
from app.services.integrity import ready_documents_missing_embeddings
from app.services.queue import JobQueue

logger = logging.getLogger(__name__)


@dataclass
class SweepReport:
    requeued_pending: list[uuid.UUID] = field(default_factory=list)
    reset_stale: list[uuid.UUID] = field(default_factory=list)
    failed_stale: list[uuid.UUID] = field(default_factory=list)
    enqueue_failed: list[uuid.UUID] = field(default_factory=list)
    missing_embeddings: int = 0


async def sweep(db: AsyncSession, queue: JobQueue, settings: Settings | None = None) -> SweepReport:
    settings = settings or get_settings()
    now = clock.utcnow()
    report = SweepReport()

    # A processing document's last sign of life. (Documents claimed before
    # heartbeats existed fall back to when processing started.)
    last_seen = func.coalesce(
        Document.heartbeat_at, Document.processing_started_at, Document.created_at
    )
    stale = (Document.status == DocumentStatus.processing) & (
        last_seen < now - timedelta(minutes=settings.STALE_PROCESSING_MINUTES)
    )

    # 1. Stuck processing with no attempts left: give up.
    for document_id, attempts in await _update_returning(
        db,
        stale & (Document.attempts >= settings.MAX_JOB_ATTEMPTS),
        status=DocumentStatus.failed,
        error_message=(
            f"Processing stopped responding on each of {settings.MAX_JOB_ATTEMPTS} attempts; "
            "giving up"
        ),
        processing_run_id=None,
        heartbeat_at=None,
        finished_at=now,
    ):
        report.failed_stale.append(document_id)
        logger.warning(
            "Sweeper: document %s stuck after %d attempts, marked failed", document_id, attempts
        )

    # 2. Stuck processing with attempts left: back to pending and re-enqueue.
    # Clearing the run token means the old run, if somehow still alive,
    # stops at its next heartbeat and can never write.
    for document_id, attempts in await _update_returning(
        db,
        stale & (Document.attempts < settings.MAX_JOB_ATTEMPTS),
        status=DocumentStatus.pending,
        error_message="Processing was interrupted; it will be retried",
        processing_run_id=None,
        heartbeat_at=None,
        queued_at=now,
    ):
        report.reset_stale.append(document_id)
        logger.warning(
            "Sweeper: document %s had no heartbeat for %s min (attempt %d), reset to pending",
            document_id,
            settings.STALE_PROCESSING_MINUTES,
            attempts,
        )
        await _enqueue(queue, document_id, report)

    # 3. Pending for too long: the enqueue failed or the message was lost.
    # queued_at moves to now, so the next sweep waits another full period.
    waiting_since = func.coalesce(Document.queued_at, Document.created_at)
    for document_id, _ in await _update_returning(
        db,
        (Document.status == DocumentStatus.pending)
        & (waiting_since < now - timedelta(minutes=settings.SWEEP_PENDING_MINUTES)),
        queued_at=now,
    ):
        report.requeued_pending.append(document_id)
        logger.info("Sweeper: document %s pending too long, re-enqueued", document_id)
        await _enqueue(queue, document_id, report)

    # 4. An invariant check: a ready document must have a vector on every chunk.
    problems = await ready_documents_missing_embeddings(db)
    report.missing_embeddings = len(problems)
    for problem in problems:
        logger.warning(
            "Sweeper: ready document %s has %d of %d chunks without embeddings; reprocess it",
            problem.document_id,
            problem.chunks_without_vector,
            problem.total_chunks,
        )

    if report.requeued_pending or report.reset_stale or report.failed_stale:
        logger.info(
            "Sweeper: re-enqueued %d pending, reset %d stuck, failed %d stuck",
            len(report.requeued_pending),
            len(report.reset_stale),
            len(report.failed_stale),
        )
    return report


async def _update_returning(db: AsyncSession, where, **values) -> list[tuple[uuid.UUID, int]]:
    rows = (
        await db.execute(
            update(Document)
            .where(where)
            .values(**values)
            .returning(Document.id, Document.attempts)
            .execution_options(synchronize_session=False)
        )
    ).all()
    await db.commit()
    return [(row[0], row[1]) for row in rows]


async def _enqueue(queue: JobQueue, document_id: uuid.UUID, report: SweepReport) -> None:
    if not await queue.enqueue(document_id):
        # Still pending with a fresh queued_at: the next sweep tries again.
        report.enqueue_failed.append(document_id)
