"""The background job that processes one document.

A queue may deliver the same job more than once (and the sweeper may enqueue
a document again), so the job is idempotent: it starts by claiming the
document, which only succeeds while the document is "pending". A duplicate
delivery finds it already claimed (or finished) and does nothing.

Errors are sorted into two kinds:
- permanent: retrying can't help (scanned or corrupt file, unsupported type,
  too large, invalid API key...). The document is marked "failed" at once.
- transient: the same request may succeed later (rate limit, provider
  outage, timeout, a database or network blip). The document goes back to
  "pending" and is retried after a growing delay, until MAX_JOB_ATTEMPTS.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from enum import StrEnum

from redis.exceptions import RedisError
from sqlalchemy import update
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.core.config import Settings, get_settings
from app.models import Document, DocumentStatus
from app.services import usage
from app.services.embeddings import (
    EmbeddingBusyError,
    EmbeddingError,
    EmbeddingProvider,
    RetryableEmbeddingError,
)
from app.services.extraction import ExtractionError
from app.services.processing import ClaimLostError, ours, run_pipeline
from app.services.storage import StorageBackend, StorageError
from app.services.usage import UsageRecorder

logger = logging.getLogger(__name__)

GENERIC_ERROR = "Could not read this file"
MAX_ERROR_LENGTH = 500
UNREACHABLE = (OperationalError, InterfaceError, RedisError, ConnectionError, StorageError)


class Outcome(StrEnum):
    ready = "ready"
    failed = "failed"
    retry = "retry"  # back to "pending"; enqueue again after retry_in seconds
    skipped = "skipped"  # nothing to do: deleted, or not pending (duplicate delivery)
    lost = "lost"  # stopped midway: deleted, or handed to another run by the sweeper


@dataclass(frozen=True)
class JobResult:
    outcome: Outcome
    retry_in: float | None = None
    message: str | None = None


@dataclass(frozen=True)
class Failure:
    transient: bool
    message: str  # safe to show users: no internals, no API key
    retry_after: float | None = None  # the provider's own hint, if any


def classify(exc: BaseException) -> Failure:
    """Decide whether retrying could help, and what to tell the user."""
    if isinstance(exc, EmbeddingBusyError):
        # The provider asked us to wait longer than we sit through in one go.
        return Failure(True, str(exc), retry_after=exc.retry_after)
    if isinstance(exc, RetryableEmbeddingError):
        # Rate limit, provider 5xx or timeout, after the client's own retries.
        return Failure(True, str(exc))
    if isinstance(exc, TimeoutError):
        return Failure(True, "Processing took too long and was stopped; it will be retried")
    if isinstance(exc, UNREACHABLE):
        # The database, Redis or the file storage was briefly unreachable.
        return Failure(True, "A temporary problem interrupted processing; it will be retried")
    if isinstance(exc, ExtractionError | EmbeddingError):
        # Scanned or corrupt file, unsupported type, over a limit, chunk too
        # large, invalid API key or input: the same request fails the same way.
        return Failure(False, str(exc))
    # Anything else is unexpected (e.g. a parser crashing on a malformed
    # file). Retrying the same file won't change it; details go to the log.
    return Failure(False, GENERIC_ERROR)


def retry_delay(attempts: int, settings: Settings, retry_after: float | None = None) -> float:
    """BASE after the 1st attempt, then 2x, 4x ...; never sooner than the
    provider asked for."""
    delay = settings.JOB_RETRY_BASE_SECONDS * 2 ** max(0, attempts - 1)
    return max(delay, retry_after or 0.0)


async def claim(db: AsyncSession, document_id: uuid.UUID) -> tuple[uuid.UUID, int] | None:
    """Atomically take a "pending" document: mark it "processing", count the
    attempt and give this run its token. Returns (run id, attempt number), or
    None if the document is gone or not pending (someone else has it, it's
    already done, or this is a duplicate delivery)."""
    now = clock.utcnow()
    run_id = uuid.uuid4()
    attempts = await db.scalar(
        update(Document)
        .where(Document.id == document_id, Document.status == DocumentStatus.pending)
        .values(
            status=DocumentStatus.processing,
            attempts=Document.attempts + 1,
            processing_run_id=run_id,
            processing_started_at=now,
            heartbeat_at=now,
            finished_at=None,
        )
        .returning(Document.attempts)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return None if attempts is None else (run_id, attempts)


async def run_job(
    db: AsyncSession,
    document_id: uuid.UUID,
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    settings: Settings | None = None,
    recorder: UsageRecorder | None = None,
) -> JobResult:
    """Process one document from the queue. Never raises for problems with
    the document itself: the outcome says what happened and, for a retry,
    when to enqueue it again. Embedding usage is recorded for the document's
    owner through `recorder` (none: nothing recorded, e.g. the eval runner)."""
    settings = settings or get_settings()
    claimed = await claim(db, document_id)
    if claimed is None:
        logger.info("Document %s: not pending (deleted or already handled), skipping", document_id)
        return JobResult(Outcome.skipped)
    run_id, attempts = claimed
    document = await db.get(Document, document_id, populate_existing=True)
    if document is None:  # deleted between the claim and now
        return JobResult(Outcome.lost)
    logger.info("Document %s: attempt %d (run %s)", document_id, attempts, run_id)

    try:
        async with asyncio.timeout(settings.JOB_TIMEOUT_MINUTES * 60):
            with usage.scoped(recorder, document.owner_id, document_id=document.id):
                await run_pipeline(db, storage, embedder, document, run_id, settings)
    except ClaimLostError:
        await db.rollback()
        logger.warning(
            "Document %s: run %s stopped, document deleted or reassigned", document_id, run_id
        )
        return JobResult(Outcome.lost)
    except Exception as exc:
        await db.rollback()
        failure = classify(exc)
        if failure.message == GENERIC_ERROR:
            logger.exception("Unexpected error processing document %s", document_id)
        else:
            logger.info(
                "Document %s: attempt %d failed: %s", document_id, attempts, failure.message
            )
        if failure.transient and attempts < settings.MAX_JOB_ATTEMPTS:
            delay = retry_delay(attempts, settings, failure.retry_after)
            message = f"Attempt {attempts} of {settings.MAX_JOB_ATTEMPTS} failed: {failure.message}"
            if not await _back_to_pending(db, document_id, run_id, message):
                return JobResult(Outcome.lost)
            logger.info("Document %s: will retry in %.0fs", document_id, delay)
            return JobResult(Outcome.retry, retry_in=delay, message=message)
        if failure.transient:
            message = f"Gave up after {attempts} attempts: {failure.message}"
        else:
            message = failure.message
        if not await mark_failed(db, document_id, run_id, message):
            return JobResult(Outcome.lost)
        return JobResult(Outcome.failed, message=message)

    logger.info("Document %s: ready", document_id)
    return JobResult(Outcome.ready)


async def fail_without_processing(
    db: AsyncSession, document_id: uuid.UUID, message: str
) -> JobResult:
    """The worker can't process anything (e.g. embedding misconfigured): claim
    the document and fail it with the reason, so it doesn't sit pending."""
    claimed = await claim(db, document_id)
    if claimed is None:
        return JobResult(Outcome.skipped)
    run_id, _ = claimed
    logger.error("Document %s: cannot process: %s", document_id, message)
    if not await mark_failed(db, document_id, run_id, message):
        return JobResult(Outcome.lost)
    return JobResult(Outcome.failed, message=message)


async def mark_failed(
    db: AsyncSession, document_id: uuid.UUID, run_id: uuid.UUID | None, message: str
) -> bool:
    """Set "failed", only while this run still holds the document (run_id None:
    whoever holds it, used by the sweeper). False if it was deleted or taken."""
    where = ours(document_id, run_id) if run_id else (Document.id == document_id)
    updated = await db.scalar(
        update(Document)
        .where(where)
        .values(
            status=DocumentStatus.failed,
            error_message=message[:MAX_ERROR_LENGTH],
            processing_run_id=None,
            heartbeat_at=None,
            finished_at=clock.utcnow(),
        )
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return updated is not None


async def _back_to_pending(
    db: AsyncSession, document_id: uuid.UUID, run_id: uuid.UUID, message: str
) -> bool:
    updated = await db.scalar(
        update(Document)
        .where(ours(document_id, run_id))
        .values(
            status=DocumentStatus.pending,
            error_message=message[:MAX_ERROR_LENGTH],
            processing_run_id=None,
            heartbeat_at=None,
            queued_at=clock.utcnow(),
        )
        .returning(Document.id)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return updated is not None
