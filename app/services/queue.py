"""Putting documents on the processing queue."""

import logging
import uuid
from abc import ABC, abstractmethod
from functools import partial

import anyio

from app.core.request_id import get_request_id

logger = logging.getLogger(__name__)


class JobQueue(ABC):
    @abstractmethod
    async def enqueue(self, document_id: uuid.UUID, countdown: float | None = None) -> bool:
        """Ask a worker to process the document (after `countdown` seconds).
        Returns False instead of raising if the queue can't be reached: the
        document is already saved as "pending", and the sweeper will enqueue
        it again later."""

    def has_room(self) -> bool:
        """Whether a new job would be accepted now (checked before storing an
        upload, so a full queue refuses the upload instead of piling up)."""
        return True


class CeleryJobQueue(JobQueue):
    async def enqueue(self, document_id: uuid.UUID, countdown: float | None = None) -> bool:
        from app.workers.celery_app import PROCESS_DOCUMENT, REQUEST_ID_HEADER, celery_app

        send = partial(
            celery_app.send_task,
            PROCESS_DOCUMENT,
            args=[str(document_id)],
            countdown=countdown,
            retry=False,  # don't keep retrying the broker; the sweeper is the fallback
            # The worker logs the job under the id of the request that queued it.
            headers={REQUEST_ID_HEADER: get_request_id()},
        )
        try:
            # Celery's client is blocking; keep it off the event loop.
            await anyio.to_thread.run_sync(send)
        except Exception:
            logger.warning(
                "Could not enqueue document %s (is Redis reachable?); the sweeper will retry",
                document_id,
                exc_info=True,
            )
            return False
        return True


def get_queue() -> JobQueue:
    """FastAPI dependency (tests use a fake queue): Celery, or in inline
    mode this process's own queue."""
    from app.core.config import get_settings

    if get_settings().INGESTION_MODE == "inline":
        from app.core.errors import ApiError
        from app.services import background

        runtime = background.current()
        if runtime is None:  # only between startup and shutdown
            raise ApiError(503, "Document processing is not running", code="ingestion_unavailable")
        return runtime.queue
    return CeleryJobQueue()


def queue_full():
    from app.core.errors import ApiError

    return ApiError(
        503,
        "Too many documents are waiting to be processed; please try again in a few minutes",
        code="ingestion_queue_full",
        headers={"Retry-After": "120"},
        extra={"retry_after": 120},
    )
