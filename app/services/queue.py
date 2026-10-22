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
    """FastAPI dependency, so tests can use a fake queue."""
    return CeleryJobQueue()
