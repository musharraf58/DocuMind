"""Inline ingestion (INGESTION_MODE=inline): the API process processes
uploaded documents itself, for hosts with no free background worker.

    upload / reprocess ──► InlineJobQueue ──► InlineWorker ──► jobs.process_document
                          (bounded, in memory)  (ONE job at a time)   (the same code
                                                                       the Celery task runs)

Why this is still safe, although the queue lives in memory and vanishes on
every restart: the queue was never the source of truth. Postgres is. A
document is committed as "pending" before it is enqueued, the job claims it
atomically (only a "pending" document can be claimed, so a duplicate does
nothing), and the job's heartbeat shows it's alive. Whatever the memory
loses, the sweeper finds again in Postgres: "pending" for too long ->
enqueued again; "processing" with a stale heartbeat -> back to "pending".
That is exactly the recovery the Celery mode already relied on.

One job at a time, on purpose: the embedding provider's free tier allows 3
requests a minute, so a second parallel job would only wait behind the
first while doubling the memory in use on a 512 MB instance.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.services.queue import JobQueue

logger = logging.getLogger(__name__)


class InlineJobQueue(JobQueue):
    """A bounded FIFO of document ids, with no duplicates.

    The sweeper re-enqueues every document that has been "pending" for a few
    minutes, which includes the ones simply waiting their turn here; without
    de-duplication they would fill the queue with copies.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._queue: asyncio.Queue[uuid.UUID] = asyncio.Queue()
        self._waiting: set[uuid.UUID] = set()
        self._timers: set[asyncio.TimerHandle] = set()
        self._closed = False

    def __len__(self) -> int:
        return len(self._waiting)

    def has_room(self) -> bool:
        return not self._closed and len(self._waiting) < self.capacity

    async def enqueue(self, document_id: uuid.UUID, countdown: float | None = None) -> bool:
        """False when full or shutting down: the document stays "pending" in
        Postgres, and the sweeper enqueues it again later."""
        if countdown:
            # A retry. Waiting retries don't take a place in the queue; when
            # the timer fires, the id joins like any other (or, if the queue
            # is full or the process restarted by then, the sweeper does it).
            loop = asyncio.get_running_loop()
            timer = loop.call_later(countdown, self._put_later, document_id)
            self._timers.add(timer)
            return True
        return self._put(document_id)

    def _put_later(self, document_id: uuid.UUID) -> None:
        self._timers = {timer for timer in self._timers if not timer.cancelled()}
        if not self._put(document_id):
            logger.info("Document %s: retry not queued (queue full); the sweeper will", document_id)

    def _put(self, document_id: uuid.UUID) -> bool:
        if document_id in self._waiting:
            return True  # already waiting: one copy is enough
        if not self.has_room():
            return False
        self._waiting.add(document_id)
        self._queue.put_nowait(document_id)
        return True

    async def get(self) -> uuid.UUID:
        document_id = await self._queue.get()
        # Taken: from now on the same id may be queued again (e.g. reprocess).
        self._waiting.discard(document_id)
        return document_id

    def close(self) -> None:
        """Stop accepting jobs and drop scheduled retries. Whatever was still
        waiting stays "pending" in Postgres for the next start."""
        self._closed = True
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()


class InlineWorker:
    """Takes ids from the queue and runs them, strictly one after another."""

    def __init__(self, queue: InlineJobQueue, run: Callable[[uuid.UUID], Awaitable[Any]]) -> None:
        self.queue = queue
        self._run = run
        self._task: asyncio.Task | None = None
        self._job: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def busy(self) -> bool:
        return self._job is not None and not self._job.done()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="inline-worker")

    async def _loop(self) -> None:
        while True:
            document_id = await self.queue.get()
            # The job runs in its own task so that stop() can give it a grace
            # period, but it is awaited here: never more than one at a time.
            self._job = asyncio.create_task(self._run(document_id), name=f"job-{document_id}")
            try:
                await asyncio.shield(self._job)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The job handles the document's own problems; anything that
                # escapes is a bug or an outage. Log it and keep the worker
                # alive: the sweeper retries the document later.
                logger.exception("Document %s: the job crashed", document_id)

    async def stop(self, grace_seconds: float) -> None:
        """Take no new jobs; let the running one finish for up to
        `grace_seconds`, then cancel it (it hands its document back to
        "pending" as it stops, see jobs.run_job)."""
        self.queue.close()
        job = self._job
        if job is not None and not job.done():
            logger.info("Waiting up to %.0fs for the running job to finish", grace_seconds)
            await asyncio.wait({job}, timeout=grace_seconds)
            if not job.done():
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
