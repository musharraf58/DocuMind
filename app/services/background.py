"""The work the API process does in the background in inline mode
(INGESTION_MODE=inline), started and stopped with the app (see the lifespan
in app/main.py):

- the inline worker, processing documents one at a time
  (app/services/inline_worker.py);
- maintenance every MAINTENANCE_INTERVAL_SECONDS (2 minutes by default),
  and once right at startup: the sweeper, and retention.

Maintenance uses Postgres only. Nothing here polls Redis: on Upstash's free
tier every command counts against a monthly allowance, and an idle app
checking Redis every few seconds would spend it on nothing.

In Celery mode none of this runs here: the worker process and Celery beat
do the same jobs (app/workers/).
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.request_id import new_request_id, set_request_id
from app.services import jobs, sweeper
from app.services.inline_worker import InlineJobQueue, InlineWorker

logger = logging.getLogger(__name__)

Sessions = async_sessionmaker[AsyncSession]


class InlineRuntime:
    def __init__(
        self,
        settings: Settings,
        sessions: Sessions,
        redis: Redis | None,
        maintenance: Callable[[Sessions, InlineJobQueue], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.sessions = sessions
        self.redis = redis
        self.queue = InlineJobQueue(settings.INLINE_QUEUE_CAPACITY)
        self.worker = InlineWorker(self.queue, self._process)
        self._maintenance = maintenance or run_maintenance
        self._maintenance_task: asyncio.Task | None = None

    async def _process(self, document_id: uuid.UUID) -> jobs.JobResult:
        set_request_id(new_request_id())  # its own id in the log, like a Celery job
        return await jobs.process_document(document_id, self.sessions, self.redis, self.queue)

    def start(self) -> None:
        self.worker.start()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="maintenance")
        logger.info(
            "Inline ingestion started (queue capacity %d, maintenance every %.0fs)",
            self.queue.capacity,
            self.settings.MAINTENANCE_INTERVAL_SECONDS,
        )

    async def _maintenance_loop(self) -> None:
        while True:
            set_request_id(new_request_id())
            try:
                await self._maintenance(self.sessions, self.queue)
            except Exception:
                # A database blip must not end maintenance for good.
                logger.exception("Maintenance failed; trying again next round")
            set_request_id(None)
            await asyncio.sleep(self.settings.MAINTENANCE_INTERVAL_SECONDS)

    async def stop(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
        await self.worker.stop(self.settings.INLINE_SHUTDOWN_GRACE_SECONDS)
        logger.info("Inline ingestion stopped")


async def run_maintenance(sessions: Sessions, queue: InlineJobQueue) -> None:
    async with sessions() as db:
        await sweeper.sweep(db, queue)


_runtime: InlineRuntime | None = None


def current() -> InlineRuntime | None:
    """The running inline runtime, if this process has one."""
    return _runtime


def job_sessions() -> Sessions:
    """Sessions for jobs: their own small pool, with the worker's longer
    statement timeout (storing a large document's chunks is one big write).
    Tests replace this."""
    from app.core.database import make_engine

    settings = get_settings()
    engine = make_engine(
        statement_timeout_ms=settings.DB_WORKER_STATEMENT_TIMEOUT_MS, pool_size=2, max_overflow=1
    )
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def running(settings: Settings, redis: Redis | None) -> AsyncIterator[InlineRuntime | None]:
    """Run the inline runtime for the app's lifetime (inline mode only)."""
    global _runtime
    if settings.INGESTION_MODE != "inline":
        yield None
        return
    sessions = job_sessions()
    _runtime = InlineRuntime(settings, sessions, redis)
    _runtime.start()
    try:
        yield _runtime
    finally:
        await _runtime.stop()
        _runtime = None
        await sessions.kw["bind"].dispose()
