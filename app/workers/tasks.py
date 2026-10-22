"""Celery tasks. Thin wrappers: each runs async code from app.services in a
fresh event loop.

Celery tasks are synchronous, and every task gets its own event loop
(asyncio.run). Database and Redis connections belong to the loop that made
them, so each task makes its own engine and client and closes them after.
"""

import asyncio
import logging
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.redis import make_redis
from app.services import jobs, sweeper
from app.services.embeddings import EmbeddingError, get_embedding_provider
from app.services.queue import CeleryJobQueue
from app.services.storage import get_storage
from app.services.usage import UsageRecorder
from app.workers.celery_app import PROCESS_DOCUMENT, SWEEP, celery_app

logger = logging.getLogger(__name__)


def _engine():
    # NullPool: no connections kept around between tasks (each has its own loop).
    return create_async_engine(get_settings().DATABASE_URL, poolclass=NullPool)


@celery_app.task(name=PROCESS_DOCUMENT)
def process_document(document_id: str) -> str:
    return asyncio.run(process(uuid.UUID(document_id))).outcome


async def process(document_id: uuid.UUID) -> jobs.JobResult:
    engine, redis = _engine(), make_redis()
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            try:
                embedder = get_embedding_provider(redis)
            except EmbeddingError as exc:
                # Misconfiguration (e.g. dimensions not matching the database).
                return await jobs.fail_without_processing(db, document_id, str(exc))
            recorder = UsageRecorder(async_sessionmaker(engine, expire_on_commit=False), redis)
            result = await jobs.run_job(db, document_id, get_storage(), embedder, recorder=recorder)
        if result.outcome == jobs.Outcome.retry:
            # The document is already back to "pending"; if this enqueue
            # fails, the sweeper picks it up.
            await CeleryJobQueue().enqueue(document_id, countdown=result.retry_in)
        return result
    finally:
        await redis.aclose()
        await engine.dispose()


@celery_app.task(name=SWEEP)
def sweep() -> None:
    asyncio.run(run_sweep())


async def run_sweep() -> sweeper.SweepReport:
    engine = _engine()
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            return await sweeper.sweep(db, CeleryJobQueue())
    finally:
        await engine.dispose()
