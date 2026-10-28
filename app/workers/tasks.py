"""Celery tasks. Thin wrappers: each runs async code from app.services in a
fresh event loop.

Celery tasks are synchronous, and every task gets its own event loop
(asyncio.run). Database and Redis connections belong to the loop that made
them, so each task makes its own engine and client and closes them after.
"""

import asyncio
import logging
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.database import make_engine
from app.core.redis import make_redis
from app.services import jobs, retention, sweeper
from app.services.embeddings import get_embedding_provider
from app.services.queue import CeleryJobQueue
from app.services.storage import get_storage
from app.workers.celery_app import PROCESS_DOCUMENT, RETENTION, SWEEP, celery_app

logger = logging.getLogger(__name__)


def _engine():
    # NullPool: no connections kept around between tasks (each has its own loop).
    # A longer statement timeout than the API's: storing a large document's
    # chunks and vectors is one big write.
    return make_engine(
        statement_timeout_ms=get_settings().DB_WORKER_STATEMENT_TIMEOUT_MS, poolclass=NullPool
    )


@celery_app.task(name=PROCESS_DOCUMENT)
def process_document(document_id: str) -> str:
    return asyncio.run(process(uuid.UUID(document_id))).outcome


async def process(document_id: uuid.UUID) -> jobs.JobResult:
    # The job itself is shared with inline mode: jobs.process_document.
    engine, redis = _engine(), make_redis()
    try:
        return await jobs.process_document(
            document_id,
            async_sessionmaker(engine, expire_on_commit=False),
            redis,
            CeleryJobQueue(),
            make_embedder=get_embedding_provider,
        )
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


@celery_app.task(name=RETENTION)
def purge_expired() -> None:
    asyncio.run(run_retention())


async def run_retention() -> retention.RetentionReport:
    engine, redis = _engine(), make_redis()
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        return await retention.purge_expired_accounts(sessions, get_storage(), redis)
    finally:
        await redis.aclose()
        await engine.dispose()
