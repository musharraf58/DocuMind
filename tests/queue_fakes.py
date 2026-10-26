"""A stand-in for the Celery queue: records what was enqueued, checks it was
committed first, and can run the queued jobs the way the worker would."""

import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.deps import embedding_provider
from app.core.redis import get_redis
from app.main import app
from app.models import Document, DocumentStatus
from app.services.embeddings import EmbeddingProvider
from app.services.jobs import JobResult, run_job
from app.services.queue import JobQueue
from app.services.storage import get_storage
from app.services.usage import UsageRecorder


class RecordingQueue(JobQueue):
    def __init__(self, sessions: async_sessionmaker) -> None:
        self._sessions = sessions
        self.enqueued: list[tuple[uuid.UUID, float | None]] = []
        # Documents enqueued before they were committed as "pending": a worker
        # picking the job up at once would not have found them.
        self.not_yet_committed: list[uuid.UUID] = []
        self.down = False  # simulate Redis being unreachable

    async def enqueue(self, document_id: uuid.UUID, countdown: float | None = None) -> bool:
        if self.down:
            return False
        async with self._sessions() as other:  # what a worker would see right now
            row = await other.get(Document, document_id)
            if row is None or row.status != DocumentStatus.pending:
                self.not_yet_committed.append(document_id)
        self.enqueued.append((document_id, countdown))
        return True

    @property
    def ids(self) -> list[uuid.UUID]:
        return [document_id for document_id, _ in self.enqueued]

    async def run_all(self, embedder: EmbeddingProvider | None = None) -> list[JobResult]:
        """Run every queued job, as the worker would, with the same embedding
        provider the API is configured with (including test overrides)."""
        provider = (
            embedder or app.dependency_overrides.get(embedding_provider, embedding_provider)()
        )
        # Like the worker: usage is recorded for the document's owner, into the
        # test database and the test's fake Redis.
        redis = app.dependency_overrides.get(get_redis, lambda: None)()
        recorder = UsageRecorder(self._sessions, redis)
        results = []
        while self.enqueued:
            document_id, _ = self.enqueued.pop(0)
            async with self._sessions() as session:
                results.append(
                    await run_job(session, document_id, get_storage(), provider, recorder=recorder)
                )
        return results
