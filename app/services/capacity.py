"""The database capacity guard: stop taking uploads before the database is
full, while everything else keeps working.

Neon's free tier caps a project's storage, and when Postgres runs out of
room every write fails, including the small ones that asking a question
makes (the conversation, the usage record). Uploads are what fill the
database (each document adds its chunks, vectors and full-text index), so
past MAX_DB_SIZE_MB uploads are refused with a clear 503 while ask, search,
reading and deleting keep working. Deleting documents frees room again.

The size comes from pg_database_size(), which walks the database's files,
so it is cached per process for DB_SIZE_CACHE_SECONDS: an upload burst costs
one size query a minute, not one per upload. The cache means the limit can
be overshot by what one minute of uploads adds, which is why the default
(400 MB) stays well below Neon's cap.
"""

import logging
import time
from collections.abc import Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import ApiError

logger = logging.getLogger(__name__)

MB = 1024 * 1024
RETRY_AFTER_SECONDS = 3600


class DatabaseSize:
    def __init__(self, cache_seconds: float = 60, clock: Callable[[], float] = time.monotonic):
        self.cache_seconds = cache_seconds
        self.clock = clock
        self._value: int | None = None
        self._at = 0.0

    async def bytes(self, db: AsyncSession) -> int:
        now = self.clock()
        if self._value is None or now - self._at >= self.cache_seconds:
            self._value = int(await db.scalar(text("SELECT pg_database_size(current_database())")))
            self._at = now
        return self._value

    def forget(self) -> None:
        self._value = None


# One per process: the cache must outlive single requests.
database_size = DatabaseSize()


async def check_room_for_uploads(db: AsyncSession, size: DatabaseSize = database_size) -> None:
    """503 storage_capacity_reached once the database is over MAX_DB_SIZE_MB
    (0 turns the guard off)."""
    limit_mb = get_settings().MAX_DB_SIZE_MB
    if not limit_mb:
        return
    used = await size.bytes(db)
    if used >= limit_mb * MB:
        logger.warning(
            "Uploads refused: database is %d MB, over MAX_DB_SIZE_MB=%d", used // MB, limit_mb
        )
        raise ApiError(
            503,
            "The service has reached its storage capacity and can't accept new documents "
            "right now. Asking questions and searching still work.",
            code="storage_capacity_reached",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            extra={"retry_after": RETRY_AFTER_SECONDS},
        )
