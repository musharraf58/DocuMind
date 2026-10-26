from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.core.db_url import (
    apply_statement_timeout_per_transaction,
    engine_options,
    to_asyncpg_url,
)


def make_engine(
    url: str | None = None, statement_timeout_ms: int | None = None, **kwargs: Any
) -> AsyncEngine:
    """An engine with a statement timeout: Postgres cancels any single
    statement that runs longer, so one runaway query can't hold a connection
    (and its locks) forever. `kwargs` go to create_async_engine (pool options,
    or poolclass=NullPool for the worker).

    The URL is translated for asyncpg, and with DB_PGBOUNCER the engine is
    set up for transaction pooling (see app/core/db_url.py)."""
    settings = get_settings()
    timeout = statement_timeout_ms or settings.DB_STATEMENT_TIMEOUT_MS
    options = engine_options(timeout, settings.DB_PGBOUNCER, settings.DB_POOL_RECYCLE_SECONDS)
    engine = create_async_engine(
        to_asyncpg_url(url) if url else settings.database_url, **{**options, **kwargs}
    )
    if settings.DB_PGBOUNCER:
        apply_statement_timeout_per_transaction(engine, timeout)
    return engine


_settings = get_settings()
# The API's pool. pool_size connections stay open; up to max_overflow more are
# opened under load; past that a request waits pool_timeout seconds for one,
# then fails (a clear error instead of piling up). Keep pool_size + max_overflow
# (times the number of API processes, plus the worker) below Postgres's
# max_connections (100 by default).
engine = make_engine(
    pool_size=_settings.DB_POOL_SIZE,
    max_overflow=_settings.DB_MAX_OVERFLOW,
    pool_timeout=_settings.DB_POOL_TIMEOUT_SECONDS,
)

# expire_on_commit=False: after commit, objects keep their loaded attributes.
# With async, lazily reloading expired attributes would trigger implicit I/O
# and raise errors, so we turn that off.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency: one session per request, closed when the request ends."""
    async with SessionLocal() as session:
        yield session
