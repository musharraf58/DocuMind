from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


def make_engine(
    url: str | None = None, statement_timeout_ms: int | None = None, **kwargs: Any
) -> AsyncEngine:
    """An engine with a statement timeout: Postgres cancels any single
    statement that runs longer, so one runaway query can't hold a connection
    (and its locks) forever. `kwargs` go to create_async_engine (pool options,
    or poolclass=NullPool for the worker)."""
    settings = get_settings()
    timeout = statement_timeout_ms or settings.DB_STATEMENT_TIMEOUT_MS
    return create_async_engine(
        url or settings.DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"server_settings": {"statement_timeout": str(timeout)}},
        **kwargs,
    )


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
    # Replace connections after 30 minutes, before a proxy or firewall drops
    # them silently.
    pool_recycle=1800,
)

# expire_on_commit=False: after commit, objects keep their loaded attributes.
# With async, lazily reloading expired attributes would trigger implicit I/O
# and raise errors, so we turn that off.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency: one session per request, closed when the request ends."""
    async with SessionLocal() as session:
        yield session
