from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

engine = create_async_engine(get_settings().DATABASE_URL, pool_pre_ping=True)

# expire_on_commit=False: after commit, objects keep their loaded attributes.
# With async, lazily reloading expired attributes would trigger implicit I/O
# and raise errors, so we turn that off.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency: one session per request, closed when the request ends."""
    async with SessionLocal() as session:
        yield session
