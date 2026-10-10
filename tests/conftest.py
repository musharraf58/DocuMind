from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.database import get_db
from app.main import app

# Tests use their own database so they can never touch your dev data.
# Derived from DATABASE_URL: same server and credentials, different DB name.
_dev_url = make_url(get_settings().DATABASE_URL)
TEST_DB_NAME = f"{_dev_url.database}_test"
TEST_DB_URL = _dev_url.set(database=TEST_DB_NAME)


async def _ensure_test_database() -> None:
    """Create the test DB if missing. CREATE DATABASE needs autocommit and a
    connection to some *other* database, so we connect to the 'postgres' one."""
    admin_engine = create_async_engine(
        _dev_url.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    async with admin_engine.connect() as conn:
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DB_NAME}
        )
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    await admin_engine.dispose()

    # Same thing migration 0001 does, so the test DB has pgvector too.
    test_engine = create_async_engine(TEST_DB_URL, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with test_engine.connect() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await test_engine.dispose()


@pytest.fixture(scope="session")
async def test_engine():
    await _ensure_test_database()
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession]:
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    # Swap the real get_db for one that uses the test database.
    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
