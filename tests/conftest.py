from collections.abc import AsyncGenerator
from pathlib import Path

import fakeredis
import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_session_factory
from app.core.config import get_settings
from app.core.database import get_db
from app.core.rate_limit import memory_fallback
from app.core.redis import get_redis
from app.main import app
from app.models import Base
from app.services.queue import get_queue

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
    # Build the schema from the models (fresh each run). Faster than running
    # migrations; migration correctness is checked separately by hand/CI.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(autouse=True)
async def clean_tables(test_engine) -> AsyncGenerator[None]:
    """Empty every table after each test so tests can't affect each other."""
    yield
    async with test_engine.begin() as conn:
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {tables} CASCADE"))


@pytest.fixture
def job_queue(test_engine):
    """The fake queue the API enqueues into (see tests/queue_fakes.py)."""
    from tests.queue_fakes import RecordingQueue

    return RecordingQueue(async_sessionmaker(test_engine, expire_on_commit=False))


@pytest.fixture
async def client(test_engine, job_queue) -> AsyncGenerator[AsyncClient]:
    fake_redis = fakeredis.aioredis.FakeRedis()
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    # Like production: every request gets its own fresh session.
    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    # Code that opens its own short sessions must use the TEST database too.
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_queue] = lambda: job_queue  # never a real broker
    app.dependency_overrides[get_redis] = lambda: fake_redis  # never a real Redis
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def db(test_engine) -> AsyncGenerator[AsyncSession]:
    """A session for tests that call services directly."""
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as session:
        yield session


@pytest.fixture(autouse=True)
def storage_dir(tmp_path, monkeypatch) -> Path:
    """Point file storage at a throwaway directory for every test."""
    root = tmp_path / "storage"
    monkeypatch.setattr(get_settings(), "STORAGE_PATH", str(root))
    return root


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch) -> None:
    """Every test embeds with the fake provider unless it builds its own."""
    monkeypatch.setattr(get_settings(), "EMBEDDING_PROVIDER", "fake")
    monkeypatch.setattr(get_settings(), "LLM_PROVIDER", "fake")


GENEROUS = "100000/minute"


@pytest.fixture(autouse=True)
def generous_limits(monkeypatch) -> None:
    """Limits stay switched on (so their code runs in every test) but high
    enough never to get in the way. Tests of a limit lower it themselves."""
    settings = get_settings()
    for name in type(settings).model_fields:
        if name.startswith("RATE_LIMIT_") and name != "RATE_LIMIT_ENABLED":
            monkeypatch.setattr(settings, name, GENEROUS)
    memory_fallback.clear()  # in-process state must not leak between tests


@pytest.fixture(autouse=True)
def open_registration(monkeypatch) -> None:
    """Most tests just need accounts; tests of the modes set their own."""
    monkeypatch.setattr(get_settings(), "REGISTRATION_MODE", "open")


@pytest.fixture(autouse=True)
def no_network(monkeypatch) -> None:
    """Fail loudly if anything tries a real HTTP request. (The test client
    talks to the app in-process via ASGITransport, which is unaffected.)"""

    async def refuse(self, request):
        raise RuntimeError(f"Tests must not make network calls (tried {request.url.host})")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


@pytest.fixture(autouse=True)
def generous_quotas(monkeypatch) -> None:
    """Daily quotas high enough never to get in the way; their tests lower them."""
    settings = get_settings()
    for name in [
        "DAILY_QUESTIONS_PER_USER",
        "DAILY_EMBED_TOKENS_PER_USER",
        "DAILY_UPLOADS_PER_USER",
        "GLOBAL_DAILY_LLM_BUDGET",
    ]:
        monkeypatch.setattr(settings, name, 10**9)
