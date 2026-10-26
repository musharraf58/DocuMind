import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.core.database import engine, make_engine
from app.workers import tasks
from tests.conftest import TEST_DB_URL

TEST_URL = TEST_DB_URL.render_as_string(hide_password=False)


def test_api_pool_is_configured():
    from app.core.config import get_settings

    settings = get_settings()
    assert engine.pool.size() == settings.DB_POOL_SIZE
    assert engine.pool._max_overflow == settings.DB_MAX_OVERFLOW
    assert engine.pool._timeout == settings.DB_POOL_TIMEOUT_SECONDS


async def test_statement_timeout_cancels_a_slow_query():
    slow = make_engine(TEST_URL, statement_timeout_ms=200)
    try:
        async with slow.connect() as conn:
            assert (await conn.scalar(text("SHOW statement_timeout"))) == "200ms"
            with pytest.raises(DBAPIError, match="statement timeout"):
                await conn.execute(text("SELECT pg_sleep(2)"))
    finally:
        await slow.dispose()


async def test_worker_engine_has_the_longer_timeout(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "DATABASE_URL", TEST_URL)
    monkeypatch.setattr(get_settings(), "DB_WORKER_STATEMENT_TIMEOUT_MS", 123_000)
    worker = tasks._engine()
    try:
        async with worker.connect() as conn:
            assert (await conn.scalar(text("SHOW statement_timeout"))) == "123s"
    finally:
        await worker.dispose()
