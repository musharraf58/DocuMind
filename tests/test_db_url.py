"""Database URL translation and engine options (app/core/db_url.py)."""

import pytest

from app.core.config import Settings
from app.core.db_url import engine_options, to_asyncpg_url

NEON = (
    "postgresql://alex:s3cret-pw@ep-cool-name-123456-pooler.eu-central-1.aws.neon.tech"
    "/neondb?sslmode=require&channel_binding=require"
)


def query(url) -> dict:
    return dict(url.query)


def test_neon_url_becomes_asyncpg_with_ssl():
    url = to_asyncpg_url(NEON)
    assert url.drivername == "postgresql+asyncpg"
    assert query(url) == {"ssl": "require"}  # sslmode translated, channel_binding dropped
    # Everything else is untouched.
    assert (url.username, url.password, url.database) == ("alex", "s3cret-pw", "neondb")
    assert url.host == "ep-cool-name-123456-pooler.eu-central-1.aws.neon.tech"


@pytest.mark.parametrize("scheme", ["postgres", "postgresql", "postgresql+psycopg2"])
def test_other_postgres_schemes_get_the_asyncpg_driver(scheme):
    assert to_asyncpg_url(f"{scheme}://u:p@h:5432/db").drivername == "postgresql+asyncpg"


def test_compose_url_is_unchanged():
    """Local Docker Compose keeps working exactly as before."""
    raw = "postgresql+asyncpg://documind:documind@postgres:5432/documind"
    assert to_asyncpg_url(raw).render_as_string(hide_password=False) == raw


@pytest.mark.parametrize("mode", ["disable", "prefer", "require", "verify-ca", "verify-full"])
def test_each_sslmode_maps_to_the_same_ssl_value(mode):
    assert query(to_asyncpg_url(f"postgresql://u:p@h/db?sslmode={mode}")) == {"ssl": mode}


def test_explicit_ssl_wins_over_sslmode():
    url = to_asyncpg_url("postgresql://u:p@h/db?ssl=verify-full&sslmode=require")
    assert query(url) == {"ssl": "verify-full"}


def test_other_query_parameters_are_kept():
    url = to_asyncpg_url("postgresql://u:p@h/db?sslmode=require&application_name=documind")
    assert query(url) == {"ssl": "require", "application_name": "documind"}


def test_unknown_sslmode_is_refused():
    with pytest.raises(ValueError, match="Unsupported sslmode"):
        to_asyncpg_url("postgresql://u:p@h/db?sslmode=sometimes")


@pytest.mark.parametrize("raw", ["mysql://u:secretpw@h/db", "not a url secretpw"])
def test_non_postgres_urls_are_refused_without_echoing_the_password(raw):
    with pytest.raises(ValueError) as caught:
        to_asyncpg_url(raw)
    assert "secretpw" not in str(caught.value)


def test_direct_engine_sends_the_timeout_at_connect():
    options = engine_options(15_000, pgbouncer=False, pool_recycle_seconds=300)
    assert options["pool_pre_ping"] is True
    assert options["pool_recycle"] == 300
    assert options["connect_args"] == {"server_settings": {"statement_timeout": "15000"}}


def test_pgbouncer_engine_disables_statement_caches_and_startup_parameters():
    """PgBouncer refuses statement_timeout as a startup parameter, and its
    transaction pooling breaks asyncpg's per-connection statement cache."""
    args = engine_options(15_000, pgbouncer=True, pool_recycle_seconds=300)["connect_args"]
    assert args["statement_cache_size"] == 0
    assert args["prepared_statement_cache_size"] == 0
    assert "server_settings" not in args
    name = args["prepared_statement_name_func"]
    assert name() != name()  # unique each time: no collisions between clients


BASE = {"REDIS_URL": "redis://x", "SECRET_KEY": "k"}


def test_migrations_use_the_direct_url_when_given():
    settings = Settings(**BASE, DATABASE_URL=NEON, DATABASE_URL_DIRECT=NEON.replace("-pooler", ""))
    assert "-pooler" in settings.database_url.host
    assert "-pooler" not in settings.migration_database_url.host
    assert query(settings.migration_database_url) == {"ssl": "require"}


def test_migrations_fall_back_to_database_url():
    settings = Settings(**BASE, DATABASE_URL=NEON)
    assert settings.migration_database_url == settings.database_url


def test_a_bad_database_url_fails_at_startup_without_leaking_it():
    with pytest.raises(ValueError) as caught:
        Settings(**BASE, DATABASE_URL="mysql://u:topsecretpw@h/db")
    assert "topsecretpw" not in str(caught.value)


async def test_pgbouncer_mode_sets_the_timeout_inside_each_transaction(monkeypatch):
    """Against the real test database: the timeout arrives by SET LOCAL."""
    from sqlalchemy import text
    from sqlalchemy.pool import NullPool

    from app.core.config import get_settings
    from app.core.database import make_engine
    from tests.conftest import TEST_DB_URL

    monkeypatch.setattr(get_settings(), "DB_PGBOUNCER", True)
    engine = make_engine(
        TEST_DB_URL.render_as_string(hide_password=False),
        statement_timeout_ms=1234,
        poolclass=NullPool,
    )
    try:
        async with engine.connect() as conn:
            assert await conn.scalar(text("SHOW statement_timeout")) == "1234ms"
    finally:
        await engine.dispose()
