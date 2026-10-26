import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.core.config import get_settings
from app.core.db_url import engine_options
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Alembic compares this metadata to the DB when autogenerating migrations.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting (alembic upgrade head --sql)."""
    context.configure(
        url=get_settings().migration_database_url.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # DATABASE_URL_DIRECT if set (Neon: the non-pooled host), else DATABASE_URL.
    # Only when migrations do go through PgBouncer (no direct URL given) does
    # the connection need the PgBouncer-safe options.
    settings = get_settings()
    through_pgbouncer = settings.DB_PGBOUNCER and not settings.DATABASE_URL_DIRECT
    options = engine_options(
        # Migrations may rewrite big tables: no statement timeout.
        statement_timeout_ms=0,
        pgbouncer=through_pgbouncer,
        pool_recycle_seconds=settings.DB_POOL_RECYCLE_SECONDS,
    )
    engine = create_async_engine(settings.migration_database_url, **options)
    async with engine.connect() as connection:
        # Alembic's migration runner is synchronous, so we hand it a sync view
        # of the async connection.
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
