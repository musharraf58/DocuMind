"""Turning the database URL a host gives you into what asyncpg accepts, and
the engine options that go with it.

Hosted Postgres providers (Neon, and most others) hand out libpq-style URLs:

    postgresql://user:pass@host/db?sslmode=require&channel_binding=require

asyncpg is not libpq. Through SQLAlchemy it needs the "+asyncpg" driver in
the scheme, spells TLS as `ssl=require` (an unknown `sslmode` argument makes
it refuse to connect), and has no `channel_binding` option at all. So
to_asyncpg_url() rewrites those parts and leaves everything else alone.

PgBouncer (Neon's "-pooler" URLs) needs two more things, which
engine_options() handles when DB_PGBOUNCER is on:

- In transaction mode a client gets a server connection only for the length
  of one transaction; the next transaction may run on another one. asyncpg
  prepares statements and caches them per connection, under generated names.
  A cached statement may not exist on the server connection we get next, and
  two clients may pick the same name on one server connection. So: no caches,
  and a unique name for every statement.
- PgBouncer refuses unknown connection startup parameters, including the
  statement_timeout we normally send that way. Instead each transaction
  starts with SET LOCAL statement_timeout, which lasts only until that
  transaction ends, so it can't leak to another client sharing the server
  connection.
"""

import uuid
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine

ASYNCPG_DRIVER = "postgresql+asyncpg"
POSTGRES_SCHEMES = {"postgres", "postgresql", "postgresql+asyncpg", "postgresql+psycopg2"}
# libpq sslmode values; asyncpg's `ssl` accepts the same words.
SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
# libpq-only options asyncpg doesn't know. Dropping channel_binding means
# channel binding is no longer *required*; TLS (ssl=require) still is.
DROPPED_PARAMS = {"channel_binding"}


def to_asyncpg_url(raw: str) -> URL:
    """The same database, as a URL SQLAlchemy's asyncpg dialect accepts.

    Raises ValueError for something that isn't a Postgres URL or an sslmode
    asyncpg doesn't understand. The message never includes the URL itself:
    it contains the password.
    """
    try:
        url = make_url(raw)
    except Exception as exc:
        raise ValueError("DATABASE_URL is not a valid database URL") from exc
    if url.drivername not in POSTGRES_SCHEMES:
        raise ValueError(f"DATABASE_URL must be a PostgreSQL URL (got scheme {url.drivername!r})")

    query = {k: v for k, v in url.query.items() if k not in DROPPED_PARAMS}
    sslmode = query.pop("sslmode", None)
    if sslmode is not None and "ssl" not in query:  # an explicit ssl= wins
        if sslmode not in SSL_MODES:
            raise ValueError(f"Unsupported sslmode {sslmode!r} in DATABASE_URL")
        query["ssl"] = sslmode
    return url.set(drivername=ASYNCPG_DRIVER, query=query)


def engine_options(
    statement_timeout_ms: int, pgbouncer: bool, pool_recycle_seconds: int
) -> dict[str, Any]:
    """Keyword arguments for create_async_engine."""
    options: dict[str, Any] = {
        # Test each pooled connection before use. Neon closes connections
        # when it scales to zero after a few idle minutes, and a free host
        # may sleep; without this, the first request after a pause would fail.
        "pool_pre_ping": True,
        # Replace connections older than this, before a proxy or firewall
        # drops them silently.
        "pool_recycle": pool_recycle_seconds,
    }
    if pgbouncer:
        options["connect_args"] = {
            "statement_cache_size": 0,  # asyncpg's own cache
            "prepared_statement_cache_size": 0,  # SQLAlchemy's cache
            "prepared_statement_name_func": _unique_statement_name,
        }
    else:
        options["connect_args"] = {
            "server_settings": {"statement_timeout": str(statement_timeout_ms)}
        }
    return options


def _unique_statement_name() -> str:
    return f"__asyncpg_{uuid.uuid4().hex}__"


def apply_statement_timeout_per_transaction(engine: AsyncEngine, statement_timeout_ms: int) -> None:
    """PgBouncer mode: set the timeout at the start of every transaction
    (one short extra statement each time; the price of pooling)."""
    sql = f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}"

    @event.listens_for(engine.sync_engine, "begin")
    def set_timeout(connection) -> None:
        connection.exec_driver_sql(sql)
