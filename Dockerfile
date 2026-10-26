# Three stages from one file:
#   builder  installs the locked dependencies into /app/.venv (needs uv).
#   dev      (what docker compose builds): builder + pytest, ruff, the tests,
#            so `docker compose exec api pytest -q` and `... ruff check .` work.
#   prod     (the default for a plain `docker build .`, and what Render
#            builds): a fresh slim image with only the venv and the app. No
#            uv, no build tools, no tests; runs as a non-root user.
FROM python:3.12-slim AS builder

# uv binary, copied from the official image (pinned; Dependabot bumps it)
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# tiktoken downloads its vocabulary on first use. Fetch it at build time so the
# running container never needs internet access to count tokens.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"


FROM builder AS dev
# The dev dependency group (pytest, ruff, ...), then the code, the tests and
# the retrieval evaluation (its tests import it). Docker Compose gives the
# services their own commands (see docker-compose.yml).
RUN uv sync --frozen --no-install-project
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts
COPY demo ./demo
COPY tests ./tests
COPY eval ./eval
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]


# Last stage = the default target, so a plain `docker build .` makes this image.
FROM python:3.12-slim AS prod

# MALLOC_ARENA_MAX=2: glibc gives each thread its own memory arena, and the
# app runs extraction and file I/O in worker threads. Capping the arenas
# lowered peak memory from 287 to 253 MB in the 512 MB measurements
# (docs/HARDENING.md, "Memory").
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache \
    MALLOC_ARENA_MAX=2

# A system user with no login shell and no home: if the app is ever
# compromised, the attacker is not root inside the container.
RUN groupadd --system app && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
# Owned by root and read-only to the app user: the app can't modify its own
# code or dependencies.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.tiktoken_cache /app/.tiktoken_cache
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# The only writable place, for STORAGE_BACKEND=local (on Render it's
# ephemeral; use STORAGE_BACKEND=s3 there).
RUN mkdir -p /app/storage && chown app:app /app/storage

USER app

# Render (and most platforms) say which port to listen on in $PORT.
ENV PORT=8000
EXPOSE 8000

# Liveness only (no database or Redis checks, see app/api/health.py). The
# start period covers the migrations the entrypoint runs first.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health/live\", timeout=4)"]

ENTRYPOINT ["entrypoint.sh"]
