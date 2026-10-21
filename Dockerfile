# Two images from one file:
#   dev  (what docker compose builds): the app plus pytest, ruff and the tests,
#        so `docker compose exec api pytest -q` and `... ruff check .` work.
#   prod (the default for a plain `docker build .`): only what the app needs.
FROM python:3.12-slim AS base

# uv binary, copied from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
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

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS dev
# The dev dependency group (pytest, ruff, ...), then the tests and the
# retrieval evaluation (its tests import it).
RUN uv sync --frozen --no-install-project
COPY tests ./tests
COPY eval ./eval


# Last stage = the default target, so a plain `docker build .` makes the slim image.
FROM base AS prod

