# DocuMind

RAG knowledge base API. FastAPI (async), PostgreSQL + pgvector, SQLAlchemy 2.0 async, Redis.

**Status:** Phase 0 (scaffolding only: config, DB, migrations, `/health`, tests).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker with Docker Compose

uv installs Python 3.12 for you if it's missing.

## Setup

```bash
cp .env.example .env          # then set SECRET_KEY
uv sync                       # create .venv and install deps (incl. dev)
```

## Run everything in Docker

```bash
docker compose up --build -d  # api + postgres + redis; api runs migrations on start
docker compose ps             # all three should become "healthy"
curl localhost:8000/health    # {"status":"ok","database":"ok"}
docker compose logs -f api    # follow logs
docker compose down           # stop (data kept in the postgres_data volume)
docker compose down -v        # stop AND delete the database volume
```

## Run the API locally (hot reload), with only the databases in Docker

```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
curl localhost:8000/health
```

## Migrations

```bash
uv run alembic upgrade head                         # apply all
uv run alembic current                              # show current revision
uv run alembic downgrade -1                         # undo last
uv run alembic revision --autogenerate -m "message" # new migration from model changes
```

## Tests

Tests use a separate database, `documind_test`, created automatically on the same
Postgres server (so Postgres must be running).

```bash
docker compose up -d postgres
uv run pytest
```

## Lint and format

```bash
uv run ruff check .           # lint (add --fix to auto-fix)
uv run ruff format .          # format
```
