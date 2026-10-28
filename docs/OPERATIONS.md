# Operations

Running DocuMind in full mode (Docker Compose, with a Celery worker) and in
lite mode (free tiers, one small instance), deploying it, and every
environment variable. Locally, lite mode can also run like production:
[LITE_LOCAL.md](LITE_LOCAL.md).

## Full mode (Docker Compose)

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker with Docker Compose

uv installs Python 3.12 for you if it's missing.

### Setup

```bash
cp .env.example .env          # then set SECRET_KEY (32+ random chars, see comment inside),
                              # pick an embedding provider (see [Embeddings](ARCHITECTURE.md#embeddings)), and set
                              # REGISTRATION_MODE=open for local use (default: invite only)
uv sync                       # create .venv and install deps (incl. dev)
```

### Run everything in Docker

```bash
docker compose up --build -d  # api + worker + postgres + redis; api runs migrations on start
docker compose ps             # api, postgres and redis "healthy", worker "running"
curl localhost:8000/health    # {"status":"ok","database":"ok","redis":"ok"}
docker compose logs -f api    # follow the API's logs
docker compose logs -f worker # follow processing: jobs, embedding progress, sweeper
docker compose down           # stop (data kept in the postgres_data volume)
docker compose down -v        # stop AND delete the database volume
```

### Run the API locally (hot reload), with only the databases in Docker

```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
curl localhost:8000/health

# In a second terminal: the worker, so uploaded documents get processed.
uv run celery -A app.workers.celery_app worker --beat --concurrency=1 --loglevel=INFO
```

Run only **one** worker. Celery beat (`--beat`) runs the sweeper, and two workers would run
it twice. To scale out, move beat to its own process first.

## Observability and health

- **Request ids**:
  - every response has an `X-Request-ID` (yours if well-formed, otherwise a new one);
  - every log line includes it;
  - so do the worker's logs for a job queued by that request.

  Quote it when reporting a problem.
- **`LOG_FORMAT=json`**: one JSON object per log line, for log collectors.
- **Health checks**:
  - `/health/live` checks only the process; a platform restarts the app when it fails;
  - `/health/ready` checks Postgres, Redis and the worker: in Celery mode its heartbeat
    (written every 30 s, expiring after 90 s), in inline mode the in-process worker task;
    traffic is held back while it fails;
  - `/health` is unchanged.
- **`APP_ENV=production`** refuses to start, listing every problem at once, when:
  - `SECRET_KEY` is weak or a well-known example;
  - the docs are enabled without `ALLOW_DOCS_IN_PRODUCTION=true`;
  - rate limits are off, or a fake provider is configured;
  - the invite code is shorter than 12 characters;
  - `STORAGE_BACKEND=s3` without its bucket and keys.

  Startup errors never print setting values.
- **Database**: connection pool and statement timeout settings, the index review and the
  dependency audit are in [docs/HARDENING.md](HARDENING.md).

## Lite mode (free tier)

Lite mode runs the whole service on free tiers, on one small web instance,
for a demo or a small group of invited users. Everything stays free to run.

| Part | Service (free tier, checked October 2026) | Setting |
| --- | --- | --- |
| API + document jobs | [Render](https://render.com/docs/free) web service: 512 MB, 0.1 CPU, sleeps after 15 idle minutes (about 1 minute to wake), disk wiped on restart, 750 hours a month, no free background worker | `render.yaml` |
| Postgres + pgvector | [Neon](https://neon.com/docs/introduction/plans): 1 GB per project, scales to zero after 5 idle minutes, PgBouncer pooling in transaction mode | `DATABASE_URL`, `DATABASE_URL_DIRECT`, `DB_PGBOUNCER=true` |
| Redis | [Upstash](https://upstash.com/pricing/redis): 250 MB, 500,000 commands a month | `REDIS_URL` |
| Uploaded files | [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/) (10 GB, free egress) or [Backblaze B2](https://www.backblaze.com/cloud-storage/pricing) (10 GB) | `STORAGE_BACKEND=s3`, `S3_*` |
| Embeddings, answers | Voyage (3 requests/minute), Gemini (small daily quota) | `EMBEDDING_API_KEY`, `LLM_API_KEY` |

What changes compared with Docker Compose, and why:

- **No Celery worker: `INGESTION_MODE=inline`.** Render has no free
  background worker, and an idle Celery worker sends Redis about 188,000
  commands a day (measured), the whole Upstash month in under three days.
  The API process runs the jobs itself, one at a time, from a bounded
  in-memory queue (20 by default; a full queue answers uploads with 503
  `ingestion_queue_full`). Both modes call the same job function
  (`jobs.process_document`): the same atomic claim, heartbeat, retries and
  failure handling. The queue living in memory is safe because Postgres, not
  the queue, is the source of truth: the sweeper runs in-process every 2
  minutes (Postgres only, no Redis) and re-queues anything left "pending"
  or stuck "processing". On shutdown, the running job gets 15 seconds, then
  its document goes back to "pending" without using up an attempt.
- **Files in a bucket: `STORAGE_BACKEND=s3`.** The free instance's disk is
  wiped on every deploy, restart and sleep; files there would vanish while
  their database rows remain. Uploads and downloads stream in parts, never
  whole files in memory.
- **Neon through PgBouncer.** Neon's connection strings are pasted as they
  are (`sslmode=require` is translated for asyncpg). `DB_PGBOUNCER=true`
  turns off asyncpg's prepared-statement cache (transaction pooling hands a
  client a different server connection per transaction) and sets the
  statement timeout per transaction (PgBouncer refuses it as a connection
  parameter). Migrations use `DATABASE_URL_DIRECT`, the non-pooled host.
- **Redis budget.** Each request's Redis commands were counted and trimmed
  (a question costs 7, an idle app 0); see
  [docs/HARDENING.md](HARDENING.md), "Redis command budget".
- **Capacity.** Uploads stop at `MAX_DB_SIZE_MB` (400) while asking and
  searching keep working; `RETENTION_DAYS` can expire old accounts with
  everything they own.
- **Memory.** The worst case measured under a 512 MB limit peaks at 262 MB
  (docs/HARDENING.md, "Memory").

### Deploying to Render

1. **Neon.** Create a project (pgvector is available; the first migration
   enables it). From *Connection details* copy two connection strings:
   the **pooled** one (host contains `-pooler`) for `DATABASE_URL`, and the
   **direct** one for `DATABASE_URL_DIRECT`.
2. **Upstash.** Create a Redis database in the region nearest your Render
   region and copy its `rediss://...` URL (TLS) for `REDIS_URL`.
3. **A bucket**, private, no public access:
   - *R2:* create a bucket, then an API token with *Object Read & Write* on
     that bucket only. `S3_ENDPOINT_URL=https://<account id>.r2.cloudflarestorage.com`,
     `S3_REGION=auto`.
   - *B2:* create a private bucket and an application key restricted to it.
     `S3_ENDPOINT_URL=https://s3.<region>.backblazeb2.com`,
     `S3_REGION=<region>` (e.g. `us-west-004`).
4. **Keys**: a Voyage API key and a Gemini API key.
5. **Render**: *New → Blueprint*, pick this repository. Render reads
   `render.yaml` and asks for every value marked `sync: false` (the secrets);
   generate `SECRET_KEY` with
   `python -c "import secrets; print(secrets.token_urlsafe(48))"` and pick an
   `INVITE_CODE` of at least 12 characters. The container applies the
   migrations on start, then serves on `$PORT`. Deploys of `main` wait for CI
   to pass.
6. **Check it**: `GET /health/live` answers `{"status":"ok"}`;
   `GET /health/ready` reports database, Redis and `"worker": "ok"` (the
   in-process worker; there is no heartbeat in lite mode). Register with the
   invite code and upload a document.
7. **Demo account (optional).** With the same environment as the
   deployment (copy `.env.lite.example` to `.env.lite`, fill it in, and add
   `DEMO_EMAIL` and `DEMO_PASSWORD`), from your machine:

   ```bash
   set -a; . ./.env.lite; set +a
   uv run python -m scripts.seed_demo
   ```

   It creates (or updates) the read-only demo account and its sample
   collection from `demo/corpus/`. Run it again any time; it only changes
   what changed.

`.env.lite.example` lists every variable lite mode needs. With
`APP_ENV=production` the app refuses to start on a weak configuration (weak
`SECRET_KEY`, docs on, rate limits off, fake providers, short invite code,
missing S3 settings), listing every problem without printing any value.

**`TRUSTED_PROXY_COUNT` is 0 on Render, and that was measured, not
assumed.** With 1, a client that sent made-up `X-Forwarded-For` headers was
never rate-limited: it chose its own address. With 0, the per-IP login limit
works (401 for about 10 attempts, then 429). The cost: all visitors share
the per-IP limits on login and registration, while per-email and per-user
limits and the daily quotas still apply to each person. Details and how to
re-check: [docs/HARDENING.md](HARDENING.md), "6. The client address
behind Render". A count that is too low only makes limits shared, never
bypassable; never raise it without repeating that test.

### Living with the free tier

- **The first request after 15 idle minutes waits about a minute** while
  Render wakes the instance (and Neon its database). A visitor's first
  page view is slow; the rest are not.
- **Documents wait while the service sleeps.** A document uploaded just
  before the instance goes idle is picked up again when the next request
  wakes it (the sweeper runs at startup). A document whose embedding takes
  longer than 15 minutes (roughly 75,000+ tokens, about 110 pages, at
  Voyage's free rate) can be interrupted if nobody uses the app meanwhile;
  it restarts from the beginning on wake, which spends its embedding tokens
  again. Keeping the document page open (it polls the status) keeps the
  instance awake.
- **One instance only.** Inline mode and migrations-on-start assume a single
  instance (see the comment in `docker/entrypoint.sh`).
- **Uptime monitors** should call `/health/live` (no Redis command), never
  `/health/ready`.

## Environment variables

Every setting the app reads (`app/core/config.py`), with its default.
`.env.example` (full mode) and `.env.lite.example` (lite mode) list them
with comments; a test checks that every setting appears here.

### Core and database

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_ENV` | `development` | `development` or `production` (production refuses a weak configuration at startup) |
| `DATABASE_URL` | required | Postgres URL; `postgresql+asyncpg://` or libpq style with `sslmode` (translated) |
| `DATABASE_URL_DIRECT` | (empty) | Migrations only: the direct (non-pooled) URL. Empty: `DATABASE_URL` |
| `DB_PGBOUNCER` | `false` | `true` behind PgBouncer in transaction mode (Neon `-pooler`): no prepared-statement cache |
| `DB_POOL_SIZE` | `5` | Connections kept open per process |
| `DB_MAX_OVERFLOW` | `10` | Extra connections allowed under load |
| `DB_POOL_TIMEOUT_SECONDS` | `10` | Longest wait for a free connection |
| `DB_POOL_RECYCLE_SECONDS` | `1800` | Reconnect connections older than this |
| `DB_STATEMENT_TIMEOUT_MS` | `15000` | Longest one SQL statement may run in the API |
| `DB_WORKER_STATEMENT_TIMEOUT_MS` | `300000` | The same for background jobs |
| `REDIS_URL` | required | Redis URL (`rediss://` for TLS, as Upstash gives it) |
| `LOG_FORMAT` | `text` | `text` or `json` |
| `MAX_DB_SIZE_MB` | `400` | Uploads stop when the database reaches this size; asking keeps working |
| `RETENTION_DAYS` | `0` | Delete accounts inactive this long, with everything they own (0: never) |

### Auth and registration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SECRET_KEY` | required | Signs the tokens. 32+ random characters; production refuses weak ones |
| `JWT_ALGORITHM` | `HS256` | Token signature algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token lifetime |
| `REGISTRATION_MODE` | `invite` | `open`, `invite` or `closed` |
| `INVITE_CODE` | (empty) | Needed to register in invite mode (12+ characters in production) |
| `LOGIN_FREE_ATTEMPTS` | `3` | Failed logins before the progressive delay starts |
| `LOGIN_DELAY_BASE_SECONDS` | `2` | First delay; doubles with each further failure |
| `LOGIN_DELAY_MAX_SECONDS` | `900` | Longest delay |
| `LOGIN_FAILURE_WINDOW_MINUTES` | `15` | Failures are forgotten after this long without one |
| `DEMO_EMAIL` | (empty) | The shared read-only demo account (`scripts/seed_demo.py`) |
| `DEMO_PASSWORD` | (empty) | Its password; read by the seed script only |

### Storage and uploads

| Variable | Default | Meaning |
| --- | --- | --- |
| `STORAGE_BACKEND` | `local` | `local` (a directory) or `s3` (R2, B2, any S3-compatible bucket) |
| `STORAGE_PATH` | `./storage` | The directory for `local` |
| `S3_ENDPOINT_URL` | (empty) | The bucket's S3 endpoint |
| `S3_BUCKET` | (empty) | Bucket name |
| `S3_REGION` | `us-east-1` | Region (`auto` for R2) |
| `S3_ACCESS_KEY_ID` | (empty) | Access key, restricted to the bucket |
| `S3_SECRET_ACCESS_KEY` | (empty) | Its secret |
| `MAX_UPLOAD_MB` | `20` | Largest single file |
| `MAX_STORAGE_MB_PER_USER` | `100` | Total stored per user |
| `MAX_DOCUMENTS_PER_USER` | `50` | Documents per user |
| `MAX_PENDING_DOCUMENTS_PER_USER` | `5` | Documents waiting or processing at once, per user |

### Ingestion

| Variable | Default | Meaning |
| --- | --- | --- |
| `INGESTION_MODE` | `celery` | `celery` (a worker process) or `inline` (the API runs jobs; lite mode) |
| `INLINE_QUEUE_CAPACITY` | `20` | Inline mode: jobs waiting in memory before uploads get 503 |
| `INLINE_SHUTDOWN_GRACE_SECONDS` | `15` | Inline mode: time the running job gets on shutdown |
| `MAINTENANCE_INTERVAL_SECONDS` | `120` | Inline mode: how often the sweeper and retention run |
| `MAX_JOB_ATTEMPTS` | `4` | Tries per document for transient failures |
| `JOB_RETRY_BASE_SECONDS` | `30` | First retry delay; doubles each time |
| `JOB_TIMEOUT_MINUTES` | `120` | Longest one job may run |
| `SWEEP_PENDING_MINUTES` | `3` | The sweeper re-enqueues documents pending longer than this |
| `STALE_PROCESSING_MINUTES` | `10` | ...and resets processing documents with no heartbeat for this long |
| `CHUNK_TOKENS` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `60` | Tokens repeated from the previous chunk |
| `MAX_EXTRACT_PAGES` | `500` | PDFs with more pages fail |
| `MAX_EXTRACTED_CHARS` | `2000000` | Documents with more text fail |
| `MIN_EXTRACTED_CHARS` | `50` | Documents with less text fail (typically scanned PDFs) |
| `MAX_DOCX_UNCOMPRESSED_MB` | `100` | Zip-bomb guard for DOCX |
| `MAX_DOCX_ENTRIES` | `1000` | Zip-bomb guard for DOCX |

### Embeddings

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `voyage` | `voyage` or `fake` (no network; development and tests) |
| `EMBEDDING_MODEL` | `voyage-4-lite` | Model name sent to the provider |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector size; must match the database column |
| `EMBEDDING_API_KEY` | (empty) | Voyage API key; never logged |
| `EMBEDDING_BATCH_SIZE` | `64` | Most texts per request |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `EMBEDDING_TPM_LIMIT` | `10000` | Your account's tokens per minute |
| `EMBEDDING_RPM_LIMIT` | `3` | Your account's requests per minute |
| `EMBEDDING_RATE_SAFETY` | `0.5` | Fraction of the token limit we plan to use |
| `EMBEDDING_MAX_RETRIES` | `6` | Retries per batch on 429, 5xx and timeouts |
| `EMBEDDING_RETRY_BASE_SECONDS` | `2` | First retry wait; doubles each time |
| `EMBEDDING_RETRY_MAX_SECONDS` | `60` | Longest backoff wait |
| `EMBEDDING_MIN_RETRY_WAIT_SECONDS` | (empty) | Shortest wait after a 429 (empty: 60 / `EMBEDDING_RPM_LIMIT`) |
| `EMBEDDING_QUERY_MAX_WAIT_SECONDS` | `20` | Longest a search waits for embedding capacity |
| `EMBEDDING_INGEST_RPM_SHARE` | `0.67` | Share of requests ingestion may use; the rest is kept for searches |

### Retrieval and answers

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | `gemini` or `fake` |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Gemini model |
| `LLM_API_KEY` | (empty) | Gemini API key; never logged |
| `LLM_MAX_OUTPUT_TOKENS` | `600` | Answer length limit, thinking tokens included |
| `LLM_TIMEOUT_SECONDS` | `30` | Per request |
| `LLM_RPM_LIMIT` | `10` | Requests per minute, shared through Redis |
| `LLM_TEMPERATURE` | `1.0` | Sampling temperature |
| `RAG_TOP_K` | `8` | Chunks retrieved per question |
| `RAG_MAX_DISTANCE` | `0.7` | Relevance cut-off (measured: docs/EVALUATION.md) |
| `RAG_CONTEXT_TOKEN_BUDGET` | `3000` | Source text per question |
| `ASK_MAX_QUESTION_CHARS` | `1000` | Longest question accepted |
| `RAG_SEARCH_MODE` | `hybrid` | `hybrid` or `vector` |
| `HYBRID_CANDIDATES` | `20` | Candidates from each list before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RAG_KEYWORD_MATCH` | `strict` | `strict` (every term) or `loose` (any term) |
| `TEXT_SEARCH_CONFIG` | `english` | Postgres text search configuration |
| `RAG_KEYWORD_GATE_SLACK` | `0.1` | Extra distance allowed for a strong keyword match |
| `RAG_HISTORY_MAX_TURNS` | `6` | Earlier turns included in the prompt |
| `RAG_HISTORY_TOKEN_BUDGET` | `1000` | ...within this many tokens |
| `RAG_REWRITE_TIMEOUT_SECONDS` | `10` | After this, a follow-up is searched as asked |
| `STREAM_HEARTBEAT_SECONDS` | `15` | Heartbeat interval while a stream is waiting |

### Limits and quotas

| Variable | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_ENABLED` | `true` | Master switch (refused off in production) |
| `RATE_LIMIT_LOGIN_PER_IP` | `10/minute` | Login attempts per client address |
| `RATE_LIMIT_LOGIN_PER_EMAIL` | `5/minute` | Login attempts per email, from any address |
| `RATE_LIMIT_REGISTER_PER_IP` | `5/hour` | Registrations per client address |
| `RATE_LIMIT_ACCOUNT` | `5/minute` | Password changes and account deletion, per user |
| `RATE_LIMIT_ASK` | `10/minute` | Questions per user |
| `RATE_LIMIT_UPLOAD` | `10/hour` | Uploads per user |
| `RATE_LIMIT_DEFAULT` | `120/minute` | Every other authenticated endpoint, per user |
| `DAILY_QUESTIONS_PER_USER` | `50` | Questions per user per day |
| `DAILY_EMBED_TOKENS_PER_USER` | `200000` | Embedding tokens per user per day |
| `DAILY_UPLOADS_PER_USER` | `20` | Uploads per user per day |
| `GLOBAL_DAILY_LLM_BUDGET` | `350` | LLM requests per day, all users together |
| `QUOTA_RESET_HOUR_UTC` | `0` | When the quota day starts |

### HTTP

| Variable | Default | Meaning |
| --- | --- | --- |
| `CORS_ORIGINS` | (empty) | Allowed browser origins, comma-separated (`*` refused) |
| `MAX_REQUEST_BODY_KB` | `64` | Largest request body except uploads |
| `TRUSTED_PROXY_COUNT` | `0` | Proxies whose `X-Forwarded-For` entries are trusted (Render: 0, measured) |
| `DOCS_ENABLED` | `true` | Serve `/docs`, `/redoc` and `/openapi.json` |
| `ALLOW_DOCS_IN_PRODUCTION` | `false` | Opt-in for docs in production |
| `ALLOW_FAKE_PROVIDERS_IN_PRODUCTION` | `false` | Opt-in for fake providers in production (the local lite stack only) |

### Unused

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | (empty) | Not read by the app (left from the project template) |

## Development

### Migrations

```bash
uv run alembic upgrade head                         # apply all
uv run alembic current                              # show current revision
uv run alembic downgrade -1                         # undo last
uv run alembic revision --autogenerate -m "message" # new migration from model changes
```

### Tests

Tests use a separate database, `documind_test`, created automatically on the same
Postgres server (so Postgres must be running). They never call Voyage, Gemini or S3
(simulated servers, moto's in-process S3). Redis is faked, except in
`tests/test_real_redis.py`, which runs the Lua scripts on a real Redis (database 15) and
skips without one. CI runs everything against real Postgres + pgvector and Redis
containers; see [docs/CI.md](CI.md).

```bash
docker compose up -d postgres
uv run pytest
```

Or inside the running `api` container, which includes pytest, ruff and the tests (docker
compose builds the Dockerfile's `dev` stage; a plain `docker build .` makes the slimmer
`prod` image without them):

```bash
docker compose exec api pytest -q
docker compose exec api ruff check .
```

If those commands report `executable file not found`, the container is running an older
image: run `docker compose up -d --build` once to rebuild it.

### Lint and format

```bash
uv run ruff check .           # lint (add --fix to auto-fix)
uv run ruff format .          # format
```
