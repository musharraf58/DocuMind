# DocuMind

RAG knowledge base API. FastAPI (async), PostgreSQL + pgvector, SQLAlchemy 2.0 async, Redis.

**Status:** Phase 5 (users, JWT auth, per-user collections, document upload and storage,
text extraction and chunking, embeddings and vector search, background processing).

## Architecture

```
                      ┌──────────────────────────── Postgres + pgvector ──────────────────┐
                      │  documents (status, attempts, timestamps)   chunks (text, vector) │
                      └──────▲──────────────────────────────────────────────▲────────────┘
                             │ 1. save "pending", COMMIT                   │ claim, heartbeat,
                             │                                             │ one transaction → ready
  client ──HTTP──► API (FastAPI) ──2. enqueue(document id)──►  Redis  ──► Worker (Celery, 1 job
    ▲                │   ▲                                   │ queue │      at a time)
    │ 202 + status   │   └── search: embed the query ───────►│ rate- │◄──── ingestion: embed
    └────────────────┘                                       │ limit │      the chunks
                                                             │window │      │
                                                             └───────┘      ▼
                                                     shared by both    Voyage AI (embeddings)
                                                                        sweeper (every minute
                                                                        + at worker start)
```

- **API** (`app/main.py`): accepts uploads, saves the file and the document as `pending`,
  commits, and only then puts a job on the queue. It answers `202` at once; clients poll
  `GET /documents/{id}` for the status.
- **Redis**: the job queue (Celery broker) and the shared embedding rate-limit window, so
  the API's searches and the worker's ingestion never exceed the account's limits together.
- **Worker** (`app/workers/`): one Celery process, one job at a time. A job claims the
  document (`pending` → `processing`), extracts, chunks and embeds it, refreshing a heartbeat
  as it goes, and saves the chunks in one transaction (`ready`). Transient failures are
  retried with a growing delay; permanent ones fail at once.
- **Sweeper** (runs in the worker every minute, and once at startup): re-enqueues documents
  pending too long (say the upload happened while Redis was down) and resets documents whose
  worker died mid-job (stale heartbeat), so nothing is ever stuck.
- **Postgres** holds the truth: the queue may lose or repeat a message; the database state
  decides what actually happens.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker with Docker Compose

uv installs Python 3.12 for you if it's missing.

## Setup

```bash
cp .env.example .env          # then set SECRET_KEY (32+ random chars, see comment inside)
                              # and pick an embedding provider (see "Embeddings" below)
uv sync                       # create .venv and install deps (incl. dev)
```

## Run everything in Docker

```bash
docker compose up --build -d  # api + worker + postgres + redis; api runs migrations on start
docker compose ps             # api, postgres and redis "healthy", worker "running"
curl localhost:8000/health    # {"status":"ok","database":"ok","redis":"ok"}
docker compose logs -f api    # follow the API's logs
docker compose logs -f worker # follow processing: jobs, embedding progress, sweeper
docker compose down           # stop (data kept in the postgres_data volume)
docker compose down -v        # stop AND delete the database volume
```

## Run the API locally (hot reload), with only the databases in Docker

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

## Using the API

Open http://localhost:8000/docs (Swagger UI).

1. `POST /auth/register` with an email and a password of 8+ characters.
2. Run `POST /auth/login` (email in the `username` field) and copy the `access_token`
   from the response. Click **Authorize**, paste the token into the **Value** box
   (no `Bearer ` prefix needed) and confirm. Swagger now sends
   `Authorization: Bearer <token>` on every request. Tokens expire after 30 minutes.
3. Try `/auth/me` and the `/collections` endpoints.

Or with curl:

```bash
curl -X POST localhost:8000/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"correct-horse-battery"}'
TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -d 'username=me@example.com&password=correct-horse-battery' | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
curl -X POST localhost:8000/collections -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"name":"My docs"}'
curl localhost:8000/collections -H "Authorization: Bearer $TOKEN"
```

A collection that belongs to someone else returns 404, the same as one that doesn't exist.

## Documents

Upload files into one of your collections, then list, fetch, download or delete them.
Allowed types: `.pdf`, `.docx`, `.md`, `.txt` (the content is checked, not just the
extension). Max size is `MAX_UPLOAD_MB` (default 20). New documents start as `pending` and
are processed in the background (see "Processing" below).

| Method | Path | What it does |
|---|---|---|
| POST | `/collections/{id}/documents` | Upload (multipart field `file`). `202` with the document (`pending`) |
| GET | `/collections/{id}/documents` | List. `?limit=&offset=&status=pending\|processing\|ready\|failed` |
| GET | `/documents/{id}` | One document: `status`, `attempts`, `error_message`, `queued_at`, `processing_started_at`, `heartbeat_at`, `finished_at` |
| GET | `/documents/{id}/download` | The file itself (always as an attachment) |
| DELETE | `/documents/{id}` | Delete the document and its stored file |

Errors: `415` wrong file type, `413` too large, `422` empty or content doesn't match the
type, `409` the same file is already in that collection. Documents you don't own return
`404`. Deleting a collection also deletes its documents' files.

```bash
curl -X POST localhost:8000/collections/$COLLECTION_ID/documents \
  -H "Authorization: Bearer $TOKEN" -F "file=@notes.md"
curl -OJ localhost:8000/documents/$DOC_ID/download -H "Authorization: Bearer $TOKEN"
```

Files are stored under `STORAGE_PATH` (`./storage` locally; in Docker the `storage_data`
named volume mounted at `/data/storage`). They are saved as `<owner id>/<document id>`;
the uploaded filename is kept only as display metadata.

## Processing: text extraction and chunking

Processing turns an uploaded document into **chunks**: pieces of its text of at most
`CHUNK_TOKENS` tokens (default 500), each sharing about `CHUNK_OVERLAP_TOKENS` (default 60)
with the previous one. Each chunk is then **embedded** (turned into a vector) so it can be
searched by meaning (see "Embeddings and search" below).

| Method | Path | What it does |
|---|---|---|
| POST | `/documents/{id}/reprocess` | Queue the document again. `202`; `409` if it is already `pending` or `processing` |
| GET | `/documents/{id}/chunks` | The chunks, in order. `?limit=&offset=` |

Processing happens in the background worker. Upload, then poll the status:

```bash
curl localhost:8000/documents/$DOC_ID -H "Authorization: Bearer $TOKEN"   # pending → processing → ready
curl "localhost:8000/documents/$DOC_ID/chunks?limit=5" -H "Authorization: Bearer $TOKEN"
curl -X POST localhost:8000/documents/$DOC_ID/reprocess -H "Authorization: Bearer $TOKEN"
```

How it works:

1. The worker claims the document: `pending` → `processing`, `attempts` + 1. Only a
   `pending` document can be claimed, so a job delivered twice runs once.
2. Text is extracted in a worker thread: PDF page by page (chunks get a `page_number`),
   DOCX paragraphs and tables (headings become `section_title`), Markdown sections
   (`#` headings become `section_title`), plain text as is. The text is then cleaned
   (whitespace normalized, null characters removed, words hyphenated across lines rejoined).
3. The text is split by paragraph, then line, sentence and word until pieces fit, and the
   pieces are packed into chunks. A chunk never spans two pages or sections.
4. All chunks are embedded by the embedding provider, in token-sized batches that respect
   its rate limits (see "Rate limits"). This can take minutes, and no database transaction
   is open meanwhile.
5. In a single transaction, the old chunks are deleted, the new ones inserted with their
   vectors, and the document marked `ready`.

If anything fails, the previous chunks (and their vectors) are kept, and:

- **Permanent** problems (scanned or corrupt file, unsupported type, over a limit, invalid
  API key): `failed` at once, with a short `error_message`. Retrying wouldn't change it.
- **Transient** problems (rate limit, provider outage, timeout, database or Redis blip):
  back to `pending` with `error_message` saying which attempt failed, and retried after
  30 s, 60 s, 120 s... Once `attempts` reaches `MAX_JOB_ATTEMPTS` the document is `failed`.

You can reprocess `ready` and `failed` documents at any time. A document deleted while it is
being processed is left alone: the job notices at its next step, stops, and writes nothing.

Background job settings:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_JOB_ATTEMPTS` | 4 | Tries per document for transient failures |
| `JOB_RETRY_BASE_SECONDS` | 30 | First retry delay; doubles each time |
| `JOB_TIMEOUT_MINUTES` | 120 | Longest one job may run (a very large document takes ~1-2 hours at 10,000 tokens/minute) |
| `SWEEP_PENDING_MINUTES` | 3 | The sweeper re-enqueues documents pending longer than this |
| `STALE_PROCESSING_MINUTES` | 10 | ...and resets processing documents with no heartbeat for this long |

Deleting a document or collection deletes its chunks too.

Settings (see `.env.example`):

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_TOKENS` | 500 | Max tokens per chunk (tiktoken `cl100k_base`, an approximation) |
| `CHUNK_OVERLAP_TOKENS` | 60 | Tokens repeated from the previous chunk; must be under half of `CHUNK_TOKENS` |
| `MAX_EXTRACT_PAGES` | 500 | PDFs with more pages fail |
| `MAX_EXTRACTED_CHARS` | 2000000 | Documents with more text fail |
| `MIN_EXTRACTED_CHARS` | 50 | Documents with less text fail (typically scanned PDFs) |
| `MAX_DOCX_UNCOMPRESSED_MB` | 100 | Zip-bomb guard: DOCX files that would expand past this fail |
| `MAX_DOCX_ENTRIES` | 1000 | Zip-bomb guard: DOCX files with more internal parts fail |

Known limitations:

- No OCR: scanned PDFs fail with "No readable text found".
- PDFs that need a password, or use AES encryption, fail.
- DOCX headers, footers, footnotes and comments are not extracted.
- Each job has a time limit (`JOB_TIMEOUT_MINUTES`), but extraction itself runs in a thread,
  which can't be interrupted: a pathological file is bounded by the size limits and, as a
  last resort, by the worker's hard time limit.
- tiktoken downloads its vocabulary (a few MB) the first time it runs locally. The Docker
  image includes it at build time.

## Embeddings and search

Each chunk's vector is stored in `chunks.embedding` (pgvector, 1024 dimensions) together
with `chunks.embedding_model`, the model that made it. An HNSW index (`vector_cosine_ops`)
makes nearest-neighbour search fast.

Choose the provider with `EMBEDDING_PROVIDER`:

- `fake`: deterministic vectors from a hash of the text, with no network and no key. It's
  good for local development and is what the tests use. The vectors carry **no meaning**,
  so search results are effectively random. They're stored under the model name
  `fake-sha256`, so they're never searched together with real vectors.
- `voyage` (the default): [Voyage AI](https://docs.voyageai.com/docs/embeddings). Set
  `EMBEDDING_API_KEY`. Without a key, processing fails with "Embedding API key is not
  configured".

| Setting | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | `voyage` | `voyage` or `fake` |
| `EMBEDDING_MODEL` | `voyage-4-lite` | Model name sent to the provider |
| `EMBEDDING_DIMENSIONS` | 1024 | Vector size. Must match the database column (see below) |
| `EMBEDDING_API_KEY` | (empty) | Provider API key. Never logged or shown in errors |
| `EMBEDDING_BATCH_SIZE` | 64 | Most texts per request. The token budget below usually ends a batch sooner |
| `EMBEDDING_TIMEOUT_SECONDS` | 30 | Per-request timeout |
| `EMBEDDING_TPM_LIMIT` | 10000 | Your account's tokens per minute |
| `EMBEDDING_RPM_LIMIT` | 3 | Your account's requests per minute |
| `EMBEDDING_RATE_SAFETY` | 0.5 | Fraction of the token limit we plan to use (our tiktoken count can be well below the provider's) |
| `EMBEDDING_MAX_RETRIES` | 6 | Retries per batch on rate limits (429), server errors (5xx) and timeouts |
| `EMBEDDING_RETRY_BASE_SECONDS` | 2 | First retry wait; doubles each time |
| `EMBEDDING_RETRY_MAX_SECONDS` | 60 | Longest backoff wait |
| `EMBEDDING_MIN_RETRY_WAIT_SECONDS` | 60 / RPM limit | Shortest wait after a 429 (20 s at 3 requests per minute) |
| `EMBEDDING_QUERY_MAX_WAIT_SECONDS` | 20 | Longest a search waits for embedding capacity |
| `EMBEDDING_INGEST_RPM_SHARE` | 0.67 | Share of requests per minute background ingestion may use (2 of 3); the rest is kept for searches |

A bad key or invalid input fails at once without retrying.

### Rate limits

Embedding providers cap tokens and requests per minute. DocuMind stays under your limits
instead of finding them out from errors:

- **Budgets.** With the defaults it uses at most 5,000 tokens (10,000 x 0.5) and 3 requests
  in any rolling 60 seconds, for the API and the worker together. The safety factor applies
  to tokens only: our tiktoken count can undercount the provider's (a different tokenizer,
  plus the retrieval prompt it adds to every text), while requests are counted exactly. A
  sliding-window limiter keeps a log of what was sent in the last minute and, before every
  attempt, including retries and failed ones, waits until both budgets have room.
- **Batches are sized by tokens.** A batch ends when the next chunk would push it over the
  token budget (or `EMBEDDING_BATCH_SIZE`). One request bigger than the per-minute allowance
  could never succeed, however often it was retried, which is why batches are kept below it.
- **Room for searches.** Ingestion uses at most `EMBEDDING_INGEST_RPM_SHARE` of the requests
  (2 of 3) and 95% of the tokens (4,750), so uploads can never starve interactive searches.
- **429 responses.** Retrying within seconds only uses up more of the per-minute allowance,
  so after a 429 the wait is at least `Retry-After` (when sent) and at least
  `EMBEDDING_MIN_RETRY_WAIT_SECONDS`, doubling on each further 429 up to
  `EMBEDDING_RETRY_MAX_SECONDS`, with 0-25% random extra on top. Every caller is held until
  then. Server errors and timeouts get a short backoff instead. A `Retry-After` over two
  minutes fails the document with "retry in about N seconds".
- **Learning the real count.** The provider reports how many tokens each request used. The
  limiter records that number when it's higher than our estimate, and the log compares the
  two for every batch, for example `7391 reported by the provider (+12%)`, with a warning
  when a batch went over its budget by the provider's count. Use it to tune
  `EMBEDDING_RATE_SAFETY`.
- **Startup check.** `CHUNK_TOKENS` must fit in one batch (4,750 tokens with the defaults).

What this means in practice: with the defaults, embedding goes at about **one 4,750-token
batch per minute**. A document of 33 chunks (about 12,000 tokens) takes about 2 minutes, one
of 96 chunks (about 48,000 tokens) roughly 10, and a 500-page PDF can take an hour. Progress
is logged at INFO level with timestamps, for example `embedded 18/96 chunks (batch 2/11:
~4500 tokens estimated, 5040 reported by the provider (+12%), waited 60.0s)`. Retries log
the real wait, including time spent waiting for the rate-limit window, and every failed
request logs its HTTP status and `Retry-After` header. Higher plan limits mean faster
runs: raise `EMBEDDING_TPM_LIMIT` and `EMBEDDING_RPM_LIMIT`.

Long runs happen in the worker, not in an HTTP request: the worker refreshes the document's
`heartbeat_at` before every embedding request, so the sweeper can tell a slow job from a
dead one.

**One run per document.** `reprocess` returns `409` while a document is pending or being
processed, and the worker only claims `pending` documents. A run counts as abandoned only if
its heartbeat is older than `STALE_PROCESSING_MINUTES`; the sweeper then resets the document
for a new run. Each claim carries a run id, and every write a run makes needs that id to
still be on the document, so a run that was replaced stops before its next request to the
provider and never writes chunks or marks the document failed.

**Searches never hang.** A search waits at most `EMBEDDING_QUERY_MAX_WAIT_SECONDS` for
embedding capacity. If it would have to wait longer, it answers `503` with a `Retry-After`
header and `Embedding service is busy; retry in about N seconds`.

**Shared across processes.** The limiter's log lives in Redis, and every check-and-record is
one Lua script, which Redis runs as a single indivisible step. So the API (searches) and the
worker (ingestion) can never both take the last slot, and adding processes doesn't multiply
the budget. If Redis is unreachable, searches answer `503` and jobs retry later.

**Changing the model or dimensions** is a migration, not just a setting. The column
holds vectors of exactly one size, and vectors from different models can't be compared.
See the comment in `alembic/versions/0005_chunk_embeddings.py` for the steps.

### Temporary search endpoint

| Method | Path | What it does |
|---|---|---|
| POST | `/collections/{id}/search` | The `limit` chunks in your collection closest in meaning to `query` |

Body: `{"query": "...", "limit": 5}`. The query is 1-2000 characters; `limit` is 1-20
(default 5). Each result has `content`, `distance` (cosine distance: smaller means more
similar), `document_name`, `page_number` and `section_title`. Collections you don't own
return `404`. If the embedding provider is unavailable, you get `503`.

```bash
curl -X POST localhost:8000/collections/$COLLECTION_ID/search -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"query": "how do I run the tests?", "limit": 3}'
```

This endpoint is temporary; Phase 6 replaces it with the real retrieval service. Search
only covers chunks embedded with the current `EMBEDDING_MODEL`. Documents processed before
Phase 4 (no vectors yet), or with a different model, need to be processed again.

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

Or inside the running `api` container, which includes pytest, ruff and the tests (docker
compose builds the Dockerfile's `dev` stage; a plain `docker build .` makes the slimmer
`prod` image without them):

```bash
docker compose exec api pytest -q
docker compose exec api ruff check .
```

If those commands report `executable file not found`, the container is running an older
image: run `docker compose up -d --build` once to rebuild it.

## Lint and format

```bash
uv run ruff check .           # lint (add --fix to auto-fix)
uv run ruff format .          # format
```
