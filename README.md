# DocuMind

RAG knowledge base API. FastAPI (async), PostgreSQL + pgvector, SQLAlchemy 2.0 async, Redis.

**Status:** Phase 8 (users, JWT auth, per-user collections, document upload and storage,
text extraction and chunking, embeddings and vector search, background processing,
question answering with citations, streamed answers and conversations, hybrid search and
retrieval evaluation).

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
- **Gemini** (`app/services/llm/`) writes answers to questions from the retrieved chunks,
  called by the API with its own Redis rate limit (see "Asking questions").
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

**Searches never hang.** A search or question waits at most
`EMBEDDING_QUERY_MAX_WAIT_SECONDS` for embedding capacity. If it would have to wait longer,
it answers `429` with a `Retry-After` header and `Embedding service is busy; retry in about
N seconds`.

**Shared across processes.** The limiter's log lives in Redis, and every check-and-record is
one Lua script, which Redis runs as a single indivisible step. So the API (searches) and the
worker (ingestion) can never both take the last slot, and adding processes doesn't multiply
the budget. If Redis is unreachable, searches answer `503` and jobs retry later.

**Changing the model or dimensions** is a migration, not just a setting. The column
holds vectors of exactly one size, and vectors from different models can't be compared.
See the comment in `alembic/versions/0005_chunk_embeddings.py` for the steps.

### Search (debugging view)

| Method | Path | What it does |
|---|---|---|
| POST | `/collections/{id}/search` | The `limit` chunks in your collection closest in meaning to `query` |

Body: `{"query": "...", "limit": 5}`. The query is 1-2000 characters; `limit` is 1-20
(default 5). Each result has `content`, `distance` (cosine distance: smaller means more
similar), `document_name`, `page_number` and `section_title`. Collections you don't own
return `404`. If the shared embedding quota is used up you get `429` (with `Retry-After`);
if the embedding provider is unavailable, `503`.

```bash
curl -X POST localhost:8000/collections/$COLLECTION_ID/search -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"query": "how do I run the tests?", "limit": 3}'
```

Each result also has `vector_rank`, `keyword_rank` (its position in each list, `null` if
not in it), `keyword_score`, `keyword_all_terms` and `rrf_score`, so you can see *why* it
was found.

It shows what retrieval finds for a query, with distances, without generating an answer:
useful to see why a question was or wasn't answered. `/ask` (below) uses the same
retrieval. Search only covers chunks embedded with the current `EMBEDDING_MODEL`. Documents
processed before Phase 4 (no vectors yet), or with a different model, need to be processed
again.

### Hybrid search

Embeddings capture meaning but blur rare exact terms: an error code like `ERR-4471` or a
part number can sit far from the chunk that contains it. Keyword search is the opposite.
With `RAG_SEARCH_MODE=hybrid` (the default) both run:

- **vector**: the `HYBRID_CANDIDATES` chunks closest in meaning;
- **keyword**: Postgres full-text search (`websearch_to_tsquery`, ranked by
  `ts_rank_cd`) over a generated `tsvector` column with a GIN index (migration 0009),
  so the column is recomputed by Postgres whenever a chunk is written and can't fall out of
  sync on re-processing;
- **Reciprocal Rank Fusion** merges the two lists: each chunk scores Σ 1 / (`RRF_K` +
  rank) over the lists it appears in. Only ranks are used, never the raw scores, which
  are on incomparable scales (a cosine distance and a `ts_rank_cd` value).

Every candidate gets its vector distance, keyword-only hits included, because the
relevance gate needs it.

| Setting | Default | Meaning |
|---|---|---|
| `RAG_SEARCH_MODE` | `hybrid` | `hybrid` or `vector` |
| `HYBRID_CANDIDATES` | 20 | Candidates from each list before fusion |
| `RRF_K` | 60 | The RRF constant; larger flattens the difference between ranks |
| `RAG_KEYWORD_MATCH` | `strict` | `strict`: a keyword hit needs every term; `loose`: any term |
| `TEXT_SEARCH_CONFIG` | `english` | Stemming and stop words. The index uses `english`; other values work, unindexed |
| `RAG_KEYWORD_GATE_SLACK` | 0.10 | Extra distance allowed for a strong keyword match (0 = off) |

The defaults were chosen by measurement: see [Evaluation](#evaluation).

## Asking questions

| Method | Path | What it does |
|---|---|---|
| POST | `/collections/{id}/ask` | Answer a question from your documents, with citations |
| POST | `/collections/{id}/ask/stream` | The same, streamed as server-sent events (see below) |

Both take `{"question": "...", "conversation_id": "..."}`. Without `conversation_id` a new
conversation is started; with it, the question continues that conversation (which must
belong to you and to this collection, otherwise `404`). Every question and answer is saved.

```bash
curl -X POST localhost:8000/collections/$COLLECTION_ID/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"question": "How long do refunds take?"}'
```

```json
{
  "conversation_id": "...",
  "message_id": "...",
  "answer": "Refunds are processed within 30 days [1].",
  "grounded": true,
  "citations": [
    {"number": 1, "document_id": "...", "document_name": "policy.pdf", "page_number": 3,
     "section_title": null, "snippet": "Refunds are processed within ...", "distance": 0.21}
  ],
  "usage": {"prompt_tokens": 1834, "completion_tokens": 41}
}
```

`grounded` is `true` only when the answer cites at least one of the sources it was given.
`citations` lists only those cited sources; `[n]` in the answer matches `number`.

### Streaming answers

`POST /collections/{id}/ask/stream` sends the answer as it is written, as
[server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
(`text/event-stream`). Use `curl -N` so curl prints events as they arrive:

```bash
curl -N -X POST localhost:8000/collections/$COLLECTION_ID/ask/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "How long do refunds take?"}'
```

```
event: meta
data: {"conversation_id":"...","message_id":"..."}

event: token
data: {"text":"Refunds are processed "}

event: token
data: {"text":"within 30 days [1]."}

: heartbeat

event: done
data: {"conversation_id":"...","message_id":"...","answer":"Refunds are processed within 30 days [1].","grounded":true,"citations":[...],"usage":{"prompt_tokens":1834,"completion_tokens":41}}
```

Events, always in this order:

| Event | When | Data |
|---|---|---|
| `meta` | first | `conversation_id`, `message_id` (the answer being written) |
| `token` | while generating | `text`: the next piece of the answer |
| `done` | at the end | the final `answer`, `grounded`, `citations`, `usage`: exactly what was saved |
| `error` | instead of `done` | `message` (short, safe), `status` (429 or 503), `retry_after` (seconds or null) |

Lines starting with `:` are comments: a heartbeat is sent every
`STREAM_HEARTBEAT_SECONDS` while waiting, so proxies don't close an idle-looking connection.

**Use `done`, not the tokens, as the answer.** Tokens are the model's raw output; only
after the last one can the citations be checked. `done.answer` has invalid citation markers
removed, and `done.citations` lists the verified sources. A client should show tokens as
they arrive and then replace them with `done.answer`.

If nothing in your documents is relevant, the stream is `meta`, one `token` with "I could
not find this in your documents.", and `done` with `grounded: false`, without calling the
model. Problems found before streaming starts are ordinary HTTP errors: `422` (invalid
question), `404` (collection or conversation not yours), `409` (an answer is already being
written in this conversation).

If the client disconnects mid-answer, the request to Gemini is closed at once (it stops
generating) and the answer is saved as `cancelled` with the text produced so far. If
generation fails, it is saved as `failed` with the reason. The question is always saved
first, so it is never lost.

### Conversations

```
first question ──► new conversation (title = the question, ~60 chars; no LLM call)
                   question saved, then answer saved (complete / failed / cancelled)
follow-up      ──► 1. one short LLM call rewrites it into a standalone question, using the
                      last few turns ("And for damaged items?" → "What is the refund policy
                      for damaged items?"); on any failure the original question is used
                   2. the REWRITTEN question is embedded and searched (one Voyage request)
                   3. the answer prompt gets your own question, the last RAG_HISTORY_MAX_TURNS
                      turns (within RAG_HISTORY_TOKEN_BUDGET, newest kept first) in their own
                      <history> block, and this question's sources; old sources are never
                      carried over
```

A follow-up therefore costs two Gemini requests (rewrite + answer); keep that in mind on
the free tier. Only one answer can be written at a time per conversation; a second
request gets `409`. (If a server dies mid-answer, the conversation frees itself after 5
minutes.)

Citations are saved as **snapshots** (document name, page, section, snippet, distance)
rather than links to chunks: re-processing a document replaces its chunks, and old answers
must still show what they cited.

| Method | Path | What it does |
|---|---|---|
| GET | `/conversations` | Yours, most recently active first. `?collection_id=&limit=&offset=` |
| GET | `/conversations/{id}` | One conversation with its messages, oldest first. `?limit=&offset=` |
| PATCH | `/conversations/{id}` | Rename: `{"title": "..."}` (1-200 characters) |
| DELETE | `/conversations/{id}` | Delete it and its messages |

Other users' conversations return `404`. Deleting a collection deletes its conversations.

| Setting | Default | Meaning |
|---|---|---|
| `RAG_HISTORY_MAX_TURNS` | 6 | Earlier turns included in the prompt |
| `RAG_HISTORY_TOKEN_BUDGET` | 1000 | ...within this many tokens (newest kept first) |
| `RAG_REWRITE_TIMEOUT_SECONDS` | 10 | After this, a follow-up is searched as asked |
| `STREAM_HEARTBEAT_SECONDS` | 15 | Heartbeat interval while a stream is waiting |

Logs contain ids, counts and token usage only, never question or answer text.

### How a question flows through the system

```
question
  │ 1. validate: empty, or longer than ASK_MAX_QUESTION_CHARS → 422 (nothing embedded)
  │ 2. ownership: not your collection → 404
  ▼
retrieval (app/services/retrieval.py)
  │ 3. embed the question (Voyage, through the shared rate limiter)
  │ 4. search (owner + collection filters inside the SQL). Hybrid (the default):
  │      the HYBRID_CANDIDATES nearest chunks by meaning and the best keyword matches,
  │      merged by Reciprocal Rank Fusion, top RAG_TOP_K. Vector: the RAG_TOP_K nearest.
  │ 5. relevance gate (relevance.py): keep chunks within RAG_MAX_DISTANCE; in hybrid
  │      mode, a strong keyword match within RAG_MAX_DISTANCE + RAG_KEYWORD_GATE_SLACK
  │      nothing left → "I could not find this in your documents", grounded=false,
  │                     and the LLM is NOT called
  ▼
context + prompt (context.py, prompts.py)
  │ 6. number the chunks [1], [2], ... best first, until RAG_CONTEXT_TOKEN_BUDGET
  │ 7. system message = rules only; user message = <sources>...</sources> + <question>
  ▼
LLM (app/services/llm/, Gemini)
  │ 8. generate (Redis rate limit LLM_RPM_LIMIT/min, timeout, retries for transient errors)
  ▼
citations (citations.py)
  │ 9. keep [n] that match a supplied source, remove the others; none valid → grounded=false
  ▼
response {answer, grounded, citations, usage}; one log line with counts and token usage
(never the question or document text)
```

**Why skip the LLM when nothing is relevant?** Asked to answer from nothing, a model
guesses, and a guess presented as an answer is worse than "I don't know". Skipping it also
costs nothing and spends none of the daily quota.

**Prompt injection.** Document text is untrusted: a file can contain "ignore previous
instructions...". So document text only ever appears inside the `<sources>` block, the
rules tell the model that block is data and never instructions, and anything in a document
that looks like our tags is defused so it can't close the block early. This reduces the
risk; it can't remove it, because the model still reads the text and may be persuaded.
Answers are therefore treated as untrusted output too: citations are validated, and
nothing the model says is executed.

**The relevance threshold was measured, not guessed.** `RAG_MAX_DISTANCE=0.70` (with
`RAG_KEYWORD_GATE_SLACK=0.10` in hybrid mode) comes from sweeping thresholds over the
evaluation set ([docs/EVALUATION.md](docs/EVALUATION.md)). Distances depend on the
embedding model and on your documents: after changing either, run `python -m eval.sweep`
again. Use `/search` to see the distances your questions actually get.

Errors: `422` invalid question; `404` not your collection; `429` (with `Retry-After`) when
the embedding or answer quota is used up, including Gemini's daily limit (which resets at
midnight Pacific time); `503` when a provider is down, times out or is misconfigured. Error
bodies are short fixed messages, never provider responses or keys.

### The answer model (Gemini)

Set `LLM_PROVIDER=gemini` and `LLM_API_KEY` (from Google AI Studio). Settings:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, or `fake` for scripted answers without network |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Stable, available on the free tier |
| `LLM_API_KEY` | (empty) | Sent in the `x-goog-api-key` header; never logged |
| `LLM_MAX_OUTPUT_TOKENS` | 600 | Answer length limit, including the model's thinking tokens |
| `LLM_TIMEOUT_SECONDS` | 30 | Per request |
| `LLM_RPM_LIMIT` | 10 | Requests per minute (shared through Redis) |
| `LLM_TEMPERATURE` | 1.0 | Google recommends 1.0 for Gemini 3 models; lower can cause looping |
| `RAG_TOP_K` | 8 | Chunks retrieved per question |
| `RAG_MAX_DISTANCE` | 0.70 | Relevance cut-off, measured (see Evaluation) |
| `RAG_CONTEXT_TOKEN_BUDGET` | 3000 | Source text per question (approximate tokens) |
| `ASK_MAX_QUESTION_CHARS` | 1000 | Longest question accepted |

**Free-tier limits** are no longer published by Google; see your project's page in AI
Studio. An independent measurement found 15 requests per minute and 500 per day for this
model, so `LLM_RPM_LIMIT` defaults to 10. Daily quotas reset at midnight Pacific time.

**Privacy on the free tier:** Google may use what you send on the free tier, which includes
excerpts of your documents, to improve its products. The paid tier doesn't; enabling
billing on the Google Cloud project is enough, no code change.

## Evaluation

`eval/` measures how well retrieval finds the right passage, without ever calling the LLM
(free to run, repeatable, and it isolates the part that decides whether a good answer is
possible). Details, and how to add documents and questions: [eval/README.md](eval/README.md).
Results and the reasoning behind the defaults: [docs/EVALUATION.md](docs/EVALUATION.md).

```bash
uv run python -m eval.dataset      # check questions.yaml
uv run python -m eval.run          # one evaluation with the current settings
uv run python -m eval.run --mode vector --chunk-tokens 300 --max-distance 0.65
uv run python -m eval.sweep        # rejection rates across RAG_MAX_DISTANCE values
uv run python -m eval.experiments  # vector / keyword / hybrid, strict / loose, chunk sizes
```

Each run ingests `eval/corpus/` with the real pipeline into a collection owned by
`eval@documind.local`, asks every question in `eval/questions.yaml` through the real
search code, prints a table and saves a JSON report (with the full configuration) under
`eval/reports/`. Embeddings are cached in Postgres by (model, input type, dimensions,
SHA-256 of the text), so the first run on a fresh database makes a few Voyage requests
and every later one makes none.

Reading the table:

- **hit@k**: share of answerable questions whose right chunk is in the top k.
- **MRR**: average of 1 / rank of the right chunk; rewards rank 1 over rank 3.
- **falseRej**: answerable questions the relevance gate turned away (lower is better).
- **rejected**: unanswerable questions the gate turned away (higher is better).
- **Failed questions**: each with the rank of its right chunk and its top results.

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

**Five tests fail on purpose**: `tests/test_eval_metrics.py` checks `mrr()` in
`eval/metrics.py`, which is left as an exercise. They pass once it is implemented.

If those commands report `executable file not found`, the container is running an older
image: run `docker compose up -d --build` once to rebuild it.

## Lint and format

```bash
uv run ruff check .           # lint (add --fix to auto-fix)
uv run ruff format .          # format
```
