# Hardening: database, dependencies, Redis budget, memory, client address

Sections 1–3 were measured 2026-10-22 for Phase 9, sections 4–5 for Phase
10a, and section 6 on the live Render service for Phase 10b. The limits,
quotas and HTTP hardening are described in the README ("Security and
Limits"); this file records the measurements behind them.

## 1. Indexes behind the hot queries

### How it was measured

`EXPLAIN (ANALYZE, BUFFERS)` on a **scratch database** (`documind_explain`,
never the dev database), built with `alembic upgrade head` and seeded with
synthetic data. Measuring on an empty database proves nothing: with a few
rows the planner scans whole tables because that is cheapest.

| Table | Rows | Shape |
| --- | ---: | --- |
| users | 1,000 | |
| collections | 5,000 | 5 per user |
| documents | 50,000 | 10 per collection; 1% pending, 1% processing, 3% failed |
| chunks | 60,000 | 1 per document, plus one "heavy" collection with 10,010 chunks (1024-dim vectors) |
| conversations | 20,000 | 20 per user |
| messages | 400,000 | 20 per conversation |
| usage_events | 300,000 | 300 per user over 60 days |

Each query below is the SQL the app sends (same filters, order and limit),
for a typical user, and for the heavy collection where it matters.

### Results

| # | Query (endpoint) | Plan | Time |
| --- | --- | --- | ---: |
| 1 | Documents of a collection, newest first (`GET /collections/{id}/documents`) | BitmapAnd of `uq_documents_collection_hash` (collection_id first) and `ix_documents_owner_id`, then sort 10 rows | 0.36 ms |
| 2 | Same with `?status=failed` | Same, status as a filter | 0.13 ms |
| 3 | Collections of a user (`GET /collections`) | `ix_collections_owner_id`, sort 5 rows | 0.08 ms |
| 4 | Conversations, latest first (`GET /conversations`) | `ix_conversations_owner_updated`, sort 20 rows | 0.36 ms |
| 5 | Messages of a conversation (`GET /conversations/{id}`) | `ix_messages_conversation_created` (+ owner index), sort 20 rows | 0.59 ms |
| 6 | Recent history for a follow-up (newest 16) | Index Scan **Backward** on `ix_messages_conversation_created`: no sort at all | 0.07 ms |
| 7 | Vector search, heavy collection (10,010 chunks) | Exact: the user's documents, their chunks via `uq_chunks_document_index`, distance for every chunk, top-20 heap sort | 90–100 ms warm (200 ms cold) |
| 8 | Vector search, typical collection | Bitmap on `ix_chunks_collection_id` + `ix_chunks_owner_id`, 10 distances | 1.1 ms |
| 9 | Keyword search, heavy collection | Same document loop, `@@` as a filter | 50 ms (see note) |
| 10 | Usage totals since a time (`GET /usage`) | `ix_usage_events_user_id_created_at` (new in 0012), both columns in the index condition | 0.66 ms |
| 11 | Upload caps: count, bytes, in progress | `ix_documents_owner_id`, aggregate 50 rows | 0.25 ms |
| 12 | Sweeper: pending documents queued long ago | `ix_documents_status` | 1.0 ms |

### Findings

- **No missing index.** Every per-user list (1–6, 10, 11) is under 1 ms and
  uses an index. The sorts left are over 5–20 rows, which costs microseconds.
  An index such as `(collection_id, created_at DESC)` would remove the sort in
  query 1, but there is nothing measurable to gain, and every extra index
  slows every write. **No migration was added.**
- **The upload caps bound these queries.** A user can have at most 50 documents
  (`MAX_DOCUMENTS_PER_USER`), so "this user's documents" can never become a big
  scan. Raising the caps a lot is the point at which to measure again.
- **Vector search is the slowest query and the one to watch.** For the heavy
  collection the planner skips the HNSW index and computes the exact distance
  to all 10,010 chunks (about 0.1 s warm). This is a sound choice:
  - recall is 100%;
  - forcing the HNSW index (with `hnsw.iterative_scan`, which the app already
    sets) returned the same top 20 in 100–150 ms, no faster.

  The data is synthetic (random vectors reused from a pool), and real
  embeddings cluster, which helps HNSW. So re-measure with real documents
  before changing anything. Cost grows linearly with collection size; at
  `MAX_STORAGE_MB_PER_USER=100` a collection stays in the tens of thousands of
  chunks, so a fraction of a second, below the embedding call that comes
  before it.
- **Keyword search (9)** looks slow only because every synthetic chunk in the
  heavy collection contains both search words, so there is nothing for the
  GIN index to narrow down. Real queries match a few chunks.
- **Migrations 0011 and 0012** were checked on the scratch database: upgrade,
  downgrade to 0010, upgrade again, all clean.

Reproduce: create the scratch database, `DATABASE_URL=... alembic upgrade
head`, seed with `generate_series` inserts like the table above, then run the
queries with `EXPLAIN (ANALYZE, BUFFERS)`. Drop the scratch database afterwards.

## 2. Connection pool and statement timeout

`app/core/database.py`, settings in `.env.example`:

| Setting | Default | Why |
| --- | ---: | --- |
| `DB_POOL_SIZE` | 5 | Connections the API keeps open |
| `DB_MAX_OVERFLOW` | 10 | Extra connections opened under load, closed afterwards |
| `DB_POOL_TIMEOUT_SECONDS` | 10 | How long a request waits for a free connection before failing with an error, instead of piling up |
| `DB_STATEMENT_TIMEOUT_MS` | 15,000 | Postgres cancels any single API statement running longer. The slowest measured query takes ~0.1 s, so 15 s only ever stops a runaway query |
| `DB_WORKER_STATEMENT_TIMEOUT_MS` | 300,000 | The worker stores a large document's chunks and vectors in one write |

Connections are also recycled after 30 minutes (`pool_recycle`), before a
proxy or firewall drops them silently, and checked before use
(`pool_pre_ping`).

**Budget:** (pool_size + max_overflow) × API processes + worker must stay below
Postgres's `max_connections` (100 by default). One API process and the worker
use at most 15 + 1 connections. Managed free tiers often allow fewer, so
check the host's limit in Phase 10.

The app never holds a transaction while waiting on the network (embedding,
LLM, streaming), so a pool of 5 serves far more concurrent requests than 5.
The exception is an upload: it holds a lock on the user's row until it
commits (the upload caps), which only blocks that user's other uploads.

## 3. Dependency audit

`pip-audit` 2.10.1, run once on the locked dependencies (200 packages,
including dev tools):

```bash
uv export --format requirements-txt --no-hashes --no-emit-project --all-groups > /tmp/req.txt
uvx pip-audit -r /tmp/req.txt --disable-pip --no-deps
```

| Package | Was | Advisory | Affects DocuMind? | Action |
| --- | --- | --- | --- | --- |
| certifi | 2026.7.22 | PYSEC-2026-4320 / GHSA-6q48-6c7w-9gw2: trusted a root CA whose intermediates' keys were compromised | **Yes**: certifi supplies the trusted CAs for our HTTPS calls to Voyage and Gemini, so an on-path attacker with a compromised key could read the API keys in transit | Upgraded to 2026.10.19 |
| pypdf | 6.20.1 | CVE-2026-113786 / GHSA-5hx5-vpf9-5wcv: a crafted PDF can make pypdf call arbitrary image parsers (`/DCTDecode`) | Not today: it needs Pillow, which isn't installed. But we parse PDFs from strangers, and Pillow could arrive as a dependency later | Upgraded to 6.21.2; minimum raised to `pypdf>=6.21.1` |

After the upgrades: **no known vulnerabilities found.** Rebuild the Docker
image (`docker compose build`) to pick up the new versions.

pip-audit sends package names and versions (nothing else) to the PyPI
advisory database. CI now runs it on every pull request (docs/CI.md).

## 4. Redis command budget

Upstash's free Redis allows **500,000 commands a month** (checked on
[upstash.com/pricing/redis](https://upstash.com/pricing/redis), October 2026), about
**16,400 a day**. Unlike a server you run yourself, idle chatter costs the
same as real work, so every command per request was counted.

### How it was measured

`tests/test_redis_budget.py` sends each kind of request through the app
with the **real** Voyage and Gemini clients (their real Redis rate limiters
and the real global LLM budget) against simulated HTTP servers, and a fake
Redis that records every command. The test asserts the exact list, so a
change that adds Redis traffic to a request fails CI until this table is
updated. Counts are steady state: the first use of a Lua script on a fresh
Redis also sends a refused `EVALSHA` and a `SCRIPT LOAD`, once per server.

### Commands per request

| Request | Before | Now | Commands (now) |
| --- | ---: | ---: | --- |
| `GET /health/live` (Render's health check) | 0 | **0** | none |
| `GET /health/ready` | 2 | 2 (Celery) / **1** (inline) | `PING`; `EXISTS` on the worker heartbeat in Celery mode only |
| `POST /auth/register` | 1 | 1 | rate limit |
| `POST /auth/login` | 4 | **3** | rate limit per IP, per email; failed-login record (`HMGET`). The `DEL` that cleared it now runs only when there was a record |
| any other authenticated request | 1 | 1 | rate limit |
| `GET /usage` | 4 | **2** | rate limit; the three daily counters in one `MGET` |
| `POST .../documents` (upload) | 3 | 3 | rate limit; embedding tokens left (`GET`); daily upload quota |
| document job | 1 + 3 per batch | 1 + 2–3 per batch | `GET` quota; per embedding batch: rate-limit window, token correction (only if the provider counted more than we estimated), daily token counter |
| `POST .../search` | 5 | 5 | rate limit; tokens left; embedding rate limit + correction; daily token counter |
| `POST .../ask` and `.../ask/stream` | 9 | **7** | rate limit; **admission** (global answer capacity + tokens left + take a question: one script, was three commands); embedding rate limit + correction + token counter; global LLM budget; LLM rate limit. A follow-up question adds 2 (the rewrite's LLM budget and rate limit) |
| idle app, **inline mode** | – | **0** | the worker waits on an in-memory queue; maintenance uses Postgres only (`tests/test_inline_mode.py` asserts zero) |
| idle app, **Celery mode** | 131 per minute | (unchanged) | measured with `INFO commandstats`: the worker's `BRPOP` once a second, `PUBLISH`, `PING`, beat, the heartbeat |
| `DELETE /auth/me`, retention | `SCAN` per pattern | **1** | one `DEL` of the user's known key names (no `SCAN`, whose cost grows with the whole keyspace) |

Fixes made: the question admission script, the `MGET`, skipping a token
correction that can't change anything (the limiter never lowers a count),
skipping the login `DEL` when there is nothing to clear, and deleting a
user's keys by name.

### What the free tier carries

At 16,400 commands a day:

- **Questions:** 7 each, so about 2,300 a day, but the global LLM budget
  (`GLOBAL_DAILY_LLM_BUDGET=350`) stops answers long before: 350 questions
  use about 2,450 commands, 15% of the day.
- **A typical visit** (log in, ~20 page requests, 5 questions, 2 searches):
  3 + 20 + 35 + 10 = **68 commands**, so roughly **240 visits a day**.
- **Ingestion:** a 300-page book (about 200,000 tokens, ~45 embedding
  batches) costs about 135 commands.
- **Idle:** 0 in inline mode. A Celery worker on Upstash would spend about
  188,000 commands a day doing nothing, the whole month's allowance in under
  3 days. This alone rules out Celery-on-Upstash for the free tier.

**If Upstash bills each command inside a Lua script** (its docs don't say;
`EVALSHA` might count as 1 or as every `redis.call` it makes), the worst
case is about 5x: a question would cost about 40, and the free tier would
carry about 400 questions a day, still above the 350 the LLM budget allows.
Watch the Upstash dashboard during the first week to see which it is.

### REDIS_COMMAND_BUDGET (a note, not a setting)

There is deliberately no runtime counter: counting commands would itself
cost commands. Keep the budget in mind instead:

- Point uptime monitors at `/health/live` (0 commands), never at
  `/health/ready` (2 commands: once a minute is 2,880 a day, 18% of the
  budget).
- Don't lower the rate limits' windows or add Redis calls to hot paths
  without updating `tests/test_redis_budget.py` and this table.
- If the budget runs out, Upstash refuses commands: rate limits and quotas
  then **fail open** (logged loudly), and the LLM rate limiter refuses, so
  answers stop with 503 until the month resets. Nothing is lost.


## 5. Memory in 512 MB (Render free)

Render's free web service has 512 MB, and in lite mode the API process also
runs the document jobs. Measured on 2026-10-27 with the production image
under the same limit: `docker run --memory=512m --memory-swap=512m`, lite
mode (`INGESTION_MODE=inline`, `STORAGE_BACKEND=s3` against a local S3
server, Postgres, Redis), fake embedding and LLM providers (the providers'
own memory use is tiny; what matters is our process). A sampler read the
container's cgroup memory and the app process's RSS every 0.1 s.

Test files (generated for the measurement, not committed): a real-world
14-page academic paper with embedded fonts (1.0 MB), a 222-page
LibreOffice-made book (0.4 MB), a 40-page PDF with one incompressible image
per page close to the upload limit (19.3 MB), and a 480-page dense text PDF
at the page limit (2.1 MB, 1.74 million characters, about 447,000 tokens,
fully embedded).

| Phase | Peak RSS | Peak RSS with `MALLOC_ARENA_MAX=2` |
| --- | ---: | ---: |
| idle after startup | 124 MB | 124 MB |
| upload + process the 14-page paper | 177 MB | 173 MB |
| upload + process the 222-page book | 183 MB | 180 MB |
| upload + process the 19.3 MB image PDF | 234 MB | 229 MB |
| 5 concurrent ask / ask-stream, 3 rounds | 198 MB | 193 MB |
| 5 concurrent downloads of the 19.3 MB file | 225 MB | 204 MB |
| upload + process the 480-page document | 287 MB | **253 MB** |
| steady state after all jobs | 282 MB | 250 MB |
| **all at once**: the 480-page job + 5 questions + 5 downloads, 4 rounds | – | **262 MB** |

No run was OOM-killed. The worst case uses about half of the 512 MB.

Why it fits without further changes:

- Jobs run **one at a time** (inline mode), so two large documents never
  sit in memory together.
- Uploads and downloads **stream** in 1 MB chunks (S3: 8 MB multipart
  parts): five concurrent downloads of the 19.3 MB file peaked lower than
  processing that same file once.
- The limits already bound a single document: `MAX_UPLOAD_MB=20`,
  `MAX_EXTRACT_PAGES=500`, `MAX_EXTRACTED_CHARS=2,000,000`, and the daily
  embedding quota (200,000 tokens) refuses the largest documents before
  their vectors are built.

The one change made: `MALLOC_ARENA_MAX=2` in the production image (-34 MB
at the peak, -32 MB in steady state). Python keeps freed memory for reuse
rather than returning it to the system, which is why steady state stays near
the peak; it doesn't grow with more documents, since the same memory is
reused.

If a future change pushes memory up, the cheap levers, in order: lower
`MAX_EXTRACT_PAGES` / `MAX_EXTRACTED_CHARS`; spool the stored file to a
temporary file instead of reading it whole (the job holds the upload, at
most 20 MB, while extracting); lower `EMBEDDING_BATCH_SIZE`.

## 6. The client address behind Render

Per-IP rate limits need the client's address. Behind a proxy, the app finds
it in `X-Forwarded-For`, a header that every proxy appends to and that the
client can also fill with anything it likes. `TRUSTED_PROXY_COUNT=N` tells
the app to believe only the N rightmost entries, the ones our proxies wrote
(`app/core/client_ip.py`). Only two limits key on the address: login
(`RATE_LIMIT_LOGIN_PER_IP`, 10 a minute) and registration
(`RATE_LIMIT_REGISTER_PER_IP`, 5 an hour). Everything else is per user.

### What was tested on the live service

The blueprint first set `TRUSTED_PROXY_COUNT=1`, on the assumption that
Render's proxy appends one entry. Render doesn't document its proxy chain,
so it was tested on the deployed app:

- **With 1:** a client sending made-up `X-Forwarded-For` headers was
  **never rate-limited**. The entry the app trusted was one the client
  controlled, so every attempt looked like a new address. 1 was too high
  for Render's chain: anyone could choose their own address and walk past
  the per-IP limits.
- **With 0:** the header is ignored and the address of the connection is
  used. The per-IP login limit works: about 10 attempts answer 401, then
  429.

So `render.yaml` and `.env.lite.example` now set 0, and a test
(`tests/test_lite_config.py`) keeps them there.

### The trade-off

With 0, the connection's address is Render's proxy, not the visitor, so
**all visitors share the per-IP limits**: 10 login attempts a minute and 5
registrations an hour, for everyone together. One person hammering the
login form can make other people's logins wait up to a minute.

What still applies to each person individually:

- the per-email login limit (`RATE_LIMIT_LOGIN_PER_EMAIL`, 5 a minute) and
  the progressive delay after failed logins (`app/services/login_guard.py`;
  keyed by account and address, so with a shared address it acts per
  account);
- every per-user limit (asking, uploading, account changes, everything
  else);
- the daily quotas (questions, uploads, embedding and LLM tokens).

And the important part: **nobody can bypass a limit by faking a header.**
A count that is too low only makes limits coarser; a count that is too high
makes them optional. For a small invite-only demo, a shared login limit is
an acceptable price.

### Re-checking after a platform change

From one machine, send 11 login attempts within a minute, each with a
different made-up email (so the per-email limit stays out of it) and each
with a different made-up `X-Forwarded-For` header. The 11th must answer
429. If it doesn't, the count is too high. Only consider raising it if
Render documents a fixed proxy chain, and repeat this test after any change.
