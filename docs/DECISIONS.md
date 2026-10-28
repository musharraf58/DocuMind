# Decisions

One short entry per significant decision: what the situation was, what the
alternatives were, what evidence there is in this repository, what was
decided and what it costs. The last line of each entry is for the author's
own reasoning.

1. [pgvector inside Postgres](#1-pgvector-inside-postgres)
2. [Voyage AI for embeddings](#2-voyage-ai-for-embeddings)
3. [Hybrid search with RRF and a measured threshold](#3-hybrid-search-with-rrf-and-a-measured-threshold)
4. [Celery in full mode, inline jobs in lite mode](#4-celery-in-full-mode-inline-jobs-in-lite-mode)
5. [A storage abstraction with an S3 backend](#5-a-storage-abstraction-with-an-s3-backend)
6. [Fail-open rate limiting](#6-fail-open-rate-limiting)
7. [token_version to revoke tokens](#7-token_version-to-revoke-tokens)
8. [One shared, read-only demo account](#8-one-shared-read-only-demo-account)
9. [Trusting no proxy on Render](#9-trusting-no-proxy-on-render)
10. [Gemini on the free tier](#10-gemini-on-the-free-tier)

## 1. pgvector inside Postgres

**Context.** Chunks need a vector for semantic search, next to their text,
page, section, owner and collection. Every search must be filtered by owner
and collection.

**Options considered.** A dedicated vector database (Pinecone, Qdrant,
Weaviate) next to Postgres; pgvector in the same Postgres.

**Evidence.**
- Re-processing replaces a document's chunks and vectors in one
  transaction (`replace_chunks` in `app/services/processing.py`); with two
  stores, a crash between them would leave text and vectors out of step.
- The owner and collection filters are ordinary `WHERE` clauses in the same
  SQL as the distance (`app/services/retrieval.py`).
- Hybrid search needs full-text search too, which Postgres has (a
  generated `tsvector` column with a GIN index, migration 0009).
- Measured with 10,010 chunks of 1024 dimensions in one collection: 90–100
  ms warm, with 100% recall; a typical collection: 1.1 ms
  ([HARDENING.md](HARDENING.md), section 1).
- Neon's free tier includes pgvector, so lite mode needs no extra service.

**Decision.** pgvector in the application's Postgres, with an HNSW index
(`vector_cosine_ops`).

**Consequences.** One database to back up, migrate and secure; vector and
data changes are atomic. Search cost grows linearly in the worst case (the
planner chose an exact scan for the large collection), which the upload
caps bound. Changing the embedding model or dimensions needs a migration.

My reasoning: TODO(me)

## 2. Voyage AI for embeddings

**Context.** Every chunk and every question must be embedded, on a free
tier.

**Options considered.** Voyage AI; OpenAI embeddings (no free tier); Gemini
embeddings; a local model (sentence-transformers) inside the container.

**Evidence.**
- Voyage's free tier: 3 requests and 10,000 tokens a minute
  (`EMBEDDING_RPM_LIMIT`, `EMBEDDING_TPM_LIMIT` in `app/core/config.py`).
- On the evaluation set, `voyage-4-lite` (1024 dimensions) put the right
  chunk first for 76% of answerable questions and in the top 5 for 100%
  (MRR 0.858; [EVALUATION.md](EVALUATION.md)).
- The whole evaluation matrix cost 10 Voyage requests (about 8,000
  tokens), thanks to the embedding cache ([EVALUATION.md](EVALUATION.md)).
- A local model would have to fit next to the API in 512 MB; the API alone
  peaks at 262 MB ([HARDENING.md](HARDENING.md), section 5).

**Decision.** Voyage `voyage-4-lite`, behind a provider interface
(`app/services/embeddings/base.py`) with a fake provider for development
and tests.

**Consequences.** Ingestion is slow on the free tier (about one batch of
4,750 tokens a minute; a 500-page PDF can take an hour), so a shared,
Redis-backed rate limiter keeps ingestion and searches within the limits
together. Vectors from different models can't be mixed: each chunk records
its `embedding_model`.

My reasoning: TODO(me)

## 3. Hybrid search with RRF and a measured threshold

**Context.** Embeddings capture meaning but blur rare exact terms (error
codes, part numbers). And the app must decide when nothing retrieved is
relevant enough to answer from.

**Options considered.** Vector search only; keyword search only; hybrid
with score blending; hybrid with Reciprocal Rank Fusion. For the threshold:
a guessed value, or one chosen by sweeping the evaluation set.

**Evidence** ([EVALUATION.md](EVALUATION.md)):
- Keyword-only (strict) found the right chunk for just 24% of questions;
  loose keyword matching dropped hybrid's hit@5 from 100% to 88%.
- "What is part ZX-9?" has its right chunk at rank 1 in both lists, but at
  vector distance 0.769, so a vector-only gate rejects it at any threshold
  that also rejects something unanswerable.
- The sweep: rejecting every unanswerable question needs a threshold of
  0.54 or less, which turns away 41% of answerable ones. At 0.70 with a
  0.10 slack for strong keyword matches, hybrid has 0% false rejections and
  rejects 25% of unanswerable questions (vector alone: 6% false
  rejections).
- 0.70 rather than 0.68: 0.68 rejects "no-revenue" by a margin of 0.0002,
  which is luck.

**Decision.** `RAG_SEARCH_MODE=hybrid`: pgvector and Postgres full-text
search, fused by RRF (ranks only, `RRF_K=60`), `RAG_KEYWORD_MATCH=strict`,
`RAG_MAX_DISTANCE=0.70`, `RAG_KEYWORD_GATE_SLACK=0.10`.

**Consequences.** The gate prefers false acceptances to false rejections:
3 of 4 unanswerable test questions still reach the LLM, where the
grounding rules and citation validation are the next defence. The threshold
is specific to the model and corpus; re-run `python -m eval.sweep` after
changing either. The evaluation set is small (21 questions).

My reasoning: TODO(me)

## 4. Celery in full mode, inline jobs in lite mode

**Context.** Processing a document takes minutes, so it can't happen inside
the upload request. Full mode has room for a worker; Render's free tier
has no background worker and 512 MB.

**Options considered.** Celery with Redis everywhere; a paid worker on
Render; jobs inside the API process; FastAPI's `BackgroundTasks`.

**Evidence.**
- An idle Celery worker sent Redis 131 commands a minute, about 188,000 a
  day; Upstash's free tier allows 500,000 a month, so it would be gone in
  under three days. Idle inline mode sends 0 ([HARDENING.md](HARDENING.md),
  section 4).
- Both modes call the same `jobs.process_document`
  (`app/services/jobs.py`); `tests/test_inline_mode.py` runs the inline
  path end to end.
- With one inline job at a time, the worst measured case (a 480-page job,
  5 questions and 5 downloads at once) peaked at 262 MB of 512
  ([HARDENING.md](HARDENING.md), section 5).

**Decision.** `INGESTION_MODE=celery` for full mode, `inline` for lite
mode: a bounded, de-duplicated in-memory queue and one worker task in the
API process. Postgres stays the source of truth; the sweeper recovers
whatever the queue loses.

**Consequences.** Lite mode runs on one instance only, and its queue is
lost on restart (recovered by the sweeper at startup). A long job can be
interrupted when the free instance sleeps; it restarts from the beginning.
Jobs and requests share one process's CPU and memory.

My reasoning: TODO(me)

## 5. A storage abstraction with an S3 backend

**Context.** Uploaded files must outlive restarts. Render's free disk is
wiped on every deploy, restart and sleep.

**Options considered.** Local disk only; files in Postgres (`bytea`); an
S3-compatible bucket behind an interface.

**Evidence.**
- `StorageBackend` (`app/services/storage.py`) has a local and an S3
  implementation (`app/services/s3_storage.py`); uploads and downloads
  stream in parts instead of holding whole files in memory.
- Neon's free tier is 1 GB, which the vectors need; R2 and B2 give 10 GB
  free (the lite-mode table in [OPERATIONS.md](OPERATIONS.md#lite-mode-free-tier)).
- `tests/test_s3_storage.py` covers multipart upload, streamed reads and a
  failure midway leaving no object; `scripts/smoke_lite.sh` checks a
  byte-identical download through a local S3 server.
- Five concurrent downloads of a 19.3 MB file peaked at 204 MB
  ([HARDENING.md](HARDENING.md), section 5).

**Decision.** `STORAGE_BACKEND=local` for full mode, `s3` (R2 or B2) for
lite mode, chosen by a setting; storage keys are generated
(`<owner id>/<document id>`), never taken from the filename.

**Consequences.** The database and the bucket can disagree (a row whose
file is missing answers 404 on download). Production refuses to start with
`s3` and missing bucket settings. Storage errors map to 503
`storage_unavailable`.

My reasoning: TODO(me)

## 6. Fail-open rate limiting

**Context.** Inbound rate limits and quotas live in Redis. Redis can be
unreachable (an outage, the free tier's command limit).

**Options considered.** Fail closed (refuse every request that can't be
counted); fail open (let it through uncounted); fail open with exceptions.

**Evidence.**
- `tests/test_rate_limit_inbound.py::test_redis_down_fails_open_with_a_loud_log`
  and `tests/test_rate_limit_api.py::test_redis_down_general_endpoints_fail_open`.
- Login keeps a limit without Redis: an in-process fallback with the same
  limits, capped at 10,000 keys
  (`test_redis_down_login_still_limited_by_fallback`).
- The provider calls stay protected: the outbound limiter needs Redis and
  refuses without it, so an outage can't drain the free tiers
  ([SECURITY.md](SECURITY.md#when-redis-is-down-fail-open)).

**Decision.** Fail open, logged at ERROR at most every 30 seconds per
process, except login, which falls back to a per-process limiter.

**Consequences.** During a Redis outage, cheap endpoints are unlimited and
the daily counters are lost if Redis restarts (everyone gets a fresh day).
Expensive actions still fail, because embedding and answering need Redis.

My reasoning: TODO(me)

## 7. token_version to revoke tokens

**Context.** Access tokens are stateless JWTs. After a password change (a
suspected leak), tokens issued before it must stop working, on every
device.

**Options considered.** Short-lived tokens only; a denylist of revoked
token ids in Redis; a per-user version number copied into each token.

**Evidence.**
- `users.token_version`, copied into every token and checked on every
  request; a password change increments it
  ([SECURITY.md](SECURITY.md#passwords-and-tokens)).
- `tests/test_token_version.py`; the demo seed also uses it to end old
  sessions (`tests/test_demo.py::test_a_new_password_replaces_the_old_and_ends_old_sessions`).

**Decision.** A `token_version` column, checked on every authenticated
request (the user row is loaded anyway).

**Consequences.** No Redis dependency for revocation, and nothing to
expire. Revocation is all-or-nothing per user: one device can't be logged
out alone. Tokens issued before the column existed are refused once.

My reasoning: TODO(me)

## 8. One shared, read-only demo account

**Context.** Visitors should be able to try the app without an invite,
without being able to change or delete anything, on free-tier quotas.

**Options considered.** Open registration; a fresh temporary account per
visitor; one shared account with every write refused.

**Evidence.**
- The `not_demo` dependency (`app/api/demo.py`) refuses every change with
  `403 demo_read_only`; `tests/test_demo.py::test_every_change_is_refused`
  covers each endpoint, and `test_reading_searching_and_asking_work` the
  allowed ones.
- The demo's quotas still apply
  (`test_the_demo_accounts_quotas_still_apply`).
- `scripts/seed_demo.py` builds it from `demo/corpus/` (self-written
  texts), idempotently and through the embedding cache.

**Decision.** `users.is_demo`, one seeded account and collection, read-only
everywhere; asking and searching allowed.

**Consequences.** Visitors share the account's daily question quota and
see each other's conversations, so the demo page warns not to type anything
private. No registration abuse, no per-visitor data to clean up.

My reasoning: TODO(me)

## 9. Trusting no proxy on Render

**Context.** Per-IP limits (login, registration) need the client's address.
Behind a proxy it comes from `X-Forwarded-For`, which clients can also
write.

**Options considered.** `TRUSTED_PROXY_COUNT=1`, assuming Render appends
one entry; `0`, ignoring the header.

**Evidence.** Tested on the live service ([HARDENING.md](HARDENING.md),
section 6): with 1, a client sending made-up `X-Forwarded-For` headers was
never rate-limited; with 0, the per-IP login limit worked (401 for about 10
attempts, then 429). `tests/test_lite_config.py` keeps the blueprint at 0.

**Decision.** `TRUSTED_PROXY_COUNT=0` in `render.yaml` and
`.env.lite.example`.

**Consequences.** All visitors share the per-IP limits (10 logins a minute,
5 registrations an hour); per-email and per-user limits and the daily
quotas still apply individually; nobody can bypass a limit by faking a
header.

My reasoning: TODO(me)

## 10. Gemini on the free tier

**Context.** Answers need an LLM, at no cost.

**Options considered.** Gemini's free tier; a paid API (OpenAI, Anthropic);
a local model.

**Evidence.**
- Google no longer publishes the free-tier limits; an independent
  measurement found 15 requests a minute and 500 a day for
  `gemini-3.5-flash-lite`, so `LLM_RPM_LIMIT` is 10 and
  `GLOBAL_DAILY_LLM_BUDGET` 350 ([ARCHITECTURE.md](ARCHITECTURE.md#the-answer-model-gemini),
  [SECURITY.md](SECURITY.md#daily-quotas)).
- The free tier may use inputs to improve Google's products
  ([SECURITY.md](SECURITY.md#privacy-the-gemini-free-tier)).
- The LLM sits behind an interface (`app/services/llm/`) with a fake
  provider; the relevance gate skips it entirely when nothing is relevant.

**Decision.** `LLM_PROVIDER=gemini`, `gemini-3.5-flash-lite`, with a global
daily budget shared by all users.

**Consequences.** At most about 350 answers a day for everyone together;
after that, `503 daily_capacity_reached` until the next quota day. Users
are warned not to enter anything private. Enabling billing removes the
data-use caveat without a code change.

My reasoning: TODO(me)
