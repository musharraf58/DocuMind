# Security and limits

DocuMind runs on free tiers: Voyage allows 3 requests and 10,000 tokens a minute, and
Gemini's free tier about 500 requests a day. Once the app is public, anyone can sign up
and try to use that up. This section lists every limit, its default and why it exists.
Every default can be changed in `.env` (see `.env.example`).

## Two kinds of rate limit

- **Outbound** (Phase 5, `app/services/embeddings/redis_rate_limit.py`; see
  [Outbound limits](#outbound-limits-the-embedding-provider)): how fast *we*
  call Voyage and Gemini. The provider's limit is fixed and shared by everyone, so a
  request that doesn't fit *waits* for room.
- **Inbound** (Phase 9, `app/core/rate_limit.py`): how fast *a client* may call *us*.
  Here the client is the one who must slow down, so we *refuse* at once with
  `429 Too Many Requests`. Waiting would let one client tie up the server.

## Rate limits per endpoint class

Each endpoint belongs to exactly one class. Limits use a sliding window in Redis, shared
by every API process.

| Class | Endpoints | Counted per | Default | Setting | Why |
| --- | --- | --- | --- | --- | --- |
| login | `POST /auth/login` | client IP | 10/minute | `RATE_LIMIT_LOGIN_PER_IP` | Password guessing from one address |
| login | `POST /auth/login` | email (any IP) | 5/minute | `RATE_LIMIT_LOGIN_PER_EMAIL` | Guessing one account from many addresses |
| register | `POST /auth/register` | client IP | 5/hour | `RATE_LIMIT_REGISTER_PER_IP` | Mass account creation to multiply quotas |
| account | `POST /auth/change-password`, `DELETE /auth/me` | user | 5/minute | `RATE_LIMIT_ACCOUNT` | A stolen token used to guess the password |
| ask | `/ask`, `/ask/stream` | user | 10/minute | `RATE_LIMIT_ASK` | Each question costs 1–2 Gemini calls |
| upload | `POST /collections/{id}/documents` | user | 10/hour | `RATE_LIMIT_UPLOAD` | Each document costs embedding tokens and worker time |
| default | every other authenticated endpoint | user | 120/minute | `RATE_LIMIT_DEFAULT` | Scripts hammering cheap endpoints |
| — | `/health`, `/health/live`, `/health/ready` | — | none | — | Probes must always answer |

Format: `N/second`, `N/minute`, `N/hour` or `N/day`. A typo stops the app at startup.
`RATE_LIMIT_ENABLED=false` switches them all off (refused in production).

Per-user limits count by user id, not IP, so people behind one office address don't share
a limit. A rejection looks like this:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 42
RateLimit-Limit: 10
RateLimit-Remaining: 0
RateLimit-Reset: 42

{"detail": "Too many requests; retry in 42 seconds", "code": "rate_limited",
 "retry_after": 42, "request_id": "..."}
```

Most successful responses carry the `RateLimit-*` headers too, so a client can slow down
before it hits the limit. Streams, downloads and empty `204` responses don't.

## The client's IP address

Per-IP limits need the client's real address. `X-Forwarded-For` is a header that **any
client can set to anything**, so it is only trusted behind proxies we run:

- `TRUSTED_PROXY_COUNT=0` (default): the connection's own address is used and the header
  is ignored.
- `TRUSTED_PROXY_COUNT=N`: each of our N proxies appends the address it received the
  request from, so the client is the **Nth address from the right**. Everything to its
  left was written by the client.

The compose file starts uvicorn with `--no-proxy-headers`, so uvicorn doesn't rewrite the
client address itself and only this setting decides. On Render it is **0**: tested on the
live service, 1 let a client pick its own address with a fake header (see
[docs/HARDENING.md](HARDENING.md), "6. The client address behind Render").

## When Redis is down: fail open

The limits and quotas live in Redis. If Redis can't be reached, two choices exist:

- **Fail closed**: refuse every request, because we can't count it.
- **Fail open**: let requests through without counting them.

**DocuMind fails open**, and logs it loudly (ERROR, at most every 30 s per process:
`RATE LIMITING DEGRADED ...`). Failing closed would turn a Redis problem into a complete
outage, including for logged-in users doing cheap things. The expensive paths stay
protected anyway: Voyage and Gemini calls go through the outbound limiter, which also
needs Redis and refuses without it. So an outage can't be used to drain the free tier.

**Login is the exception**: password guessing is what attackers do while limits are off,
so login falls back to an in-process limiter (same limits, per process, capped at 10,000
remembered keys). The failed-login delay below is skipped during an outage.

## Login protection

- **Progressive delay per account + IP**: the first 3 failures are free (typos). Then
  each failure blocks the next attempt for 2, 4, 8 … seconds, up to 15 minutes
  (`LOGIN_FREE_ATTEMPTS`, `LOGIN_DELAY_BASE_SECONDS`, `LOGIN_DELAY_MAX_SECONDS`). A
  success, or 15 minutes without a failure (`LOGIN_FAILURE_WINDOW_MINUTES`), resets it.
  It's per account *and* IP, so an attacker failing from their address doesn't lock the
  real user out of theirs.
- **The same answer either way**: unknown emails are counted and delayed exactly like
  real ones, and the 401 always says "Incorrect email or password". Neither the message
  nor the delay reveals whether an account exists.
- **The cost of the per-email limit**: someone flooding your email's login can make your
  own logins wait for up to a minute. That's the price of stopping guesses from many
  addresses.

## Registration

`REGISTRATION_MODE`: `open` (anyone), `invite` (default: needs `INVITE_CODE`, sent as
`invite_code` in the register request) or `closed`. In invite mode with no `INVITE_CODE`
set, nobody can register. The code is compared in constant time, so response timing
doesn't reveal how much of a guess was right. It is never logged and never returned.

For local development, set `REGISTRATION_MODE=open` in `.env`.

## Passwords and tokens

- **Password policy** (registration and password changes): 10–128 characters, not one of
  about 150 very common passwords, not your email address. The maximum stops a different
  attack: hashing a huge "password" with argon2 burns server CPU. Existing shorter
  passwords keep working at login.
- **`POST /auth/change-password`** (current and new password) returns a fresh token, and
  every token issued before it stops working, on every device.
- **How**: each user has a `token_version`. It is copied into every token and checked on
  every request. A password change bumps it, so older tokens no longer match. Tokens
  issued before this phase have no version and are refused: log in again once.
- **`DELETE /auth/me`** (with the password) deletes the account and everything stored for
  it: the rows (by cascade), the stored files and the Redis counters. Files go first, so a
  failure midway can be retried; the Redis cleanup is best-effort (counters expire within
  two days anyway).

## Daily quotas

Counted in Redis per "quota day", which starts at `QUOTA_RESET_HOUR_UTC` (default 0:
midnight UTC). Taking from a quota is one atomic step, so parallel requests can't both
take the last unit. Keys expire by themselves after 48 hours.

| Quota | Default | Setting | When it's used up |
| --- | ---: | --- | --- |
| Questions per user | 50 | `DAILY_QUESTIONS_PER_USER` | 429 `daily_questions_quota_reached` |
| Embedding tokens per user | 200,000 | `DAILY_EMBED_TOKENS_PER_USER` | 429 for search, ask and upload; the worker fails a document that doesn't fit, with "reprocess after 00:00 UTC" |
| Uploads per user | 20 | `DAILY_UPLOADS_PER_USER` | 429 `daily_uploads_quota_reached` |
| **LLM requests, all users together** | 350 | `GLOBAL_DAILY_LLM_BUDGET` | 503 `daily_capacity_reached` on `/ask` and `/ask/stream`; `/search` keeps working |

- **Questions** are counted when asked. A question that turns out to be a 404 or 409 is
  given back.
- **Embedding tokens** are counted from what Voyage reports, after each request. Cached
  embeddings cost nothing. 200,000 tokens is about a 300-page book.
- **The global budget** protects the free-tier Gemini quota (about 500 a day when
  measured) from a few heavy users, or one attacker with many accounts. One unit is
  taken before *every* Gemini request, so a follow-up costs 2 (rewrite + answer) and a
  retry counts too. A rewrite never takes the last unit: it is skipped (the original
  question is used) so the answer still gets one.
- **Why 503 and not 429**: 429 means "*you* sent too many requests"; this client did
  nothing wrong. The service as a whole is out of capacity until tomorrow, which is what
  `503 Service Unavailable` with `Retry-After` means.

## Usage tracking

Every successful Voyage or Gemini request adds a row to `usage_events`:

- kind (`embed_document`, `embed_query`, `llm_answer`, `llm_rewrite`), tokens in and
  out, model, time;
- the related document or conversation id;
- **never** any question, answer, prompt or document text.

The worker records its embedding too. `GET /usage` returns your own totals per kind for
today and this month, plus your quota counters and when they reset.

**Redis counters versus Postgres events.** Both count usage, for different jobs:

- **Redis counters** are for enforcement: atomic, fast enough to check on every request,
  and they expire by themselves. If Redis restarts they're lost and everyone gets a
  fresh day (fail open again).
- **`usage_events`** is the durable history, for `/usage`, auditing and cost analysis.
  Counting it on every request would be slow and racy.
- **They can differ**: a question asked counts a question even when nothing relevant is
  found and no LLM call is made, and a request that failed isn't an event.

## Upload caps

| Cap | Default | Setting | Response |
| --- | ---: | --- | --- |
| Total storage per user | 100 MB | `MAX_STORAGE_MB_PER_USER` | 403 `storage_limit_reached` |
| Documents per user | 50 | `MAX_DOCUMENTS_PER_USER` | 403 `document_limit_reached` |
| Pending or processing at once | 5 | `MAX_PENDING_DOCUMENTS_PER_USER` | 429 `pending_limit_reached` + `Retry-After` |

They keep disk use and the single worker's queue bounded.

- **Status codes**: storage and count are 403, because only the user deleting something
  frees them. Pending is 429, because it clears by itself.
- **Race safety**: the check locks the user's row until the upload commits, so two
  parallel uploads can't both take the last slot.
- **Size**: a single file is still limited by `MAX_UPLOAD_MB` (20).

## HTTP hardening

- **Security headers** on every response: `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
  `Cross-Origin-Resource-Policy: same-origin`, a `Content-Security-Policy` that allows
  nothing (except on the docs pages, and a strict one of its own on the
  [demo page](#the-demo-page)), and HSTS in production.
- **`Cache-Control: no-store`** on authenticated responses and on `/auth/*`, so proxies
  and browsers never keep a copy of your data or a token.
- **CORS**: closed unless `CORS_ORIGINS` lists origins (comma-separated). `*` is refused
  at startup. No credentials mode: tokens travel in the `Authorization` header.
- **Request body limit**: `MAX_REQUEST_BODY_KB` (64) for everything but uploads,
  enforced before the app reads the body, including bodies sent without a
  `Content-Length`.
- **API docs**: `DOCS_ENABLED=false` removes `/docs`, `/redoc` and `/openapi.json`.
- **One error shape**: `{"detail": "...", "code": "...", "request_id": "..."}`, with
  extra fields where useful (`errors` for validation, `retry_after`, `limit`).
  Validation errors never echo the submitted values, so a rejected password isn't sent
  back.
- **Unexpected errors** return a generic `500 Internal server error` with only the
  request id. The stack trace goes to the log under the same id.

## Outbound limits: the embedding provider

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

**Searches never hang.** A search or question waits at most
`EMBEDDING_QUERY_MAX_WAIT_SECONDS` for embedding capacity. If it would have to wait longer,
it answers `429` with a `Retry-After` header and `Embedding service is busy; retry in about
N seconds`.

**Shared across processes.** The limiter's log lives in Redis, and every check-and-record is
one Lua script, which Redis runs as a single indivisible step. So the API (searches) and the
worker (ingestion) can never both take the last slot, and adding processes doesn't multiply
the budget. If Redis is unreachable, searches answer `503` and jobs retry later.

## Prompt injection

**Prompt injection.** Document text is untrusted: a file can contain "ignore previous
instructions...". So document text only ever appears inside the `<sources>` block, the
rules tell the model that block is data and never instructions, and anything in a document
that looks like our tags is defused so it can't close the block early. This reduces the
risk; it can't remove it, because the model still reads the text and may be persuaded.
Answers are therefore treated as untrusted output too: citations are validated, and
nothing the model says is executed.

## Privacy: the Gemini free tier

**Privacy on the free tier:** Google may use what you send on the free tier, which includes
excerpts of your documents, to improve its products. The paid tier doesn't; enabling
billing on the Google Cloud project is enough, no code change.

## The demo account

The shared demo account (`DEMO_EMAIL`, created by `scripts/seed_demo.py`) is
read-only: every endpoint that changes something (uploading, deleting,
reprocessing, creating or renaming collections and conversations, changing
the password, deleting the account) answers `403` with code
`demo_read_only` (`app/api/demo.py`). It may log in, list, read, search and
ask, within the same per-user limits and daily quotas as anyone. Everyone
who uses it shares its data and its quotas, so nothing private belongs in it.

## The demo page

The page at `/demo` ([OPERATIONS.md](OPERATIONS.md#the-demo-page)) shows
text that nobody controls: model output, document snippets, file names,
conversation titles. It is built so that none of it can ever run as code:

- **Text goes in as text.** Every dynamic value is inserted with
  `textContent` or as a text node, never parsed as HTML. A test
  (`tests/test_demo_ui.py`) fails if the page's JavaScript contains
  `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `eval` or
  `document.write`.
- **A strict Content-Security-Policy** on every `/demo` response:
  `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self';
  connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none';
  form-action 'self'; frame-ancestors 'none'`. No inline script or style
  and no other origin: even if markup got in, the browser would refuse to
  run it or send data anywhere else. The API keeps its own
  `default-src 'none'`.
- **The token stays in memory**, never in `localStorage`, `sessionStorage`
  or a cookie (a test checks this too), so there is nothing stored for a
  script to steal, and no cookie for another site to ride on.
- **No credential in the source.** The login hint comes from the
  `DEMO_UI_HINT` setting, set in the hosting dashboard.
- **A visible warning** that the demo account is shared and public, and
  that Gemini's free tier may use inputs.
