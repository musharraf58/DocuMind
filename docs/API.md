# API

Every endpoint, with examples. Base URL locally: `http://localhost:8000`. In
development the interactive docs are at `/docs` (off in production).
Authenticated endpoints need `Authorization: Bearer <token>` from
`POST /auth/login`. Every error has one shape, `{"detail": "...", "code":
"...", "request_id": "..."}` ([SECURITY.md](SECURITY.md#http-hardening)); the
rate limit and quota on each endpoint are in [SECURITY.md](SECURITY.md).

## Endpoint index

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health`, `/health/live`, `/health/ready` | Health checks ([OPERATIONS.md](OPERATIONS.md#observability-and-health)) |
| POST | `/auth/register` | Create an account (invite code in invite mode) |
| POST | `/auth/login` | Get a token (form fields `username`, `password`) |
| GET | `/auth/me` | Your account, including `is_demo` |
| POST | `/auth/change-password` | New password; older tokens stop working |
| DELETE | `/auth/me` | Delete your account and everything it owns |
| POST, GET | `/collections` | Create, list your collections |
| GET, PATCH, DELETE | `/collections/{id}` | One collection: read, rename, delete |
| POST, GET | `/collections/{id}/documents` | [Upload, list documents](#documents) |
| GET, DELETE | `/documents/{id}` | One document: status, delete |
| GET | `/documents/{id}/download` | The stored file |
| POST | `/documents/{id}/reprocess` | [Process again](#processing) |
| GET | `/documents/{id}/chunks` | The chunks, in order |
| POST | `/collections/{id}/search` | [Retrieval only, no answer](#search-debugging-view) |
| POST | `/collections/{id}/ask` | [A grounded answer with citations](#asking-questions) |
| POST | `/collections/{id}/ask/stream` | [The same, streamed](#streaming-answers) |
| GET | `/conversations` | [Your conversations](#conversations) |
| GET, PATCH, DELETE | `/conversations/{id}` | One conversation: messages, rename, delete |
| GET | `/usage` | Your usage and quotas today and this month ([SECURITY.md](SECURITY.md#usage-tracking)) |
| GET | `/demo`, `/` | The demo page; `/` redirects to it. Only with `DEMO_UI_ENABLED=true` ([OPERATIONS.md](OPERATIONS.md#the-demo-page)) |

The demo account gets `403 demo_read_only` on every endpoint that changes
something ([SECURITY.md](SECURITY.md#the-demo-account)).

## Getting started

Open http://localhost:8000/docs (Swagger UI).

1. `POST /auth/register` with an email and a password of 10+ characters (not a common
   one). In invite mode (the default) add `"invite_code"`; see [Registration](SECURITY.md#registration).
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
are processed in the background (see [Ingestion](ARCHITECTURE.md#ingestion-extraction-and-chunking)).

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

## Processing

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

## Search (debugging view)

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

Errors: `422` invalid question; `404` not your collection; `429` (with `Retry-After`) when
the embedding or answer quota is used up, including Gemini's daily limit (which resets at
midnight Pacific time); `503` when a provider is down, times out or is misconfigured. Error
bodies are short fixed messages, never provider responses or keys.

## Streaming answers

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

## Conversations

| Method | Path | What it does |
|---|---|---|
| GET | `/conversations` | Yours, most recently active first. `?collection_id=&limit=&offset=` |
| GET | `/conversations/{id}` | One conversation with its messages, oldest first. `?limit=&offset=` |
| PATCH | `/conversations/{id}` | Rename: `{"title": "..."}` (1-200 characters) |
| DELETE | `/conversations/{id}` | Delete it and its messages |

Other users' conversations return `404`. Deleting a collection deletes its conversations.
