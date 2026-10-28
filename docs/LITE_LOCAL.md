# Lite mode, locally

`docker-compose.lite.yml` runs DocuMind the way it runs on Render's free
tier, on your machine:

| Production ([OPERATIONS.md](OPERATIONS.md#lite-mode-free-tier)) | Locally |
| --- | --- |
| Render free web service, 512 MB, no worker | the `api` container: the production image, `mem_limit: 512m`, no swap, `INGESTION_MODE=inline`, no Celery worker |
| `APP_ENV=production`, docs off, invite-only | the same |
| Neon through its PgBouncer pooler | Postgres (pgvector) behind PgBouncer in transaction mode (`DB_PGBOUNCER=true`); migrations use the direct connection |
| Upstash Redis | Redis |
| R2 / B2 (`STORAGE_BACKEND=s3`) | moto's S3 server (in memory, see below) |
| Voyage, Gemini | **fake providers**, unless you export real keys (below) |

It is a separate Compose project (`documind-lite`): it shares no
containers, volumes or ports with the development stack in
`docker-compose.yml`, and both can run at once. Everything listens on
127.0.0.1 only.

Production refuses the fake providers. This stack sets
`ALLOW_FAKE_PROVIDERS_IN_PRODUCTION=true` so the rest of the production
checks still apply locally; the app logs a warning at startup when it is
used. Never set it on a real deployment (a test checks `render.yaml` doesn't).

**About the S3 server.** Locally, files go to
[moto](https://github.com/getmoto/moto)'s standalone S3 server
(`docker/moto/Dockerfile`, a pinned `moto[server]`). It is an **in-memory
emulator**: everything stored in it disappears when its container stops or
restarts, while the database rows survive (in the `postgres_data` volume),
so after a restart old documents' downloads answer 404 "Stored file is
missing". That's fine for a smoke test. The app talks to it exactly as it
talks to Cloudflare R2 or Backblaze B2: the same boto3 calls (multipart
uploads, streamed downloads, bulk deletes) over the S3 API, so real object
storage behaves the same; only durability differs. To start clean after a
restart, use `down -v` (step 7).

## 1. One-time setup: local secrets

```bash
python3 - > .env.lite.local <<'EOF'
import secrets
print(f"LITE_SECRET_KEY={secrets.token_urlsafe(48)}")
print(f"LITE_INVITE_CODE={secrets.token_urlsafe(12)}")
print("DEMO_EMAIL=demo@example.com")
print(f"DEMO_PASSWORD={secrets.token_urlsafe(16)}")
EOF
```

`.env.lite.local` is gitignored. Compose reads it with `--env-file` (not
your development `.env`, whose weak `SECRET_KEY` production would refuse).

## 2. Start

```bash
docker compose -f docker-compose.lite.yml --env-file .env.lite.local up -d --build
docker compose -f docker-compose.lite.yml --env-file .env.lite.local ps
```

Startup order: Postgres and the S3 server, then the one-off `bucket`
service that creates the bucket, then the API, which applies the migrations
and starts. The API answers on <http://localhost:8080> (`LITE_PORT` changes
it). The S3 server is reachable only from inside the stack.

Check the API log for the provider line and the fake-provider warning:

```bash
docker compose -f docker-compose.lite.yml --env-file .env.lite.local logs api | head
```

### Using real providers

Export the keys in the shell before `up`; each provider switches to the
real one when its key is set:

```bash
export EMBEDDING_API_KEY=...   # Voyage
export LLM_API_KEY=...         # Gemini
docker compose -f docker-compose.lite.yml --env-file .env.lite.local up -d
```

Never put real keys in `.env.lite.local` if you might share it. Note that
search compares only vectors from the same model: documents embedded with
the fake provider must be reprocessed after switching.

## 3. Create the bucket (manually)

The `bucket` service does this on every `up`. To run it yourself (it does
nothing if the bucket exists):

```bash
docker compose -f docker-compose.lite.yml --env-file .env.lite.local run --rm bucket
```

## 4. Seed the demo user

```bash
docker compose -f docker-compose.lite.yml --env-file .env.lite.local \
  exec api python -m scripts.seed_demo
```

It uses `DEMO_EMAIL` and `DEMO_PASSWORD` from `.env.lite.local`, stores the
files from `demo/corpus/` in the S3 server and processes them. Run it again
any time. Because the S3 server is in memory, seed again after restarting
the stack (or start clean with `down -v`).

## 5. Smoke test

```bash
scripts/smoke_lite.sh
```

Needs `curl` and `python3`. It reads `.env.lite.local` and checks:

1. `/health/live` and `/health/ready` (the inline worker is "ok" without any
   heartbeat), and that `/docs` is disabled;
2. a wrong invite code is refused, the right one registers, login works;
3. an upload reaches "ready" through inline ingestion;
4. a streamed answer with `curl -N`: token events, then `done`;
5. the download comes back from the S3 server byte-identical;
6. the demo user may ask and search, and gets 403 `demo_read_only` on each
   forbidden action (upload, delete and reprocess a document, create,
   rename and delete a collection, rename and delete a conversation, change
   the password, delete the account).

It prints PASS/FAIL per check and exits 1 if anything failed. Each run
registers a new user; registration allows 5 per hour per address, so more
than 5 runs an hour will fail at step 2.

## 6. Watch memory

```bash
docker stats documind-lite-api-1
```

The limit shows as 512 MiB. The measured worst case is in
[HARDENING.md](HARDENING.md), "Memory".

## 7. Tear down

```bash
# Stop, keep the data:
docker compose -f docker-compose.lite.yml --env-file .env.lite.local down
# Stop and delete everything (the database; stored files vanish with the S3 server anyway):
docker compose -f docker-compose.lite.yml --env-file .env.lite.local down -v
```

`down -v` removes only this stack's volumes (`documind-lite_*`), never the
development stack's.
