#!/bin/sh
# Container start (production image): apply database migrations, then run the
# server (or whatever command was given instead, e.g. a one-off script).
#
# Migrating on start is safe for ONE instance, which is what the free tier
# runs. With several instances starting at once, each would run `alembic
# upgrade head` at the same time, and two could try to apply the same
# migration concurrently. Before scaling out, run migrations as a separate
# release step (e.g. Render's preDeployCommand on a paid plan) and set
# RUN_MIGRATIONS=0 here.
set -eu

if [ "${RUN_MIGRATIONS:-1}" = "1" ]; then
    echo "entrypoint: applying database migrations"
    alembic upgrade head
fi

if [ "$#" -eq 0 ]; then
    # exec: uvicorn replaces this shell, so it receives SIGTERM directly and
    # shuts down cleanly (finishing requests, stopping background tasks).
    # --no-proxy-headers: the app decides about X-Forwarded-For itself
    # (TRUSTED_PROXY_COUNT); see docs/SECURITY.md.
    set -- uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
        --no-proxy-headers --timeout-graceful-shutdown 20
fi
exec "$@"
