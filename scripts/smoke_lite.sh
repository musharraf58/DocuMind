#!/usr/bin/env bash
# Smoke test for the local lite-mode stack (docker-compose.lite.yml).
# See docs/LITE_LOCAL.md. Needs: curl, python3.
#
#   scripts/smoke_lite.sh                 # reads .env.lite.local
#   ENV_FILE=other.env BASE_URL=http://localhost:8081 scripts/smoke_lite.sh
#
# Checks, in order:
#   1. /health/live and /health/ready (inline worker "ok", no heartbeat needed)
#   2. registration with the invite code, and login
#   3. an upload that reaches "ready" through inline ingestion
#   4. a streamed answer (curl -N): token events, then a done event
#   5. the download comes back from S3 byte-identical
#   6. the demo user gets 403 demo_read_only on every forbidden action
# Prints PASS/FAIL per check; exits 1 if anything failed. Never prints a
# password, the invite code or a token.
set -uo pipefail

ENV_FILE="${ENV_FILE:-.env.lite.local}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi
BASE_URL="${BASE_URL:-http://localhost:${LITE_PORT:-8081}}"
INVITE_CODE="${LITE_INVITE_CODE:?LITE_INVITE_CODE missing (see docs/LITE_LOCAL.md)}"
DEMO_EMAIL="${DEMO_EMAIL:-}"
DEMO_PASSWORD="${DEMO_PASSWORD:-}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FAILURES=0

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
section() { printf '\n%s\n' "$1"; }

# json FILE KEY [KEY...]: print a value from a JSON file (keys or list indexes).
json() {
    python3 - "$@" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2:]:
    value = value[int(key)] if isinstance(value, list) else value.get(key)
print("" if value is None else value)
PY
}

# call METHOD PATH [curl args...]: the body goes to $WORK/body, prints the status.
call() {
    local method="$1" path="$2"
    shift 2
    curl -sS -o "$WORK/body" -w '%{http_code}' -X "$method" "$BASE_URL$path" "$@"
}

auth() { printf 'Authorization: Bearer %s' "$1"; }

section "1. Health"
status=$(call GET /health/live)
[[ "$status" == 200 ]] && pass "/health/live 200" || fail "/health/live answered $status"
status=$(call GET /health/ready)
worker=$(json "$WORK/body" worker 2>/dev/null)
if [[ "$status" == 200 && "$worker" == ok ]]; then
    pass "/health/ready 200, inline worker ok"
else
    fail "/health/ready answered $status: $(cat "$WORK/body")"
fi
status=$(call GET /docs)
[[ "$status" == 404 ]] && pass "API docs disabled (/docs 404)" || fail "/docs answered $status"

section "2. Register (invite) and log in"
EMAIL="smoke-$(date +%s)-$RANDOM@example.com"
PASSWORD="smoke-$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"
status=$(call POST /auth/register -H 'Content-Type: application/json' \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"invite_code\":\"wrong-invite-code\"}")
[[ "$status" == 403 ]] && pass "a wrong invite code is refused (403)" || fail "wrong invite: $status"
status=$(call POST /auth/register -H 'Content-Type: application/json' \
    -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"invite_code\":\"$INVITE_CODE\"}")
[[ "$status" == 201 ]] && pass "registered $EMAIL" || fail "register answered $status: $(cat "$WORK/body")"
status=$(call POST /auth/login --data-urlencode "username=$EMAIL" --data-urlencode "password=$PASSWORD")
TOKEN=$(json "$WORK/body" access_token 2>/dev/null)
if [[ "$status" == 200 && -n "$TOKEN" ]]; then pass "login"; else fail "login answered $status"; fi

section "3. Upload -> ready (inline ingestion)"
call POST /collections -H "$(auth "$TOKEN")" -H 'Content-Type: application/json' \
    -d '{"name":"Smoke test"}' >/dev/null
COLLECTION=$(json "$WORK/body" id)
python3 - "$WORK/upload.txt" <<'PY'
import secrets, sys
lines = [
    "Smoke test notes. Bicycle chains need a drop of oil on every roller.",
    "Houseplants drown more often than they dry out.",
    "Coffee tastes best within a few weeks of roasting.",
]
with open(sys.argv[1], "w") as f:
    f.write(f"Run {secrets.token_hex(8)}\n")
    for i in range(400):
        f.write(f"{i}: {lines[i % 3]}\n")
PY
status=$(call POST "/collections/$COLLECTION/documents" -H "$(auth "$TOKEN")" \
    -F "file=@$WORK/upload.txt;type=text/plain")
DOCUMENT=$(json "$WORK/body" id 2>/dev/null)
[[ "$status" == 202 ]] && pass "upload accepted (202, pending)" || fail "upload answered $status: $(cat "$WORK/body")"
state=""
for ((i = 0; i < READY_TIMEOUT; i += 2)); do
    call GET "/documents/$DOCUMENT" -H "$(auth "$TOKEN")" >/dev/null
    state=$(json "$WORK/body" status)
    [[ "$state" == ready || "$state" == failed ]] && break
    sleep 2
done
if [[ "$state" == ready ]]; then
    pass "document ready after ~${i}s"
else
    fail "document is '$state' after ${i}s: $(json "$WORK/body" error_message)"
fi

section "4. Streamed answer (curl -N)"
curl -sS -N --max-time 90 -X POST "$BASE_URL/collections/$COLLECTION/ask/stream" \
    -H "$(auth "$TOKEN")" -H 'Content-Type: application/json' \
    -d '{"question":"How often should a bicycle chain be oiled?"}' >"$WORK/stream"
tokens=$(grep -c '^event: token' "$WORK/stream" || true)
if grep -q '^event: done' "$WORK/stream" && ((tokens > 0)); then
    pass "stream: $tokens token events, then done"
else
    fail "stream ended without token + done events: $(head -c 300 "$WORK/stream")"
fi

section "5. Download from S3, byte-identical"
curl -sS -o "$WORK/download" -H "$(auth "$TOKEN")" "$BASE_URL/documents/$DOCUMENT/download"
if cmp -s "$WORK/upload.txt" "$WORK/download"; then
    pass "download matches the upload ($(wc -c <"$WORK/download") bytes, sha256 $(sha256sum "$WORK/download" | cut -c1-16)...)"
else
    fail "download differs from the upload"
fi

section "6. Demo account is read-only"
if [[ -z "$DEMO_EMAIL" || -z "$DEMO_PASSWORD" ]]; then
    fail "DEMO_EMAIL / DEMO_PASSWORD not set: seed the demo first (docs/LITE_LOCAL.md)"
else
    status=$(call POST /auth/login --data-urlencode "username=$DEMO_EMAIL" \
        --data-urlencode "password=$DEMO_PASSWORD")
    DEMO=$(json "$WORK/body" access_token 2>/dev/null)
    if [[ "$status" != 200 || -z "$DEMO" ]]; then
        fail "demo login answered $status (was the demo seeded?)"
    else
        pass "demo login"
        call GET /collections -H "$(auth "$DEMO")" >/dev/null
        DCOL=$(json "$WORK/body" 0 id)
        call GET "/collections/$DCOL/documents" -H "$(auth "$DEMO")" >/dev/null
        DDOC=$(json "$WORK/body" 0 id)
        status=$(call POST "/collections/$DCOL/ask" -H "$(auth "$DEMO")" \
            -H 'Content-Type: application/json' -d '{"question":"How do I water a houseplant?"}')
        DCONV=$(json "$WORK/body" conversation_id 2>/dev/null)
        [[ "$status" == 200 ]] && pass "demo may ask (200)" || fail "demo ask answered $status"
        status=$(call POST "/collections/$DCOL/search" -H "$(auth "$DEMO")" \
            -H 'Content-Type: application/json' -d '{"query":"coffee"}')
        [[ "$status" == 200 ]] && pass "demo may search (200)" || fail "demo search answered $status"

        forbidden() { # name method path [curl args...]
            local name="$1" method="$2" path="$3"
            shift 3
            local status code
            status=$(call "$method" "$path" -H "$(auth "$DEMO")" "$@")
            code=$(json "$WORK/body" code 2>/dev/null)
            if [[ "$status" == 403 && "$code" == demo_read_only ]]; then
                pass "demo: $name -> 403 demo_read_only"
            else
                fail "demo: $name -> $status ${code:-}"
            fi
        }
        printf 'demo upload attempt\n' >"$WORK/demo.txt"
        J=(-H 'Content-Type: application/json')
        forbidden "upload" POST "/collections/$DCOL/documents" -F "file=@$WORK/demo.txt;type=text/plain"
        forbidden "delete document" DELETE "/documents/$DDOC"
        forbidden "reprocess" POST "/documents/$DDOC/reprocess"
        forbidden "create collection" POST /collections "${J[@]}" -d '{"name":"Mine"}'
        forbidden "rename collection" PATCH "/collections/$DCOL" "${J[@]}" -d '{"name":"Renamed"}'
        forbidden "delete collection" DELETE "/collections/$DCOL"
        forbidden "rename conversation" PATCH "/conversations/$DCONV" "${J[@]}" -d '{"title":"x"}'
        forbidden "delete conversation" DELETE "/conversations/$DCONV"
        forbidden "change password" POST /auth/change-password "${J[@]}" \
            -d "{\"current_password\":\"$DEMO_PASSWORD\",\"new_password\":\"another-demo-pass-123\"}"
        forbidden "delete account" DELETE /auth/me "${J[@]}" -d "{\"password\":\"$DEMO_PASSWORD\"}"
    fi
fi

echo
if ((FAILURES == 0)); then
    printf '\033[32mAll checks passed.\033[0m\n'
else
    printf '\033[31m%d check(s) failed.\033[0m\n' "$FAILURES"
    exit 1
fi
