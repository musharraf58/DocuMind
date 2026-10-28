#!/usr/bin/env bash
# Read-only smoke check against a deployed DocuMind (docs/OPERATIONS.md).
#
#   DEMO_EMAIL=... DEMO_PASSWORD=... scripts/smoke_live.sh https://your-app.onrender.com
#
# Checks, in order:
#   1. /health/live (waits up to WAKE_TIMEOUT seconds for a sleeping free instance)
#   2. /health/ready
#   3. /docs is 404 (API docs off in production)
#   4. login as the demo account
#   5. a streamed question: at least one token event, then a done event with a citation
#   6. a delete attempt by the demo account answers 403 demo_read_only
# Prints PASS/FAIL per check; exits 1 if anything failed. Never prints the
# password or the token.
#
# Read-only except for one thing: asking saves a conversation in the shared
# demo account (the demo account can't delete it, which is what check 6 shows).
# It also uses one of the demo account's daily questions and one LLM request.
set -uo pipefail

BASE_URL="${1:?usage: scripts/smoke_live.sh <base-url>}"
BASE_URL="${BASE_URL%/}"
: "${DEMO_EMAIL:?set DEMO_EMAIL}"
: "${DEMO_PASSWORD:?set DEMO_PASSWORD}"
QUESTION="${QUESTION:-How often should a bicycle chain be oiled?}"
WAKE_TIMEOUT="${WAKE_TIMEOUT:-180}"

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
try:
    value = json.load(open(sys.argv[1]))
    for key in sys.argv[2:]:
        value = value[int(key)] if isinstance(value, list) else value.get(key)
except (ValueError, LookupError, AttributeError):
    value = None
print("" if value is None else value)
PY
}

# call METHOD PATH [curl args...]: the body goes to $WORK/body, prints the status.
call() {
    local method="$1" path="$2"
    shift 2
    curl -sS --max-time 60 -o "$WORK/body" -w '%{http_code}' -X "$method" "$BASE_URL$path" "$@" \
        2>/dev/null || echo 000
}

auth() { printf 'Authorization: Bearer %s' "$1"; }

section "1. Wake up (a sleeping free instance takes about a minute)"
start=$SECONDS
status=000
while ((SECONDS - start < WAKE_TIMEOUT)); do
    status=$(curl -sS --max-time 120 -o "$WORK/body" -w '%{http_code}' "$BASE_URL/health/live" 2>/dev/null || true)
    [[ "$status" == 200 ]] && break
    sleep 5
done
if [[ "$status" == 200 ]]; then
    pass "/health/live 200 after $((SECONDS - start))s"
else
    fail "/health/live answered ${status:-nothing} for ${WAKE_TIMEOUT}s"
fi

section "2. Readiness and production settings"
status=$(call GET /health/ready)
if [[ "$status" == 200 ]]; then
    pass "/health/ready 200 (database $(json "$WORK/body" database), redis $(json "$WORK/body" redis), worker $(json "$WORK/body" worker))"
else
    fail "/health/ready answered $status: $(head -c 300 "$WORK/body")"
fi
status=$(call GET /docs)
if [[ "$status" == 404 ]]; then pass "API docs off (/docs 404)"; else fail "/docs answered $status"; fi

section "3. Demo login"
status=$(call POST /auth/login --data-urlencode "username=$DEMO_EMAIL" --data-urlencode "password=$DEMO_PASSWORD")
TOKEN=$(json "$WORK/body" access_token)
if [[ "$status" == 200 && -n "$TOKEN" ]]; then
    pass "login"
else
    fail "login answered $status"
    printf '\nCannot continue without a token.\n'
    exit 1
fi
call GET /collections -H "$(auth "$TOKEN")" >/dev/null
COLLECTION=$(json "$WORK/body" 0 id)
if [[ -n "$COLLECTION" ]]; then
    pass "demo collection: $(json "$WORK/body" 0 name)"
else
    fail "the demo account has no collection (was it seeded?)"
fi

section "4. Streamed answer"
QUESTION_JSON=$(python3 -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$QUESTION")
status=$(curl -sS -N --max-time 120 -o "$WORK/stream" -w '%{http_code}' -X POST \
    "$BASE_URL/collections/$COLLECTION/ask/stream" -H "$(auth "$TOKEN")" \
    -H 'Content-Type: application/json' -d "$QUESTION_JSON" 2>/dev/null || echo 000)
# The data line after "event: done" (or "event: error").
python3 - "$WORK/stream" "$WORK" <<'PY'
import json, sys
event, out = None, sys.argv[2]
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.rstrip("\r\n")
    if line.startswith("event: "):
        event = line[7:]
    elif line.startswith("data: ") and event in ("done", "error"):
        with open(f"{out}/{event}.json", "w") as f:
            f.write(line[6:])
PY
tokens=$(grep -c '^event: token' "$WORK/stream" 2>/dev/null || true)
if [[ "$status" != 200 ]]; then
    fail "ask/stream answered $status: $(head -c 300 "$WORK/stream")"
elif [[ -f "$WORK/error.json" ]]; then
    fail "the stream ended with an error: $(json "$WORK/error.json" message) (status $(json "$WORK/error.json" status))"
elif [[ "$tokens" -lt 1 || ! -f "$WORK/done.json" ]]; then
    fail "expected token events and a done event; got $tokens token(s): $(head -c 300 "$WORK/stream")"
else
    pass "$tokens token event(s), then done"
    citations=$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["citations"]))' "$WORK/done.json")
    if ((citations > 0)); then
        pass "done cites $citations source(s), first: $(json "$WORK/done.json" citations 0 document_name)"
    else
        fail "done has no citation (grounded: $(json "$WORK/done.json" grounded)); answer: $(json "$WORK/done.json" answer | head -c 200)"
    fi
fi
CONVERSATION=$(json "$WORK/done.json" conversation_id 2>/dev/null)

section "5. The demo account is read-only"
if [[ -z "$CONVERSATION" ]]; then
    call GET /conversations -H "$(auth "$TOKEN")" >/dev/null
    CONVERSATION=$(json "$WORK/body" 0 id)
fi
if [[ -z "$CONVERSATION" ]]; then
    fail "no conversation to try deleting"
else
    status=$(call DELETE "/conversations/$CONVERSATION" -H "$(auth "$TOKEN")")
    code=$(json "$WORK/body" code)
    if [[ "$status" == 403 && "$code" == demo_read_only ]]; then
        pass "delete refused: 403 demo_read_only"
    else
        fail "delete answered $status ${code}"
    fi
fi

echo
if ((FAILURES == 0)); then
    printf '\033[32mAll checks passed.\033[0m\n'
else
    printf '\033[31m%d check(s) failed.\033[0m\n' "$FAILURES"
    exit 1
fi
