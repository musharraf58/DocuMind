#!/usr/bin/env bash
# Lists every TODO(me) left in README.md and docs/: the places only the
# author can fill in (live URL, demo login, measurements, screenshots, "My
# reasoning", "My answer"). For the author's own use; CI never runs it
# (tests/test_ci_workflow.py checks that).
#
#   scripts/check_readme_todos.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

matches=$(grep -rn --include='*.md' -F 'TODO(me)' README.md docs/ || true)
if [[ -z "$matches" ]]; then
    echo "No TODO(me) left in README.md or docs/."
    exit 0
fi
printf '%s\n' "$matches"
echo
printf '%s\n' "$matches" | cut -d: -f1 | sort | uniq -c | sort -rn
echo "$(printf '%s\n' "$matches" | wc -l) TODO(me) left."
