# Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. (Not on pushes to other branches: a branch with an open PR would
otherwise run everything twice.) Four jobs run in parallel:

| Job | What it checks |
| --- | --- |
| **Lint (ruff)** | `ruff check`, `ruff format --check`, actionlint (with shellcheck) on the workflow files, and the demo page's JavaScript tests (`node --test tests/js/`, no npm packages) |
| **Tests** | `alembic upgrade head` + `alembic check` (the migrations apply and match the models), then the full pytest suite, against real `pgvector/pgvector:pg16` and `redis:7` service containers |
| **Dependency audit** | `pip-audit` on the exact locked versions (`uv.lock`), dev tools included |
| **Production image** | `docker build --target prod`, as Render builds it, and that it runs as a non-root user |

Dependencies are cached: uv's download cache (by `setup-uv`) and Docker
layers (the GitHub Actions cache).

The tests use fake embedding and LLM providers (no network, no keys), and
`REQUIRE_REAL_SERVICES=1` makes the real-Redis tests (`tests/test_real_redis.py`)
fail instead of skip if the service container were missing.

**Test count:** 1,037 tests, counted on 2026-10-28 with
`uv run pytest --collect-only -q` (the README quotes this number; recount
and update both when it changes noticeably).

**No secrets.** The workflow references no `secrets.*` at all (a test
enforces this). The `SECRET_KEY` in its `env` is a labelled throwaway value
for a disposable test run. The GitHub token is read-only (`permissions:
contents: read`).

## Tests in the dev container

`docker compose exec api pytest -q` runs the same suite inside the
development image. A few tests check repository files the app itself never
reads: `.env.example`, `.env.lite.example`, `.github/workflows/ci.yml`,
`render.yaml`, `docker-compose.lite.yml`, the `Dockerfile`, `README.md` and
`docs/` (the link check). CI has them from the checkout.
The dev stage of the `Dockerfile` copies them in, next to the tests (the
production stage doesn't).

Why copy them rather than the alternatives:

- **Not a bind mount.** The tests themselves are copied in at build time, so
  their input files come from the same snapshot. Mounting the files live
  would pair them with tests from an older build.
- **Not a skip.** These tests catch real regressions: an unknown setting in an
  example file, a secret in the workflow, a weak lite config. A skip would
  pass silently wherever the files are missing.

To keep this from drifting:

- Tests read these files through `tests/repo_files.py` (`repo_file(...)`),
  which only accepts names listed in `REPO_FILES`. A missing file fails with
  the fix ("rebuild the image") instead of a bare `FileNotFoundError`. It
  never skips, in CI or anywhere else.
- `test_the_dev_image_has_every_repo_file_the_tests_read` checks that the
  dev stage copies every file in `REPO_FILES`.

A test that needs a new repository file:

1. Read it with `repo_file(...)`.
2. Add its name to `REPO_FILES`.
3. Add it to the dev stage's `COPY` lines.

After changing one of these files, rebuild before testing in the container:
`docker compose build api && docker compose up -d api worker`.

## Status badge



```markdown
[![CI](https://github.com/musharraf58/DocuMind/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/musharraf58/DocuMind/actions/workflows/ci.yml)
```

## Branch protection for `main` (do this once, by hand)

CI only reports; branch protection is what makes it binding. On GitHub:
**Settings → Rules → Rulesets → New ruleset → New branch ruleset**
(or the older **Settings → Branches → Add branch protection rule**, which has
the same options):

1. **Name:** `main`. **Enforcement status:** Active.
2. **Target branches:** Add target → *Include default branch*.
3. **Restrict deletions:** on.
4. **Block force pushes:** on. History on `main` is never rewritten.
5. **Require a pull request before merging:** on. This is what forbids
   direct pushes to `main`: every change arrives through a PR.
   - Required approvals: 0 while you work alone (GitHub doesn't let you
     approve your own PR), 1 or more once someone else reviews.
   - *Dismiss stale pull request approvals when new commits are pushed:* on.
   - *Require conversation resolution before merging:* on.
6. **Require status checks to pass:** on.
   - *Require branches to be up to date before merging:* on (the checks
     must have run against the latest `main`).
   - Add these checks (they appear in the list after CI has run once on a PR):
     `Lint (ruff)`, `Tests (Postgres + pgvector, Redis)`,
     `Dependency audit (pip-audit)`, `Production image`.
7. **Bypass list:** leave empty, so the rules apply to you as well. (An
   admin bypass is how "just this once" pushes happen.)

Then **Settings → General → Pull Requests**: allow the merge style you use and
turn on *Automatically delete head branches*.

Note on the audit job: a newly published advisory can turn it red without
any change on our side. That is the point (it tells you to upgrade), but it
means a PR can fail for a reason unrelated to its diff; fix it with a
separate dependency-update PR.

## Dependabot

`.github/dependabot.yml` opens weekly update PRs for:

- **Python packages** (the `uv` ecosystem: it updates `uv.lock` as well as
  `pyproject.toml`; the plain `pip` ecosystem would leave the lockfile, which
  is what we actually install, untouched). Minor and patch updates are
  grouped into one PR.
- **GitHub Actions** used by the workflow.
- **Docker base images** in the `Dockerfile`.

Every Dependabot PR runs CI, so a breaking update shows up as a red check.

## Running the same checks locally

```bash
uv run ruff check . && uv run ruff format --check .
docker compose up -d postgres redis
uv run alembic upgrade head && uv run alembic check
uv run pytest -q
uv export --format requirements-txt --no-hashes --no-emit-project --all-groups > /tmp/req.txt
uvx pip-audit==2.10.1 -r /tmp/req.txt --disable-pip --no-deps
docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:1.7.7
node --test tests/js/                 # the demo page's SSE parser (Node 20+)
docker build --target prod -t documind:prod .
```
