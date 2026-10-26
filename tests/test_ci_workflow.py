"""The CI workflow file: it checks what we rely on, and stores no secrets."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_runs_on_pull_requests_and_pushes_to_main(workflow):
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    triggers = workflow[True]
    assert "pull_request" in triggers
    assert triggers["push"]["branches"] == ["main"]


def test_has_every_required_job(workflow):
    assert set(workflow["jobs"]) == {"lint", "test", "audit", "docker"}


def test_tests_run_against_real_postgres_and_redis(workflow):
    job = workflow["jobs"]["test"]
    assert job["services"]["postgres"]["image"].startswith("pgvector/pgvector:")
    assert job["services"]["redis"]["image"].startswith("redis:")
    assert job["env"]["REQUIRE_REAL_SERVICES"] == "1"


def test_token_is_read_only(workflow):
    assert workflow["permissions"] == {"contents": "read"}


def test_no_secrets_are_referenced():
    """Nothing in CI needs a secret: providers are fake, services are
    throwaway containers. A `secrets.` reference would mean one crept in."""
    assert not re.search(r"\$\{\{\s*secrets\.", WORKFLOW.read_text())


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not installed")
def test_actionlint_passes():
    # CI also runs it (in the lint job, from its Docker image).
    result = subprocess.run(["actionlint", "-no-color"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
