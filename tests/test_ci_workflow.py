"""The CI workflow file: it checks what we rely on, and stores no secrets."""

import re
import shutil
import subprocess

import pytest
import yaml

from tests.repo_files import REPO_FILES, ROOT, repo_file


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(repo_file(".github/workflows/ci.yml").read_text())


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
    assert not re.search(r"\$\{\{\s*secrets\.", repo_file(".github/workflows/ci.yml").read_text())


def dev_stage_copies() -> set[str]:
    """Sources of the COPY lines in the Dockerfile's dev stage."""
    dockerfile = repo_file("Dockerfile").read_text()
    stage = dockerfile.split("AS dev", 1)[1].split("\nFROM ", 1)[0]
    sources = set()
    for line in stage.splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            sources.update(line.split()[1:-1])
    return sources


def test_the_dev_image_has_every_repo_file_the_tests_read():
    """Otherwise `docker compose exec api pytest` fails where CI passes."""
    assert set(REPO_FILES) <= dev_stage_copies()


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not installed")
def test_actionlint_passes():
    # CI also runs it (in the lint job, from its Docker image).
    result = subprocess.run(["actionlint", "-no-color"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
