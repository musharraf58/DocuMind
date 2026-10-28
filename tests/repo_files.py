"""Repository files that tests read (config examples, the CI workflow, the
Render blueprint, the documentation). Not part of the app, so the dev image copies them in
explicitly (Dockerfile, "dev" stage; docs/CI.md, "Tests in the dev
container"). Read them through repo_file() so a missing one fails with the
fix instead of a bare FileNotFoundError, and never skips.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Every repo-root file (or folder) a test reads. tests/test_ci_workflow.py
# checks that the dev stage of the Dockerfile copies each one.
REPO_FILES = (
    ".env.example",
    ".env.lite.example",
    ".github/workflows/ci.yml",
    "Dockerfile",
    "README.md",
    "docker-compose.lite.yml",
    "docs",
    "render.yaml",
)


def repo_file(name: str) -> Path:
    assert name in REPO_FILES, f"add {name} to REPO_FILES (and to the Dockerfile's dev stage)"
    path = ROOT / name
    if not path.exists():
        pytest.fail(
            f"{name} is missing from {ROOT}. In the dev container, rebuild the image "
            "(docker compose build api): the Dockerfile's dev stage copies it in.",
            pytrace=False,
        )
    return path
