"""render.yaml and .env.lite.example: complete, secret-free, and the
production checks still refuse a weak configuration in lite mode."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.core.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SECRETS = {
    "SECRET_KEY",
    "INVITE_CODE",
    "DATABASE_URL",
    "DATABASE_URL_DIRECT",
    "REDIS_URL",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "EMBEDDING_API_KEY",
    "LLM_API_KEY",
    "DEMO_PASSWORD",
}


@pytest.fixture(scope="module")
def service() -> dict:
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text())
    (web,) = blueprint["services"]
    return web


@pytest.fixture(scope="module")
def render_env(service) -> dict[str, dict]:
    return {entry["key"]: entry for entry in service["envVars"]}


@pytest.fixture(scope="module")
def lite_example() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.lite.example").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_free_docker_web_service_with_liveness_health_check(service):
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["plan"] == "free"
    assert service["healthCheckPath"] == "/health/live"


def test_lite_mode_settings(render_env):
    expected = {
        "APP_ENV": "production",
        "INGESTION_MODE": "inline",
        "STORAGE_BACKEND": "s3",
        "REGISTRATION_MODE": "invite",
        "DOCS_ENABLED": "false",
    }
    for key, value in expected.items():
        assert render_env[key]["value"] == value


def test_every_secret_is_sync_false_with_no_value(render_env):
    for key in SECRETS - {"DEMO_PASSWORD"}:  # the demo password is for the seed script only
        assert render_env[key] == {"key": key, "sync": False}, key


def test_every_variable_is_a_real_setting(render_env, lite_example):
    fields = set(Settings.model_fields)
    assert set(render_env) <= fields
    assert set(lite_example) <= fields


def test_the_example_lists_what_render_declares_and_holds_no_secret(render_env, lite_example):
    assert set(render_env) <= set(lite_example)
    assert set(lite_example) - set(render_env) == {"DEMO_EMAIL", "DEMO_PASSWORD"}
    for key in SECRETS | {"S3_ENDPOINT_URL", "S3_BUCKET", "S3_REGION", "CORS_ORIGINS"}:
        assert lite_example[key] == "", key


def filled(lite_example: dict[str, str], **overrides) -> dict[str, str]:
    """The example with plausible (fake) values in every blank."""
    values = {key: value for key, value in lite_example.items() if value}
    values.update(
        SECRET_KEY="lite-test-0123456789-abcdefghijklmnopqrstuvwxyz-ABCDEFG",
        INVITE_CODE="invite-code-long-enough",
        DATABASE_URL="postgresql://u:p@ep-x-pooler.neon.tech/db?sslmode=require",
        DATABASE_URL_DIRECT="postgresql://u:p@ep-x.neon.tech/db?sslmode=require",
        REDIS_URL="rediss://default:pw@x.upstash.io:6379",
        S3_ENDPOINT_URL="https://acct.r2.cloudflarestorage.com",
        S3_BUCKET="documind",
        S3_REGION="auto",
        S3_ACCESS_KEY_ID="key-id",
        S3_SECRET_ACCESS_KEY="key-secret",
        EMBEDDING_API_KEY="voyage-key",
        LLM_API_KEY="gemini-key",
    )
    values.update(overrides)
    return values


def test_a_complete_lite_config_starts(lite_example):
    settings = Settings(_env_file=None, **filled(lite_example))
    assert settings.APP_ENV == "production"
    assert settings.INGESTION_MODE == "inline"
    assert settings.database_url.query == {"ssl": "require"}


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"SECRET_KEY": "change-me"}, "SECRET_KEY"),
        ({"SECRET_KEY": "short-but-random-1"}, "SECRET_KEY"),
        ({"DOCS_ENABLED": "true"}, "docs"),
        ({"RATE_LIMIT_ENABLED": "false"}, "RATE_LIMIT_ENABLED"),
        ({"EMBEDDING_PROVIDER": "fake"}, "fake"),
        ({"INVITE_CODE": "short"}, "INVITE_CODE"),
        ({"S3_BUCKET": ""}, "S3_BUCKET"),
        ({"S3_SECRET_ACCESS_KEY": ""}, "S3_SECRET_ACCESS_KEY"),
    ],
)
def test_weak_lite_configs_are_refused(lite_example, override, problem):
    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **filled(lite_example, **override))
    assert problem in str(caught.value)
    # The rejected values themselves are never echoed.
    assert "lite-test-0123456789" not in str(caught.value)
    assert "key-secret" not in str(caught.value)
