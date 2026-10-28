"""render.yaml and .env.lite.example: complete, secret-free, and the
production checks still refuse a weak configuration in lite mode."""

import pytest
import yaml
from pydantic import ValidationError

from app.core.config import Settings
from tests.repo_files import repo_file

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
    blueprint = yaml.safe_load(repo_file("render.yaml").read_text())
    (web,) = blueprint["services"]
    return web


@pytest.fixture(scope="module")
def render_env(service) -> dict[str, dict]:
    return {entry["key"]: entry for entry in service["envVars"]}


@pytest.fixture(scope="module")
def lite_example() -> dict[str, str]:
    values = {}
    for line in repo_file(".env.lite.example").read_text().splitlines():
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


def test_render_never_allows_fake_providers(render_env, lite_example):
    assert "ALLOW_FAKE_PROVIDERS_IN_PRODUCTION" not in render_env
    assert "ALLOW_FAKE_PROVIDERS_IN_PRODUCTION" not in lite_example


def test_fake_providers_in_production_need_the_explicit_opt_in(lite_example):
    fake = filled(lite_example, EMBEDDING_PROVIDER="fake", LLM_PROVIDER="fake")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **fake)
    settings = Settings(_env_file=None, ALLOW_FAKE_PROVIDERS_IN_PRODUCTION="true", **fake)
    assert settings.LLM_PROVIDER == "fake"
    # The opt-in covers only the providers: a weak key is still refused.
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(
            _env_file=None,
            ALLOW_FAKE_PROVIDERS_IN_PRODUCTION="true",
            **{**fake, "SECRET_KEY": "change-me"},
        )


@pytest.fixture(scope="module")
def lite_compose() -> dict:
    return yaml.safe_load(repo_file("docker-compose.lite.yml").read_text())


def test_local_lite_stack_mirrors_production(lite_compose):
    services = lite_compose["services"]
    assert "worker" not in services  # no Celery worker: inline ingestion
    api = services["api"]
    assert api["build"]["target"] == "prod"
    assert api["mem_limit"] == api["memswap_limit"] == "512m"
    env = api["environment"]
    expected = {
        "APP_ENV": "production",
        "INGESTION_MODE": "inline",
        "STORAGE_BACKEND": "s3",
        "REGISTRATION_MODE": "invite",
        "DOCS_ENABLED": "false",
        "DB_PGBOUNCER": "true",
    }
    for key, value in expected.items():
        assert env[key] == value, key
    assert env["S3_ENDPOINT_URL"] == "http://s3:9000"
    assert {"s3", "bucket", "postgres", "pgbouncer", "redis"} <= set(services)
    # Secrets come from the generated env file, not the compose file.
    assert env["SECRET_KEY"].startswith("${LITE_SECRET_KEY:?")
    assert env["INVITE_CODE"].startswith("${LITE_INVITE_CODE:?")
    # Every published port is bound to this machine only.
    for service in services.values():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), port


def test_render_trusts_no_forwarded_for_entry(render_env, lite_example):
    """Measured on Render: with 1, a client's fake X-Forwarded-For entry was
    trusted and per-IP limits never applied (docs/HARDENING.md, section 6)."""
    assert render_env["TRUSTED_PROXY_COUNT"]["value"] == "0"
    assert lite_example["TRUSTED_PROXY_COUNT"] == "0"


def test_render_serves_the_demo_page_with_the_hint_set_in_the_dashboard(render_env, lite_example):
    assert render_env["DEMO_UI_ENABLED"]["value"] == "true"
    # Shown to every visitor, but it may hold the demo login: never committed.
    assert render_env["DEMO_UI_HINT"] == {"key": "DEMO_UI_HINT", "sync": False}
    assert lite_example["DEMO_UI_ENABLED"] == "true"
    assert lite_example["DEMO_UI_HINT"] == ""
