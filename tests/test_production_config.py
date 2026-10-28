import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings

STRONG_SECRET = "q8Zr3TnV0xLw5Kd1Ys7Hm2Pf9Bc4Jg6UaE"


def production(**overrides) -> Settings:
    """Settings for production, from explicit values only (never .env)."""
    values = {
        "DATABASE_URL": "postgresql+asyncpg://u:p@db/documind",
        "REDIS_URL": "redis://redis:6379/0",
        "SECRET_KEY": STRONG_SECRET,
        "APP_ENV": "production",
        "DOCS_ENABLED": False,
        "EMBEDDING_PROVIDER": "voyage",
        "LLM_PROVIDER": "gemini",
        "REGISTRATION_MODE": "invite",
        "INVITE_CODE": SecretStr("a-long-invite-code-2026"),
        "CORS_ORIGINS": "https://app.example.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_a_good_production_config_starts():
    settings = production()
    assert settings.APP_ENV == "production"


@pytest.mark.parametrize(
    "secret", ["change-me", "CHANGE-ME", "secret", "your-secret-key", "short-but-random-9"]
)
def test_weak_secret_key_is_refused(secret):
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        production(SECRET_KEY=secret)


def test_repetitive_secret_key_is_refused():
    with pytest.raises(ValidationError, match="too repetitive"):
        production(SECRET_KEY="ab" * 30)


def test_the_example_files_secret_is_refused():
    """The value shipped in .env.example must never pass."""
    from tests.repo_files import repo_file

    example = repo_file(".env.example")
    lines = example.read_text().splitlines()
    line = next(text for text in lines if text.startswith("SECRET_KEY="))
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        production(SECRET_KEY=line.split("=", 1)[1])


def test_cors_wildcard_is_refused():
    with pytest.raises(ValidationError, match="not allowed"):
        production(CORS_ORIGINS="*")


def test_docs_need_an_explicit_opt_in():
    with pytest.raises(ValidationError, match="API docs are enabled"):
        production(DOCS_ENABLED=True)
    assert production(DOCS_ENABLED=True, ALLOW_DOCS_IN_PRODUCTION=True).DOCS_ENABLED


def test_rate_limits_cannot_be_switched_off():
    with pytest.raises(ValidationError, match="RATE_LIMIT_ENABLED"):
        production(RATE_LIMIT_ENABLED=False)


def test_fake_providers_are_refused():
    with pytest.raises(ValidationError, match="fake"):
        production(LLM_PROVIDER="fake")
    with pytest.raises(ValidationError, match="fake"):
        production(EMBEDDING_PROVIDER="fake")


def test_short_invite_code_is_refused():
    with pytest.raises(ValidationError, match="INVITE_CODE"):
        production(INVITE_CODE=SecretStr("abc123"))


def test_all_problems_are_listed_at_once():
    with pytest.raises(ValidationError) as caught:
        production(SECRET_KEY="change-me", DOCS_ENABLED=True, LLM_PROVIDER="fake")
    message = str(caught.value)
    for part in ["SECRET_KEY", "API docs", "fake"]:
        assert part in message


def test_no_input_value_is_printed_in_the_error():
    """The startup error is logged. Pydantic would print the input values
    (shortened, so whether a secret shows depends on where it lands), which
    can include SECRET_KEY and API keys. With hide_input_in_errors none are."""
    with pytest.raises(ValidationError) as caught:
        production(SECRET_KEY="my-own-weak-key-123", LLM_API_KEY="AIza-leak-me")
    message = str(caught.value)
    assert "input_value" not in message
    assert "my-own-weak-key-123" not in message
    assert "AIza-leak-me" not in message


def test_development_allows_a_weak_secret():
    settings = production(APP_ENV="development", SECRET_KEY="change-me", DOCS_ENABLED=True)
    assert settings.secret_key_problem() is not None
