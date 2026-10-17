import pytest
from pydantic import ValidationError

from app.core.config import Settings

BASE = {"DATABASE_URL": "postgresql+asyncpg://x", "REDIS_URL": "redis://x", "SECRET_KEY": "k"}


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **BASE, **overrides)


def test_default_budgets_follow_the_documented_limits():
    settings = make()
    assert settings.embedding_token_budget == 8000  # 10,000 x 0.8
    assert settings.embedding_request_budget == 2  # 3 x 0.8 = 2.4, rounded down
    assert settings.embedding_query_reserve_tokens == 400  # 5%
    assert settings.embedding_document_token_budget == 7600
    assert settings.embedding_query_reserve_requests == 1
    assert settings.EMBEDDING_MAX_RETRIES == 6


def test_request_budget_is_never_below_one():
    settings = make(EMBEDDING_RPM_LIMIT=1)
    assert settings.embedding_request_budget == 1
    assert settings.embedding_query_reserve_requests == 0  # nothing to spare


def test_chunk_larger_than_a_batch_is_refused_at_startup():
    with pytest.raises(ValidationError, match="CHUNK_TOKENS"):
        make(EMBEDDING_TPM_LIMIT=500, CHUNK_TOKENS=500)


def test_chunk_that_fits_is_accepted():
    make(EMBEDDING_TPM_LIMIT=1000, CHUNK_TOKENS=500)


def test_retry_max_below_base_is_refused():
    with pytest.raises(ValidationError, match="RETRY_MAX"):
        make(EMBEDDING_RETRY_BASE_SECONDS=10, EMBEDDING_RETRY_MAX_SECONDS=5)
