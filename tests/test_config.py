import pytest
from pydantic import ValidationError

from app.core.config import Settings

BASE = {"DATABASE_URL": "postgresql+asyncpg://x", "REDIS_URL": "redis://x", "SECRET_KEY": "k"}


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **BASE, **overrides)


def test_default_budgets_follow_the_documented_limits():
    settings = make()
    assert settings.embedding_token_budget == 5000  # 10,000 x 0.5
    assert settings.embedding_request_budget == 3  # the account's limit, shared by all
    assert settings.embedding_ingest_request_budget == 2  # 3 x 0.67, rounded down
    assert settings.embedding_query_reserve_tokens == 250  # 5%
    assert settings.embedding_document_token_budget == 4750
    assert settings.embedding_min_retry_wait == 20  # 60 s / 3 requests
    assert settings.embedding_query_reserve_requests == 1  # kept free for searches
    assert settings.EMBEDDING_MAX_RETRIES == 6


def test_safety_factor_applies_to_tokens_only():
    settings = make(EMBEDDING_RATE_SAFETY=0.2, EMBEDDING_RPM_LIMIT=10)
    assert settings.embedding_token_budget == 2000
    assert settings.embedding_request_budget == 10  # not 10 x 0.2
    assert settings.embedding_ingest_request_budget == 6  # 10 x 0.67


def test_min_retry_wait_can_be_set_explicitly():
    assert make(EMBEDDING_MIN_RETRY_WAIT_SECONDS=45).embedding_min_retry_wait == 45
    assert make(EMBEDDING_RPM_LIMIT=60).embedding_min_retry_wait == 1


def test_empty_min_retry_wait_in_env_file_means_computed(tmp_path):
    env = tmp_path / ".env"
    env.write_text("EMBEDDING_MIN_RETRY_WAIT_SECONDS=\n")
    settings = Settings(_env_file=env, **BASE)
    assert settings.embedding_min_retry_wait == 20


def test_llm_and_rag_settings_defaults():
    settings = make()
    assert settings.LLM_PROVIDER == "gemini"  # real by default: fail loudly if unconfigured
    assert settings.LLM_MODEL == "gemini-3.5-flash-lite"
    assert (settings.LLM_MAX_OUTPUT_TOKENS, settings.LLM_TIMEOUT_SECONDS) == (600, 30)
    assert settings.LLM_RPM_LIMIT == 10
    assert settings.LLM_TEMPERATURE == 1.0
    assert (settings.RAG_TOP_K, settings.RAG_MAX_DISTANCE) == (8, 0.72)
    assert settings.RAG_CONTEXT_TOKEN_BUDGET == 3000
    assert settings.ASK_MAX_QUESTION_CHARS == 1000
    assert "SecretStr" in repr(settings.LLM_API_KEY) or "*" in str(settings.LLM_API_KEY)


def test_conversation_settings_defaults():
    settings = make()
    assert (settings.RAG_HISTORY_MAX_TURNS, settings.RAG_HISTORY_TOKEN_BUDGET) == (6, 1000)
    assert settings.RAG_REWRITE_TIMEOUT_SECONDS == 10
    assert settings.STREAM_HEARTBEAT_SECONDS == 15


def test_job_settings_defaults():
    settings = make()
    assert settings.MAX_JOB_ATTEMPTS == 4
    assert settings.SWEEP_PENDING_MINUTES == 3
    assert settings.STALE_PROCESSING_MINUTES == 10
    assert settings.EMBEDDING_INGEST_RPM_SHARE == 0.67


def test_request_budget_is_never_below_one():
    settings = make(EMBEDDING_RPM_LIMIT=1)
    assert settings.embedding_request_budget == 1
    assert settings.embedding_ingest_request_budget == 1
    assert settings.embedding_query_reserve_requests == 0  # nothing to spare


def test_chunk_larger_than_a_batch_is_refused_at_startup():
    with pytest.raises(ValidationError, match="CHUNK_TOKENS"):
        make(EMBEDDING_TPM_LIMIT=500, CHUNK_TOKENS=500)


def test_chunk_that_fits_is_accepted():
    make(EMBEDDING_TPM_LIMIT=2000, CHUNK_TOKENS=500)  # 2000 x 0.5 - 5% = 950


def test_retry_max_below_base_is_refused():
    with pytest.raises(ValidationError, match="RETRY_MAX"):
        make(EMBEDDING_RETRY_BASE_SECONDS=10, EMBEDDING_RETRY_MAX_SECONDS=5)
