from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.redis import get_redis
from app.services.embeddings.redis_rate_limit import RedisRateLimiter
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage
from app.services.llm.budget import GlobalLLMBudget
from app.services.llm.errors import (
    LLMBudgetExhausted,
    LLMBusyError,
    LLMError,
    LLMRefusalError,
    RetryableLLMError,
)
from app.services.llm.fake import FakeLLMProvider
from app.services.llm.gemini import GeminiLLMProvider

__all__ = [
    "FakeLLMProvider",
    "GeminiLLMProvider",
    "GlobalLLMBudget",
    "LLMBudgetExhausted",
    "LLMBusyError",
    "LLMError",
    "LLMProvider",
    "LLMRefusalError",
    "LLMResult",
    "LLMUsage",
    "RetryableLLMError",
    "get_llm_provider",
]

LLM_RATE_LIMIT_KEY = "documind:llm-rate-limit"
# Only requests are limited: a question is ~4,000 tokens, so even at the
# request limit the token rate stays far below any published token limit.
UNLIMITED_TOKENS = 10**9


def get_llm_provider(redis: Redis | None = None) -> LLMProvider:
    """The provider selected by LLM_PROVIDER. Also a FastAPI dependency, so
    tests can swap in their own."""
    settings = get_settings()
    if settings.LLM_PROVIDER == "fake":
        return FakeLLMProvider()
    redis = redis if redis is not None else get_redis()
    limiter = RedisRateLimiter(
        redis, UNLIMITED_TOKENS, settings.LLM_RPM_LIMIT, key_prefix=LLM_RATE_LIMIT_KEY
    )
    return GeminiLLMProvider(
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        temperature=settings.LLM_TEMPERATURE,
        model=settings.LLM_MODEL,
        limiter=limiter,
        budget=GlobalLLMBudget(redis),
    )
