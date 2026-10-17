from functools import lru_cache

from app.core.config import get_settings
from app.models.chunk import EMBEDDING_COLUMN_DIMENSIONS
from app.services.embeddings.base import EmbeddingProvider, Vector
from app.services.embeddings.errors import (
    EmbeddingBusyError,
    EmbeddingError,
    RequestTooLargeError,
    RetryableEmbeddingError,
)
from app.services.embeddings.fake import FakeEmbeddingProvider
from app.services.embeddings.rate_limit import SlidingWindowRateLimiter
from app.services.embeddings.voyage import VoyageEmbeddingProvider

__all__ = [
    "EmbeddingBusyError",
    "EmbeddingError",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "RequestTooLargeError",
    "RetryableEmbeddingError",
    "Vector",
    "VoyageEmbeddingProvider",
    "get_embedding_provider",
    "get_rate_limiter",
    "reset_rate_limiters",
]

FAKE_MODEL_NAME = "fake-sha256"


@lru_cache
def _limiter(
    token_budget: int, request_budget: int, reserve_tokens: int, reserve_requests: int
) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(token_budget, request_budget, reserve_tokens, reserve_requests)


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """The one limiter for this process. A new provider object is built for
    every request, but they must all draw on the SAME 60-second log, or each
    request would believe the whole budget was free."""
    settings = get_settings()
    return _limiter(
        settings.embedding_token_budget,
        settings.embedding_request_budget,
        settings.embedding_query_reserve_tokens,
        settings.embedding_query_reserve_requests,
    )


def reset_rate_limiters() -> None:
    """Forget what has been sent (for tests)."""
    _limiter.cache_clear()


def get_embedding_provider() -> EmbeddingProvider:
    """The provider selected by EMBEDDING_PROVIDER. Also a FastAPI dependency,
    so tests can swap in their own."""
    settings = get_settings()
    if settings.EMBEDDING_DIMENSIONS != EMBEDDING_COLUMN_DIMENSIONS:
        raise EmbeddingError(
            "Embedding settings don't match the database (EMBEDDING_DIMENSIONS is "
            f"{settings.EMBEDDING_DIMENSIONS}, the column holds {EMBEDDING_COLUMN_DIMENSIONS})"
        )
    common = {
        "model": settings.EMBEDDING_MODEL,
        "dimensions": settings.EMBEDDING_DIMENSIONS,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "max_retries": settings.EMBEDDING_MAX_RETRIES,
        "retry_base_seconds": settings.EMBEDDING_RETRY_BASE_SECONDS,
        "retry_max_seconds": settings.EMBEDDING_RETRY_MAX_SECONDS,
    }
    if settings.EMBEDDING_PROVIDER == "fake":
        # Its own model name, never the real one: fake vectors stored during
        # development must not be mistaken for real ones after switching to
        # the real provider (search compares only vectors from the same model).
        # No rate limiter either: it makes no requests, and throttling local
        # development to the real account's limits would only slow it down.
        return FakeEmbeddingProvider(**{**common, "model": FAKE_MODEL_NAME})
    return VoyageEmbeddingProvider(
        api_key=settings.EMBEDDING_API_KEY,
        timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
        limiter=get_rate_limiter(),
        **common,
    )
