from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.redis import get_redis
from app.models.chunk import EMBEDDING_COLUMN_DIMENSIONS
from app.services.embeddings.base import EmbeddingProvider, Vector
from app.services.embeddings.errors import (
    EmbeddingBusyError,
    EmbeddingError,
    RequestTooLargeError,
    RetryableEmbeddingError,
)
from app.services.embeddings.fake import FakeEmbeddingProvider
from app.services.embeddings.redis_rate_limit import RedisRateLimiter
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
]

FAKE_MODEL_NAME = "fake-sha256"


def make_rate_limiter(redis: Redis) -> RedisRateLimiter:
    """The rate limiter for the embedding account, stored in Redis so that the
    API (searches) and the worker (ingestion) draw on one shared window."""
    settings = get_settings()
    return RedisRateLimiter(
        redis,
        settings.embedding_token_budget,
        settings.embedding_request_budget,
        settings.embedding_query_reserve_tokens,
        settings.embedding_query_reserve_requests,
    )


def get_embedding_provider(redis: Redis | None = None) -> EmbeddingProvider:
    """The provider selected by EMBEDDING_PROVIDER. Also a FastAPI dependency,
    so tests can swap in their own. `redis`: the client for the shared rate
    limiter (default: this process's shared client)."""
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
        "query_max_wait": settings.EMBEDDING_QUERY_MAX_WAIT_SECONDS,
        "min_retry_wait": settings.embedding_min_retry_wait,
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
        limiter=make_rate_limiter(redis if redis is not None else get_redis()),
        **common,
    )
