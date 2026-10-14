from app.core.config import get_settings
from app.services.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    RetryableEmbeddingError,
    Vector,
)
from app.services.embeddings.fake import FakeEmbeddingProvider
from app.services.embeddings.voyage import VoyageEmbeddingProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "RetryableEmbeddingError",
    "Vector",
    "VoyageEmbeddingProvider",
    "get_embedding_provider",
]


def get_embedding_provider() -> EmbeddingProvider:
    """The provider selected by EMBEDDING_PROVIDER. Also a FastAPI dependency,
    so tests can swap in their own."""
    settings = get_settings()
    common = {
        "model": settings.EMBEDDING_MODEL,
        "dimensions": settings.EMBEDDING_DIMENSIONS,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "max_retries": settings.EMBEDDING_MAX_RETRIES,
    }
    if settings.EMBEDDING_PROVIDER == "fake":
        return FakeEmbeddingProvider(**common)
    return VoyageEmbeddingProvider(
        api_key=settings.EMBEDDING_API_KEY,
        timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
        **common,
    )
