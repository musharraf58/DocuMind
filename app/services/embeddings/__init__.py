from app.core.config import get_settings
from app.models.chunk import EMBEDDING_COLUMN_DIMENSIONS
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

FAKE_MODEL_NAME = "fake-sha256"


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
    }
    if settings.EMBEDDING_PROVIDER == "fake":
        # Its own model name, never the real one: fake vectors stored during
        # development must not be mistaken for real ones after switching to
        # the real provider (search compares only vectors from the same model).
        return FakeEmbeddingProvider(**{**common, "model": FAKE_MODEL_NAME})
    return VoyageEmbeddingProvider(
        api_key=settings.EMBEDDING_API_KEY,
        timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
        **common,
    )
