import math


class EmbeddingError(Exception):
    """Embedding failed. The message is stored on documents and shown to users,
    so it must be short and must never contain the API key or anything the
    provider sent back."""


class RetryableEmbeddingError(EmbeddingError):
    """A failure that may go away if we wait and try again (rate limit, server
    error, timeout). retry_after is the provider's own hint, in seconds."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class EmbeddingBusyError(EmbeddingError):
    """There is no embedding capacity right now and waiting would take longer
    than the caller is willing to wait. Not retried internally: the caller
    should tell its own client to come back in retry_after seconds."""

    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        seconds = max(1, math.ceil(retry_after))
        super().__init__(f"Embedding service is busy; retry in about {seconds} seconds")


class RequestTooLargeError(EmbeddingError):
    """One request is bigger than the per-minute allowance, so no amount of
    waiting or retrying could ever get it through."""
