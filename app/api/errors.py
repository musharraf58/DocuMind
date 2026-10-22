"""Turning provider failures into clean HTTP responses: a status, a short
message, and a retry hint where one makes sense. Never a stack trace, never
the API key or the provider's own response text (our error messages are
fixed strings written for this)."""

import math

from fastapi import HTTPException

from app.services.embeddings import EmbeddingBusyError, EmbeddingError, RetryableEmbeddingError
from app.services.llm import LLMBudgetExhausted, LLMBusyError, LLMError, RetryableLLMError
from app.services.quotas import capacity_reached

DEFAULT_RETRY_AFTER_SECONDS = 30


def provider_error(exc: EmbeddingError | LLMError) -> HTTPException:
    """429 when it's a rate limit (with Retry-After), 503 for everything else:
    provider down, timed out, or misconfigured; and 503 "daily capacity
    reached" when our own global LLM budget is used up."""
    if isinstance(exc, LLMBudgetExhausted):
        return capacity_reached(max(1, math.ceil(exc.retry_after)))
    if isinstance(exc, EmbeddingBusyError | LLMBusyError):
        return _too_many_requests(str(exc), exc.retry_after)
    if isinstance(exc, RetryableEmbeddingError | RetryableLLMError) and exc.rate_limited:
        return _too_many_requests(str(exc), exc.retry_after or DEFAULT_RETRY_AFTER_SECONDS)
    return HTTPException(status_code=503, detail=str(exc))


def _too_many_requests(message: str, retry_after: float) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=message,
        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
    )
