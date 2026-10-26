"""Turning provider failures into clean HTTP responses: a status, a short
message, and a retry hint where one makes sense. Never a stack trace, never
the API key or the provider's own response text (our error messages are
fixed strings written for this)."""

import math

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.errors import error_body
from app.services.embeddings import EmbeddingBusyError, EmbeddingError, RetryableEmbeddingError
from app.services.llm import LLMBudgetExhausted, LLMBusyError, LLMError, RetryableLLMError
from app.services.quotas import capacity_reached
from app.services.storage import StorageError

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


async def storage_unavailable(request: Request, exc: StorageError) -> JSONResponse:
    """The file storage (bucket) failed: 503 with its fixed, safe message.
    What went wrong has been logged without credentials (s3_storage.py)."""
    return JSONResponse(
        error_body(503, str(exc), "storage_unavailable"),
        status_code=503,
        headers={"Retry-After": str(DEFAULT_RETRY_AFTER_SECONDS)},
    )
