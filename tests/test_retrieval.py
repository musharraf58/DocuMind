import uuid

import pytest

from app.api.errors import provider_error
from app.services.embeddings import EmbeddingBusyError, EmbeddingError, RetryableEmbeddingError
from app.services.llm import LLMBusyError, LLMError, RetryableLLMError
from app.services.retrieval import RetrievedChunk, filter_relevant


def chunk(distance: float) -> RetrievedChunk:
    return RetrievedChunk(uuid.uuid4(), uuid.uuid4(), "a.txt", "text", None, None, distance)


def test_filter_relevant_keeps_chunks_at_or_below_the_threshold():
    chunks = [chunk(0.2), chunk(0.72), chunk(0.7201), chunk(1.1)]
    assert [c.distance for c in filter_relevant(chunks, 0.72)] == [0.2, 0.72]
    assert filter_relevant(chunks, 0.1) == []


@pytest.mark.parametrize(
    ("error", "status", "retry_after"),
    [
        (EmbeddingBusyError(retry_after=12.3), 429, "13"),
        (LLMBusyError(retry_after=40), 429, "40"),
        (RetryableLLMError("rate limited", rate_limited=True), 429, "30"),
        (RetryableEmbeddingError("rate limited", retry_after=5, rate_limited=True), 429, "5"),
        (RetryableLLMError("The answer service timed out"), 503, None),
        (RetryableEmbeddingError("Embedding service is unavailable"), 503, None),
        (LLMError("The answer service rejected the API key"), 503, None),
        (EmbeddingError("Embedding service rejected the API key"), 503, None),
    ],
)
def test_provider_errors_map_to_clean_statuses(error, status, retry_after):
    http = provider_error(error)
    assert http.status_code == status
    assert http.detail == str(error)
    assert (http.headers or {}).get("Retry-After") == retry_after
