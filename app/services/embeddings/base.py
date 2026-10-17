import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from app.services.chunking import count_tokens
from app.services.embeddings.errors import EmbeddingError, RetryableEmbeddingError

logger = logging.getLogger(__name__)

Vector = list[float]
InputType = Literal["document", "query"]

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
MAX_RETRY_AFTER_SECONDS = 60.0


@dataclass(frozen=True)
class BatchResult:
    vectors: list[Vector]
    total_tokens: int


class EmbeddingProvider(ABC):
    """Turns text into vectors.

    Subclasses implement _embed_batch (one request). This base class does the
    rest, identically for every provider: refuses blank text, splits work
    into batches within the provider's limits, retries transient failures
    with backoff, and checks that every vector has the expected size.
    """

    # The provider's own per-request limits. Subclasses override.
    max_batch_texts: int = 1000
    max_batch_tokens: int | None = None

    def __init__(
        self,
        model: str,
        dimensions: int,
        batch_size: int = 64,
        max_retries: int = 4,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = min(batch_size, self.max_batch_texts)
        self.max_retries = max_retries
        self._sleep = sleep  # replaceable so tests don't actually wait

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts for storage, returning one vector per text, in order."""
        _reject_blank(texts)
        vectors: list[Vector] = []
        total_tokens = requests = 0
        for batch in self._batches(texts):
            result = await self._embed_with_retry(batch, "document")
            vectors.extend(result.vectors)
            total_tokens += result.total_tokens
            requests += 1
        logger.info(
            "Embedded %d texts in %d requests (%d tokens, model %s)",
            len(texts),
            requests,
            total_tokens,
            self.model,
        )
        return vectors

    async def embed_query(self, text: str) -> Vector:
        """Embed a search query. Providers that tell queries and documents apart
        do better when told which is which."""
        _reject_blank([text])
        [vector] = (await self._embed_with_retry([text], "query")).vectors
        return vector

    @abstractmethod
    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        """Send ONE request. Raise RetryableEmbeddingError for failures worth
        retrying and EmbeddingError for everything else."""

    def _batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        """Group texts so each request stays within the batch size and the
        provider's token limit. Our tokenizer differs from the provider's, so
        we use only half the token limit as a safety margin."""
        token_budget = self.max_batch_tokens // 2 if self.max_batch_tokens else None
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            tokens = count_tokens(text) if token_budget else 0
            too_many_tokens = token_budget is not None and batch_tokens + tokens > token_budget
            if batch and (len(batch) == self.batch_size or too_many_tokens):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += tokens
        if batch:
            yield batch

    async def _embed_with_retry(self, texts: list[str], input_type: InputType) -> BatchResult:
        attempt = 0
        while True:
            try:
                result = await self._embed_batch(texts, input_type)
                break
            except RetryableEmbeddingError as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self._backoff(attempt, exc.retry_after)
                logger.warning(
                    "Embedding request failed (%s); retry %d/%d in %.1fs",
                    exc,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await self._sleep(delay)
                attempt += 1
        self._check_vectors(result.vectors, len(texts))
        return result

    @staticmethod
    def _backoff(attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter: about 1s, 2s, 4s, ... capped at 30s,
        each scaled by a random 50-100% so that many clients failing at once
        don't all retry at the same instant. A provider's Retry-After wins if
        it asks for longer."""
        delay = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt)
        delay *= random.uniform(0.5, 1.0)
        if retry_after is not None:
            delay = max(delay, min(retry_after, MAX_RETRY_AFTER_SECONDS))
        return delay

    def _check_vectors(self, vectors: list[Vector], expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingError("Embedding service returned an unexpected response")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise EmbeddingError(
                f"Embedding service returned vectors of the wrong size "
                f"(expected {self.dimensions} dimensions)"
            )


def _reject_blank(texts: Sequence[str]) -> None:
    # Callers filter blank text out first; reaching here with any is a bug,
    # and sending it would waste a request or get it rejected.
    if any(not text.strip() for text in texts):
        raise ValueError("Cannot embed empty or whitespace-only text")
