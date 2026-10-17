import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.services.chunking import count_tokens
from app.services.embeddings.errors import (
    EmbeddingError,
    RequestTooLargeError,
    RetryableEmbeddingError,
)

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
        token_budget: int | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = min(batch_size, self.max_batch_texts)
        self.max_retries = max_retries
        self._sleep = sleep  # replaceable so tests don't actually wait
        # Most tokens we put in one request. Our tokenizer only approximates the
        # provider's, so the provider's own cap counts at half.
        caps = [token_budget, self.max_batch_tokens // 2 if self.max_batch_tokens else None]
        self.token_budget = min((cap for cap in caps if cap), default=None)

    async def embed_documents(
        self, texts: Sequence[str], token_counts: Sequence[int] | None = None
    ) -> list[Vector]:
        """Embed texts for storage, returning one vector per text, in order.

        token_counts: how many tokens each text has, if the caller already knows
        (counting again is wasted CPU, and this runs on the event loop).
        """
        _reject_blank(texts)
        batches = self._plan_batches(texts, token_counts)  # fails before any request is sent
        vectors: list[Vector] = []
        total_tokens = requests = 0
        for batch, _estimated_tokens in batches:
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

    def _plan_batches(
        self, texts: Sequence[str], token_counts: Sequence[int] | None
    ) -> list[tuple[list[str], int]]:
        """Group texts into requests of (texts, estimated tokens).

        A batch ends when adding the next text would push it over the token
        budget or the batch size. A text that alone exceeds the token budget can
        never be sent, however often we retry, so it is refused up front,
        before any quota is spent on the texts ahead of it.
        """
        if token_counts is not None and len(token_counts) != len(texts):
            raise ValueError("token_counts must have one entry per text")
        budget = self.token_budget
        if budget is None:
            counts = [0] * len(texts)  # no token limit to respect: don't bother counting
        else:
            if token_counts is None:
                counts = [count_tokens(text) for text in texts]
            else:
                counts = list(token_counts)
            largest = max(counts, default=0)
            if largest > budget:
                raise RequestTooLargeError(
                    f"A chunk is too large to embed ({largest} tokens; at most {budget} "
                    "per request). Try a smaller CHUNK_TOKENS."
                )

        batches: list[tuple[list[str], int]] = []
        batch: list[str] = []
        batch_tokens = 0
        for text, tokens in zip(texts, counts, strict=True):
            full = len(batch) == self.batch_size
            over_budget = budget is not None and batch_tokens + tokens > budget
            if batch and (full or over_budget):
                batches.append((batch, batch_tokens))
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += tokens
        if batch:
            batches.append((batch, batch_tokens))
        return batches

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
