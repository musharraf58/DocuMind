import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.services.chunking import count_tokens
from app.services.embeddings.errors import (
    EmbeddingBusyError,
    EmbeddingError,
    RequestTooLargeError,
    RetryableEmbeddingError,
)
from app.services.embeddings.rate_limit import Kind, RateLimiter, estimate_seconds

logger = logging.getLogger(__name__)

Vector = list[float]
InputType = Literal["document", "query"]
# Called before every attempt to send: "I'm still alive", for callers that need
# to prove a long run hasn't been abandoned.
Heartbeat = Callable[[], Awaitable[None]]

# The longest Retry-After we will sit through. Beyond this the work is failed
# with a "busy, retry in N seconds" message rather than tying a request up.
RETRY_AFTER_CEILING_SECONDS = 120.0


@dataclass(frozen=True)
class BatchResult:
    vectors: list[Vector]
    total_tokens: int


class EmbeddingProvider(ABC):
    """Turns text into vectors.

    Subclasses implement _embed_batch (one request). This base class does the
    rest, identically for every provider: refuses blank text, splits work
    into batches within the token budget, waits for the rate limiter before
    every request, retries transient failures (honouring Retry-After), and
    checks that every vector has the expected size.
    """

    # The provider's own per-request limits. Subclasses override.
    max_batch_texts: int = 1000
    max_batch_tokens: int | None = None

    def __init__(
        self,
        model: str,
        dimensions: int,
        batch_size: int = 64,
        max_retries: int = 6,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        token_budget: int | None = None,
        limiter: RateLimiter | None = None,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
        query_max_wait: float | None = 20.0,
        min_retry_wait: float = 20.0,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = min(batch_size, self.max_batch_texts)
        self.max_retries = max_retries
        self.limiter = limiter
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.query_max_wait = query_max_wait
        self.min_retry_wait = min_retry_wait
        self._sleep = sleep  # replaceable so tests don't actually wait
        # Most tokens we put in one request. Our tokenizer only approximates the
        # provider's, so the provider's own cap counts at half.
        caps = [
            token_budget,
            limiter.document_token_budget if limiter else None,
            self.max_batch_tokens // 2 if self.max_batch_tokens else None,
        ]
        self.token_budget = min((cap for cap in caps if cap), default=None)

    async def embed_documents(
        self,
        texts: Sequence[str],
        token_counts: Sequence[int] | None = None,
        heartbeat: Heartbeat | None = None,
    ) -> list[Vector]:
        """Embed texts for storage, returning one vector per text, in order.

        token_counts: how many tokens each text has, if the caller already knows
        (counting again is wasted CPU, and this runs on the event loop).
        heartbeat: awaited before each attempt, so a caller can show the run
        is still alive while the rate limiter makes it wait for minutes.
        """
        return await self._embed_many(texts, "document", "chunks", token_counts, heartbeat)

    async def embed_queries(
        self, texts: Sequence[str], heartbeat: Heartbeat | None = None
    ) -> list[Vector]:
        """Embed many queries at once, as queries, in as few requests as
        possible. For offline work such as the evaluation: unlike embed_query
        nobody is waiting, so it waits for the rate limiter as long as needed.
        """
        return await self._embed_many(texts, "query", "queries", None, heartbeat)

    async def _embed_many(
        self,
        texts: Sequence[str],
        input_type: InputType,
        noun: str,
        token_counts: Sequence[int] | None,
        heartbeat: Heartbeat | None,
    ) -> list[Vector]:
        _reject_blank(texts)
        batches = self._plan_batches(texts, token_counts)  # fails before any request is sent
        estimated_total = sum(tokens for _, tokens in batches)
        if self.limiter is not None and batches:
            seconds = estimate_seconds(estimated_total, self.limiter.document_token_budget)
            logger.info(
                "Embedding %d %s (~%d tokens) in %d batches; at the current rate limit "
                "this takes %s",
                len(texts),
                noun,
                estimated_total,
                len(batches),
                f"about {round(seconds / 60)} min" if seconds >= 30 else "under a minute",
            )

        vectors: list[Vector] = []
        reported_total = 0
        waited_total = 0.0
        for number, (batch, estimated) in enumerate(batches, start=1):
            # Bulk work of either input type draws on the bulk ("document")
            # share of the rate limit, leaving the reserve for live searches.
            result, waited = await self._send(
                batch, input_type, estimated, "document", heartbeat=heartbeat
            )
            vectors.extend(result.vectors)
            reported_total += result.total_tokens
            waited_total += waited
            logger.info(
                "embedded %d/%d %s (batch %d/%d: ~%d tokens estimated, %s, waited %.1fs)",
                len(vectors),
                len(texts),
                noun,
                number,
                len(batches),
                estimated,
                _compare(estimated, result.total_tokens),
                waited,
            )
            if self.token_budget and result.total_tokens > self.token_budget:
                # We planned this batch to fit the budget, but by the provider's
                # count it didn't: our estimate undercounts by more than the
                # safety factor allows for.
                logger.warning(
                    "The provider counted %d tokens for a batch estimated at %d, more than "
                    "the %d-token batch budget; consider lowering EMBEDDING_RATE_SAFETY",
                    result.total_tokens,
                    estimated,
                    self.token_budget,
                )
        logger.info(
            "Embedding finished: %d %s, %d requests, waited %.0fs; tokens ~%d estimated, "
            "%s (model %s)",
            len(texts),
            noun,
            len(batches),
            waited_total,
            sum(tokens for _, tokens in batches),
            _compare(sum(tokens for _, tokens in batches), reported_total),
            self.model,
        )
        return vectors

    async def embed_query(self, text: str) -> Vector:
        """Embed a search query. Providers that tell queries and documents apart
        do better when told which is which.

        Someone is waiting on the answer, so unlike document embedding this never
        waits for long: if getting through (rate limiter plus retries) would
        take longer than query_max_wait seconds in total, it raises
        EmbeddingBusyError saying when to try again.
        """
        _reject_blank([text])
        result, _waited = await self._send(
            [text], "query", count_tokens(text), "query", max_wait=self.query_max_wait
        )
        [vector] = result.vectors
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

    async def _send(
        self,
        texts: list[str],
        input_type: InputType,
        estimated_tokens: int,
        kind: Kind,
        max_wait: float | None = None,
        heartbeat: Heartbeat | None = None,
    ) -> tuple[BatchResult, float]:
        """Send one batch: wait for the rate limiter, send, retry on transient
        failures. Returns the result and the seconds spent waiting.

        Every attempt, including retries, goes through the limiter, because
        the provider counts a failed request against the quota too.
        max_wait bounds the total waiting; past it EmbeddingBusyError is raised.
        """
        attempt = 0
        waited = 0.0
        reservation = None
        while True:
            if heartbeat is not None:
                await heartbeat()
            if self.limiter is not None:
                remaining = None if max_wait is None else max(0.0, max_wait - waited)
                reservation = await self.limiter.acquire(estimated_tokens, kind, max_wait=remaining)
                waited += reservation.waited
            try:
                result = await self._embed_batch(texts, input_type)
                break
            except RetryableEmbeddingError as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(attempt, exc.retry_after, exc.rate_limited)
                attempt += 1
                if exc.rate_limited and self.limiter is not None:
                    # Hold every caller, not just this one, until the provider is
                    # ready again; the next acquire() does the actual waiting,
                    # which can be longer than `delay` if the window is full.
                    await self.limiter.penalize(delay)
                    real_wait = await self.limiter.time_until_available(estimated_tokens, kind)
                    logger.warning(
                        "Embedding request failed (%s); retry %d/%d in about %.0fs%s",
                        exc,
                        attempt,
                        self.max_retries,
                        real_wait,
                        " (rate-limit window)" if real_wait > delay + 1 else "",
                    )
                    continue
                logger.warning(
                    "Embedding request failed (%s); retry %d/%d in about %.0fs",
                    exc,
                    attempt,
                    self.max_retries,
                    delay,
                )
                if max_wait is not None and waited + delay > max_wait:
                    raise EmbeddingBusyError(retry_after=delay) from exc
                await self._sleep(delay)
                waited += delay
        if reservation is not None:
            await reservation.correct(result.total_tokens)
        self._check_vectors(result.vectors, len(texts))
        return result, waited

    def _retry_delay(
        self, attempt: int, retry_after: float | None, rate_limited: bool = False
    ) -> float:
        """How long to wait before retry number attempt+1.

        After a 429 ("too many requests"), retrying within seconds only spends
        more of the per-minute allowance and keeps the window full. So the wait
        is at least the provider's Retry-After and at least min_retry_wait
        (by default the spacing at which requests stay under the limit),
        doubling on each further 429 up to the cap, plus 0-25% jitter added on
        top, never subtracted.

        Other failures (server errors, timeouts) say nothing about our quota:
        Retry-After if given, else short exponential backoff with jitter.
        Either way, many clients that failed together don't all retry at the
        same instant.
        """
        if retry_after is not None and retry_after > RETRY_AFTER_CEILING_SECONDS:
            raise EmbeddingBusyError(retry_after=retry_after)
        if rate_limited:
            growing = min(self.retry_max_seconds, self.min_retry_wait * 2**attempt)
            wait = max(self.min_retry_wait, growing, retry_after or 0.0)
            return wait + random.uniform(0, 0.25 * wait)
        if retry_after is not None:
            return retry_after
        delay = min(self.retry_max_seconds, self.retry_base_seconds * 2**attempt)
        return delay * random.uniform(0.5, 1.0)

    def _check_vectors(self, vectors: list[Vector], expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingError("Embedding service returned an unexpected response")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise EmbeddingError(
                f"Embedding service returned vectors of the wrong size "
                f"(expected {self.dimensions} dimensions)"
            )


def _compare(estimated: int, reported: int) -> str:
    """'7391 reported by the provider (+12%)', for tuning EMBEDDING_RATE_SAFETY."""
    if reported <= 0 or estimated <= 0:
        return "no count reported by the provider"
    return f"{reported} reported by the provider ({(reported - estimated) / estimated:+.0%})"


def _reject_blank(texts: Sequence[str]) -> None:
    # Callers filter blank text out first; reaching here with any is a bug,
    # and sending it would waste a request or get it rejected.
    if any(not text.strip() for text in texts):
        raise ValueError("Cannot embed empty or whitespace-only text")
