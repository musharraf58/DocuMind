import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.services.chunking import count_tokens
from app.services.embeddings.errors import EmbeddingBusyError
from app.services.embeddings.rate_limit import RateLimiter
from app.services.llm.errors import LLMBusyError, RetryableLLMError

logger = logging.getLogger(__name__)

# Someone is waiting for the answer: never sit in rate-limit waits and retry
# backoff for longer than this in total; answer "busy, retry in N s" instead.
INTERACTIVE_MAX_WAIT_SECONDS = 20.0


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    """Generates text. Subclasses implement _generate (one request); this base
    class adds the shared behaviour: waiting for the rate limiter, retrying
    transient failures with backoff and jitter (honouring Retry-After), and
    giving up quickly with LLMBusyError when the wait would be too long."""

    def __init__(
        self,
        model: str,
        limiter: RateLimiter | None = None,
        max_retries: int = 2,
        max_wait: float = INTERACTIVE_MAX_WAIT_SECONDS,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self.limiter = limiter
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._sleep = sleep

    async def generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        """Generate a reply to `user` following the `system` instructions."""
        estimated_tokens = count_tokens(system) + count_tokens(user) + max_tokens
        waited = 0.0
        attempt = 0
        while True:
            if self.limiter is not None:
                try:
                    reservation = await self.limiter.acquire(
                        estimated_tokens, "query", max_wait=max(0.0, self.max_wait - waited)
                    )
                except EmbeddingBusyError as exc:  # the limiter's "would wait too long"
                    raise LLMBusyError(exc.retry_after) from exc
                waited += reservation.waited
            try:
                return await self._generate(system, user, max_tokens)
            except RetryableLLMError as exc:
                if attempt >= self.max_retries:
                    if exc.rate_limited:
                        raise LLMBusyError(self._delay(attempt, exc.retry_after)) from exc
                    raise
                delay = self._delay(attempt, exc.retry_after)
                if waited + delay > self.max_wait:
                    if exc.rate_limited:
                        raise LLMBusyError(delay) from exc
                    raise
                attempt += 1
                logger.warning(
                    "LLM request failed (%s); retry %d/%d in about %.0fs",
                    exc,
                    attempt,
                    self.max_retries,
                    delay,
                )
                if exc.rate_limited and self.limiter is not None:
                    await self.limiter.penalize(delay)  # the next acquire() waits it out
                else:
                    await self._sleep(delay)
                    waited += delay

    @abstractmethod
    async def _generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        """Send ONE request. Raise RetryableLLMError for failures worth retrying,
        LLMRefusalError if the provider declines, LLMError for anything else."""

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter; never sooner than the provider asked."""
        delay = min(self.retry_max_seconds, self.retry_base_seconds * 2**attempt)
        delay *= random.uniform(0.5, 1.0)
        return max(delay, retry_after or 0.0)
