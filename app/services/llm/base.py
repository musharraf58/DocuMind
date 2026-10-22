import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from app.models import UsageKind
from app.services import usage
from app.services.chunking import count_tokens
from app.services.embeddings.errors import EmbeddingBusyError
from app.services.embeddings.rate_limit import RateLimiter
from app.services.llm.errors import LLMBusyError, LLMError, RetryableLLMError

logger = logging.getLogger(__name__)

# Someone is waiting for the answer: never sit in rate-limit waits and retry
# backoff for longer than this in total; answer "busy, retry in N s" instead.
INTERACTIVE_MAX_WAIT_SECONDS = 20.0


class Budget(Protocol):
    async def reserve(self, keep: int = 0) -> None:
        """Take one request from the budget (leaving at least `keep`), or
        raise LLMBudgetExhausted."""


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int


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
        budget: "Budget | None" = None,
    ) -> None:
        self.model = model
        self.limiter = limiter
        # The global daily request budget (app/services/llm/budget.py).
        self.budget = budget
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._sleep = sleep

    async def generate(
        self,
        system: str,
        user: str,
        max_tokens: int,
        usage_kind: UsageKind = UsageKind.llm_answer,
    ) -> LLMResult:
        """Generate a reply to `user` following the `system` instructions.
        usage_kind: what the request is recorded as (an answer or a rewrite)."""
        retry = _RetryState(
            self,
            count_tokens(system) + count_tokens(user) + max_tokens,
            # A rewrite must leave one request for the answer it serves.
            keep_budget=1 if usage_kind == UsageKind.llm_rewrite else 0,
        )
        while True:
            await retry.wait_for_capacity()
            try:
                result = await self._generate(system, user, max_tokens)
            except RetryableLLMError as exc:
                await retry.before_retry(exc)
                continue
            await usage.record(
                usage_kind, self.model, result.prompt_tokens, result.completion_tokens
            )
            return result

    async def stream(
        self, system: str, user: str, max_tokens: int
    ) -> AsyncIterator[str | LLMUsage]:
        """Like generate(), but yields the reply as it is produced: text pieces,
        then one final LLMUsage.

        Rate limiting, fail-fast and retries are the same as generate(), with
        one difference: a request is retried only if it fails BEFORE its first
        piece of text. Once text has been passed on, it can't be taken back, so
        a later failure is final.

        Stopping early (aclose(), or cancelling the task that iterates) closes
        the provider request at once.
        """
        retry = _RetryState(self, count_tokens(system) + count_tokens(user) + max_tokens)
        while True:
            await retry.wait_for_capacity()
            pieces = self._stream(system, user, max_tokens)
            try:
                first = await anext(pieces)
            except StopAsyncIteration:
                raise LLMError("The answer service returned an empty answer") from None
            except RetryableLLMError as exc:
                await pieces.aclose()
                await retry.before_retry(exc)
                continue
            except BaseException:
                await pieces.aclose()
                raise
            break
        try:
            piece = first
            while True:
                if isinstance(piece, LLMUsage):
                    # The final piece: the request is complete, record it.
                    # (A stream stopped early has no usage to record.)
                    await usage.record(
                        UsageKind.llm_answer,
                        self.model,
                        piece.prompt_tokens,
                        piece.completion_tokens,
                    )
                yield piece
                try:
                    piece = await anext(pieces)
                except StopAsyncIteration:
                    break
        finally:
            await pieces.aclose()  # stops the provider request if we stopped early

    @abstractmethod
    async def _generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        """Send ONE request. Raise RetryableLLMError for failures worth retrying,
        LLMRefusalError if the provider declines, LLMError for anything else."""

    @abstractmethod
    def _stream(self, system: str, user: str, max_tokens: int) -> AsyncIterator[str | LLMUsage]:
        """Send ONE streaming request: yield text pieces, then one LLMUsage.
        Same exceptions as _generate. Must stop the request when closed early."""

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter; never sooner than the provider asked."""
        delay = min(self.retry_max_seconds, self.retry_base_seconds * 2**attempt)
        delay *= random.uniform(0.5, 1.0)
        return max(delay, retry_after or 0.0)


class _RetryState:
    """The waiting and retrying shared by generate() and stream(): wait for the
    rate limiter, back off between attempts, and give up (busy) rather than
    keep a waiting user hanging for more than max_wait in total."""

    def __init__(self, provider: LLMProvider, estimated_tokens: int, keep_budget: int = 0) -> None:
        self.provider = provider
        self.estimated_tokens = estimated_tokens
        self.keep_budget = keep_budget
        self.waited = 0.0
        self.attempt = 0

    async def wait_for_capacity(self) -> None:
        provider = self.provider
        if provider.budget is not None:
            # Before every attempt: each one is a request the provider counts.
            await provider.budget.reserve(self.keep_budget)
        if provider.limiter is None:
            return
        try:
            reservation = await provider.limiter.acquire(
                self.estimated_tokens, "query", max_wait=max(0.0, provider.max_wait - self.waited)
            )
        except EmbeddingBusyError as exc:  # the limiter's "would wait too long"
            raise LLMBusyError(exc.retry_after) from exc
        self.waited += reservation.waited

    async def before_retry(self, exc: RetryableLLMError) -> None:
        """Wait before the next attempt, or raise if we should give up."""
        provider = self.provider
        if self.attempt >= provider.max_retries:
            if exc.rate_limited:
                raise LLMBusyError(provider._delay(self.attempt, exc.retry_after)) from exc
            raise exc
        delay = provider._delay(self.attempt, exc.retry_after)
        if self.waited + delay > provider.max_wait:
            if exc.rate_limited:
                raise LLMBusyError(delay) from exc
            raise exc
        self.attempt += 1
        logger.warning(
            "LLM request failed (%s); retry %d/%d in about %.0fs",
            exc,
            self.attempt,
            provider.max_retries,
            delay,
        )
        if exc.rate_limited and provider.limiter is not None:
            await provider.limiter.penalize(delay)  # the next acquire() waits it out
        else:
            await provider._sleep(delay)
            self.waited += delay
