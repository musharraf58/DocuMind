"""Sliding-window rate limiting for embedding requests.

Providers limit requests and tokens per rolling minute. A limiter keeps a log
of what was sent in the last 60 seconds and, before each new request, waits
until sending it would keep BOTH totals within budget.

RateLimiter holds the budgets and the waiting logic; where the log lives is up
to a subclass. SlidingWindowRateLimiter keeps it in this process's memory,
which is only correct while one process talks to the provider.
RedisRateLimiter (redis_rate_limit.py) keeps it in Redis, shared by every
process.
"""

import asyncio
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

from app.services.embeddings.errors import EmbeddingBusyError, RequestTooLargeError

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
Kind = Literal["document", "query"]

WINDOW_SECONDS = 60.0
# Sleep a hair past the computed moment, so rounding or a slightly early wake-up
# can't make us re-check a few microseconds too soon and sleep again.
WAKE_PADDING_SECONDS = 0.01


@dataclass
class Reservation:
    """A request the limiter has let through and recorded."""

    waited: float
    _correct: Callable[[int], Awaitable[None]]
    tokens: int = 0  # what was recorded for this request (our estimate)

    async def correct(self, actual_tokens: int) -> None:
        """The provider reported how many tokens this request really used. If
        that is more than we estimated, record the real number, so the window
        reflects what the provider is counting and the next request waits
        accordingly. Never lowered: a missing (0) or smaller count changes
        nothing, so the limiter can only become more cautious. (Which is also
        why such a count isn't even sent: with a shared Redis limiter, that's
        one command saved per request.)"""
        if actual_tokens <= self.tokens:
            return
        await self._correct(actual_tokens)


class RateLimiter(ABC):
    """Limits requests and tokens per rolling window.

    Document batches (ingestion) may use only part of the budget:
    query_reserve_* is kept free for search queries, so a long document run
    can't starve them. Queries may use the whole budget.

    `clock` and `sleep` are injectable so tests run instantly. The clock is
    the wall clock by default: when several processes share one window, their
    timestamps must be comparable, which time.monotonic()'s are not.
    """

    def __init__(
        self,
        token_budget: int,
        request_budget: int,
        query_reserve_tokens: int = 0,
        query_reserve_requests: int = 0,
        clock: Clock = time.time,
        sleep: Sleep = asyncio.sleep,
        window: float = WINDOW_SECONDS,
    ) -> None:
        self.token_budget = token_budget
        self.request_budget = request_budget
        self.document_token_budget = token_budget - query_reserve_tokens
        self.document_request_budget = max(1, request_budget - query_reserve_requests)
        self._clock = clock
        self._sleep = sleep
        self._window = window

    def max_tokens(self, kind: Kind) -> int:
        """The most tokens one request of this kind may carry, ever."""
        return self._caps(kind)[0]

    def _caps(self, kind: Kind) -> tuple[int, int]:
        if kind == "query":
            return self.token_budget, self.request_budget
        return self.document_token_budget, self.document_request_budget

    async def usage(self) -> tuple[int, int]:
        """(requests, tokens) sent within the current window."""
        return await self._usage(self._clock())

    async def time_until_available(self, tokens: int, kind: Kind = "document") -> float:
        """Seconds until a request of `tokens` tokens could be sent, without
        reserving anything (for logging and planning)."""
        token_cap, request_cap = self._caps(kind)
        wait, _ = await self._try(self._clock(), tokens, token_cap, request_cap, record=False)
        return wait

    async def penalize(self, seconds: float) -> None:
        """The provider said "come back in `seconds`": hold everyone until then."""
        await self._hold_until(self._clock() + seconds)

    async def acquire(
        self, tokens: int, kind: Kind = "document", max_wait: float | None = None
    ) -> Reservation:
        """Wait until a request of `tokens` tokens may be sent, record it as sent,
        and return the reservation (with how long we waited).

        Raises RequestTooLargeError at once if it could never fit, and
        EmbeddingBusyError (without recording anything) if it would have to
        wait longer than `max_wait`.

        Checking and recording are one indivisible step (see _try), so two
        callers can never both take the last slot. The sleep happens outside
        that step, so a waiting document never blocks a query from checking
        (and failing fast).
        """
        token_cap, request_cap = self._caps(kind)
        if tokens > token_cap:
            raise RequestTooLargeError(
                f"Text is too large for the embedding rate limit "
                f"({tokens} tokens; at most {token_cap} per request)"
            )

        waited = 0.0
        while True:
            wait, handle = await self._try(
                self._clock(), tokens, token_cap, request_cap, record=True
            )
            if wait <= 0:
                return Reservation(waited, partial(self._correct, handle), tokens)
            if max_wait is not None and waited + wait > max_wait:
                raise EmbeddingBusyError(retry_after=wait)
            await self._sleep(wait + WAKE_PADDING_SECONDS)
            waited += wait + WAKE_PADDING_SECONDS

    @abstractmethod
    async def _try(
        self, now: float, tokens: int, token_cap: int, request_cap: int, record: bool
    ) -> tuple[float, Any]:
        """In ONE indivisible step: drop expired entries, compute how long until
        the request fits, and if it fits now (and `record`), record it.
        Returns (seconds to wait or 0, a handle to the recorded entry)."""

    @abstractmethod
    async def _correct(self, handle: Any, actual_tokens: int) -> None:
        """Raise the recorded token count of an entry to `actual_tokens` if higher."""

    @abstractmethod
    async def _hold_until(self, until: float) -> None:
        """Block every request until `until` (keep the later of old and new)."""

    @abstractmethod
    async def _usage(self, now: float) -> tuple[int, int]:
        """(requests, tokens) in the window ending at `now`."""


@dataclass
class _Sent:
    at: float
    tokens: int


class SlidingWindowRateLimiter(RateLimiter):
    """Keeps the log in this process's memory: correct only while a single
    process sends requests (used in tests and as the reference behaviour)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sent: deque[_Sent] = deque()
        self._blocked_until = 0.0

    async def _try(
        self, now: float, tokens: int, token_cap: int, request_cap: int, record: bool
    ) -> tuple[float, Any]:
        # Nothing here awaits, so on one event loop no other task can run
        # between the check and the record: that makes it indivisible.
        wait = self._wait_needed(now, tokens, token_cap, request_cap)
        if wait > 0 or not record:
            return wait, None
        sent = _Sent(now, tokens)
        self._sent.append(sent)
        return 0.0, sent

    async def _correct(self, handle: _Sent, actual_tokens: int) -> None:
        handle.tokens = max(handle.tokens, actual_tokens)

    async def _hold_until(self, until: float) -> None:
        self._blocked_until = max(self._blocked_until, until)

    async def _usage(self, now: float) -> tuple[int, int]:
        self._expire(now)
        return len(self._sent), sum(sent.tokens for sent in self._sent)

    def _expire(self, now: float) -> None:
        # An entry stays in the window until exactly `window` seconds later.
        while self._sent and self._sent[0].at + self._window <= now:
            self._sent.popleft()

    def _wait_needed(self, now: float, tokens: int, token_cap: int, request_cap: int) -> float:
        """Seconds until the request fits, or 0 if it fits now."""
        self._expire(now)
        wait = max(0.0, self._blocked_until - now)

        # Requests: the (n+1)th must wait for enough old ones to leave the window.
        excess_requests = len(self._sent) + 1 - request_cap
        if excess_requests > 0:
            wait = max(wait, self._sent[excess_requests - 1].at + self._window - now)

        # Tokens: wait until enough of the oldest requests leave to make room.
        excess_tokens = sum(sent.tokens for sent in self._sent) + tokens - token_cap
        for sent in self._sent:
            if excess_tokens <= 0:
                break
            excess_tokens -= sent.tokens
            wait = max(wait, sent.at + self._window - now)
        return wait


def estimate_seconds(total_tokens: int, token_budget: int, window: float = WINDOW_SECONDS) -> int:
    """Rough time to send `total_tokens` at the token budget: the first window's
    worth goes immediately, each further window's worth takes another window."""
    return max(0, math.ceil(total_tokens / token_budget) - 1) * int(window)
