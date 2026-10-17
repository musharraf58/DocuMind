"""A sliding-window rate limiter for embedding requests.

Providers limit requests and tokens per rolling minute. This keeps a log of
what we sent in the last 60 seconds and, before each new request, waits until
sending it would keep BOTH totals within budget.

State lives in this process's memory. That is correct only while a single
process talks to the provider; see the README for what several workers need.
"""

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from app.services.embeddings.errors import EmbeddingBusyError, RequestTooLargeError

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
Kind = Literal["document", "query"]

WINDOW_SECONDS = 60.0
# Sleep a hair past the computed moment, so rounding or a slightly early wake-up
# can't make us re-check a few microseconds too soon and sleep again.
WAKE_PADDING_SECONDS = 0.01


@dataclass(frozen=True)
class _Sent:
    at: float
    tokens: int


class SlidingWindowRateLimiter:
    """Limits requests and tokens per rolling window.

    Document batches may use only part of the budget: query_reserve_* is kept
    free for search queries, so a long document run can't starve them.

    `clock` and `sleep` are injectable so tests run instantly.
    """

    def __init__(
        self,
        token_budget: int,
        request_budget: int,
        query_reserve_tokens: int = 0,
        query_reserve_requests: int = 0,
        clock: Clock = time.monotonic,
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
        self._sent: deque[_Sent] = deque()
        self._blocked_until = 0.0

    def max_tokens(self, kind: Kind) -> int:
        """The most tokens one request of this kind may carry, ever."""
        return self.token_budget if kind == "query" else self.document_token_budget

    def usage(self) -> tuple[int, int]:
        """(requests, tokens) sent within the current window."""
        self._expire(self._clock())
        return len(self._sent), sum(sent.tokens for sent in self._sent)

    def penalize(self, seconds: float) -> None:
        """The provider said "come back in `seconds`": hold everyone until then."""
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)

    async def acquire(
        self, tokens: int, kind: Kind = "document", max_wait: float | None = None
    ) -> float:
        """Wait until a request of `tokens` tokens may be sent, record it as sent,
        and return how long we waited.

        Raises RequestTooLargeError at once if it could never fit, and
        EmbeddingBusyError (without recording anything) if it would have to
        wait longer than `max_wait`.

        Nothing awaits between checking and recording, so on one event loop no
        other task can slip in between: no lock is needed. The sleep happens
        outside the check, so a waiting document never blocks a query from
        checking (and failing fast).
        """
        token_cap = self.max_tokens(kind)
        request_cap = self.request_budget if kind == "query" else self.document_request_budget
        if tokens > token_cap:
            raise RequestTooLargeError(
                f"Text is too large for the embedding rate limit "
                f"({tokens} tokens; at most {token_cap} per request)"
            )

        waited = 0.0
        while True:
            now = self._clock()
            wait = self._wait_needed(now, tokens, token_cap, request_cap)
            if wait <= 0:
                self._sent.append(_Sent(now, tokens))
                return waited
            if max_wait is not None and waited + wait > max_wait:
                raise EmbeddingBusyError(retry_after=wait)
            await self._sleep(wait + WAKE_PADDING_SECONDS)
            waited += wait + WAKE_PADDING_SECONDS

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
