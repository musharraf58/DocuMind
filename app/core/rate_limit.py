"""Inbound rate limiting: how often a CLIENT may call US.

(Not to be confused with the outbound limiter in
app/services/embeddings/redis_rate_limit.py, which paces OUR calls to Voyage
and Gemini so we stay inside their limits. That one makes callers wait; this
one refuses at once with 429, because the caller is the one to slow down.)

Algorithm: a sliding-window log. For each key (e.g. "ask requests of user
X") Redis keeps a sorted set of the times of recent allowed requests. A new
request is allowed if fewer than `limit` of them fall inside the last
`window` seconds. Unlike a fixed window ("10 per calendar minute"), a client
can't fire 10 at 12:00:59 and 10 more at 12:01:00. Refused requests are not
recorded, so hammering a limit doesn't push the reset further away.

Each check is one Lua script, which Redis runs as one indivisible step: two
parallel requests can't both take the last slot. The time comes from the
caller (the injectable clock), so tests run on a fake clock.
"""

import logging
import math
import re
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core import clock

logger = logging.getLogger(__name__)

UNITS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
RATE_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*(second|minute|hour|day)s?\s*$")


@dataclass(frozen=True)
class Rate:
    limit: int
    window: float  # seconds

    @classmethod
    def parse(cls, text: str) -> "Rate":
        """ "10/minute" -> Rate(10, 60). Also second, hour, day (plural ok)."""
        match = RATE_PATTERN.match(text)
        if not match or int(match.group(1)) < 1:
            raise ValueError(f"Invalid rate {text!r}; expected e.g. '10/minute'")
        return cls(int(match.group(1)), UNITS[match.group(2)])

    def __str__(self) -> str:
        unit = next(name for name, seconds in UNITS.items() if seconds == self.window)
        return f"{self.limit}/{unit}"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    # Seconds until the oldest counted request leaves the window (a slot frees).
    reset_after: float

    @property
    def retry_after(self) -> int:
        """Whole seconds, at least 1, for the Retry-After header."""
        return max(1, math.ceil(self.reset_after))

    def headers(self) -> dict[str, str]:
        """RateLimit-* headers (IETF draft "RateLimit header fields for HTTP")."""
        headers = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(self.remaining),
            "RateLimit-Reset": str(max(0, math.ceil(self.reset_after))),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after)
        return headers


def now_seconds() -> float:
    return clock.utcnow().timestamp()


# KEYS: the log. ARGV: now, window, limit, member.
# Returns {allowed (0/1), count after, seconds until a slot frees} as strings
# (a Lua number returned to Redis is cut to an integer).
_HIT = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local count = redis.call('ZCARD', KEYS[1])
local allowed = 0
if count < limit then
  redis.call('ZADD', KEYS[1], now, ARGV[4])
  redis.call('PEXPIRE', KEYS[1], math.ceil(window * 1000))
  count = count + 1
  allowed = 1
end
local reset = 0
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
if #oldest > 0 then reset = tonumber(oldest[2]) + window - now end
return {tostring(allowed), tostring(count), tostring(reset)}
"""


class RedisRateLimiter:
    """The shared limiter: every API process sees the same counts."""

    def __init__(self, redis: Redis, now: Callable[[], float] = now_seconds) -> None:
        self.redis = redis
        self.now = now
        self._hit = redis.register_script(_HIT)

    async def hit(self, key: str, rate: Rate) -> Decision:
        """Count one request if allowed. Raises RedisError if Redis is down."""
        now = self.now()
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"  # unique even at the same instant
        allowed, count, reset = await self._hit(
            keys=[key], args=[repr(now), rate.window, rate.limit, member]
        )
        return Decision(
            allowed=allowed in (b"1", "1"),
            limit=rate.limit,
            remaining=max(0, rate.limit - int(count)),
            reset_after=max(0.0, float(reset)),
        )


class MemoryRateLimiter:
    """The same sliding window in this process's memory: the fallback for
    login when Redis is down. Per process (each API process counts on its
    own), and capped at max_keys so a flood of distinct keys (many IPs or
    emails) can't grow it without bound: the least recently used key goes."""

    def __init__(self, max_keys: int = 10_000, now: Callable[[], float] = now_seconds) -> None:
        self.max_keys = max_keys
        self.now = now
        self._logs: OrderedDict[str, deque[float]] = OrderedDict()

    async def hit(self, key: str, rate: Rate) -> Decision:
        now = self.now()
        log = self._logs.pop(key, None) or deque()
        while log and log[0] <= now - rate.window:
            log.popleft()
        allowed = len(log) < rate.limit
        if allowed:
            log.append(now)
        if log:
            self._logs[key] = log  # re-inserted at the end: most recently used
            while len(self._logs) > self.max_keys:
                self._logs.popitem(last=False)
        reset = log[0] + rate.window - now if log else 0.0
        return Decision(allowed, rate.limit, max(0, rate.limit - len(log)), max(0.0, reset))

    def clear(self) -> None:
        self._logs.clear()


class OutageLog:
    """Logs "Redis is down, limits are off" loudly (ERROR), but at most once
    per `every` seconds per process, so an outage doesn't flood the log."""

    def __init__(self, every: float = 30.0, now: Callable[[], float] = time.monotonic) -> None:
        self.every = every
        self.now = now
        self._last: float | None = None
        self.suppressed = 0

    def report(self, what: str, exc: Exception) -> None:
        now = self.now()
        if self._last is not None and now - self._last < self.every:
            self.suppressed += 1
            return
        logger.error(
            "RATE LIMITING DEGRADED: Redis unreachable (%s: %s); %s "
            "(%d similar messages suppressed)",
            type(exc).__name__,
            exc,
            what,
            self.suppressed,
        )
        self._last = now
        self.suppressed = 0


class InboundLimiter:
    """What the API uses: the Redis limiter, with the behaviour for a Redis
    outage decided per check.

    - fallback=False (most endpoints): FAIL OPEN. The request is allowed and
      the outage is logged. Refusing everyone because our own Redis is down
      would turn a Redis problem into a full outage, and these endpoints are
      still protected by authentication and by the outbound provider limits.
    - fallback=True (login): use the in-process limiter instead. Login is
      what attackers hammer (password guessing), so it keeps a limit even
      during an outage, if only per process.
    """

    def __init__(
        self,
        redis: Redis,
        fallback: MemoryRateLimiter,
        outage_log: OutageLog,
        now: Callable[[], float] = now_seconds,
    ) -> None:
        self.shared = RedisRateLimiter(redis, now)
        self.fallback = fallback
        self.outage_log = outage_log

    async def hit(self, key: str, rate: Rate, *, fallback: bool = False) -> Decision | None:
        """The decision, or None when Redis is down and the request is let
        through without a limit (fail open)."""
        try:
            return await self.shared.hit(key, rate)
        except (RedisError, OSError) as exc:
            if fallback:
                self.outage_log.report("login uses the in-process fallback limiter", exc)
                return await self.fallback.hit(key, rate)
            self.outage_log.report("requests are allowed WITHOUT rate limits (fail open)", exc)
            return None


# One of each per process: the fallback's memory and the log throttle must
# outlive single requests.
memory_fallback = MemoryRateLimiter()
outage_log = OutageLog()
