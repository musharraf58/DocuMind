"""Progressive delay after failed logins, per account + IP.

The first few failures are free (typos happen). After that, each failure
makes the next attempt wait longer: 2 s, 4 s, 8 s ... up to a maximum. One
success, or a quiet period with no failures, starts the count over.

Why account + IP, and not just the account: keyed by account alone, anyone
could lock a real user out by failing logins in their name. Here the
attacker only slows down themselves; the real user's own IP has its own
count. (Guessing one account from MANY IPs is stopped by the per-email rate
limit instead, see app/api/limits.py.)

Unknown emails are counted exactly like real ones, so the delay never
reveals whether an account exists. The 401 message stays identical too.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core import redis_keys
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.rate_limit import now_seconds, outage_log

logger = logging.getLogger(__name__)

# KEYS: the hash. ARGV: now, free attempts, base delay, max delay, window.
# Returns the delay now imposed, in seconds (0 while still free), as a string.
_FAIL = """
local count = redis.call('HINCRBY', KEYS[1], 'count', 1)
local now = tonumber(ARGV[1])
local free = tonumber(ARGV[2])
local delay = 0
if count > free then
  delay = math.min(tonumber(ARGV[3]) * 2 ^ (count - free - 1), tonumber(ARGV[4]))
  redis.call('HSET', KEYS[1], 'blocked_until', tostring(now + delay))
end
-- Forget the failures after a quiet window (but never before a block ends).
redis.call('EXPIRE', KEYS[1], math.ceil(math.max(tonumber(ARGV[5]), delay)))
return tostring(delay)
"""


@dataclass(frozen=True)
class LoginPolicy:
    free_attempts: int
    base_delay: float
    max_delay: float
    window: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "LoginPolicy":
        return cls(
            settings.LOGIN_FREE_ATTEMPTS,
            settings.LOGIN_DELAY_BASE_SECONDS,
            settings.LOGIN_DELAY_MAX_SECONDS,
            settings.LOGIN_FAILURE_WINDOW_MINUTES * 60,
        )


class LoginGuard:
    """Redis being down never blocks a login here (fail open, logged): the
    in-process login rate limit still applies in that case."""

    def __init__(
        self,
        redis: Redis,
        policy: LoginPolicy | None = None,
        now: Callable[[], float] = now_seconds,
    ) -> None:
        self.redis = redis
        self.policy = policy or LoginPolicy.from_settings(get_settings())
        self.now = now
        self._fail = redis.register_script(_FAIL)
        # Keys this guard has seen to hold no failures: a successful login
        # then has nothing to clear and skips the DEL (one Redis command less
        # per login; a guard lives for one request).
        self._no_record: set[str] = set()

    async def wait_needed(self, email: str, ip: str) -> float:
        """Seconds until this account + IP may try again (0 = now)."""
        key = redis_keys.login_failures(email, ip)
        try:
            blocked_until, count = await self.redis.hmget(key, "blocked_until", "count")
        except (RedisError, OSError) as exc:
            outage_log.report("the failed-login delay is off (fail open)", exc)
            return 0.0
        if blocked_until is None and count is None:
            self._no_record.add(key)
        return max(0.0, float(blocked_until) - self.now()) if blocked_until else 0.0

    async def check(self, email: str, ip: str) -> None:
        """Raise 429 while this account + IP must wait."""
        wait = await self.wait_needed(email, ip)
        if wait > 0:
            seconds = max(1, math.ceil(wait))
            raise ApiError(
                429,
                f"Too many failed login attempts; try again in {seconds} seconds",
                code="login_delayed",
                headers={"Retry-After": str(seconds)},
                extra={"retry_after": seconds},
            )

    async def failed(self, email: str, ip: str) -> float:
        """Record a failure; returns the delay it imposes (0 while free)."""
        self._no_record.discard(redis_keys.login_failures(email, ip))
        p = self.policy
        try:
            delay = await self._fail(
                keys=[redis_keys.login_failures(email, ip)],
                args=[repr(self.now()), p.free_attempts, p.base_delay, p.max_delay, p.window],
            )
        except (RedisError, OSError) as exc:
            outage_log.report("the failed-login delay is off (fail open)", exc)
            return 0.0
        return float(delay)

    async def succeeded(self, email: str, ip: str) -> None:
        key = redis_keys.login_failures(email, ip)
        if key in self._no_record:
            return  # nothing to clear
        try:
            await self.redis.delete(key)
        except (RedisError, OSError) as exc:
            outage_log.report("the failed-login delay is off (fail open)", exc)
