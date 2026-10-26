"""Daily quotas, enforced with atomic Redis counters.

Per user and day: questions asked, embedding tokens used, documents
uploaded. And one GLOBAL counter: LLM requests made by everyone together
today, so the free-tier Gemini quota can't be spent by a handful of users
(or one attacker with many accounts).

Each counter is one Redis key per day, e.g.
documind:quota:u:<user id>:2026-10-22:questions. When the day changes, new
keys start at zero and old ones expire by themselves (48 hours), so nothing
ever has to reset them. The "day" starts at QUOTA_RESET_HOUR_UTC.

Taking from a quota is one Lua script: "if used + amount fits under the
limit, add it" happens as one step, so two parallel requests can't both
take the last unit.

Redis down: quotas fail OPEN (allowed, logged loudly), like the rate limits.
LLM calls are still bounded then, because the outbound provider limiter
needs Redis too and refuses without it.

Postgres (usage_events) is the durable history; these counters are only for
enforcement. See the README ("Security and Limits") for how they relate.
"""

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core import clock, redis_keys
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.rate_limit import outage_log

QUESTIONS = "questions"
UPLOADS = "uploads"
EMBED_TOKENS = "embed_tokens"
GLOBAL_LLM = "llm"
# Old days' keys linger at most this long (the day itself plus a margin).
KEY_TTL_SECONDS = 48 * 3600

# KEYS: counter. ARGV: amount, limit, ttl. Returns {taken (0/1), used after}.
_TAKE = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
if used + amount > tonumber(ARGV[2]) then return {0, used} end
used = redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {1, used}
"""

# KEYS: counter. ARGV: amount, ttl. Adds unconditionally; returns the total.
_ADD = """
local used = redis.call('INCRBY', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return used
"""


def quota_day(now: datetime, reset_hour: int) -> date:
    """The quota day `now` belongs to: days start at reset_hour UTC."""
    return (now - timedelta(hours=reset_hour)).date()


def next_reset(now: datetime, reset_hour: int) -> datetime:
    day = quota_day(now, reset_hour)
    return datetime(day.year, day.month, day.day, reset_hour, tzinfo=now.tzinfo) + timedelta(days=1)


def day_start(now: datetime, reset_hour: int) -> datetime:
    return next_reset(now, reset_hour) - timedelta(days=1)


@dataclass(frozen=True)
class Taken:
    allowed: bool
    used: int | None  # None: Redis was down (allowed without counting)
    limit: int


class Quotas:
    def __init__(
        self,
        redis: Redis,
        settings: Settings | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.redis = redis
        self.settings = settings or get_settings()
        self._now = now
        self._take = redis.register_script(_TAKE)
        self._add = redis.register_script(_ADD)

    # --- time -------------------------------------------------------------------

    def now(self) -> datetime:
        return self._now() if self._now else clock.utcnow()

    @property
    def reset_hour(self) -> int:
        return self.settings.QUOTA_RESET_HOUR_UTC

    def day(self) -> str:
        return quota_day(self.now(), self.reset_hour).isoformat()

    def resets_at(self) -> datetime:
        return next_reset(self.now(), self.reset_hour)

    def seconds_until_reset(self) -> int:
        return max(1, math.ceil((self.resets_at() - self.now()).total_seconds()))

    # --- limits --------------------------------------------------------------------

    def limit(self, name: str) -> int:
        s = self.settings
        return {
            QUESTIONS: s.DAILY_QUESTIONS_PER_USER,
            UPLOADS: s.DAILY_UPLOADS_PER_USER,
            EMBED_TOKENS: s.DAILY_EMBED_TOKENS_PER_USER,
            GLOBAL_LLM: s.GLOBAL_DAILY_LLM_BUDGET,
        }[name]

    def _user_key(self, user_id: uuid.UUID, name: str) -> str:
        return redis_keys.quota_user(user_id, self.day(), name)

    def _global_key(self) -> str:
        return redis_keys.quota_global(self.day(), GLOBAL_LLM)

    # --- counters ------------------------------------------------------------------

    async def _take_from(self, key: str, limit: int, amount: int) -> Taken:
        try:
            taken, used = await self._take(keys=[key], args=[amount, limit, KEY_TTL_SECONDS])
        except (RedisError, OSError) as exc:
            outage_log.report("daily quotas are NOT enforced (fail open)", exc)
            return Taken(True, None, limit)
        return Taken(bool(int(taken)), int(used), limit)

    async def take(self, user_id: uuid.UUID, name: str, amount: int = 1) -> Taken:
        return await self._take_from(self._user_key(user_id, name), self.limit(name), amount)

    async def give_back(self, user_id: uuid.UUID, name: str, amount: int = 1) -> None:
        """Undo a take() for a request that turned out not to count (e.g. a
        404 before anything was done)."""
        try:
            await self.redis.decrby(self._user_key(user_id, name), amount)
        except (RedisError, OSError) as exc:
            outage_log.report("daily quotas are NOT enforced (fail open)", exc)

    async def add(self, user_id: uuid.UUID, name: str, amount: int) -> None:
        """Count usage that has already happened (tokens the provider reported),
        even if it goes over: it was spent."""
        try:
            await self._add(keys=[self._user_key(user_id, name)], args=[amount, KEY_TTL_SECONDS])
        except (RedisError, OSError) as exc:
            outage_log.report("daily quotas are NOT enforced (fail open)", exc)

    async def used(self, user_id: uuid.UUID, name: str) -> int | None:
        try:
            value = await self.redis.get(self._user_key(user_id, name))
        except (RedisError, OSError) as exc:
            outage_log.report("daily quotas are NOT enforced (fail open)", exc)
            return None
        return int(value or 0)

    async def global_llm_used(self) -> int | None:
        try:
            value = await self.redis.get(self._global_key())
        except (RedisError, OSError) as exc:
            outage_log.report("the global LLM budget is NOT enforced (fail open)", exc)
            return None
        return int(value or 0)

    async def take_global_llm(self, keep: int = 0) -> Taken:
        return await self._take_from(self._global_key(), self.limit(GLOBAL_LLM) - keep, 1)

    # --- what endpoints call ---------------------------------------------------------

    def _over(self, name: str, what: str, taken_or_used: int | None, limit: int) -> ApiError:
        seconds = self.seconds_until_reset()
        return ApiError(
            429,
            f"Daily {what} limit reached ({limit}); it resets in about "
            f"{_human(seconds)} ({self.resets_at():%H:%M} UTC)",
            code=f"daily_{name}_quota_reached",
            headers={"Retry-After": str(seconds)},
            extra={"limit": limit, "used": taken_or_used, "retry_after": seconds},
        )

    async def check_llm_capacity(self) -> None:
        """503 if the global LLM budget for today is used up (checked before a
        question does any work, so nothing is spent on a question that can't
        be answered)."""
        used = await self.global_llm_used()
        limit = self.limit(GLOBAL_LLM)
        if used is not None and used >= limit:
            raise capacity_reached(self.seconds_until_reset())

    async def check_embed_tokens(self, user_id: uuid.UUID, needed: int = 1) -> None:
        """429 if `needed` more embedding tokens don't fit in today's quota."""
        used = await self.used(user_id, EMBED_TOKENS)
        limit = self.limit(EMBED_TOKENS)
        if used is not None and used + needed > limit:
            raise self._over(EMBED_TOKENS, "embedding token", used, limit)

    async def take_question(self, user_id: uuid.UUID) -> None:
        taken = await self.take(user_id, QUESTIONS)
        if not taken.allowed:
            raise self._over(QUESTIONS, "question", taken.used, taken.limit)

    async def take_upload(self, user_id: uuid.UUID) -> None:
        taken = await self.take(user_id, UPLOADS)
        if not taken.allowed:
            raise self._over(UPLOADS, "upload", taken.used, taken.limit)


CAPACITY_MESSAGE = "Daily capacity reached; please try again tomorrow"


def capacity_reached(retry_after: int) -> ApiError:
    """503, not 429: the client did nothing wrong (429 means "YOU sent too
    many"); the service as a whole is out of answer capacity for today."""
    return ApiError(
        503,
        CAPACITY_MESSAGE,
        code="daily_capacity_reached",
        headers={"Retry-After": str(retry_after)},
        extra={"retry_after": retry_after},
    )


def _human(seconds: int) -> str:
    hours, minutes = divmod(math.ceil(seconds / 60), 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"
