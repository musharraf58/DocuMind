"""The rate limiter's log kept in Redis, so every process shares one window.

The API (search queries) and the worker (document ingestion) are separate
processes. With a log in each process's memory, each would believe it had
the whole per-minute budget, and together they would exceed it. Here the log
lives in Redis:

    <prefix>:log      sorted set: request id -> time it was sent
    <prefix>:tokens   hash:       request id -> tokens it used
    <prefix>:blocked  string:     time until which everyone must wait (after a 429)

Each operation is a Lua script. Redis runs a script as one indivisible step,
so "check whether it fits, then record it" can't be interleaved with another
process doing the same: two processes can never both take the last slot.
The current time is passed in by the caller (the injectable clock) rather
than read inside Redis, so tests can run on a fake clock.
"""

import uuid
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.services.embeddings.errors import RetryableEmbeddingError
from app.services.embeddings.rate_limit import RateLimiter

DEFAULT_KEY_PREFIX = "documind:embedding-rate-limit"

# KEYS: log, tokens, blocked.
# ARGV: now, window, tokens, token_cap, request_cap, record (0/1), request id.
# Returns the seconds to wait as a string ("0" = fits, and recorded if asked).
# Same rules as SlidingWindowRateLimiter._wait_needed.
_TRY = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local tokens = tonumber(ARGV[3])
local token_cap = tonumber(ARGV[4])
local request_cap = tonumber(ARGV[5])

-- Drop entries that have left the window (sent at or before now - window).
local cutoff = now - window
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', cutoff)
if #expired > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
  redis.call('HDEL', KEYS[2], unpack(expired))
end

local wait = 0
local blocked = tonumber(redis.call('GET', KEYS[3]) or '0')
if blocked - now > wait then wait = blocked - now end

-- Oldest first: a flat list of id, time, id, time, ...
local entries = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
local n = #entries / 2

-- Requests: the (n+1)th must wait for enough old ones to leave the window.
local excess_requests = n + 1 - request_cap
if excess_requests > 0 then
  local at = tonumber(entries[excess_requests * 2])
  if at + window - now > wait then wait = at + window - now end
end

-- Tokens: wait until enough of the oldest requests leave to make room.
local counts = {}
local used = 0
if n > 0 then
  local ids = {}
  for i = 1, n do ids[i] = entries[i * 2 - 1] end
  counts = redis.call('HMGET', KEYS[2], unpack(ids))
  for i = 1, n do used = used + tonumber(counts[i] or '0') end
end
local excess_tokens = used + tokens - token_cap
local i = 1
while excess_tokens > 0 and i <= n do
  excess_tokens = excess_tokens - tonumber(counts[i] or '0')
  local at = tonumber(entries[i * 2])
  if at + window - now > wait then wait = at + window - now end
  i = i + 1
end

if wait > 0 or ARGV[6] ~= '1' then return tostring(wait) end
redis.call('ZADD', KEYS[1], now, ARGV[7])
redis.call('HSET', KEYS[2], ARGV[7], tokens)
-- Let idle keys disappear on their own.
local ttl = math.ceil(window * 2000)
redis.call('PEXPIRE', KEYS[1], ttl)
redis.call('PEXPIRE', KEYS[2], ttl)
return '0'
"""

# KEYS: tokens. ARGV: request id, actual tokens. Raises the count, never lowers it.
_CORRECT = """
local current = redis.call('HGET', KEYS[1], ARGV[1])
if current and tonumber(ARGV[2]) > tonumber(current) then
  redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
end
return 0
"""

# KEYS: blocked. ARGV: until, ttl in ms. Keeps the later of old and new.
_HOLD = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if tonumber(ARGV[1]) > current then
  redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
end
return 0
"""

# KEYS: log, tokens. ARGV: now, window. Returns {requests, tokens}.
_USAGE = """
local cutoff = tonumber(ARGV[1]) - tonumber(ARGV[2])
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', cutoff)
if #expired > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
  redis.call('HDEL', KEYS[2], unpack(expired))
end
local used = 0
for _, count in ipairs(redis.call('HVALS', KEYS[2])) do used = used + tonumber(count) end
return {redis.call('ZCARD', KEYS[1]), used}
"""

UNAVAILABLE = "Embedding rate limiter is unavailable; try again later"


class RedisRateLimiter(RateLimiter):
    """RateLimiter whose log is shared through Redis by every process."""

    def __init__(self, redis: Redis, *args, key_prefix: str = DEFAULT_KEY_PREFIX, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._keys = [f"{key_prefix}:log", f"{key_prefix}:tokens", f"{key_prefix}:blocked"]
        self._try_script = redis.register_script(_TRY)
        self._correct_script = redis.register_script(_CORRECT)
        self._hold_script = redis.register_script(_HOLD)
        self._usage_script = redis.register_script(_USAGE)

    async def _try(
        self, now: float, tokens: int, token_cap: int, request_cap: int, record: bool
    ) -> tuple[float, Any]:
        request_id = uuid.uuid4().hex
        args = [repr(now), self._window, tokens, token_cap, request_cap, int(record), request_id]
        wait = float(await self._run(self._try_script, self._keys, args))
        return wait, (request_id if wait <= 0 and record else None)

    async def _correct(self, handle: str, actual_tokens: int) -> None:
        await self._run(self._correct_script, [self._keys[1]], [handle, actual_tokens])

    async def _hold_until(self, until: float) -> None:
        ttl_ms = max(1, int((until - self._clock()) * 1000) + 1000)
        await self._run(self._hold_script, [self._keys[2]], [repr(until), ttl_ms])

    async def _usage(self, now: float) -> tuple[int, int]:
        requests, tokens = await self._run(
            self._usage_script, self._keys[:2], [repr(now), self._window]
        )
        return int(requests), int(tokens)

    @staticmethod
    async def _run(script, keys: list[str], args: list) -> Any:
        """Run a script; Redis being unreachable is a transient embedding
        failure (the job retries later, a search answers 503), not a crash."""
        try:
            return await script(keys=keys, args=args)
        except RedisError as exc:
            raise RetryableEmbeddingError(UNAVAILABLE) from exc
