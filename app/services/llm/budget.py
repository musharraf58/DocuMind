"""The global daily LLM budget, as seen by the LLM client: one unit is taken
before EVERY request to the provider (retries and rewrites included, since
the provider counts those too)."""

from collections.abc import Callable
from datetime import datetime

from redis.asyncio import Redis

from app.core.config import Settings
from app.services.llm.errors import LLMBudgetExhausted
from app.services.quotas import Quotas


class GlobalLLMBudget:
    def __init__(
        self,
        redis: Redis,
        settings: Settings | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.quotas = Quotas(redis, settings, now)

    async def reserve(self, keep: int = 0) -> None:
        """Take one request from today's budget, or raise LLMBudgetExhausted.
        keep: units that must remain afterwards. A follow-up's rewrite passes
        1, so it never takes the last unit and leaves the answer without one
        (the rewrite is optional: it falls back to the original question).
        Redis down: allowed (fail open, logged); the outbound rate limiter
        needs Redis as well and stops the request in that case anyway."""
        taken = await self.quotas.take_global_llm(keep)
        if not taken.allowed:
            raise LLMBudgetExhausted(self.quotas.seconds_until_reset())
