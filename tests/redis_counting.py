"""A fake Redis that counts every command sent to it, the way a hosted Redis
bills them (Upstash counts commands, not requests or bytes)."""

import hashlib
from collections import Counter
from functools import cache

import fakeredis


@cache
def script_names() -> dict[str, str]:
    """SHA1 of every Lua script in the app -> a readable name."""
    from app.core import rate_limit
    from app.services import login_guard, quotas
    from app.services.embeddings import redis_rate_limit

    scripts = {
        "rate_limit.hit": rate_limit._HIT,
        "quota.take": quotas._TAKE,
        "quota.add": quotas._ADD,
        "login.fail": login_guard._FAIL,
        "outbound.try": redis_rate_limit._TRY,
        "outbound.correct": redis_rate_limit._CORRECT,
        "outbound.hold": redis_rate_limit._HOLD,
        "outbound.usage": redis_rate_limit._USAGE,
    }
    for name in ("_ADMIT", "_ASK"):
        if hasattr(quotas, name):
            scripts[f"quota.{name[1:].lower()}"] = getattr(quotas, name)
    return {hashlib.sha1(text.encode()).hexdigest(): name for name, text in scripts.items()}


class CountingRedis(fakeredis.aioredis.FakeRedis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.commands: list[str] = []

    async def execute_command(self, *args, **options):
        name = str(args[0]).upper()
        if name == "EVALSHA":
            name = f"EVALSHA:{script_names().get(str(args[1]), args[1][:8])}"
        self.commands.append(name)
        return await super().execute_command(*args, **options)

    def reset(self) -> None:
        self.commands.clear()

    @property
    def steady(self) -> list[str]:
        """The commands as they are once Redis knows every script: the first
        use of a script on a fresh server is EVALSHA (refused: unknown),
        SCRIPT LOAD, EVALSHA again. The first two happen once per server, not
        per request, so they are left out."""
        steady: list[str] = []
        for name in self.commands:
            if name == "SCRIPT LOAD":
                steady.pop()  # the refused EVALSHA just before it
            else:
                steady.append(name)
        return steady

    def by_name(self) -> Counter:
        return Counter(self.commands)
