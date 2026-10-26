"""A fake Redis that counts every command sent to it, the way a hosted Redis
bills them (Upstash counts commands, not requests or bytes)."""

from collections import Counter

import fakeredis


class CountingRedis(fakeredis.aioredis.FakeRedis):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.commands: list[str] = []

    async def execute_command(self, *args, **options):
        self.commands.append(str(args[0]).upper())
        return await super().execute_command(*args, **options)

    def reset(self) -> None:
        self.commands.clear()

    @property
    def count(self) -> int:
        return len(self.commands)

    def by_name(self) -> Counter:
        return Counter(self.commands)
