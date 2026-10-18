from functools import lru_cache

from redis.asyncio import Redis

from app.core.config import get_settings


def make_redis() -> Redis:
    """A new client. Short timeouts: a down Redis must fail fast, not hang."""
    return Redis.from_url(get_settings().REDIS_URL, socket_timeout=2, socket_connect_timeout=2)


@lru_cache
def get_redis() -> Redis:
    """The API process's shared client (one event loop for the whole process).
    The worker makes its own per job, since each job runs in a new event loop."""
    return make_redis()
