"""The worker's "I'm alive" signal, for GET /health/ready.

A background thread writes the current time to Redis every
HEARTBEAT_INTERVAL seconds, with an expiry of HEARTBEAT_TTL. If the worker
dies or hangs, the key expires and readiness reports the worker as down.

A thread, not a Celery beat task: the worker runs one job at a time
(concurrency 1), so a scheduled task would wait behind a document that
takes an hour to embed, and the worker would look dead while it's busy.
"""

import logging
import threading
import time

from redis import Redis
from redis.exceptions import RedisError

from app.core.redis_keys import WORKER_HEARTBEAT

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30
# Three missed beats before the worker counts as gone.
HEARTBEAT_TTL = 90


def beat(redis: Redis, now: float | None = None) -> None:
    redis.set(WORKER_HEARTBEAT, repr(now if now is not None else time.time()), ex=HEARTBEAT_TTL)


def start_heartbeat(redis_url: str, stop: threading.Event) -> threading.Thread:
    """Start beating in a daemon thread (it dies with the process) until
    `stop` is set."""
    redis = Redis.from_url(redis_url, socket_timeout=2, socket_connect_timeout=2)

    def run() -> None:
        while not stop.is_set():
            try:
                beat(redis)
            except (RedisError, OSError) as exc:
                logger.warning("Worker heartbeat failed (%s); retrying", type(exc).__name__)
            stop.wait(HEARTBEAT_INTERVAL)
        try:
            # A clean shutdown: say so at once instead of waiting for the expiry.
            redis.delete(WORKER_HEARTBEAT)
        except (RedisError, OSError):
            pass

    thread = threading.Thread(target=run, name="worker-heartbeat", daemon=True)
    thread.start()
    return thread
