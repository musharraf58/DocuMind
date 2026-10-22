"""Every Redis key the app's limits use, in one place.

    documind:rl:{class}:u:{user_id}               sliding log, per user
    documind:rl:{class}:ip:{ip}                   sliding log, per client IP
    documind:rl:login:email:{sha256(email)}       sliding log, per login email
    documind:login-fail:{sha256(email|ip)}        failed-login counter (progressive delay)
    documind:quota:u:{user_id}:{day}:{name}       daily quota counter, per user
    documind:quota:global:{day}:llm               daily LLM requests, all users
    documind:worker:heartbeat                     the worker is alive

Per-user keys all contain "u:{user_id}", so everything kept about one user
can be found by pattern (user_key_patterns). Emails are hashed: Redis never
holds an address. The outbound provider limiter's keys
(documind:embedding-rate-limit:*, documind:llm-rate-limit:*) live in their
own modules and hold no user data.
"""

import hashlib
import uuid

PREFIX = "documind"
WORKER_HEARTBEAT = f"{PREFIX}:worker:heartbeat"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rate_user(limit_class: str, user_id: uuid.UUID) -> str:
    return f"{PREFIX}:rl:{limit_class}:u:{user_id}"


def rate_ip(limit_class: str, ip: str) -> str:
    return f"{PREFIX}:rl:{limit_class}:ip:{ip}"


def rate_email(limit_class: str, email: str) -> str:
    return f"{PREFIX}:rl:{limit_class}:email:{_digest(email)}"


def login_failures(email: str, ip: str) -> str:
    return f"{PREFIX}:login-fail:{_digest(f'{email}|{ip}')}"


def quota_user(user_id: uuid.UUID, day: str, name: str) -> str:
    return f"{PREFIX}:quota:u:{user_id}:{day}:{name}"


def quota_global(day: str, name: str) -> str:
    return f"{PREFIX}:quota:global:{day}:{name}"


def user_key_patterns(user_id: uuid.UUID) -> list[str]:
    """SCAN MATCH patterns for every key holding this user's counters (rate
    limit logs and daily quotas). Failed-login counters are keyed by email
    and IP, not user id; see login_failures."""
    return [f"{PREFIX}:rl:*:u:{user_id}", f"{PREFIX}:quota:u:{user_id}:*"]
