"""Inbound rate limits as FastAPI dependencies, one per endpoint class.

Each endpoint belongs to exactly one class:

    login     POST /auth/login             per IP, and per email (in the route)
    register  POST /auth/register          per IP
    account   change password, delete me   per user
    ask       /ask, /ask/stream            per user
    upload    POST .../documents           per user
    default   every other authenticated    per user

As dependencies they run after authentication, so per-user limits key by
the user id (not the IP: many users can share one IP, e.g. an office).
"""

from enum import StrEnum
from typing import Annotated

from fastapi import Depends, Request, Response
from redis.asyncio import Redis

from app.api.deps import CurrentUser
from app.core import redis_keys
from app.core.client_ip import client_ip
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.rate_limit import Decision, InboundLimiter, Rate, memory_fallback, outage_log
from app.core.redis import get_redis


class LimitClass(StrEnum):
    login = "login"
    register = "register"
    account = "account"
    ask = "ask"
    upload = "upload"
    default = "default"


def rate_for(limit_class: LimitClass, settings: Settings) -> Rate:
    setting = {
        LimitClass.login: settings.RATE_LIMIT_LOGIN_PER_IP,
        LimitClass.register: settings.RATE_LIMIT_REGISTER_PER_IP,
        LimitClass.account: settings.RATE_LIMIT_ACCOUNT,
        LimitClass.ask: settings.RATE_LIMIT_ASK,
        LimitClass.upload: settings.RATE_LIMIT_UPLOAD,
        LimitClass.default: settings.RATE_LIMIT_DEFAULT,
    }[limit_class]
    return Rate.parse(setting)


def get_inbound_limiter(redis: Annotated[Redis, Depends(get_redis)]) -> InboundLimiter:
    return InboundLimiter(redis, memory_fallback, outage_log)


Limiter = Annotated[InboundLimiter, Depends(get_inbound_limiter)]


def too_many_requests(decision: Decision, what: str = "requests") -> ApiError:
    return ApiError(
        429,
        f"Too many {what}; retry in {decision.retry_after} seconds",
        code="rate_limited",
        headers=decision.headers(),
        extra={"retry_after": decision.retry_after},
    )


async def enforce(
    limiter: InboundLimiter,
    key: str,
    rate: Rate,
    response: Response | None = None,
    *,
    fallback: bool = False,
    what: str = "requests",
) -> None:
    """Count one request against `key`; raise 429 if over the limit. On
    success, the RateLimit-* headers go on the response (FastAPI copies the
    injected Response's headers onto the real one)."""
    if not get_settings().RATE_LIMIT_ENABLED:
        return
    decision = await limiter.hit(key, rate, fallback=fallback)
    if decision is None:  # Redis down: fail open (already logged)
        return
    if not decision.allowed:
        raise too_many_requests(decision, what)
    if response is not None:
        response.headers.update(decision.headers())


def per_user(limit_class: LimitClass):
    """Dependency: limit the current user's requests in this class."""

    async def dependency(user: CurrentUser, limiter: Limiter, response: Response) -> None:
        rate = rate_for(limit_class, get_settings())
        await enforce(limiter, redis_keys.rate_user(limit_class, user.id), rate, response)

    dependency.__name__ = f"rate_limit_{limit_class}"
    return dependency


def per_ip(limit_class: LimitClass, *, fallback: bool = False):
    """Dependency: limit requests in this class by client IP (for endpoints
    used before logging in)."""

    async def dependency(request: Request, limiter: Limiter, response: Response) -> None:
        rate = rate_for(limit_class, get_settings())
        key = redis_keys.rate_ip(limit_class, client_ip(request))
        await enforce(limiter, key, rate, response, fallback=fallback)

    dependency.__name__ = f"rate_limit_{limit_class}_by_ip"
    return dependency


async def enforce_login_email(limiter: InboundLimiter, email: str, response: Response) -> None:
    """Login attempts per email, from any IP: stops one account being guessed
    from many addresses. (The cost: a flood can make that account's own
    logins wait too, for at most one window. See the README.)"""
    rate = Rate.parse(get_settings().RATE_LIMIT_LOGIN_PER_EMAIL)
    key = redis_keys.rate_email(LimitClass.login, email)
    await enforce(limiter, key, rate, response, fallback=True, what="login attempts")


limit_default = Depends(per_user(LimitClass.default))
limit_ask = Depends(per_user(LimitClass.ask))
limit_upload = Depends(per_user(LimitClass.upload))
limit_account = Depends(per_user(LimitClass.account))
limit_register = Depends(per_ip(LimitClass.register))
limit_login = Depends(per_ip(LimitClass.login, fallback=True))
