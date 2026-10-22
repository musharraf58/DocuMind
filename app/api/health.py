import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import get_redis
from app.core.redis_keys import WORKER_HEARTBEAT

router = APIRouter(tags=["health"])

CHECK_TIMEOUT_SECONDS = 2


async def _check(probe) -> str:
    try:
        await asyncio.wait_for(probe, CHECK_TIMEOUT_SECONDS)
    except Exception:
        return "unavailable"
    return "ok"


async def _worker(redis: Redis) -> str:
    """ "ok" if a worker wrote its heartbeat recently (the key expires after
    90 s without one, see app/workers/heartbeat.py)."""
    try:
        alive = await asyncio.wait_for(redis.exists(WORKER_HEARTBEAT), CHECK_TIMEOUT_SECONDS)
    except Exception:
        return "unknown"  # Redis itself is down; reported separately
    return "ok" if alive else "unavailable"


def _report(checks: dict[str, str]) -> JSONResponse:
    healthy = all(result == "ok" for result in checks.values())
    return JSONResponse(
        {"status": "ok" if healthy else "unavailable", **checks},
        status_code=200 if healthy else 503,
    )


@router.get("/health")
async def health(
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> JSONResponse:
    """Liveness plus dependency checks. 503 if the database or Redis (the job
    queue and the shared rate limiter) can't be reached. Kept for
    compatibility (docker-compose uses it); prefer /health/live and
    /health/ready."""
    return _report(
        {
            "database": await _check(db.execute(text("SELECT 1"))),
            "redis": await _check(redis.ping()),
        }
    )


@router.get("/health/live")
async def live() -> dict:
    """The process is up and answering. Checks nothing else on purpose: a
    platform restarts the container when this fails, and restarting the API
    doesn't fix a database outage."""
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> JSONResponse:
    """Ready to serve: Postgres and Redis answer, and a worker is alive
    (otherwise uploads would never be processed). 503 with the failing part
    named otherwise; a load balancer stops sending traffic until it's 200."""
    return _report(
        {
            "database": await _check(db.execute(text("SELECT 1"))),
            "redis": await _check(redis.ping()),
            "worker": await _worker(redis),
        }
    )
