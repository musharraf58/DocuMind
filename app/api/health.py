import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import get_redis

router = APIRouter()

CHECK_TIMEOUT_SECONDS = 2


async def _check(probe) -> str:
    try:
        await asyncio.wait_for(probe, CHECK_TIMEOUT_SECONDS)
    except Exception:
        return "unavailable"
    return "ok"


@router.get("/health")
async def health(
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> JSONResponse:
    """Liveness plus dependency checks. 503 if the database or Redis (the job
    queue and the shared rate limiter) can't be reached."""
    checks = {
        "database": await _check(db.execute(text("SELECT 1"))),
        "redis": await _check(redis.ping()),
    }
    healthy = all(result == "ok" for result in checks.values())
    return JSONResponse(
        {"status": "ok" if healthy else "unavailable", **checks},
        status_code=200 if healthy else 503,
    )
