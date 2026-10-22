from datetime import datetime

from fastapi import APIRouter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession, QuotasDep
from app.api.limits import limit_default
from app.models import UsageEvent, UsageKind
from app.schemas.usage import KindTotals, PeriodUsage, QuotaStatus, UsageRead
from app.schemas.usage import Quotas as QuotasRead
from app.services.quotas import EMBED_TOKENS, QUESTIONS, UPLOADS, Quotas, day_start

router = APIRouter(tags=["usage"], dependencies=[limit_default])


@router.get("/usage", response_model=UsageRead)
async def get_usage(user: CurrentUser, db: DbSession, quotas: QuotasDep):
    """Your usage: provider requests and tokens for today and this month, per
    kind (from the usage history in Postgres), and your daily quotas (the
    live counters the limits enforce). Only ever your own."""
    now = quotas.now()
    today = day_start(now, quotas.reset_hour)
    month = today.replace(day=1)
    return UsageRead(
        today=await _period(db, user.id, today),
        month=await _period(db, user.id, month),
        quotas=await _quotas(quotas, user.id),
    )


async def _period(db: AsyncSession, user_id, since: datetime) -> PeriodUsage:
    rows = await db.execute(
        select(
            UsageEvent.kind,
            func.count(),
            func.coalesce(func.sum(UsageEvent.tokens_in), 0),
            func.coalesce(func.sum(UsageEvent.tokens_out), 0),
        )
        # user_id first: owner-scoped, and it matches the (user_id, created_at) index.
        .where(UsageEvent.user_id == user_id, UsageEvent.created_at >= since)
        .group_by(UsageEvent.kind)
    )
    by_kind = {kind: KindTotals() for kind in UsageKind}
    for kind, requests, tokens_in, tokens_out in rows:
        by_kind[kind] = KindTotals(requests=requests, tokens_in=tokens_in, tokens_out=tokens_out)
    total = KindTotals(
        requests=sum(t.requests for t in by_kind.values()),
        tokens_in=sum(t.tokens_in for t in by_kind.values()),
        tokens_out=sum(t.tokens_out for t in by_kind.values()),
    )
    return PeriodUsage(since=since, total=total, by_kind=by_kind)


async def _quotas(quotas: Quotas, user_id) -> QuotasRead | None:
    statuses = {}
    for name in (QUESTIONS, UPLOADS, EMBED_TOKENS):
        used = await quotas.used(user_id, name)
        if used is None:  # Redis down
            return None
        statuses[name] = QuotaStatus(used=max(0, used), limit=quotas.limit(name))
    return QuotasRead(resets_at=quotas.resets_at(), **statuses)
