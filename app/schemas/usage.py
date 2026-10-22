from datetime import datetime

from pydantic import BaseModel

from app.models import UsageKind


class KindTotals(BaseModel):
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class PeriodUsage(BaseModel):
    since: datetime
    total: KindTotals
    # Every kind is listed, with zeros when unused.
    by_kind: dict[UsageKind, KindTotals]


class QuotaStatus(BaseModel):
    used: int
    limit: int


class Quotas(BaseModel):
    resets_at: datetime
    questions: QuotaStatus
    uploads: QuotaStatus
    embed_tokens: QuotaStatus


class UsageRead(BaseModel):
    today: PeriodUsage
    month: PeriodUsage
    # Live counters from Redis (what the limits enforce); null if Redis is down.
    quotas: Quotas | None
