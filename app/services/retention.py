"""Retention: delete ordinary accounts older than RETENTION_DAYS, with
everything they own. Off by default (0).

Meant for a public demo deployment on free tiers: visitors create accounts
and upload documents, and without an expiry the database and the bucket
only ever grow. Each account goes through users.purge_user, the same cleanup
DELETE /auth/me uses: stored files, then the rows (cascade), then the Redis
counters. The demo account (is_demo) is never deleted.

Runs with the periodic maintenance (inline mode) or from Celery beat. The
log says how many accounts and files went, never which: an account id or
email in a log would outlive the account it describes.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import clock
from app.core.config import Settings, get_settings
from app.models import User
from app.services.storage import StorageBackend
from app.services.users import purge_user

logger = logging.getLogger(__name__)

# At most this many accounts per run; the next run continues.
BATCH = 50


@dataclass
class RetentionReport:
    accounts: int = 0
    files: int = 0
    failed: int = 0


async def purge_expired_accounts(
    sessions: async_sessionmaker[AsyncSession],
    storage: StorageBackend,
    redis: Redis | None,
    settings: Settings | None = None,
) -> RetentionReport:
    settings = settings or get_settings()
    report = RetentionReport()
    if settings.RETENTION_DAYS <= 0:
        return report
    cutoff = clock.utcnow() - timedelta(days=settings.RETENTION_DAYS)
    async with sessions() as db:
        expired = (
            await db.scalars(
                select(User.id)
                .where(User.created_at < cutoff, User.is_demo.is_(False))
                .order_by(User.created_at)
                .limit(BATCH)
            )
        ).all()
    for user_id in expired:
        # A fresh session per account: one failure doesn't stop the rest.
        try:
            async with sessions() as db:
                user = await db.get(User, user_id)
                if user is not None:
                    report.files += await purge_user(db, storage, redis, user)
                    report.accounts += 1
        except Exception:
            report.failed += 1
            logger.exception("Retention: could not delete an expired account; retrying next run")
    if report.accounts or report.failed:
        logger.info(
            "Retention: deleted %d accounts older than %d days (%d stored files); %d failed",
            report.accounts,
            settings.RETENTION_DAYS,
            report.files,
            report.failed,
        )
    return report
