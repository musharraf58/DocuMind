"""Per-user upload caps: total storage, number of documents, and documents
waiting or being processed at once.

The check runs under a lock on the user's row (SELECT ... FOR UPDATE), held
until the upload commits. Without it, two uploads sent at the same moment
could both see "49 of 50 documents" and both be stored. With it, the second
waits until the first has committed and then sees 50.
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.models import Document, DocumentStatus, User

MB = 1024 * 1024
# A pending document usually starts within seconds; this is only a hint.
PENDING_RETRY_AFTER_SECONDS = 30


def storage_full(limit_bytes: int, used_bytes: int) -> ApiError:
    return ApiError(
        403,
        f"Storage limit reached: {used_bytes / MB:.1f} of {limit_bytes // MB} MB used. "
        "Delete documents to make room.",
        code="storage_limit_reached",
        extra={"limit_bytes": limit_bytes, "used_bytes": used_bytes},
    )


async def check_upload_caps(
    db: AsyncSession, user_id: uuid.UUID, settings: Settings | None = None
) -> int:
    """Raise if any cap is reached; otherwise return the storage bytes left,
    which becomes the size limit for this upload. Locks the user's row until
    the caller's transaction ends (the upload's commit)."""
    settings = settings or get_settings()
    await db.execute(select(User.id).where(User.id == user_id).with_for_update())
    count, used_bytes, in_progress = (
        await db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(Document.size_bytes), 0),
                func.count().filter(
                    Document.status.in_([DocumentStatus.pending, DocumentStatus.processing])
                ),
            ).where(Document.owner_id == user_id)
        )
    ).one()
    used_bytes = int(used_bytes)  # sum() of a bigint is a numeric (Decimal) in Postgres

    if count >= settings.MAX_DOCUMENTS_PER_USER:
        raise ApiError(
            403,
            f"Document limit reached: {count} of {settings.MAX_DOCUMENTS_PER_USER} documents. "
            "Delete documents to upload more.",
            code="document_limit_reached",
            extra={"limit": settings.MAX_DOCUMENTS_PER_USER, "used": count},
        )
    if in_progress >= settings.MAX_PENDING_DOCUMENTS_PER_USER:
        raise ApiError(
            429,
            f"{in_progress} documents are still waiting or being processed (at most "
            f"{settings.MAX_PENDING_DOCUMENTS_PER_USER} at once). Try again when one is ready.",
            code="pending_limit_reached",
            headers={"Retry-After": str(PENDING_RETRY_AFTER_SECONDS)},
            extra={"limit": settings.MAX_PENDING_DOCUMENTS_PER_USER, "used": in_progress},
        )
    limit_bytes = settings.MAX_STORAGE_MB_PER_USER * MB
    if used_bytes >= limit_bytes:
        raise storage_full(limit_bytes, used_bytes)
    return limit_bytes - used_bytes
