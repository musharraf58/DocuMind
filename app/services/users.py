import hashlib
import hmac
import logging

from fastapi import HTTPException, status
from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.passwords import password_problem
from app.core.security import DUMMY_HASH, hash_password, verify_password
from app.models import Document, User
from app.services.quotas import Quotas
from app.services.storage import StorageBackend

logger = logging.getLogger(__name__)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def check_registration_allowed(
    invite_code: SecretStr | None, settings: Settings | None = None
) -> None:
    """403 unless REGISTRATION_MODE lets this request create an account."""
    settings = settings or get_settings()
    mode = settings.REGISTRATION_MODE
    expected = settings.INVITE_CODE.get_secret_value()
    if mode == "closed" or (mode == "invite" and not expected):
        raise ApiError(403, "Registration is closed", code="registration_closed")
    if mode == "invite":
        given = invite_code.get_secret_value() if invite_code else ""
        if not invite_code_matches(given, expected):
            raise ApiError(403, "A valid invite code is required", code="invalid_invite_code")


def invite_code_matches(given: str, expected: str) -> bool:
    """Constant-time comparison. A plain == stops at the first differing
    character, so its timing would tell an attacker how much of a guess was
    right. compare_digest always takes the same time for equal-length inputs;
    comparing digests makes the inputs equal-length, so not even the code's
    length leaks."""
    return hmac.compare_digest(
        hashlib.sha256(given.encode()).digest(), hashlib.sha256(expected.encode()).digest()
    )


async def register_user(db: AsyncSession, email: str, password: str) -> User:
    """Create an account. REGISTRATION_MODE is checked by the API route (it's
    a rule for strangers on the internet, not for trusted code such as the
    evaluation runner)."""
    user = User(email=normalize_email(email), hashed_password=hash_password(password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        # The unique index is the real guard. Checking "does it exist?" first
        # would race with a concurrent request registering the same email.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        ) from exc
    return user


async def change_password(db: AsyncSession, user: User, new_password: str) -> int:
    """Set a new password (the caller has verified the current one) and bump
    token_version, which invalidates every token issued before now. Returns
    the new version, for the fresh token handed back to the caller."""
    problem = password_problem(new_password, user.email)
    if problem:
        raise ApiError(422, problem, code="weak_password")
    if verify_password(new_password, user.hashed_password):
        raise ApiError(422, "The new password must differ from the current one", "weak_password")
    # One UPDATE with "+ 1" in SQL: two concurrent changes can't both write
    # the same version.
    version = await db.scalar(
        update(User)
        .where(User.id == user.id)
        .values(hashed_password=hash_password(new_password), token_version=User.token_version + 1)
        .returning(User.token_version)
    )
    await db.commit()
    return version


async def delete_account(
    db: AsyncSession, storage: StorageBackend, redis: Redis, user: User, password: str
) -> None:
    """DELETE /auth/me: the password confirms it, then purge_user."""
    if not verify_password(password, user.hashed_password):
        raise ApiError(403, "Password is incorrect", code="wrong_password")
    await purge_user(db, storage, redis, user)


async def purge_user(
    db: AsyncSession, storage: StorageBackend, redis: Redis | None, user: User
) -> int:
    """Delete the account and everything kept about it: the stored files, the
    rows (by ON DELETE CASCADE: collections, documents, chunks, conversations,
    messages, usage events) and the user's Redis counters. The one cleanup
    path, used by DELETE /auth/me and by retention. Returns how many stored
    files were deleted.

    Order matters. Files go first, then the rows, like delete_collection: if
    something fails midway the user can simply retry (deleting a missing file
    is fine), whereas rows first would leave files nobody could ever find.
    Redis goes last: the counters expire within two days anyway, so a Redis
    outage after the rows are gone is only logged, never an error for a user
    whose account no longer exists.

    A document the worker is processing right now loses its row; the job
    notices (ClaimLostError) and stops without writing anything."""
    user_id = user.id  # read before the commit below, which ends the session's hold on `user`
    keys = (
        await db.scalars(select(Document.storage_key).where(Document.owner_id == user_id))
    ).all()
    await storage.delete_many(keys)
    await db.delete(user)
    await db.commit()
    if redis is not None:
        try:
            await redis.delete(*Quotas(redis).user_keys(user_id))  # one command
        except (RedisError, OSError):
            logger.warning("Deleted account %s but could not clear its Redis keys", user_id)
    return len(keys)


async def authenticate(db: AsyncSession, email: str, password: str) -> User | None:
    """Return the user if the credentials are valid, else None.

    Callers must not reveal *which* part was wrong.
    """
    user = await db.scalar(select(User).where(User.email == normalize_email(email)))
    if user is None:
        # Spend the same hashing time as a real check so response time doesn't
        # reveal whether the email is registered.
        verify_password(password, DUMMY_HASH)
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
