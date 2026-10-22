import hashlib
import hmac

from fastapi import HTTPException, status
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.security import DUMMY_HASH, hash_password, verify_password
from app.models import User


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
