from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import DUMMY_HASH, hash_password, verify_password
from app.models import User


def normalize_email(email: str) -> str:
    return email.strip().lower()


async def register_user(db: AsyncSession, email: str, password: str) -> User:
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
