import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from app.core.config import get_settings

# PasswordHash.recommended() uses argon2id with sane default parameters.
_password_hash = PasswordHash.recommended()

# Verified against when a login names an email that doesn't exist, so that
# "unknown email" costs the same time as "wrong password" (see authenticate()).
DUMMY_HASH = _password_hash.hash("not-a-real-password")


@dataclass(frozen=True)
class TokenClaims:
    user_id: uuid.UUID
    token_version: int


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return _password_hash.verify(password, hashed_password)


def create_access_token(user_id: uuid.UUID, token_version: int) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        # The user's token_version when the token was issued. A password change
        # bumps the user's version, so every older token stops matching.
        "ver": token_version,
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenClaims | None:
    """The claims inside a valid token, or None if it is invalid or expired.
    Tokens without "ver" (issued before token versions existed) are invalid."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            # Pin the allowed algorithm; never trust the token's own header.
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub", "ver"]},
        )
        version = payload["ver"]
        if not isinstance(version, int) or isinstance(version, bool):
            return None
        return TokenClaims(uuid.UUID(payload["sub"]), version)
    except (jwt.InvalidTokenError, ValueError):
        return None
