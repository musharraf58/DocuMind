import time

import jwt
from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import create_access_token, decode_access_token
from app.models import User
from tests.helpers import PASSWORD, auth_headers, login

NEW_PASSWORD = "a-brand-new-passphrase"


async def change(client, headers, current=PASSWORD, new=NEW_PASSWORD):
    return await client.post(
        "/auth/change-password",
        json={"current_password": current, "new_password": new},
        headers=headers,
    )


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_token_carries_the_version():
    import uuid

    user_id = uuid.uuid4()
    claims = decode_access_token(create_access_token(user_id, 7))
    assert claims.user_id == user_id
    assert claims.token_version == 7


def test_token_without_version_is_invalid():
    """Tokens issued before token versions existed stop working."""
    settings = get_settings()
    old = jwt.encode(
        {"sub": "6f9c0b2e-1c1a-4f0e-9a43-6a1c5f6e0b11", "exp": int(time.time()) + 600},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_access_token(old) is None


def test_non_integer_version_is_invalid():
    settings = get_settings()
    for bad in ["0", None, 1.5, True]:
        token = jwt.encode(
            {"sub": "6f9c0b2e-1c1a-4f0e-9a43-6a1c5f6e0b11", "ver": bad, "exp": time.time() + 60},
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        assert decode_access_token(token) is None, bad


async def test_password_change_invalidates_older_tokens(client):
    headers = await auth_headers(client, "alice@example.com")
    # A second session, e.g. on another device.
    other_device = bearer((await login(client, "alice@example.com")).json()["access_token"])

    response = await change(client, headers)
    assert response.status_code == 200
    fresh = bearer(response.json()["access_token"])

    # Both old tokens are dead, though neither has expired.
    assert (await client.get("/auth/me", headers=headers)).status_code == 401
    assert (await client.get("/auth/me", headers=other_device)).status_code == 401
    # The fresh one works, and so does logging in with the new password.
    assert (await client.get("/auth/me", headers=fresh)).status_code == 200
    assert (await login(client, "alice@example.com", NEW_PASSWORD)).status_code == 200
    assert (await login(client, "alice@example.com", PASSWORD)).status_code == 401


async def test_version_is_bumped_in_the_database(client, db):
    headers = await auth_headers(client, "bob@example.com")
    await change(client, headers)
    user = await db.scalar(select(User).where(User.email == "bob@example.com"))
    assert user.token_version == 1


async def test_wrong_current_password_changes_nothing(client):
    headers = await auth_headers(client, "carol@example.com")
    response = await change(client, headers, current="definitely-not-it")
    assert response.status_code == 403
    assert response.json()["code"] == "wrong_password"
    assert (await client.get("/auth/me", headers=headers)).status_code == 200
    assert (await login(client, "carol@example.com")).status_code == 200


async def test_new_password_follows_the_policy(client):
    headers = await auth_headers(client, "dave@example.com")
    for new in ["short", "password1234", "dave@example.com", PASSWORD]:
        response = await change(client, headers, new=new)
        assert response.status_code == 422, new
        assert new not in response.text
    assert (await client.get("/auth/me", headers=headers)).status_code == 200


async def test_requires_authentication(client):
    response = await change(client, {})
    assert response.status_code == 401


async def test_wrong_current_password_is_slowed_down_like_login(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "LOGIN_FREE_ATTEMPTS", 2)
    headers = await auth_headers(client, "erin@example.com")
    statuses = [
        (await change(client, headers, current="nope-nope-nope")).status_code for _ in range(4)
    ]
    assert statuses == [403, 403, 403, 429]


async def test_change_password_is_rate_limited(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "RATE_LIMIT_ACCOUNT", "2/minute")
    headers = await auth_headers(client, "frank@example.com")
    statuses = [(await change(client, headers, new="x" * 5)).status_code for _ in range(3)]
    assert statuses == [422, 422, 429]
