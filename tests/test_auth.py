import time
import uuid

import jwt
from httpx import AsyncClient

from app.core.config import get_settings
from tests.helpers import PASSWORD, auth_headers, error_body, login, register


async def test_register_returns_user_without_password_data(client: AsyncClient) -> None:
    response = await register(client, "Alice@Example.com")

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "alice@example.com"  # stored lowercase
    assert set(body) == {"id", "email", "created_at"}


async def test_duplicate_email_is_rejected_case_insensitively(client: AsyncClient) -> None:
    assert (await register(client, "alice@example.com")).status_code == 201

    for variant in ("alice@example.com", "ALICE@example.com", "Alice@Example.COM"):
        assert (await register(client, variant)).status_code == 409


async def test_register_validates_input(client: AsyncClient) -> None:
    assert (await register(client, "not-an-email")).status_code == 422
    assert (await register(client, "bob@example.com", "short")).status_code == 422


async def test_login_succeeds_and_email_case_does_not_matter(client: AsyncClient) -> None:
    await register(client, "alice@example.com")

    response = await login(client, "ALICE@example.com")

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]


async def test_wrong_password_and_wrong_email_give_identical_errors(client: AsyncClient) -> None:
    await register(client, "alice@example.com")

    wrong_password = await login(client, "alice@example.com", "not-the-password")
    wrong_email = await login(client, "nobody@example.com")

    assert wrong_password.status_code == wrong_email.status_code == 401
    assert error_body(wrong_password) == error_body(wrong_email)


async def test_me_returns_current_user(client: AsyncClient) -> None:
    headers = await auth_headers(client, "alice@example.com")

    response = await client.get("/auth/me", headers=headers)

    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"


async def test_missing_token_is_401(client: AsyncClient) -> None:
    response = await client.get("/auth/me")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_garbage_token_is_401(client: AsyncClient) -> None:
    response = await client.get("/auth/me", headers={"Authorization": "Bearer not.a.token"})

    assert response.status_code == 401


async def test_expired_token_is_401(client: AsyncClient) -> None:
    headers = await auth_headers(client, "alice@example.com")
    me = (await client.get("/auth/me", headers=headers)).json()
    settings = get_settings()
    expired = jwt.encode(
        {"sub": me["id"], "exp": int(time.time()) - 60},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})

    assert response.status_code == 401


async def test_token_signed_with_wrong_key_is_401(client: AsyncClient) -> None:
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": int(time.time()) + 600},
        "a-completely-different-secret-key-0123456789",
        algorithm="HS256",
    )

    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401


async def test_valid_token_for_nonexistent_user_is_401(client: AsyncClient) -> None:
    settings = get_settings()
    token = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": int(time.time()) + 600},
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401


async def test_no_response_ever_contains_the_hash_or_password(client: AsyncClient) -> None:
    responses = [
        await register(client, "alice@example.com"),
        await login(client, "alice@example.com"),
    ]
    headers = {"Authorization": f"Bearer {responses[1].json()['access_token']}"}
    responses.append(await client.get("/auth/me", headers=headers))

    for response in responses:
        assert "$argon2" not in response.text
        assert "hashed_password" not in response.text
        assert PASSWORD not in response.text


async def test_non_bearer_scheme_is_401(client: AsyncClient) -> None:
    response = await client.get("/auth/me", headers={"Authorization": "Basic YWxpY2U6cGFzcw=="})

    assert response.status_code == 401


async def test_openapi_advertises_plain_http_bearer(client: AsyncClient) -> None:
    schemes = (await client.get("/openapi.json")).json()["components"]["securitySchemes"]

    assert schemes == {"HTTPBearer": {"type": "http", "scheme": "bearer"}}
