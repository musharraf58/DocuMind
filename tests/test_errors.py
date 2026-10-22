import logging

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app
from tests.helpers import auth_headers

SECRET = "connection string postgresql://user:hunter2@db/prod"


@pytest.fixture
async def broken_client(client):
    """The app with a dependency that raises an unexpected exception."""

    async def explode():
        raise RuntimeError(SECRET)
        yield  # pragma: no cover

    app.dependency_overrides[get_db] = explode
    # raise_app_exceptions=False: behave like a real server, which never sees
    # the exception (our middleware has already answered).
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as ac:
        yield ac


async def test_unhandled_exception_is_a_generic_500(broken_client, caplog):
    with caplog.at_level(logging.ERROR, logger="app.core.errors"):
        response = await broken_client.post(
            "/auth/register", json={"email": "a@example.com", "password": "x" * 12}
        )
    assert response.status_code == 500
    body = response.json()
    assert body == {
        "detail": "Internal server error",
        "code": "internal_error",
        "request_id": response.headers["X-Request-ID"],
    }
    # Nothing about the cause in the response...
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text
    # ...but the log has the stack trace, under the same request id.
    record = next(r for r in caplog.records if r.name == "app.core.errors")
    assert record.exc_info is not None
    assert record.request_id == body["request_id"]


async def test_http_errors_have_the_shape(client):
    response = await client.get("/auth/me")
    assert response.status_code == 401
    body = response.json()
    assert body["detail"] == "Could not validate credentials"
    assert body["code"] == "unauthorized"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_not_found_route_has_the_shape(client):
    response = await client.get("/no-such-route")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_not_found_resource_has_the_shape(client):
    headers = await auth_headers(client, "shape@example.com")
    response = await client.get(
        "/collections/00000000-0000-0000-0000-000000000000", headers=headers
    )
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_validation_errors_do_not_echo_input(client):
    response = await client.post(
        "/auth/register", json={"email": "not-an-email", "password": "pw-secret"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["detail"] == "Request validation failed"
    assert {tuple(e["loc"]) for e in body["errors"]} >= {("body", "email")}
    assert "pw-secret" not in response.text
    assert "not-an-email" not in response.text
    assert body["request_id"]


async def test_body_too_large_has_the_shape(client):
    response = await client.post(
        "/auth/register", content=b"{}", headers={"Content-Length": str(10**12)}
    )
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
