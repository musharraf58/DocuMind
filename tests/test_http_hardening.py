import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app
from tests.helpers import auth_headers


def make_settings(**changes):
    return get_settings().model_copy(update=changes)


@pytest.fixture
async def build_client():
    """A client for an app built with different settings (for things decided
    when the app is created: docs, CORS)."""
    clients = []

    async def build(**changes) -> AsyncClient:
        app = create_app(make_settings(**changes))
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client

    yield build
    for client in clients:
        await client.aclose()


# --- security headers -------------------------------------------------------


async def test_security_headers_on_every_response(client):
    for response in [await client.get("/health"), await client.get("/no-such-route")]:
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert "default-src 'none'" in response.headers["Content-Security-Policy"]
        # Development: no HSTS (it would pin localhost to HTTPS in the browser).
        assert "Strict-Transport-Security" not in response.headers


async def test_authenticated_responses_are_not_stored(client):
    headers = await auth_headers(client, "cache@example.com")
    response = await client.get("/collections", headers=headers)
    assert response.headers["Cache-Control"] == "no-store"


async def test_auth_responses_are_not_stored(client):
    await auth_headers(client, "login-cache@example.com")
    response = await client.post(
        "/auth/login",
        data={"username": "login-cache@example.com", "password": "correct-horse-battery"},
    )
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"


async def test_public_responses_have_no_no_store(client):
    response = await client.get("/health")
    assert response.headers.get("Cache-Control") != "no-store"


async def test_hsts_in_production(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "APP_ENV", "production")
    response = await client.get("/health")
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")


# --- request body size --------------------------------------------------------


async def test_large_json_body_is_refused(client):
    limit = get_settings().MAX_REQUEST_BODY_KB * 1024
    response = await client.post(
        "/auth/register",
        content=b'{"email": "' + b"a" * limit + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


async def test_large_chunked_body_is_refused(client):
    """No Content-Length (chunked): counted as it arrives."""
    limit = get_settings().MAX_REQUEST_BODY_KB * 1024

    async def body():
        for _ in range(limit // 1024 + 2):
            yield b"a" * 1024

    response = await client.post(
        "/auth/register", content=body(), headers={"Content-Type": "application/json"}
    )
    assert "content-length" not in response.request.headers
    assert response.status_code == 413


async def test_small_body_passes(client):
    response = await client.post(
        "/auth/register", json={"email": "small@example.com", "password": "correct-horse-battery"}
    )
    assert response.status_code == 201


async def test_uploads_keep_their_own_limit(client):
    """A file bigger than the JSON limit but within MAX_UPLOAD_MB is fine."""
    from tests.helpers import create_collection, upload

    headers = await auth_headers(client, "upload-size@example.com")
    collection_id = await create_collection(client, headers)
    content = b"word " * (get_settings().MAX_REQUEST_BODY_KB * 1024 // 5 + 1000)
    response = await upload(client, headers, collection_id, "big.txt", content)
    assert response.status_code == 202


# --- docs ---------------------------------------------------------------------


async def test_docs_can_be_disabled(build_client):
    client = await build_client(DOCS_ENABLED=False)
    for path in ["/docs", "/redoc", "/openapi.json"]:
        assert (await client.get(path)).status_code == 404


async def test_docs_enabled_by_default_in_development(build_client):
    client = await build_client(DOCS_ENABLED=True)
    assert (await client.get("/openapi.json")).status_code == 200
    docs = await client.get("/docs")
    assert docs.status_code == 200
    # Swagger UI loads from a CDN: the strict API CSP would break it.
    assert "Content-Security-Policy" not in docs.headers


# --- CORS -----------------------------------------------------------------------


async def test_cors_allows_listed_origin(build_client):
    client = await build_client(CORS_ORIGINS="https://app.example.com, https://other.example.com")
    response = await client.options(
        "/collections",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://app.example.com"
    assert "Access-Control-Allow-Credentials" not in response.headers


async def test_cors_refuses_other_origins(build_client):
    client = await build_client(CORS_ORIGINS="https://app.example.com")
    preflight = await client.options(
        "/collections",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "Access-Control-Allow-Origin" not in preflight.headers
    simple = await client.get("/no-such-route", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in simple.headers


async def test_cors_exposes_rate_limit_and_request_id_headers(build_client):
    client = await build_client(CORS_ORIGINS="https://app.example.com")
    response = await client.get("/no-such-route", headers={"Origin": "https://app.example.com"})
    assert response.headers["Access-Control-Allow-Origin"] == "https://app.example.com"
    exposed = response.headers["Access-Control-Expose-Headers"].lower()
    for name in ["retry-after", "ratelimit-remaining", "x-request-id"]:
        assert name in exposed


async def test_no_cors_by_default(client):
    response = await client.get("/health", headers={"Origin": "https://app.example.com"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_wildcard_is_refused():
    with pytest.raises(ValueError, match="'\\*' is not allowed"):
        get_settings().model_validate({**get_settings().model_dump(), "CORS_ORIGINS": "*"})
    with pytest.raises(ValueError, match="not allowed"):
        get_settings().model_validate(
            {**get_settings().model_dump(), "CORS_ORIGINS": "https://a.example, *"}
        )
