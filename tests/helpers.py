from httpx import AsyncClient

PASSWORD = "correct-horse-battery"


async def register(client: AsyncClient, email: str, password: str = PASSWORD):
    return await client.post("/auth/register", json={"email": email, "password": password})


async def login(client: AsyncClient, email: str, password: str = PASSWORD):
    # OAuth2 login is form-encoded, with the email in the "username" field.
    return await client.post("/auth/login", data={"username": email, "password": password})


async def auth_headers(client: AsyncClient, email: str, password: str = PASSWORD) -> dict:
    """Register a user, log in, and return ready-to-use Authorization headers."""
    assert (await register(client, email, password)).status_code == 201
    response = await login(client, email, password)
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
