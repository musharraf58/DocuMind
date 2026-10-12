import io
import zipfile
from pathlib import Path

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


async def create_collection(client: AsyncClient, headers: dict, name: str = "Docs") -> str:
    response = await client.post("/collections", json={"name": name}, headers=headers)
    assert response.status_code == 201
    return response.json()["id"]


async def upload(
    client: AsyncClient,
    headers: dict,
    collection_id: str,
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
):
    return await client.post(
        f"/collections/{collection_id}/documents",
        files={"file": (filename, content, content_type)},
        headers=headers,
    )


def files_on_disk(root: Path) -> list[Path]:
    """Every file (including leftover .part files) under the storage root."""
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def make_docx(text: str = "hello") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f"<w:document>{text}</w:document>")
    return buffer.getvalue()


def make_pdf(text: str = "hello") -> bytes:
    return f"%PDF-1.4\n{text}\n%%EOF".encode()
