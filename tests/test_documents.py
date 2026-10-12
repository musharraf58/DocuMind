import hashlib

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from tests.helpers import (
    auth_headers,
    create_collection,
    files_on_disk,
    make_docx,
    make_pdf,
    upload,
)

MB = 1024 * 1024

VALID_FILES = {
    "report.pdf": (make_pdf(), "application/pdf"),
    "letter.docx": (
        make_docx(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "notes.md": ("# Title\n\nCafé ☕\n".encode(), "text/markdown"),
    "plain.txt": (b"just some text", "text/plain"),
}


@pytest.fixture
async def alice(client: AsyncClient) -> dict:
    return await auth_headers(client, "alice@example.com")


@pytest.fixture
async def bob(client: AsyncClient) -> dict:
    return await auth_headers(client, "bob@example.com")


# ---------- upload: happy path ----------


@pytest.mark.parametrize("filename", VALID_FILES)
async def test_valid_upload_of_each_type(client, alice, storage_dir, filename) -> None:
    content, expected_type = VALID_FILES[filename]
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, filename, content)

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == filename
    assert body["content_type"] == expected_type
    assert body["size_bytes"] == len(content)
    assert body["content_hash"] == hashlib.sha256(content).hexdigest()
    assert body["status"] == "pending"
    assert body["error_message"] is None
    assert body["collection_id"] == collection_id
    # storage_key must never be exposed
    assert "storage_key" not in body
    assert "storage_key" not in response.text
    # exactly one stored file, with the exact bytes
    (stored,) = files_on_disk(storage_dir)
    assert stored.read_bytes() == content


async def test_client_content_type_is_not_trusted(client, alice) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(
        client, alice, collection_id, "a.pdf", make_pdf(), content_type="text/html"
    )

    assert response.json()["content_type"] == "application/pdf"


async def test_uppercase_extension_is_accepted(client, alice) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, "SCAN.PDF", make_pdf())

    assert response.status_code == 201


async def test_multibyte_character_split_across_chunks_is_valid_utf8(
    client, alice, storage_dir
) -> None:
    collection_id = await create_collection(client, alice)
    # "é" is 2 bytes; place it so it straddles the 1 MB chunk boundary.
    content = b"a" * (MB - 1) + "é".encode() + b"tail"

    response = await upload(client, alice, collection_id, "big.txt", content)

    assert response.status_code == 201
    assert response.json()["size_bytes"] == len(content)


# ---------- upload: rejections (each must leave no file behind) ----------


@pytest.mark.parametrize(
    "filename", ["virus.exe", "image.png", "archive.zip", "noextension", ".pdf"]
)
async def test_wrong_extension_is_415(client, alice, storage_dir, filename) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, filename, make_pdf())

    assert response.status_code == 415
    assert files_on_disk(storage_dir) == []


@pytest.mark.parametrize("filename", VALID_FILES)
async def test_empty_file_is_rejected(client, alice, storage_dir, filename) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, filename, b"")

    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()
    assert files_on_disk(storage_dir) == []


async def test_fake_pdf_is_rejected(client, alice, storage_dir) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(
        client, alice, collection_id, "fake.pdf", b"<html>not a pdf</html>", "application/pdf"
    )

    assert response.status_code == 422
    assert files_on_disk(storage_dir) == []


async def test_fake_docx_is_rejected(client, alice, storage_dir) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, "fake.docx", b"plain text, not a zip")

    assert response.status_code == 422
    assert files_on_disk(storage_dir) == []


@pytest.mark.parametrize("filename", ["bad.txt", "bad.md"])
async def test_invalid_utf8_is_rejected(client, alice, storage_dir, filename) -> None:
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, filename, b"ok then \xff\xfe broken")

    assert response.status_code == 422
    assert "utf-8" in response.json()["detail"].lower()
    assert files_on_disk(storage_dir) == []


async def test_truncated_utf8_at_end_is_rejected(client, alice, storage_dir) -> None:
    collection_id = await create_collection(client, alice)

    # Ends in the first byte of a 2-byte character.
    response = await upload(client, alice, collection_id, "cut.txt", b"hello \xc3")

    assert response.status_code == 422
    assert files_on_disk(storage_dir) == []


async def test_oversize_declared_up_front_is_413(client, alice, storage_dir, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_MB", 1)
    collection_id = await create_collection(client, alice)

    # Content-Length is far over the limit, so the middleware refuses it unread.
    response = await upload(client, alice, collection_id, "huge.txt", b"a" * (3 * MB))

    assert response.status_code == 413
    assert files_on_disk(storage_dir) == []


async def test_oversize_mid_stream_leaves_no_partial_file(
    client, alice, storage_dir, monkeypatch
) -> None:
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_MB", 1)
    collection_id = await create_collection(client, alice)

    # 10 bytes over the limit: small enough to pass the Content-Length pre-check,
    # so the limit is hit in the streaming code *after* the first 1 MB chunk has
    # already been written to disk.
    response = await upload(client, alice, collection_id, "edge.txt", b"a" * (MB + 10))

    assert response.status_code == 413
    assert files_on_disk(storage_dir) == []  # no final file and no .part file
    assert (await client.get(f"/collections/{collection_id}", headers=alice)).status_code == 200


async def test_file_exactly_at_the_limit_is_accepted(
    client, alice, storage_dir, monkeypatch
) -> None:
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_MB", 1)
    collection_id = await create_collection(client, alice)

    response = await upload(client, alice, collection_id, "exact.txt", b"a" * MB)

    assert response.status_code == 201


# ---------- upload: duplicates ----------


async def test_duplicate_in_same_collection_is_409_and_leaves_no_extra_file(
    client, alice, storage_dir
) -> None:
    collection_id = await create_collection(client, alice)
    assert (await upload(client, alice, collection_id, "a.txt", b"same bytes")).status_code == 201

    # Even under a different filename, identical bytes are a duplicate.
    again = await upload(client, alice, collection_id, "renamed.txt", b"same bytes")

    assert again.status_code == 409
    assert len(files_on_disk(storage_dir)) == 1


async def test_same_file_is_allowed_in_another_collection(client, alice, storage_dir) -> None:
    first = await create_collection(client, alice, "one")
    second = await create_collection(client, alice, "two")

    assert (await upload(client, alice, first, "a.txt", b"shared")).status_code == 201
    assert (await upload(client, alice, second, "a.txt", b"shared")).status_code == 201
    assert len(files_on_disk(storage_dir)) == 2


async def test_same_file_is_allowed_for_another_user(client, alice, bob, storage_dir) -> None:
    alices = await create_collection(client, alice)
    bobs = await create_collection(client, bob)

    assert (await upload(client, alice, alices, "a.txt", b"shared")).status_code == 201
    assert (await upload(client, bob, bobs, "a.txt", b"shared")).status_code == 201


# ---------- upload: filename safety ----------


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("../../evil.pdf", "evil.pdf"),
        ("..\\..\\evil.pdf", "evil.pdf"),
        ("/etc/passwd.txt", "passwd.txt"),
        ("C:\\Windows\\sys.txt", "sys.txt"),
    ],
)
async def test_malicious_filename_cannot_affect_storage_paths(
    client, alice, storage_dir, sent, expected
) -> None:
    collection_id = await create_collection(client, alice)
    content = make_pdf() if sent.endswith(".pdf") else b"text"

    response = await upload(client, alice, collection_id, sent, content)

    assert response.status_code == 201
    assert response.json()["original_filename"] == expected
    (stored,) = files_on_disk(storage_dir)
    # Stored as <storage>/<owner uuid>/<document uuid>: nothing from the filename.
    assert stored.relative_to(storage_dir).parts[1] == response.json()["id"]
    assert "evil" not in str(stored) and "passwd" not in str(stored)
    # and nothing was written outside the storage root
    outside = [
        p for p in storage_dir.parent.rglob("*") if p.is_file() and storage_dir not in p.parents
    ]
    assert outside == []


# ---------- upload: auth and ownership ----------


async def test_upload_requires_auth(client, alice) -> None:
    collection_id = await create_collection(client, alice)

    response = await client.post(
        f"/collections/{collection_id}/documents", files={"file": ("a.txt", b"x")}
    )

    assert response.status_code == 401


async def test_upload_to_another_users_collection_is_404_and_stores_nothing(
    client, alice, bob, storage_dir
) -> None:
    alices = await create_collection(client, alice)

    response = await upload(client, bob, alices, "a.txt", b"sneaky")

    assert response.status_code == 404
    assert files_on_disk(storage_dir) == []
