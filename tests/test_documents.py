import hashlib

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from tests.helpers import (
    auth_headers,
    create_collection,
    error_body,
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

    assert response.status_code == 202
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

    assert response.status_code == 202


async def test_multibyte_character_split_across_chunks_is_valid_utf8(
    client, alice, storage_dir
) -> None:
    collection_id = await create_collection(client, alice)
    # "é" is 2 bytes; place it so it straddles the 1 MB chunk boundary.
    content = b"a" * (MB - 1) + "é".encode() + b"tail"

    response = await upload(client, alice, collection_id, "big.txt", content)

    assert response.status_code == 202
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

    assert response.status_code == 202


# ---------- upload: duplicates ----------


async def test_duplicate_in_same_collection_is_409_and_leaves_no_extra_file(
    client, alice, storage_dir
) -> None:
    collection_id = await create_collection(client, alice)
    assert (await upload(client, alice, collection_id, "a.txt", b"same bytes")).status_code == 202

    # Even under a different filename, identical bytes are a duplicate.
    again = await upload(client, alice, collection_id, "renamed.txt", b"same bytes")

    assert again.status_code == 409
    assert len(files_on_disk(storage_dir)) == 1


async def test_same_file_is_allowed_in_another_collection(client, alice, storage_dir) -> None:
    first = await create_collection(client, alice, "one")
    second = await create_collection(client, alice, "two")

    assert (await upload(client, alice, first, "a.txt", b"shared")).status_code == 202
    assert (await upload(client, alice, second, "a.txt", b"shared")).status_code == 202
    assert len(files_on_disk(storage_dir)) == 2


async def test_same_file_is_allowed_for_another_user(client, alice, bob, storage_dir) -> None:
    alices = await create_collection(client, alice)
    bobs = await create_collection(client, bob)

    assert (await upload(client, alice, alices, "a.txt", b"shared")).status_code == 202
    assert (await upload(client, bob, bobs, "a.txt", b"shared")).status_code == 202


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

    assert response.status_code == 202
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


# ---------- read / list ----------


async def _uploaded(client, headers, collection_id, filename="a.txt", content=b"hello") -> dict:
    response = await upload(client, headers, collection_id, filename, content)
    assert response.status_code == 202
    return response.json()


async def test_get_document(client, alice) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id)

    response = await client.get(f"/documents/{doc['id']}", headers=alice)

    assert response.status_code == 200
    assert response.json() == doc
    assert "storage_key" not in response.text


async def test_list_documents_paginates_and_filters_by_status(client, alice) -> None:
    collection_id = await create_collection(client, alice)
    for i in range(5):
        await _uploaded(client, alice, collection_id, f"f{i}.txt", f"content {i}".encode())
    url = f"/collections/{collection_id}/documents"

    everything = (await client.get(url, headers=alice)).json()
    page1 = (await client.get(f"{url}?limit=2&offset=0", headers=alice)).json()
    page2 = (await client.get(f"{url}?limit=2&offset=2", headers=alice)).json()
    page3 = (await client.get(f"{url}?limit=2&offset=4", headers=alice)).json()

    assert len(everything) == 5
    assert page1 + page2 + page3 == everything
    assert everything[0]["original_filename"] == "f4.txt"  # newest first

    assert len((await client.get(f"{url}?status=pending", headers=alice)).json()) == 5
    assert (await client.get(f"{url}?status=ready", headers=alice)).json() == []
    assert (await client.get(f"{url}?status=bogus", headers=alice)).status_code == 422
    assert (await client.get(f"{url}?limit=0", headers=alice)).status_code == 422


async def test_list_only_shows_documents_of_that_collection(client, alice) -> None:
    one = await create_collection(client, alice, "one")
    two = await create_collection(client, alice, "two")
    await _uploaded(client, alice, one, "in-one.txt", b"1")
    await _uploaded(client, alice, two, "in-two.txt", b"2")

    names = [
        d["original_filename"]
        for d in (await client.get(f"/collections/{one}/documents", headers=alice)).json()
    ]

    assert names == ["in-one.txt"]


# ---------- download ----------


async def test_download_returns_exact_bytes_with_safe_headers(client, alice) -> None:
    collection_id = await create_collection(client, alice)
    content = make_pdf("download me")
    doc = await _uploaded(client, alice, collection_id, "report.pdf", content)

    response = await client.get(f"/documents/{doc['id']}/download", headers=alice)

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-length"] == str(len(content))
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert 'filename="report.pdf"' in response.headers["content-disposition"]


async def test_download_streams_large_files_intact(client, alice) -> None:
    collection_id = await create_collection(client, alice)
    content = bytes(range(256)) * (3 * 4096)  # ~3 MB, spans several chunks
    big = await _uploaded(client, alice, collection_id, "big.pdf", b"%PDF" + content)

    response = await client.get(f"/documents/{big['id']}/download", headers=alice)

    assert response.content == b"%PDF" + content


@pytest.mark.parametrize(
    "hostile",
    [
        'we"ird.txt',
        "semi;colon.txt",
        "résumé.txt",
        "tab\there.txt",
        "new\nline.txt",
        "a" * 400 + ".txt",
    ],
)
async def test_content_disposition_cannot_be_broken_by_filename(client, alice, hostile) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id, hostile, b"text")

    response = await client.get(f"/documents/{doc['id']}/download", headers=alice)

    header = response.headers["content-disposition"]
    assert response.status_code == 200
    assert "\n" not in header and "\r" not in header
    # exactly one attachment disposition with one quoted fallback name
    assert header.count("attachment") == 1
    fallback = header.split('filename="', 1)[1].split('"', 1)[0]
    assert fallback.isascii() and '"' not in fallback and ";" not in fallback


async def test_download_when_stored_file_is_missing_is_404(client, alice, storage_dir) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id)
    for f in files_on_disk(storage_dir):
        f.unlink()

    response = await client.get(f"/documents/{doc['id']}/download", headers=alice)

    assert response.status_code == 404


# ---------- delete ----------


async def test_delete_removes_file_and_row(client, alice, storage_dir) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id)
    assert len(files_on_disk(storage_dir)) == 1

    response = await client.delete(f"/documents/{doc['id']}", headers=alice)

    assert response.status_code == 204
    assert files_on_disk(storage_dir) == []
    assert (await client.get(f"/documents/{doc['id']}", headers=alice)).status_code == 404
    assert (await client.get(f"/collections/{collection_id}/documents", headers=alice)).json() == []


async def test_delete_tolerates_an_already_missing_file(client, alice, storage_dir) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id)
    for f in files_on_disk(storage_dir):
        f.unlink()

    response = await client.delete(f"/documents/{doc['id']}", headers=alice)

    assert response.status_code == 204
    assert (await client.get(f"/documents/{doc['id']}", headers=alice)).status_code == 404


async def test_deleted_document_can_be_uploaded_again(client, alice) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id, "a.txt", b"same")
    await client.delete(f"/documents/{doc['id']}", headers=alice)

    assert (await upload(client, alice, collection_id, "a.txt", b"same")).status_code == 202


# ---------- auth ----------


async def test_document_endpoints_require_auth(client, alice) -> None:
    collection_id = await create_collection(client, alice)
    doc = await _uploaded(client, alice, collection_id)

    for method, url in [
        ("GET", f"/collections/{collection_id}/documents"),
        ("GET", f"/documents/{doc['id']}"),
        ("GET", f"/documents/{doc['id']}/download"),
        ("DELETE", f"/documents/{doc['id']}"),
    ]:
        assert (await client.request(method, url)).status_code == 401, (method, url)


# ---------- cross-user isolation ----------


async def test_other_user_gets_404_on_every_document_endpoint(
    client, alice, bob, storage_dir
) -> None:
    alices = await create_collection(client, alice)
    doc = await _uploaded(client, alice, alices, "secret.txt", b"alice only")
    before = files_on_disk(storage_dir)

    list_resp = await client.get(f"/collections/{alices}/documents", headers=bob)
    get_resp = await client.get(f"/documents/{doc['id']}", headers=bob)
    download_resp = await client.get(f"/documents/{doc['id']}/download", headers=bob)
    delete_resp = await client.delete(f"/documents/{doc['id']}", headers=bob)
    upload_resp = await upload(client, bob, alices, "x.txt", b"intrusion")

    for response in (list_resp, get_resp, download_resp, delete_resp, upload_resp):
        assert response.status_code == 404
    assert b"alice only" not in download_resp.content

    # Alice's data is completely untouched
    assert files_on_disk(storage_dir) == before
    assert (await client.get(f"/documents/{doc['id']}", headers=alice)).json() == doc


async def test_foreign_and_nonexistent_documents_look_identical(client, alice, bob) -> None:
    import uuid

    alices = await create_collection(client, alice)
    doc = await _uploaded(client, alice, alices)

    foreign = await client.get(f"/documents/{doc['id']}", headers=bob)
    missing = await client.get(f"/documents/{uuid.uuid4()}", headers=bob)

    assert foreign.status_code == missing.status_code == 404
    assert error_body(foreign) == error_body(missing)


# ---------- deleting a collection also deletes its files ----------


async def test_deleting_a_collection_removes_all_its_files_and_rows(
    client, alice, storage_dir
) -> None:
    doomed = await create_collection(client, alice, "doomed")
    survivor = await create_collection(client, alice, "survivor")
    docs = [
        await _uploaded(client, alice, doomed, "a.txt", b"one"),
        await _uploaded(client, alice, doomed, "b.pdf", make_pdf("two")),
        await _uploaded(client, alice, doomed, "c.md", b"# three"),
    ]
    kept = await _uploaded(client, alice, survivor, "keep.txt", b"keep me")
    assert len(files_on_disk(storage_dir)) == 4

    response = await client.delete(f"/collections/{doomed}", headers=alice)

    assert response.status_code == 204
    # Only the other collection's file remains on disk...
    remaining = files_on_disk(storage_dir)
    assert len(remaining) == 1
    assert remaining[0].read_bytes() == b"keep me"
    # ...and the rows are gone too (cascade), while the other document is intact.
    for doc in docs:
        assert (await client.get(f"/documents/{doc['id']}", headers=alice)).status_code == 404
    assert (await client.get(f"/documents/{kept['id']}", headers=alice)).status_code == 200


async def test_deleting_a_collection_tolerates_already_missing_files(
    client, alice, storage_dir
) -> None:
    collection_id = await create_collection(client, alice)
    await _uploaded(client, alice, collection_id, "a.txt", b"one")
    await _uploaded(client, alice, collection_id, "b.txt", b"two")
    files_on_disk(storage_dir)[0].unlink()

    response = await client.delete(f"/collections/{collection_id}", headers=alice)

    assert response.status_code == 204
    assert files_on_disk(storage_dir) == []


async def test_other_user_cannot_delete_collection_or_its_files(
    client, alice, bob, storage_dir
) -> None:
    alices = await create_collection(client, alice)
    await _uploaded(client, alice, alices, "a.txt", b"mine")

    response = await client.delete(f"/collections/{alices}", headers=bob)

    assert response.status_code == 404
    assert len(files_on_disk(storage_dir)) == 1


async def test_document_shows_job_status_fields(client):
    headers = await auth_headers(client, "status@example.com")
    collection_id = await create_collection(client, headers)
    doc_id = (await upload(client, headers, collection_id, "a.txt", b"hello there")).json()["id"]

    body = (await client.get(f"/documents/{doc_id}", headers=headers)).json()
    assert body["status"] == "pending"
    assert body["attempts"] == 0
    for field in ("queued_at", "processing_started_at", "heartbeat_at", "finished_at"):
        assert field in body
    assert "storage_key" not in body and "processing_run_id" not in body
