import asyncio

from sqlalchemy import func, select, update

from app.core.config import get_settings
from app.models import Document, DocumentStatus
from tests.helpers import auth_headers, create_collection, files_on_disk, upload

MB = 1024 * 1024


def set_cap(monkeypatch, name: str, value: int) -> None:
    monkeypatch.setattr(get_settings(), name, value)


async def account(client, email: str):
    headers = await auth_headers(client, email)
    return headers, await create_collection(client, headers)


async def mark_ready(db) -> None:
    await db.execute(update(Document).values(status=DocumentStatus.ready))
    await db.commit()


async def test_document_count_cap(client, db, monkeypatch):
    set_cap(monkeypatch, "MAX_DOCUMENTS_PER_USER", 2)
    headers, collection_id = await account(client, "count@example.com")
    for i in range(2):
        assert (
            await upload(client, headers, collection_id, f"{i}.txt", f"t{i}".encode())
        ).status_code == 202
    blocked = await upload(client, headers, collection_id, "3.txt", b"t3")
    assert blocked.status_code == 403
    body = blocked.json()
    assert body["code"] == "document_limit_reached"
    assert (body["limit"], body["used"]) == (2, 2)
    assert "2 of 2 documents" in body["detail"]


async def test_deleting_a_document_makes_room(client, monkeypatch):
    set_cap(monkeypatch, "MAX_DOCUMENTS_PER_USER", 1)
    headers, collection_id = await account(client, "room@example.com")
    first = await upload(client, headers, collection_id, "a.txt", b"first")
    assert (await upload(client, headers, collection_id, "b.txt", b"second")).status_code == 403
    await client.delete(f"/documents/{first.json()['id']}", headers=headers)
    assert (await upload(client, headers, collection_id, "b.txt", b"second")).status_code == 202


async def test_pending_cap(client, db, monkeypatch):
    set_cap(monkeypatch, "MAX_PENDING_DOCUMENTS_PER_USER", 2)
    headers, collection_id = await account(client, "pending@example.com")
    for i in range(2):
        assert (
            await upload(client, headers, collection_id, f"{i}.txt", f"t{i}".encode())
        ).status_code == 202
    blocked = await upload(client, headers, collection_id, "3.txt", b"t3")
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "pending_limit_reached"
    assert blocked.headers["Retry-After"] == "30"
    # Once they're processed, uploading works again.
    await mark_ready(db)
    assert (await upload(client, headers, collection_id, "3.txt", b"t3")).status_code == 202


async def test_storage_cap(client, db, monkeypatch, storage_dir):
    set_cap(monkeypatch, "MAX_STORAGE_MB_PER_USER", 1)
    headers, collection_id = await account(client, "storage@example.com")
    big = b"x" * (700 * 1024)
    assert (await upload(client, headers, collection_id, "a.txt", big)).status_code == 202
    # 700 KB stored; another 700 KB would pass the 20 MB file limit but not the 1 MB total.
    blocked = await upload(client, headers, collection_id, "b.txt", b"y" * (700 * 1024))
    assert blocked.status_code == 403
    body = blocked.json()
    assert body["code"] == "storage_limit_reached"
    assert body["limit_bytes"] == MB
    # The partial file was removed and no row was created.
    assert len(files_on_disk(storage_dir)) == 1
    assert await db.scalar(select(func.count()).select_from(Document)) == 1
    # Something small still fits.
    assert (await upload(client, headers, collection_id, "c.txt", b"small")).status_code == 202


async def test_storage_full_is_refused_before_reading_the_file(client, db, monkeypatch):
    set_cap(monkeypatch, "MAX_STORAGE_MB_PER_USER", 1)
    headers, collection_id = await account(client, "full@example.com")
    await upload(client, headers, collection_id, "a.txt", b"x" * MB)
    blocked = await upload(client, headers, collection_id, "b.txt", b"y")
    assert blocked.status_code == 403
    assert blocked.json()["code"] == "storage_limit_reached"


async def test_caps_are_per_user(client, monkeypatch):
    set_cap(monkeypatch, "MAX_DOCUMENTS_PER_USER", 1)
    alice, alice_collection = await account(client, "alice-cap@example.com")
    bob, bob_collection = await account(client, "bob-cap@example.com")
    assert (await upload(client, alice, alice_collection, "a.txt", b"alice")).status_code == 202
    assert (await upload(client, alice, alice_collection, "b.txt", b"alice2")).status_code == 403
    assert (await upload(client, bob, bob_collection, "a.txt", b"bob")).status_code == 202


async def test_a_refused_upload_does_not_use_the_daily_quota(client, monkeypatch):
    set_cap(monkeypatch, "MAX_DOCUMENTS_PER_USER", 1)
    set_cap(monkeypatch, "DAILY_UPLOADS_PER_USER", 2)
    headers, collection_id = await account(client, "quota-cap@example.com")
    first = await upload(client, headers, collection_id, "a.txt", b"one")
    assert (await upload(client, headers, collection_id, "b.txt", b"two")).status_code == 403
    await client.delete(f"/documents/{first.json()['id']}", headers=headers)
    # 1 of 2 daily uploads used: the refused one didn't count.
    assert (await upload(client, headers, collection_id, "b.txt", b"two")).status_code == 202


async def test_parallel_uploads_cannot_both_take_the_last_slot(client, db, monkeypatch):
    set_cap(monkeypatch, "MAX_DOCUMENTS_PER_USER", 1)
    headers, collection_id = await account(client, "race@example.com")
    responses = await asyncio.gather(
        upload(client, headers, collection_id, "a.txt", b"first file"),
        upload(client, headers, collection_id, "b.txt", b"second file"),
    )
    assert sorted(r.status_code for r in responses) == [202, 403]
    assert await db.scalar(select(func.count()).select_from(Document)) == 1
