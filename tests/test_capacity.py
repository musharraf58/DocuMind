"""The database capacity guard: uploads stop at MAX_DB_SIZE_MB, everything
else keeps working."""

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.models import Document
from app.services.capacity import DatabaseSize, database_size
from tests.helpers import auth_headers, create_collection, files_on_disk, upload


@pytest.fixture(autouse=True)
def fresh_size_cache():
    database_size.forget()
    yield
    database_size.forget()


async def test_uploads_work_under_the_limit(client):
    headers = await auth_headers(client, "room@example.com")
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, "a.txt", b"some text " * 30)
    assert response.status_code == 202


async def test_over_the_limit_uploads_are_refused_but_reading_and_asking_work(
    client, job_queue, db, storage_dir, monkeypatch
):
    headers = await auth_headers(client, "full@example.com")
    collection_id = await create_collection(client, headers)
    first = await upload(client, headers, collection_id, "cats.txt", b"Cats sleep a lot. " * 60)
    await job_queue.run_all()

    monkeypatch.setattr(get_settings(), "MAX_DB_SIZE_MB", 1)  # the test database is bigger
    refused = await upload(client, headers, collection_id, "more.txt", b"more text " * 30)
    assert refused.status_code == 503
    body = refused.json()
    assert body["code"] == "storage_capacity_reached"
    assert "Asking questions and searching still work" in body["detail"]
    assert refused.headers["Retry-After"] == "3600"
    # Nothing stored, and it didn't count against the daily upload quota.
    assert await db.scalar(select(func.count()).select_from(Document)) == 1
    assert len(files_on_disk(storage_dir)) == 1
    usage = (await client.get("/usage", headers=headers)).json()
    assert usage["quotas"]["uploads"]["used"] == 1

    base = f"/collections/{collection_id}"
    search = await client.post(f"{base}/search", json={"query": "cats"}, headers=headers)
    assert search.status_code == 200
    ask = await client.post(f"{base}/ask", json={"question": "Do cats sleep?"}, headers=headers)
    assert ask.status_code == 200
    document_id = first.json()["id"]
    assert (await client.get(f"/documents/{document_id}", headers=headers)).status_code == 200
    # Deleting is how room is made again.
    assert (await client.delete(f"/documents/{document_id}", headers=headers)).status_code == 204


async def test_zero_turns_the_guard_off(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "MAX_DB_SIZE_MB", 0)
    headers = await auth_headers(client, "nolimit@example.com")
    collection_id = await create_collection(client, headers)
    response = await upload(client, headers, collection_id, "a.txt", b"some text " * 30)
    assert response.status_code == 202


class CountingSession:
    def __init__(self, size: int) -> None:
        self.size = size
        self.queries = 0

    async def scalar(self, statement):
        assert "pg_database_size" in str(statement)
        self.queries += 1
        return self.size


async def test_the_size_is_cached_for_a_minute():
    now = [0.0]
    size = DatabaseSize(cache_seconds=60, clock=lambda: now[0])
    db = CountingSession(5_000_000)
    for _ in range(10):
        assert await size.bytes(db) == 5_000_000
    assert db.queries == 1
    now[0] = 59
    await size.bytes(db)
    assert db.queries == 1
    now[0] = 60
    db.size = 7_000_000
    assert await size.bytes(db) == 7_000_000  # refreshed
    assert db.queries == 2


async def test_the_real_query_reports_a_plausible_size(db):
    assert await DatabaseSize().bytes(db) > 1024 * 1024
