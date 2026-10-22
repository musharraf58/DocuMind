"""DELETE /auth/me. These tests FAIL ON PURPOSE until
app.services.users.delete_account is implemented (see its docstring)."""

import uuid

from sqlalchemy import func, select

from app.core import redis_keys
from app.core.redis import get_redis
from app.main import app
from app.models import Chunk, Collection, Conversation, Document, Message, User
from tests.helpers import (
    PASSWORD,
    auth_headers,
    create_collection,
    files_on_disk,
    login,
    upload,
)


async def delete_me(client, headers, password=PASSWORD):
    # httpx's .delete() takes no body; .request() does.
    return await client.request("DELETE", "/auth/me", json={"password": password}, headers=headers)


async def make_account_with_data(client, job_queue, email: str) -> dict:
    """A user with a collection, a processed document (rows, chunks, a file
    on disk), a conversation, and rate-limit counters in Redis."""
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    content = f"{email} notes: cats sleep a lot. ".encode() * 50
    response = await upload(client, headers, collection_id, "notes.txt", content)
    assert response.status_code == 202
    await job_queue.run_all()
    ask = await client.post(
        f"/collections/{collection_id}/ask", json={"question": "Do cats sleep?"}, headers=headers
    )
    assert ask.status_code == 200
    return {"headers": headers, "collection_id": collection_id}


async def count(db, model, **filters) -> int:
    query = select(func.count()).select_from(model)
    for name, value in filters.items():
        query = query.where(getattr(model, name) == value)
    return await db.scalar(query)


async def user_id(client, headers) -> uuid.UUID:
    return uuid.UUID((await client.get("/auth/me", headers=headers)).json()["id"])


async def test_wrong_password_deletes_nothing(client, job_queue, db):
    account = await make_account_with_data(client, job_queue, "keep@example.com")
    response = await delete_me(client, account["headers"], password="not-my-password")
    assert response.status_code == 403
    assert response.json()["code"] == "wrong_password"
    assert await count(db, User, email="keep@example.com") == 1
    assert (await client.get("/auth/me", headers=account["headers"])).status_code == 200


async def test_deletes_all_rows_by_cascade(client, job_queue, db):
    account = await make_account_with_data(client, job_queue, "gone@example.com")
    uid = await user_id(client, account["headers"])
    assert await count(db, Chunk) > 0

    response = await delete_me(client, account["headers"])
    assert response.status_code == 204

    assert await count(db, User, id=uid) == 0
    assert await count(db, Collection, owner_id=uid) == 0
    assert await count(db, Document, owner_id=uid) == 0
    assert await count(db, Chunk) == 0
    assert await count(db, Conversation) == 0
    assert await count(db, Message) == 0


async def test_deletes_stored_files(client, job_queue, storage_dir):
    account = await make_account_with_data(client, job_queue, "files@example.com")
    assert files_on_disk(storage_dir)
    assert (await delete_me(client, account["headers"])).status_code == 204
    assert files_on_disk(storage_dir) == []


async def test_deletes_redis_keys(client, job_queue):
    account = await make_account_with_data(client, job_queue, "keys@example.com")
    uid = await user_id(client, account["headers"])
    redis = app.dependency_overrides[get_redis]()

    async def user_keys() -> list[bytes]:
        found = []
        for pattern in redis_keys.user_key_patterns(uid):
            found += [key async for key in redis.scan_iter(match=pattern)]
        return found

    assert await user_keys()  # rate-limit logs exist for this user
    assert (await delete_me(client, account["headers"])).status_code == 204
    assert await user_keys() == []


async def test_old_token_and_login_stop_working(client, job_queue):
    account = await make_account_with_data(client, job_queue, "after@example.com")
    assert (await delete_me(client, account["headers"])).status_code == 204
    assert (await client.get("/auth/me", headers=account["headers"])).status_code == 401
    assert (await login(client, "after@example.com")).status_code == 401


async def test_other_users_are_untouched(client, job_queue, db, storage_dir):
    alice = await make_account_with_data(client, job_queue, "alice-del@example.com")
    bob = await make_account_with_data(client, job_queue, "bob-del@example.com")
    bob_id = await user_id(client, bob["headers"])
    bob_files = len(files_on_disk(storage_dir)) // 2

    assert (await delete_me(client, alice["headers"])).status_code == 204

    assert await count(db, User, id=bob_id) == 1
    assert await count(db, Document, owner_id=bob_id) == 1
    assert len(files_on_disk(storage_dir)) == bob_files
    assert (await client.get("/auth/me", headers=bob["headers"])).status_code == 200
