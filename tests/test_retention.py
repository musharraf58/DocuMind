"""Retention: accounts older than RETENTION_DAYS are deleted with their rows,
stored files (local or S3) and Redis keys; demo and young accounts stay."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core import redis_keys
from app.core.config import get_settings
from app.core.redis import get_redis
from app.main import app
from app.models import Chunk, Collection, Document, User
from app.services import background
from app.services.retention import purge_expired_accounts
from app.services.storage import get_storage
from tests.helpers import auth_headers, create_collection, files_on_disk, upload


async def make_account(client, job_queue, db, email, *, age_days=0, demo=False) -> uuid.UUID:
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    await upload(client, headers, collection_id, "a.txt", f"{email} writes. ".encode() * 40)
    await job_queue.run_all()
    await client.get("/usage", headers=headers)  # leaves rate-limit keys in Redis
    user_id = uuid.UUID((await client.get("/auth/me", headers=headers)).json()["id"])
    created = datetime.now(UTC) - timedelta(days=age_days)
    await db.execute(
        update(User).where(User.id == user_id).values(created_at=created, is_demo=demo)
    )
    await db.commit()
    return user_id


async def count(db, model, **where) -> int:
    query = select(func.count()).select_from(model)
    for name, value in where.items():
        query = query.where(getattr(model, name) == value)
    return await db.scalar(query)


async def redis_keys_of(redis, user_id) -> list:
    found = []
    for pattern in redis_keys.user_key_patterns(user_id):
        found += [key async for key in redis.scan_iter(match=pattern)]
    return found


@pytest.fixture
def sessions(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture
def retention_days(monkeypatch):
    monkeypatch.setattr(get_settings(), "RETENTION_DAYS", 30)


async def test_off_by_default(client, job_queue, db, sessions):
    assert get_settings().RETENTION_DAYS == 0
    await make_account(client, job_queue, db, "ancient@example.com", age_days=400)
    report = await purge_expired_accounts(sessions, get_storage(), None)
    assert report.accounts == 0
    assert await count(db, User) == 1


async def test_expired_accounts_go_with_everything_they_own(
    client, job_queue, db, sessions, storage_dir, retention_days, caplog
):
    caplog.set_level(logging.INFO)
    redis = app.dependency_overrides[get_redis]()
    old = await make_account(client, job_queue, db, "old@example.com", age_days=31)
    young = await make_account(client, job_queue, db, "young@example.com", age_days=29)
    demo = await make_account(client, job_queue, db, "demo@example.com", age_days=365, demo=True)
    assert len(files_on_disk(storage_dir)) == 3
    assert await redis_keys_of(redis, old)

    report = await purge_expired_accounts(sessions, get_storage(), redis)

    assert (report.accounts, report.files, report.failed) == (1, 1, 0)
    assert await count(db, User, id=old) == 0
    for model in (Collection, Document, Chunk):
        assert await count(db, model, owner_id=old) == 0
    assert await redis_keys_of(redis, old) == []
    remaining = {path.parent.name for path in files_on_disk(storage_dir)}
    assert remaining == {str(young), str(demo)}  # keys are "{owner}/{document}"
    for kept in (young, demo):
        assert await count(db, User, id=kept) == 1
        assert await count(db, Document, owner_id=kept) == 1
        assert await redis_keys_of(redis, kept)
    # Counts only: no email or id of the deleted account in the log.
    assert "Retention: deleted 1 accounts older than 30 days (1 stored files)" in caplog.text
    assert "old@example.com" not in caplog.text and str(old) not in caplog.text


async def test_expired_accounts_objects_are_deleted_from_s3(
    client, job_queue, db, sessions, retention_days, s3
):
    from tests.s3_fakes import objects

    old = await make_account(client, job_queue, db, "olds3@example.com", age_days=40)
    young = await make_account(client, job_queue, db, "youngs3@example.com")
    assert len(objects()) == 2

    report = await purge_expired_accounts(sessions, get_storage(), None)

    assert report.accounts == 1
    remaining = objects()
    assert len(remaining) == 1 and remaining[0].startswith(str(young))
    assert not any(key.startswith(str(old)) for key in remaining)


async def test_the_periodic_maintenance_runs_retention(
    client, job_queue, db, sessions, retention_days
):
    from app.services.inline_worker import InlineJobQueue

    old = await make_account(client, job_queue, db, "maint@example.com", age_days=90)
    await background.run_maintenance(sessions, InlineJobQueue(5), None)
    assert await count(db, User, id=old) == 0


async def test_the_celery_task_runs_retention(
    client, job_queue, db, test_engine, retention_days, monkeypatch
):
    import fakeredis

    from app.core.database import make_engine
    from app.workers import tasks
    from tests.conftest import TEST_DB_URL

    old = await make_account(client, job_queue, db, "beat@example.com", age_days=90)
    test_url = TEST_DB_URL.render_as_string(hide_password=False)
    monkeypatch.setattr(tasks, "_engine", lambda: make_engine(test_url))
    monkeypatch.setattr(tasks, "make_redis", lambda: fakeredis.aioredis.FakeRedis())
    report = await tasks.run_retention()
    assert report.accounts == 1
    assert await count(db, User, id=old) == 0
