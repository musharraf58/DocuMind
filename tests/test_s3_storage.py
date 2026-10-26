"""The S3-compatible storage backend, against moto's in-process fake S3 (no
network: moto intercepts boto3's requests before they leave the process)."""

import logging
import uuid

import boto3
import pytest
from moto import mock_aws
from pydantic import SecretStr

from app.core.config import get_settings
from app.services import s3_storage, storage
from app.services.s3_storage import S3StorageBackend, make_client
from app.services.storage import StorageError
from tests.helpers import PASSWORD, auth_headers, create_collection, upload

BUCKET = "documind-test"
ACCESS_KEY = "AKIATESTACCESSKEY123"
SECRET_KEY = "test-secret-key-must-never-be-logged-0123456789"
MiB = 1024 * 1024


def key() -> str:
    return f"{uuid.uuid4()}/{uuid.uuid4()}"


async def stream(*parts: bytes, fail_after: int | None = None):
    for i, part in enumerate(parts):
        if fail_after is not None and i == fail_after:
            raise RuntimeError("client went away")
        yield part


async def read_all(backend, k) -> bytes:
    return b"".join([chunk async for chunk in await backend.open(k)])


@pytest.fixture
def s3(monkeypatch):
    """A fake bucket, and the app configured to use it (STORAGE_BACKEND=s3)."""
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        settings = get_settings()
        for name, value in {
            "STORAGE_BACKEND": "s3",
            "S3_BUCKET": BUCKET,
            "S3_ENDPOINT_URL": "",
            "S3_REGION": "us-east-1",
            "S3_ACCESS_KEY_ID": ACCESS_KEY,
            "S3_SECRET_ACCESS_KEY": SECRET_KEY,
        }.items():
            if name in ("S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
                value = SecretStr(value)
            monkeypatch.setattr(settings, name, value)
        storage._s3_backend.cache_clear()
        yield storage.get_storage()
        storage._s3_backend.cache_clear()


def objects(prefix: str = "") -> list[str]:
    listed = boto3.client("s3", region_name="us-east-1").list_objects_v2(
        Bucket=BUCKET, Prefix=prefix
    )
    return [item["Key"] for item in listed.get("Contents", [])]


def unfinished_uploads() -> list:
    client = boto3.client("s3", region_name="us-east-1")
    return client.list_multipart_uploads(Bucket=BUCKET).get("Uploads", [])


def small_parts_backend() -> S3StorageBackend:
    """S3's minimum part size, so multipart uploads happen with small test data."""
    client = make_client("", ACCESS_KEY, SECRET_KEY, "us-east-1")
    return S3StorageBackend(client, BUCKET, part_size=5 * MiB)


async def test_get_storage_picks_the_s3_backend(s3):
    assert isinstance(s3, S3StorageBackend)
    assert storage.get_storage() is s3  # one client, reused


async def test_small_file_round_trip(s3):
    k = key()
    assert await s3.save(k, stream(b"hello ", b"world")) == 11
    assert await read_all(s3, k) == b"hello world"
    assert objects() == [k]


async def test_large_file_is_uploaded_in_parts_and_read_back_in_chunks():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        backend = small_parts_backend()
        calls = []
        backend.client.meta.events.register(
            "before-call.s3.*", lambda model, **kw: calls.append(model.name)
        )
        data = bytes(range(256)) * (12 * MiB // 256)  # 12 MiB: parts of 5, 5, 2
        k = key()
        assert await backend.save(k, stream(*(data[i : i + MiB] for i in range(0, len(data), MiB))))
        assert calls.count("UploadPart") == 3
        assert "PutObject" not in calls  # never the whole file in one request

        chunks = [chunk async for chunk in await backend.open(k)]
        assert max(len(chunk) for chunk in chunks) <= storage.READ_CHUNK_SIZE
        assert b"".join(chunks) == data


async def test_failure_midway_leaves_no_object_and_no_unfinished_upload():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        backend = small_parts_backend()
        k = key()
        parts = [b"x" * MiB] * 8
        with pytest.raises(RuntimeError):
            await backend.save(k, stream(*parts, fail_after=7))  # after one part was sent
        assert objects() == []
        assert unfinished_uploads() == []  # aborted, so not billed as storage


async def test_small_upload_failing_leaves_nothing(s3):
    with pytest.raises(RuntimeError):
        await s3.save(key(), stream(b"abc", b"def", fail_after=1))
    assert objects() == []


async def test_missing_key_raises_before_iteration(s3):
    with pytest.raises(FileNotFoundError):
        await s3.open(key())


async def test_delete_is_idempotent(s3):
    k = key()
    await s3.save(k, stream(b"data"))
    await s3.delete(k)
    await s3.delete(k)  # already gone: fine
    assert objects() == []


async def test_delete_many_in_batches(s3, monkeypatch):
    monkeypatch.setattr(s3_storage, "DELETE_BATCH", 2)
    keys = [key() for _ in range(5)]
    keep = key()
    for k in [*keys, keep]:
        await s3.save(k, stream(b"data"))
    await s3.delete_many(keys)
    assert objects() == [keep]


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "a/b",
        f"/{uuid.uuid4()}/{uuid.uuid4()}",
        f"{uuid.uuid4()}/../{uuid.uuid4()}",
        f"{uuid.uuid4()}".upper() + f"/{uuid.uuid4()}",
        f"{uuid.uuid4()}/{uuid.uuid4()}/extra",
        "",
    ],
)
async def test_unsafe_keys_are_refused(s3, bad):
    with pytest.raises(ValueError):
        await s3.save(bad, stream(b"x"))
    with pytest.raises(ValueError):
        await s3.open(bad)
    with pytest.raises(ValueError):
        await s3.delete(bad)
    with pytest.raises(ValueError):
        await s3.delete_many([key(), bad])
    assert objects() == []


async def test_bucket_errors_are_safe_and_never_log_credentials(caplog):
    caplog.set_level(logging.DEBUG)  # even at DEBUG, nothing secret
    with mock_aws():  # no bucket created: every call fails
        backend = small_parts_backend()
        with pytest.raises(StorageError) as caught:
            await backend.save(key(), stream(b"data"))
    assert str(caught.value) == "File storage is unavailable; please try again later"
    assert "NoSuchBucket" in caplog.text  # the provider's error code is logged
    assert SECRET_KEY not in caplog.text
    assert ACCESS_KEY not in caplog.text
    assert SECRET_KEY not in repr(backend.__dict__)


# --- through the API ------------------------------------------------------------


async def test_upload_process_download_and_delete_document(client, job_queue, s3):
    headers = await auth_headers(client, "s3user@example.com")
    collection_id = await create_collection(client, headers)
    content = b"Notes about cats. " * 100
    response = await upload(client, headers, collection_id, "cats.txt", content)
    assert response.status_code == 202
    document_id = response.json()["id"]
    assert len(objects()) == 1

    await job_queue.run_all()  # the job reads the file back from the bucket
    assert (await client.get(f"/documents/{document_id}", headers=headers)).json()["status"] == (
        "ready"
    )
    download = await client.get(f"/documents/{document_id}/download", headers=headers)
    assert download.content == content

    assert (await client.delete(f"/documents/{document_id}", headers=headers)).status_code == 204
    assert objects() == []


async def test_deleting_a_collection_deletes_its_objects(client, s3):
    headers = await auth_headers(client, "s3coll@example.com")
    gone = await create_collection(client, headers, "Gone")
    kept = await create_collection(client, headers, "Kept")
    for i in range(3):
        await upload(client, headers, gone, f"{i}.txt", f"doc {i} text".encode() * 20)
    kept_doc = await upload(client, headers, kept, "k.txt", b"keep me " * 20)
    assert len(objects()) == 4

    assert (await client.delete(f"/collections/{gone}", headers=headers)).status_code == 204
    remaining = objects()
    assert len(remaining) == 1 and remaining[0].endswith(kept_doc.json()["id"])


async def test_deleting_the_account_deletes_its_objects(client, s3):
    headers = await auth_headers(client, "s3gone@example.com")
    other = await auth_headers(client, "s3stay@example.com")
    for h in (headers, other):
        collection_id = await create_collection(client, h)
        await upload(client, h, collection_id, "a.txt", b"some text here " * 20)
    assert len(objects()) == 2
    me = (await client.get("/auth/me", headers=headers)).json()["id"]

    response = await client.request(
        "DELETE", "/auth/me", json={"password": PASSWORD}, headers=headers
    )
    assert response.status_code == 204, response.text
    remaining = objects()
    assert len(remaining) == 1 and not remaining[0].startswith(me)


async def test_storage_outage_is_a_clean_503_and_the_upload_is_not_counted(client, s3, caplog):
    headers = await auth_headers(client, "s3down@example.com")
    collection_id = await create_collection(client, headers)
    boto3.client("s3", region_name="us-east-1").delete_bucket(Bucket=BUCKET)  # "outage"

    response = await upload(client, headers, collection_id, "a.txt", b"text " * 50)
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "storage_unavailable"
    assert SECRET_KEY not in response.text and SECRET_KEY not in caplog.text
    usage = (await client.get("/usage", headers=headers)).json()
    assert usage["quotas"]["uploads"]["used"] == 0
