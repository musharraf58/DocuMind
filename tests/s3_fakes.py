"""A fake S3 bucket (moto, in-process: no network) and the app configured to
use it. The `s3` fixture is registered for every test in conftest.py."""

import uuid

import boto3
import pytest
from moto import mock_aws
from pydantic import SecretStr

from app.core.config import get_settings
from app.services import storage

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
