"""File storage in an S3-compatible bucket (Cloudflare R2, Backblaze B2,
MinIO, AWS S3).

Why not the local disk on a free host: its disk is ephemeral. Every deploy,
restart or spin-down after 15 idle minutes starts from the image again, so a
file saved there would vanish while its database row still points to it.

Streaming, in both directions, so memory use doesn't grow with file size:
- upload: bytes are collected into parts of PART_SIZE (8 MB) and each part is
  sent as soon as it is full (S3 "multipart upload"). A file smaller than one
  part goes in a single PUT. Nothing is visible under the key until the upload
  completes, and a failure midway aborts it, so a key never holds half a file.
- download: the object is read PART by PART from the response, never whole.

boto3 is a blocking library, so every call runs in a worker thread
(asyncio.to_thread) and the event loop keeps serving other requests.

Secrets: the keys live in the client object only. Errors from the bucket are
turned into StorageError with a fixed message (the provider's error code
goes to the log, never the credentials or the request), and boto's own
loggers are capped at WARNING, below which they would print request details.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.services.storage import READ_CHUNK_SIZE, StorageBackend, StorageError

logger = logging.getLogger(__name__)

for _name in ("boto3", "botocore", "s3transfer", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# S3's minimum part size is 5 MiB (except the last part). R2 also requires
# every part but the last to be the same size, which fixed slices give us.
PART_SIZE = 8 * 1024 * 1024
# DeleteObjects takes at most 1000 keys per request.
DELETE_BATCH = 1000
# Exactly the keys the app generates: "{owner uuid}/{document uuid}".
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
SAFE_KEY = re.compile(rf"^{_UUID}/{_UUID}$")


def check_key(key: str) -> str:
    """Refuse anything but a generated key. Object stores have no "../", but a
    key built from user input could still reach objects that aren't ours."""
    if not SAFE_KEY.fullmatch(key):
        raise ValueError("Invalid storage key")
    return key


def make_client(
    endpoint_url: str | None, access_key_id: str, secret_access_key: str, region: str
) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
        config=Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
            # Newer botocore adds checksum headers to every upload by default,
            # which some S3-compatible services reject. Only when required.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


class S3StorageBackend(StorageBackend):
    def __init__(self, client: Any, bucket: str, part_size: int = PART_SIZE) -> None:
        self.client = client
        self.bucket = bucket
        self.part_size = part_size

    async def _call(self, operation: str, **kwargs: Any) -> Any:
        from botocore.exceptions import BotoCoreError, ClientError

        method = getattr(self.client, operation)
        try:
            return await asyncio.to_thread(method, Bucket=self.bucket, **kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "unknown")
            if code in ("NoSuchKey", "404", "NotFound"):
                raise FileNotFoundError(kwargs.get("Key", "")) from None
            logger.error("Storage bucket refused %s (%s)", operation, code)
            raise StorageError from None
        except BotoCoreError as exc:  # connection failures, timeouts ...
            logger.error("Storage bucket unreachable during %s (%s)", operation, type(exc).__name__)
            raise StorageError from None

    async def save(self, key: str, chunks: AsyncIterator[bytes]) -> int:
        check_key(key)
        buffer = bytearray()
        written = 0
        upload_id: str | None = None
        parts: list[dict[str, Any]] = []
        try:
            async for chunk in chunks:
                buffer += chunk
                written += len(chunk)
                while len(buffer) >= self.part_size:
                    if upload_id is None:
                        created = await self._call("create_multipart_upload", Key=key)
                        upload_id = created["UploadId"]
                    part = bytes(buffer[: self.part_size])
                    del buffer[: self.part_size]
                    parts.append(await self._upload_part(key, upload_id, len(parts) + 1, part))
            if upload_id is None:
                # Smaller than one part: a single, atomic PUT.
                await self._call("put_object", Key=key, Body=bytes(buffer))
            else:
                if buffer:
                    last = bytes(buffer)
                    parts.append(await self._upload_part(key, upload_id, len(parts) + 1, last))
                await self._call(
                    "complete_multipart_upload",
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
        except BaseException:
            # Includes size-limit rejections raised by the stream and request
            # cancellation. An unfinished multipart upload is invisible but
            # still billed as storage until aborted.
            if upload_id is not None:
                await self._abort(key, upload_id)
            raise
        return written

    async def _upload_part(self, key: str, upload_id: str, number: int, data: bytes) -> dict:
        response = await self._call(
            "upload_part", Key=key, UploadId=upload_id, PartNumber=number, Body=data
        )
        return {"PartNumber": number, "ETag": response["ETag"]}

    async def _abort(self, key: str, upload_id: str) -> None:
        try:
            # Shielded: if we got here because the request was cancelled, the
            # abort must still run to completion.
            await asyncio.shield(self._call("abort_multipart_upload", Key=key, UploadId=upload_id))
        except (StorageError, asyncio.CancelledError):
            logger.warning("Could not abort an unfinished upload; the bucket may keep its parts")

    async def open(self, key: str) -> AsyncIterator[bytes]:
        check_key(key)
        response = await self._call("get_object", Key=key)  # FileNotFoundError now
        body = response["Body"]

        async def iterate() -> AsyncIterator[bytes]:
            try:
                while chunk := await asyncio.to_thread(body.read, READ_CHUNK_SIZE):
                    yield chunk
            finally:
                body.close()

        return iterate()

    async def delete(self, key: str) -> None:
        # S3 deletes are idempotent: a missing key is not an error.
        await self._call("delete_object", Key=check_key(key))

    async def delete_many(self, keys: Sequence[str]) -> None:
        for start in range(0, len(keys), DELETE_BATCH):
            batch = [{"Key": check_key(key)} for key in keys[start : start + DELETE_BATCH]]
            response = await self._call("delete_objects", Delete={"Objects": batch, "Quiet": True})
            if errors := response.get("Errors"):
                codes = sorted({error.get("Code", "unknown") for error in errors})
                logger.error("Storage bucket refused to delete %d objects (%s)", len(errors), codes)
                raise StorageError
