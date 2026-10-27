"""Create the S3 bucket named in S3_BUCKET, if it doesn't exist yet.

    python -m scripts.create_bucket

Uses the app's own settings (S3_ENDPOINT_URL, S3_REGION, the keys), so it
works against MinIO locally as well as R2 or B2. Running it again is
harmless. Never prints the keys.
"""

import sys

from app.core.config import get_settings
from app.services.s3_storage import make_client


def main() -> int:
    settings = get_settings()
    if not settings.S3_BUCKET:
        print("create_bucket: S3_BUCKET is not set", file=sys.stderr)
        return 2
    from botocore.exceptions import BotoCoreError, ClientError

    client = make_client(
        settings.S3_ENDPOINT_URL,
        settings.S3_ACCESS_KEY_ID.get_secret_value(),
        settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
        settings.S3_REGION,
    )
    bucket = settings.S3_BUCKET
    try:
        client.head_bucket(Bucket=bucket)
        print(f"Bucket {bucket!r} already exists")
        return 0
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket", "NotFound"):
            print(f"create_bucket: the storage refused ({_code(exc)})", file=sys.stderr)
            return 1
    except BotoCoreError as exc:
        print(f"create_bucket: storage unreachable ({type(exc).__name__})", file=sys.stderr)
        return 1
    try:
        client.create_bucket(Bucket=bucket)
    except ClientError as exc:
        print(f"create_bucket: could not create {bucket!r} ({_code(exc)})", file=sys.stderr)
        return 1
    print(f"Bucket {bucket!r} created")
    return 0


def _code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "unknown")


if __name__ == "__main__":
    sys.exit(main())
