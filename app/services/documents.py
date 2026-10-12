import codecs
import hashlib
import uuid
from collections.abc import AsyncIterator
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Collection, Document, DocumentStatus
from app.services.storage import StorageBackend

CHUNK_SIZE = 1024 * 1024  # 1 MB

# Allowed extensions -> the content type we record. We never store the
# client-supplied Content-Type, since the client controls it.
ALLOWED_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".txt": "text/plain",
}
TEXT_EXTENSIONS = {".md", ".txt"}
MAX_FILENAME_LENGTH = 255


def clean_filename(raw: str | None) -> str:
    """Reduce a client-supplied filename to a harmless display name.

    This value is only ever shown back to users; it is never used in a path.
    Directory parts ("../../x.pdf", "C:\\dir\\x.pdf") and control characters are removed.
    """
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if len(name) > MAX_FILENAME_LENGTH:
        suffix = PurePosixPath(name).suffix
        name = name[: MAX_FILENAME_LENGTH - len(suffix)] + suffix
    return name


class UploadInspector:
    """Looks at each chunk as it streams past: counts bytes, hashes, enforces the
    size limit, and checks that the content matches the claimed file type."""

    def __init__(self, extension: str, max_bytes: int) -> None:
        self.extension = extension
        self.max_bytes = max_bytes
        self.size = 0
        self._sha256 = hashlib.sha256()
        # Incremental, because a multi-byte UTF-8 character can straddle two chunks.
        self._decoder = (
            codecs.getincrementaldecoder("utf-8")() if extension in TEXT_EXTENSIONS else None
        )

    @property
    def content_hash(self) -> str:
        return self._sha256.hexdigest()

    def feed(self, chunk: bytes) -> None:
        if self.size == 0:
            self._check_signature(chunk)
        self.size += len(chunk)
        if self.size > self.max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {self.max_bytes // (1024 * 1024)} MB limit",
            )
        self._sha256.update(chunk)
        if self._decoder is not None:
            try:
                self._decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                raise _invalid("Text files must be valid UTF-8") from exc

    def finish(self) -> None:
        if self.size == 0:
            raise _invalid("File is empty")
        if self._decoder is not None:
            try:
                self._decoder.decode(b"", final=True)  # catches a truncated last character
            except UnicodeDecodeError as exc:
                raise _invalid("Text files must be valid UTF-8") from exc

    def _check_signature(self, first_chunk: bytes) -> None:
        if self.extension == ".pdf" and not first_chunk.startswith(b"%PDF"):
            raise _invalid("File content is not a PDF")
        if self.extension == ".docx" and not first_chunk.startswith(b"PK"):
            raise _invalid("File content is not a DOCX (ZIP) file")


def _invalid(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


async def _inspected_chunks(upload: UploadFile, inspector: UploadInspector) -> AsyncIterator[bytes]:
    """Read the upload ~1 MB at a time, inspecting each chunk before it is written.

    Any exception raised here propagates into storage.save(), which then deletes
    the partial file.
    """
    while chunk := await upload.read(CHUNK_SIZE):
        inspector.feed(chunk)
        yield chunk
    inspector.finish()


async def upload_document(
    db: AsyncSession, storage: StorageBackend, collection: Collection, upload: UploadFile
) -> Document:
    """Store an upload in `collection` (caller has already verified ownership)."""
    filename = clean_filename(upload.filename)
    extension = PurePosixPath(filename).suffix.lower()
    if extension not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only .pdf, .docx, .md and .txt files are allowed",
        )

    document_id = uuid.uuid4()
    # Generated, never derived from the filename.
    storage_key = f"{collection.owner_id}/{document_id}"
    inspector = UploadInspector(extension, get_settings().MAX_UPLOAD_MB * 1024 * 1024)

    await storage.save(storage_key, _inspected_chunks(upload, inspector))

    document = Document(
        id=document_id,
        collection_id=collection.id,
        owner_id=collection.owner_id,
        original_filename=filename,
        content_type=ALLOWED_TYPES[extension],
        size_bytes=inspector.size,
        content_hash=inspector.content_hash,
        storage_key=storage_key,
        status=DocumentStatus.pending,
    )
    db.add(document)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        await storage.delete(storage_key)  # the file must not outlive the failed insert
        if "uq_documents_collection_hash" in str(exc.orig):
            # The hash is only known after streaming, so the unique constraint
            # (race-free) is what detects the duplicate.
            raise HTTPException(
                status_code=409,
                detail="This file already exists in the collection",
            ) from exc
        raise
    except Exception:
        await db.rollback()
        await storage.delete(storage_key)
        raise
    return document
