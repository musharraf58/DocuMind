import codecs
import hashlib
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.core.config import get_settings
from app.models import Chunk, Collection, Document, DocumentStatus
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


async def get_owned_document(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> Document:
    """Fetch a document only if it belongs to user_id, otherwise raise 404.

    Filters by id AND owner_id in a single query; "not yours" and "doesn't
    exist" are indistinguishable to the caller. Every document endpoint that
    names a document goes through here.
    """
    document = await db.scalar(
        select(Document).where(Document.id == document_id, Document.owner_id == user_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


async def list_documents(
    db: AsyncSession,
    collection: Collection,
    status_filter: DocumentStatus | None,
    limit: int,
    offset: int,
) -> Sequence[Document]:
    """List documents in a collection the caller has already proven they own."""
    query = select(Document).where(
        Document.collection_id == collection.id, Document.owner_id == collection.owner_id
    )
    if status_filter is not None:
        query = query.where(Document.status == status_filter)
    result = await db.scalars(
        query.order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset)
    )
    return result.all()


async def mark_pending(db: AsyncSession, document: Document) -> None:
    """Queue the document for (re)processing: "pending", attempts reset."""
    document.status = DocumentStatus.pending
    document.error_message = None
    document.attempts = 0
    document.queued_at = clock.utcnow()
    await db.commit()


async def list_chunks(
    db: AsyncSession, document: Document, limit: int, offset: int
) -> Sequence[Chunk]:
    """Chunks of a document the caller has already proven they own, in order.

    Also filters on owner_id: cheap defense in depth should a caller ever pass
    in a document that wasn't fetched through get_owned_document.
    """
    result = await db.scalars(
        select(Chunk)
        .where(Chunk.document_id == document.id, Chunk.owner_id == document.owner_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    return result.all()


async def delete_document(db: AsyncSession, storage: StorageBackend, document: Document) -> None:
    """Remove the stored file, then the row (its chunks go with it via ON DELETE CASCADE).

    File first: if the row delete then fails, the user can simply retry (a
    missing file is tolerated). Row first would risk orphaned files that no
    row points to, which nobody could ever find or clean up.
    """
    await storage.delete(document.storage_key)
    await db.delete(document)
    await db.commit()
