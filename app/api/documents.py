import uuid
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession, QueueDep, QuotasDep, StorageDep
from app.api.limits import limit_default, limit_upload
from app.core.config import get_settings
from app.models import DocumentStatus
from app.schemas.chunk import ChunkRead
from app.schemas.document import DocumentRead
from app.services import collections as collection_service
from app.services import documents as document_service
from app.services.capacity import check_room_for_uploads
from app.services.queue import queue_full
from app.services.quotas import UPLOADS
from app.services.storage import StorageError
from app.services.upload_caps import MB, check_upload_caps, storage_full

router = APIRouter(tags=["documents"])


@router.post(
    "/collections/{collection_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[limit_upload],
)
async def upload_document(
    collection_id: uuid.UUID,
    file: UploadFile,
    db: DbSession,
    user: CurrentUser,
    storage: StorageDep,
    queue: QueueDep,
    quotas: QuotasDep,
):
    """Store the file and queue it for processing. 202: accepted, processing
    happens in the background; poll GET /documents/{id} for its status."""
    # Read once: a failed upload rolls the session back, which expires `user`
    # (reading user.id after that would try to reload it from the database).
    user_id = user.id
    collection = await collection_service.get_owned_collection(db, collection_id, user_id)
    if not queue.has_room():  # inline mode: refuse before storing anything
        raise queue_full()
    await check_room_for_uploads(db)  # 503 once the database is near its cap
    # Storage, document count, documents in progress. Locks this user's
    # uploads until this one commits, so parallel uploads can't slip past.
    storage_left = await check_upload_caps(db, user_id)
    # A document that can't be embedded today isn't worth storing now.
    await quotas.check_embed_tokens(user_id)
    await quotas.take_upload(user_id)
    limit_bytes = get_settings().MAX_STORAGE_MB_PER_USER * MB
    try:
        # upload_document has COMMITTED the "pending" document when it
        # returns, so a worker that picks the job up at once is sure to find it.
        document = await document_service.upload_document(
            db,
            storage,
            collection,
            file,
            storage_left,
            lambda: storage_full(limit_bytes, limit_bytes - storage_left),
        )
    except (HTTPException, StorageError):
        # Refused (type, size, duplicate...) or the storage failed: it
        # doesn't count as an upload.
        await quotas.give_back(user_id, UPLOADS)
        raise
    await queue.enqueue(document.id)  # if this fails, the sweeper enqueues it later
    return document


@router.get(
    "/collections/{collection_id}/documents",
    response_model=list[DocumentRead],
    dependencies=[limit_default],
)
async def list_documents(
    collection_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    collection = await collection_service.get_owned_collection(db, collection_id, user.id)
    return await document_service.list_documents(db, collection, status_filter, limit, offset)


@router.get("/documents/{document_id}", response_model=DocumentRead, dependencies=[limit_default])
async def get_document(document_id: uuid.UUID, db: DbSession, user: CurrentUser):
    return await document_service.get_owned_document(db, document_id, user.id)


def _content_disposition(filename: str) -> str:
    """Build a header value that cannot be broken out of by a hostile filename.

    The plain filename= is an ASCII-only fallback with anything risky replaced;
    filename*= carries the real name, percent-encoded (RFC 6266 / 5987).
    """
    fallback = "".join(c if c.isascii() and c.isalnum() or c in "._- " else "_" for c in filename)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


@router.get("/documents/{document_id}/download", dependencies=[limit_default])
async def download_document(
    document_id: uuid.UUID, db: DbSession, user: CurrentUser, storage: StorageDep
):
    document = await document_service.get_owned_document(db, document_id, user.id)
    try:
        chunks = await storage.open(document.storage_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Stored file is missing") from exc
    return StreamingResponse(
        chunks,
        media_type=document.content_type,
        headers={
            "Content-Disposition": _content_disposition(document.original_filename),
            "Content-Length": str(document.size_bytes),
            # Never let a browser second-guess the type and render it.
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[limit_default],
)
async def delete_document(
    document_id: uuid.UUID, db: DbSession, user: CurrentUser, storage: StorageDep
):
    document = await document_service.get_owned_document(db, document_id, user.id)
    await document_service.delete_document(db, storage, document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/documents/{document_id}/reprocess",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[limit_default],
)
async def reprocess_document(
    document_id: uuid.UUID, db: DbSession, user: CurrentUser, queue: QueueDep
):
    """Queue the document to be processed again (e.g. after a failure or a
    model change). 409 if it is already queued or being processed."""
    document = await document_service.get_owned_document(db, document_id, user.id)
    if not queue.has_room():
        raise queue_full()
    if not await document_service.queue_for_processing(db, document):
        raise HTTPException(status_code=409, detail="Document is already queued or being processed")
    await queue.enqueue(document.id)
    return document


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkRead],
    dependencies=[limit_default],
)
async def list_chunks(
    document_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    document = await document_service.get_owned_document(db, document_id, user.id)
    return await document_service.list_chunks(db, document, limit, offset)
