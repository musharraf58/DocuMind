import uuid
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession, EmbedderDep, StorageDep
from app.models import DocumentStatus
from app.schemas.chunk import ChunkRead
from app.schemas.document import DocumentRead
from app.services import collections as collection_service
from app.services import documents as document_service
from app.services import jobs

router = APIRouter(tags=["documents"])


@router.post(
    "/collections/{collection_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    collection_id: uuid.UUID,
    file: UploadFile,
    db: DbSession,
    user: CurrentUser,
    storage: StorageDep,
):
    collection = await collection_service.get_owned_collection(db, collection_id, user.id)
    return await document_service.upload_document(db, storage, collection, file)


@router.get("/collections/{collection_id}/documents", response_model=list[DocumentRead])
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


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(document_id: uuid.UUID, db: DbSession, user: CurrentUser):
    return await document_service.get_owned_document(db, document_id, user.id)


def _content_disposition(filename: str) -> str:
    """Build a header value that cannot be broken out of by a hostile filename.

    The plain filename= is an ASCII-only fallback with anything risky replaced;
    filename*= carries the real name, percent-encoded (RFC 6266 / 5987).
    """
    fallback = "".join(c if c.isascii() and c.isalnum() or c in "._- " else "_" for c in filename)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


@router.get("/documents/{document_id}/download")
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


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, db: DbSession, user: CurrentUser, storage: StorageDep
):
    document = await document_service.get_owned_document(db, document_id, user.id)
    await document_service.delete_document(db, storage, document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/documents/{document_id}/process", response_model=DocumentRead)
async def process_document(
    document_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    storage: StorageDep,
    embedder: EmbedderDep,
):
    """Process the document now, waiting for the result. (Replaced by a queued
    /reprocess shortly.) 409 if it is already being processed."""
    document = await document_service.get_owned_document(db, document_id, user.id)
    if document.status == DocumentStatus.processing:
        raise HTTPException(status_code=409, detail="Document is already being processed")
    await document_service.mark_pending(db, document)
    result = await jobs.run_job(db, document.id, storage, embedder)
    if result.outcome in (jobs.Outcome.skipped, jobs.Outcome.lost):
        raise HTTPException(status_code=409, detail="Processing was taken over by another run")
    await db.refresh(document)
    return document


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkRead])
async def list_chunks(
    document_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    document = await document_service.get_owned_document(db, document_id, user.id)
    return await document_service.list_chunks(db, document, limit, offset)
