import uuid

from fastapi import APIRouter, UploadFile, status

from app.api.deps import CurrentUser, DbSession, StorageDep
from app.schemas.document import DocumentRead
from app.services import collections as collection_service
from app.services import documents as document_service

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
