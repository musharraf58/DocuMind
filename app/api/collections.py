import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.deps import CurrentUser, DbSession, StorageDep
from app.api.limits import limit_default
from app.schemas.collection import CollectionCreate, CollectionRead, CollectionUpdate
from app.services import collections as collection_service

router = APIRouter(prefix="/collections", tags=["collections"], dependencies=[limit_default])


@router.post("", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
async def create_collection(payload: CollectionCreate, db: DbSession, user: CurrentUser):
    return await collection_service.create_collection(db, user.id, payload)


@router.get("", response_model=list[CollectionRead])
async def list_collections(
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return await collection_service.list_collections(db, user.id, limit, offset)


@router.get("/{collection_id}", response_model=CollectionRead)
async def get_collection(collection_id: uuid.UUID, db: DbSession, user: CurrentUser):
    return await collection_service.get_owned_collection(db, collection_id, user.id)


@router.patch("/{collection_id}", response_model=CollectionRead)
async def update_collection(
    collection_id: uuid.UUID, payload: CollectionUpdate, db: DbSession, user: CurrentUser
):
    return await collection_service.update_collection(db, collection_id, user.id, payload)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: uuid.UUID, db: DbSession, user: CurrentUser, storage: StorageDep
):
    await collection_service.delete_collection(db, storage, collection_id, user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
