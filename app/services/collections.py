import uuid
from collections.abc import Sequence

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Collection, Document
from app.schemas.collection import CollectionCreate, CollectionUpdate
from app.services.storage import StorageBackend


async def get_owned_collection(
    db: AsyncSession, collection_id: uuid.UUID, user_id: uuid.UUID
) -> Collection:
    """Fetch a collection only if it belongs to user_id, otherwise raise 404.

    Every collection endpoint that touches a single collection goes through
    this function, so ownership is checked in exactly one place.
    """
    collection = await db.scalar(
        select(Collection).where(
            Collection.id == collection_id,
            Collection.owner_id == user_id,
        )
    )
    if collection is None:
        # Same response whether the collection doesn't exist or belongs to
        # someone else, so IDs can't be probed to learn what exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    return collection


async def create_collection(
    db: AsyncSession, user_id: uuid.UUID, payload: CollectionCreate
) -> Collection:
    collection = Collection(owner_id=user_id, **payload.model_dump())
    db.add(collection)
    await db.commit()
    return collection


async def list_collections(
    db: AsyncSession, user_id: uuid.UUID, limit: int, offset: int
) -> Sequence[Collection]:
    result = await db.scalars(
        select(Collection)
        .where(Collection.owner_id == user_id)
        # id as a tie-breaker keeps pagination stable when timestamps collide.
        .order_by(Collection.created_at.desc(), Collection.id)
        .limit(limit)
        .offset(offset)
    )
    return result.all()


async def update_collection(
    db: AsyncSession, collection_id: uuid.UUID, user_id: uuid.UUID, payload: CollectionUpdate
) -> Collection:
    collection = await get_owned_collection(db, collection_id, user_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(collection, field, value)
    await db.commit()
    return collection


async def delete_collection(
    db: AsyncSession, storage: StorageBackend, collection_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Delete a collection together with its documents' files on disk.

    The database cascade removes the document *rows*, but it knows nothing about
    files, so we collect the storage keys first and delete those ourselves.
    Files go first (deleting a missing file is fine), then the rows, so a failure
    midway can be retried instead of leaving unreachable orphan files.
    """
    collection = await get_owned_collection(db, collection_id, user_id)
    storage_keys = (
        await db.scalars(
            select(Document.storage_key).where(Document.collection_id == collection.id)
        )
    ).all()
    await storage.delete_many(storage_keys)
    await db.delete(collection)  # ON DELETE CASCADE removes the document rows
    await db.commit()
