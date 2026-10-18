import math
import uuid

from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser, DbSession, EmbedderDep
from app.schemas.search import SearchRequest, SearchResult
from app.services import collections as collection_service
from app.services import search as search_service
from app.services.embeddings import EmbeddingBusyError, EmbeddingError

router = APIRouter(tags=["search"])


# TEMPORARY: lets you try vector search during Phase 4. Replaced by the real
# retrieval service in Phase 6.
@router.post("/collections/{collection_id}/search", response_model=list[SearchResult])
async def search_collection(
    collection_id: uuid.UUID,
    body: SearchRequest,
    db: DbSession,
    user: CurrentUser,
    embedder: EmbedderDep,
):
    # Ownership first: 404 for other users' collections, and we never pay to
    # embed a query for a collection the caller can't see.
    collection = await collection_service.get_owned_collection(db, collection_id, user.id)
    try:
        return await search_service.search_collection(
            db, embedder, collection, body.query, body.limit
        )
    except EmbeddingBusyError as exc:
        # The shared embedding quota is used up for now. 503 rather than 429:
        # this caller didn't send too much, the service as a whole is busy.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))},
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
