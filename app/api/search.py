import uuid

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession, EmbedderDep
from app.api.errors import provider_error
from app.schemas.search import SearchRequest, SearchResult
from app.services import collections as collection_service
from app.services import retrieval
from app.services.embeddings import EmbeddingError

router = APIRouter(tags=["search"])


# A debugging view: the chunks retrieval finds for a query, with their
# distances, without generating an answer. POST /collections/{id}/ask uses the
# same retrieval to answer questions.
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
        chunks = await retrieval.retrieve(db, embedder, collection, body.query, body.limit)
    except EmbeddingError as exc:
        raise provider_error(exc) from exc
    return [SearchResult.model_validate(chunk, from_attributes=True) for chunk in chunks]
