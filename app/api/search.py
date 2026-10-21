import uuid

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession, EmbedderDep
from app.api.errors import provider_error
from app.core.config import get_settings
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
    owner_id, collection_id = collection.owner_id, collection.id
    await db.commit()  # no transaction held during the embedding call
    try:
        vector = await embedder.embed_query(body.query)
    except EmbeddingError as exc:
        raise provider_error(exc) from exc
    options = retrieval.SearchOptions.from_settings(get_settings(), top_k=body.limit)
    chunks = await retrieval.search_chunks(
        db, owner_id, collection_id, embedder.model, vector, body.query, options
    )
    return [SearchResult.model_validate(chunk, from_attributes=True) for chunk in chunks]
