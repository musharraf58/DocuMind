"""TEMPORARY vector search, for trying out embeddings in Phase 4.
To be replaced by the real retrieval service in Phase 6.
"""

import uuid

from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Collection, Document
from app.schemas.search import SearchResult
from app.services.embeddings import EmbeddingProvider, Vector

# Run in the same transaction as each search. With filters, an HNSW index
# scan normally looks at only ~40 nearest candidates (hnsw.ef_search) and
# THEN applies the WHERE clause, so a user whose chunks aren't among those
# global nearest would get too few results, or none. pgvector 0.8+ "iterative
# scans" keep searching the index until enough rows pass the filter.
SEARCH_SETTINGS = ["SET LOCAL hnsw.iterative_scan = strict_order"]


def build_search_query(
    owner_id: uuid.UUID, collection_id: uuid.UUID, model: str, vector: Vector, limit: int
) -> Select:
    """Nearest chunks by cosine distance, restricted INSIDE the SQL to one
    owner's collection and to vectors from the current model."""
    distance = Chunk.embedding.cosine_distance(vector)  # the <=> operator
    return (
        select(
            Chunk.id.label("chunk_id"),
            Chunk.document_id,
            Document.original_filename.label("document_name"),
            Chunk.content,
            Chunk.page_number,
            Chunk.section_title,
            distance.label("distance"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.owner_id == owner_id,
            Chunk.collection_id == collection_id,
            Document.owner_id == owner_id,  # defense in depth on the joined row too
            # Vectors from another model are in a different space: comparing
            # them to this query's vector would give meaningless distances.
            Chunk.embedding_model == model,
            Chunk.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(limit)
    )


async def search_collection(
    db: AsyncSession,
    embedder: EmbeddingProvider,
    collection: Collection,
    query: str,
    limit: int,
) -> list[SearchResult]:
    """Search a collection the caller has already proven they own."""
    # End the ownership check's read transaction so no database connection is
    # held while we wait on the embedding provider's network call.
    await db.commit()
    vector = await embedder.embed_query(query)

    for setting in SEARCH_SETTINGS:
        await db.execute(text(setting))
    rows = await db.execute(
        build_search_query(collection.owner_id, collection.id, embedder.model, vector, limit)
    )
    results = [SearchResult.model_validate(row, from_attributes=True) for row in rows]
    await db.commit()  # ends the transaction, and with it the SET LOCAL settings
    return results
