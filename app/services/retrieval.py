"""Retrieval: find the chunks of one collection closest in meaning to a query.

Used by POST /collections/{id}/ask (to answer questions) and by
POST /collections/{id}/search (a debugging view of the same results).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Collection, Document
from app.services.embeddings import EmbeddingProvider, Vector


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    content: str
    page_number: int | None
    section_title: str | None
    distance: float  # cosine distance: 0 = same direction, larger = less similar


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


async def retrieve(
    db: AsyncSession,
    embedder: EmbeddingProvider,
    collection: Collection,
    query: str,
    limit: int,
) -> list[RetrievedChunk]:
    """The `limit` chunks nearest to `query`, best first, from a collection the
    caller has already proven they own."""
    # End the ownership check's read transaction so no database connection is
    # held while we wait on the embedding provider's network call.
    owner_id, collection_id = collection.owner_id, collection.id
    await db.commit()
    vector = await embedder.embed_query(query)
    return await search_by_vector(db, owner_id, collection_id, embedder.model, vector, limit)


async def search_by_vector(
    db: AsyncSession,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID,
    model: str,
    vector: Vector,
    limit: int,
) -> list[RetrievedChunk]:
    """The SQL half of retrieval, for callers that embedded the query themselves
    (outside any session). One short transaction."""
    for setting in SEARCH_SETTINGS:
        await db.execute(text(setting))
    rows = await db.execute(build_search_query(owner_id, collection_id, model, vector, limit))
    results = [RetrievedChunk(**row._mapping) for row in rows]
    await db.commit()  # ends the transaction, and with it the SET LOCAL settings
    return results


def filter_relevant(chunks: Sequence[RetrievedChunk], max_distance: float) -> list[RetrievedChunk]:
    """Keep only chunks close enough to count as relevant (distance <= max)."""
    return [chunk for chunk in chunks if chunk.distance <= max_distance]
