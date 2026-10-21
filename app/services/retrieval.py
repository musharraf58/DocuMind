"""Retrieval: find the chunks of one collection that best match a query.

Two ways to search, combinable (see search_chunks / fusion.py):
- vector: closest in MEANING (cosine distance between embeddings);
- keyword: containing the query's WORDS (Postgres full-text search), which
  catches exact rare terms (error codes, part numbers) that meaning-based
  search can rank low.

Owner and collection filters are always part of the SQL itself.
Used by POST /collections/{id}/ask, POST /collections/{id}/search and the
evaluation (eval/).
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from sqlalchemy import Select, cast, func, literal, select, text
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Collection, Document
from app.models.chunk import INDEXED_TEXT_SEARCH_CONFIG
from app.services.embeddings import EmbeddingProvider, Vector
from app.services.fusion import reciprocal_rank_fusion

KeywordMatch = Literal["strict", "loose"]
_WORDS = re.compile(r"\w+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    content: str
    page_number: int | None
    section_title: str | None
    distance: float  # cosine distance: 0 = same direction, larger = less similar
    # Positions (1 = best) in each list it came from, and the fused score.
    vector_rank: int | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None  # ts_rank_cd: higher = better keyword match
    # True if the chunk contains ALL the query's terms (always so in strict
    # mode). The relevance gate trusts only such matches.
    keyword_all_terms: bool | None = None
    rrf_score: float | None = None


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
    # Order ties the same way every time (chunk ids change on re-processing,
    # so they can't be the tie-breaker; the SQL orders by distance only, so
    # the index can serve it).
    results.sort(key=lambda c: (c.distance, c.document_name, c.content))
    return [replace(c, vector_rank=rank) for rank, c in enumerate(results, start=1)]


def keyword_tsquery(query: str, config: str, match: KeywordMatch):
    """The query as a tsquery. strict: websearch syntax as typed (all terms
    must match; "quoted phrases" and -excluded words work). loose: any of
    the words may match. None if the query has no searchable words."""
    if match == "loose":
        words = _WORDS.findall(query)
        if not words:
            return None
        query = " or ".join(words)
    return func.websearch_to_tsquery(cast(literal(config), REGCONFIG), query)


def _document_vector(config: str):
    if config == INDEXED_TEXT_SEARCH_CONFIG:
        return Chunk.search_vector  # the indexed, generated column
    # Another configuration: computed per row (no index; fine for small
    # collections such as the evaluation's).
    return func.to_tsvector(
        cast(literal(config), REGCONFIG),
        func.coalesce(Chunk.section_title, "") + " " + Chunk.content,
    )


async def keyword_search(
    db: AsyncSession,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID,
    model: str,
    vector: Vector,
    query: str,
    limit: int,
    config: str,
    match: KeywordMatch,
) -> list[RetrievedChunk]:
    """Chunks containing the query's words, best keyword match first
    (ts_rank_cd), with owner and collection filters in the SQL.

    The same population as vector search (chunks embedded with the current
    model), and each hit's vector distance is computed too, so keyword-only
    hits can still pass through the relevance gate."""
    tsquery = keyword_tsquery(query, config, match)
    if tsquery is None:
        return []
    document_vector = _document_vector(config)
    score = func.ts_rank_cd(document_vector, tsquery)
    all_terms = keyword_tsquery(query, config, "strict")
    rows = await db.execute(
        select(
            Chunk.id.label("chunk_id"),
            Chunk.document_id,
            Document.original_filename.label("document_name"),
            Chunk.content,
            Chunk.page_number,
            Chunk.section_title,
            Chunk.embedding.cosine_distance(vector).label("distance"),
            score.label("keyword_score"),
            (
                document_vector.bool_op("@@")(all_terms)
                if all_terms is not None
                else literal(False)
            ).label("keyword_all_terms"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.owner_id == owner_id,
            Chunk.collection_id == collection_id,
            Document.owner_id == owner_id,
            Chunk.embedding_model == model,
            Chunk.embedding.is_not(None),
            document_vector.bool_op("@@")(tsquery),
        )
        .order_by(score.desc(), Document.original_filename, Chunk.chunk_index)
        .limit(limit)
    )
    results = [RetrievedChunk(**row._mapping) for row in rows]
    await db.commit()
    return [replace(c, keyword_rank=rank) for rank, c in enumerate(results, start=1)]


async def hybrid_search(
    db: AsyncSession,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID,
    model: str,
    vector: Vector,
    query: str,
    candidates: int,
    top_k: int,
    rrf_k: int,
    config: str,
    match: KeywordMatch,
) -> list[RetrievedChunk]:
    """The top `candidates` of vector search and of keyword search, fused with
    Reciprocal Rank Fusion; the best `top_k`, each chunk once. Every result
    carries its vector distance (keyword-only hits included), vector rank,
    keyword rank and fused score."""
    by_vector = await search_by_vector(db, owner_id, collection_id, model, vector, candidates)
    by_keyword = await keyword_search(
        db, owner_id, collection_id, model, vector, query, candidates, config, match
    )
    merged = {chunk.chunk_id: chunk for chunk in by_vector}
    for hit in by_keyword:
        if hit.chunk_id in merged:
            merged[hit.chunk_id] = replace(
                merged[hit.chunk_id],
                keyword_rank=hit.keyword_rank,
                keyword_score=hit.keyword_score,
                keyword_all_terms=hit.keyword_all_terms,
            )
        else:
            merged[hit.chunk_id] = hit  # keyword-only: its distance came with it
    fused = reciprocal_rank_fusion(
        [[c.chunk_id for c in by_vector], [c.chunk_id for c in by_keyword]], rrf_k
    )
    return [replace(merged[chunk_id], rrf_score=score) for chunk_id, score in fused[:top_k]]


SearchMode = Literal["vector", "keyword", "hybrid"]


@dataclass(frozen=True)
class SearchOptions:
    """How to search. From the settings for /search and /ask; varied by the
    evaluation. "keyword" (keywords only) exists for evaluation."""

    mode: SearchMode
    top_k: int
    candidates: int
    rrf_k: int
    text_search_config: str
    keyword_match: KeywordMatch

    @classmethod
    def from_settings(cls, settings, top_k: int | None = None) -> "SearchOptions":
        top_k = top_k or settings.RAG_TOP_K
        return cls(
            mode=settings.RAG_SEARCH_MODE,
            top_k=top_k,
            candidates=max(settings.HYBRID_CANDIDATES, top_k),
            rrf_k=settings.RRF_K,
            text_search_config=settings.TEXT_SEARCH_CONFIG,
            keyword_match=settings.RAG_KEYWORD_MATCH,
        )


async def search_chunks(
    db: AsyncSession,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID,
    model: str,
    vector: Vector,
    query: str,
    options: SearchOptions,
) -> list[RetrievedChunk]:
    """Search one collection in the configured mode; best first, top_k results."""
    if options.mode == "vector":
        return await search_by_vector(db, owner_id, collection_id, model, vector, options.top_k)
    if options.mode == "keyword":
        return await keyword_search(
            db,
            owner_id,
            collection_id,
            model,
            vector,
            query,
            options.top_k,
            options.text_search_config,
            options.keyword_match,
        )
    return await hybrid_search(
        db,
        owner_id,
        collection_id,
        model,
        vector,
        query,
        options.candidates,
        options.top_k,
        options.rrf_k,
        options.text_search_config,
        options.keyword_match,
    )


def filter_relevant(chunks: Sequence[RetrievedChunk], max_distance: float) -> list[RetrievedChunk]:
    """Keep only chunks close enough to count as relevant (distance <= max)."""
    return [chunk for chunk in chunks if chunk.distance <= max_distance]
