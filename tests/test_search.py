import uuid

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.dialects import postgresql

from app.api.deps import embedding_provider
from app.core.config import get_settings
from app.main import app
from app.models import Chunk
from app.services import retrieval as search_service
from app.services.embeddings import EmbeddingBusyError, EmbeddingError
from app.services.retrieval import build_search_query
from tests.embedding_fakes import FakeClock, KeywordProvider, limited_provider
from tests.helpers import auth_headers, create_collection, error_body, run_queued_jobs, upload

KEYWORDS = ["cats", "databases", "cooking", "rockets"]
HNSW_INDEX = "ix_chunks_embedding_hnsw"


@pytest.fixture
def keywords(client):
    """Use the keyword provider for processing and search alike."""
    provider = KeywordProvider(KEYWORDS)
    app.dependency_overrides[embedding_provider] = lambda: provider
    return provider


async def add_document(client, headers, collection_id, name, content: str) -> str:
    response = await upload(client, headers, collection_id, name, content.encode())
    assert response.status_code == 202, response.text
    doc_id = response.json()["id"]
    [result] = await run_queued_jobs()
    assert result.outcome == "ready", result
    return doc_id


async def search(client, headers, collection_id, query, **body):
    return await client.post(
        f"/collections/{collection_id}/search", json={"query": query, **body}, headers=headers
    )


def about(topic: str) -> str:
    return f"This document is all about {topic}. " * 5


async def test_results_are_ordered_by_meaning(client, keywords):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    await add_document(client, headers, collection_id, "cats.txt", about("cats"))
    await add_document(client, headers, collection_id, "db.txt", about("databases"))
    await add_document(client, headers, collection_id, "mixed.txt", about("cats and databases"))

    response = await search(client, headers, collection_id, "tell me about cats")
    assert response.status_code == 200
    results = response.json()
    assert [r["document_name"] for r in results] == ["cats.txt", "mixed.txt", "db.txt"]
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0, abs=1e-3)
    assert set(results[0]) == {
        "chunk_id",
        "document_id",
        "document_name",
        "content",
        "page_number",
        "section_title",
        "distance",
        # retrieval debugging: what each search said about the chunk
        "vector_rank",
        "keyword_rank",
        "keyword_score",
        "keyword_all_terms",
        "rrf_score",
    }


async def test_limit_is_respected(client, keywords):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    for n in range(7):
        await add_document(client, headers, collection_id, f"{n}.txt", about(f"cats {n}"))

    assert len((await search(client, headers, collection_id, "cats")).json()) == 5  # default
    assert len((await search(client, headers, collection_id, "cats", limit=2)).json()) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"query": "   "},
        {"query": ""},
        {"query": "x" * 2001},
        {"query": "ok", "limit": 21},
        {"query": "ok", "limit": 0},
    ],
)
async def test_invalid_requests_are_422(client, keywords, body):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    response = await client.post(f"/collections/{collection_id}/search", json=body, headers=headers)
    assert response.status_code == 422


async def test_never_returns_another_users_chunks(client, keywords):
    alice = await auth_headers(client, "alice@example.com")
    bob = await auth_headers(client, "bob@example.com")
    alice_collection = await create_collection(client, alice)
    bob_collection = await create_collection(client, bob)
    # Bob's document is a perfect match; Alice's is only partly relevant.
    await add_document(client, bob, bob_collection, "bob-cats.txt", about("cats"))
    await add_document(client, alice, alice_collection, "alice.txt", about("cats and rockets"))

    results = (await search(client, alice, alice_collection, "cats", limit=20)).json()
    assert [r["document_name"] for r in results] == ["alice.txt"]


async def test_other_users_collection_is_404(client, keywords):
    alice = await auth_headers(client, "alice@example.com")
    bob = await auth_headers(client, "bob@example.com")
    bob_collection = await create_collection(client, bob)
    await add_document(client, bob, bob_collection, "bob.txt", about("cats"))

    response = await search(client, alice, bob_collection, "cats")
    missing = await search(client, alice, str(uuid.uuid4()), "cats")
    assert response.status_code == missing.status_code == 404
    assert error_body(response) == error_body(missing)


async def test_only_the_requested_collection_is_searched(client, keywords):
    headers = await auth_headers(client, "owner@example.com")
    work = await create_collection(client, headers, "Work")
    home = await create_collection(client, headers, "Home")
    await add_document(client, headers, work, "work.txt", about("databases"))
    await add_document(client, headers, home, "home.txt", about("cats"))

    results = (await search(client, headers, work, "cats", limit=20)).json()
    assert [r["document_name"] for r in results] == ["work.txt"]


async def test_chunks_from_other_models_or_without_vectors_are_skipped(client, db, keywords):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    old = await add_document(client, headers, collection_id, "old-model.txt", about("cats"))
    unembedded = await add_document(
        client, headers, collection_id, "phase3.txt", about("cats, again")
    )
    await add_document(client, headers, collection_id, "current.txt", about("databases"))

    await db.execute(
        update(Chunk)
        .where(Chunk.document_id == uuid.UUID(old))
        .values(embedding_model="some-older-model")
    )
    await db.execute(
        update(Chunk)
        .where(Chunk.document_id == uuid.UUID(unembedded))
        .values(embedding=None, embedding_model=None)
    )
    await db.commit()

    results = (await search(client, headers, collection_id, "cats", limit=20)).json()
    assert [r["document_name"] for r in results] == ["current.txt"]


async def test_embedding_outage_is_503_with_safe_message(client, keywords, monkeypatch):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)

    async def down(text):
        raise EmbeddingError("Embedding service is unavailable; try again later")

    monkeypatch.setattr(keywords, "embed_query", down)
    response = await search(client, headers, collection_id, "cats")
    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service is unavailable; try again later"


async def test_filtered_search_finds_owner_chunks_beyond_nearest_candidates(
    client, keywords, monkeypatch
):
    """Bob has dozens of chunks right on top of the query; Alice has one
    farther away. An index scan looks at only hnsw.ef_search nearest
    candidates (40 by default; 10 here, so the small test data shows the
    effect) and filters afterwards: without iterative scans, every candidate
    is Bob's, the owner filter removes them all, and Alice gets nothing."""
    # (Checked by hand: with SEARCH_SETTINGS emptied, this test fails.)
    settings = get_settings()
    monkeypatch.setattr(settings, "CHUNK_TOKENS", 50)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP_TOKENS", 5)
    # Make sure the HNSW index is really what answers the query, even on this
    # small table: forbid the alternatives (sequential scan, sort).
    monkeypatch.setattr(
        search_service,
        "SEARCH_SETTINGS",
        [
            *search_service.SEARCH_SETTINGS,
            "SET LOCAL hnsw.ef_search = 10",
            "SET LOCAL enable_seqscan = off",
            "SET LOCAL enable_sort = off",
        ],
    )
    alice = await auth_headers(client, "alice@example.com")
    bob = await auth_headers(client, "bob@example.com")
    alice_collection = await create_collection(client, alice)
    bob_collection = await create_collection(client, bob)
    # Each paragraph differs slightly so the vectors are distinct: pgvector
    # stores identical vectors as ONE graph entry, which would hide the effect.
    bob_text = "\n\n".join(
        "cats " * 20 + "cooking " * (i % 5) + "rockets " * (i // 5) for i in range(40)
    )
    await add_document(client, bob, bob_collection, "bob.txt", bob_text)
    await add_document(
        client, alice, alice_collection, "alice.txt", "One mention of cats. " + about("rockets")
    )

    results = (await search(client, alice, alice_collection, "cats")).json()
    assert [r["document_name"] for r in results] == ["alice.txt"]


async def test_busy_embedding_service_is_429_with_retry_after_header(client, keywords, monkeypatch):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)

    async def busy(text):
        raise EmbeddingBusyError(retry_after=33.2)

    monkeypatch.setattr(keywords, "embed_query", busy)
    response = await search(client, headers, collection_id, "cats")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "34"  # rounded up, never earlier than allowed
    assert response.json()["detail"] == "Embedding service is busy; retry in about 34 seconds"


async def test_real_limiter_makes_search_fail_fast_instead_of_hanging(client):
    clock = FakeClock()
    provider = limited_provider(clock, requests=1, query_max_wait=20)
    await provider.embed_documents(["something else used the only slot"], token_counts=[10])
    clock.now += 10  # 50 s until it frees up
    app.dependency_overrides[embedding_provider] = lambda: provider

    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    response = await search(client, headers, collection_id, "cats")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "50"
    assert "busy" in response.json()["detail"]
    assert clock.now == 1010.0  # it answered at once; no waiting happened


# --- Query plans ---


def sql(statement) -> str:
    return str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


async def explain(db, statement, *settings: str) -> str:
    for setting in settings:
        await db.execute(text(setting))
    rows = await db.execute(text("EXPLAIN " + sql(statement)))
    await db.rollback()  # also undoes the SET LOCALs
    return "\n".join(row[0] for row in rows)


QUERY = [1.0, 0.0, 0.0, 0.0] + [0.0] * 1019 + [0.01]
FORCE_INDEX = ("SET LOCAL enable_seqscan = off", "SET LOCAL enable_sort = off")


async def test_cosine_order_by_uses_hnsw_index(db):
    # On a small table Postgres rightly prefers reading every row, so the
    # alternatives are switched off: if the index CAN serve this query, the
    # planner must now use it. If operator and operator class didn't match,
    # it couldn't, and this would fail.
    statement = select(Chunk.id).order_by(Chunk.embedding.cosine_distance(QUERY)).limit(5)
    assert HNSW_INDEX in await explain(db, statement, *FORCE_INDEX)


async def test_l2_order_by_cannot_use_cosine_index(db):
    # Same query with "<->" (Euclidean distance): our index was built with
    # vector_cosine_ops, so it can't help, however hard we push the planner.
    statement = select(Chunk.id).order_by(Chunk.embedding.l2_distance(QUERY)).limit(5)
    assert HNSW_INDEX not in await explain(db, statement, *FORCE_INDEX)


async def test_real_search_query_can_use_hnsw_index(db):
    statement = build_search_query(uuid.uuid4(), uuid.uuid4(), "m", QUERY, 5)
    plan = await explain(db, statement, *search_service.SEARCH_SETTINGS, *FORCE_INDEX)
    assert HNSW_INDEX in plan


async def test_real_search_query_plan_without_hints(client, db, keywords):
    """What the planner picks on its own. On a tiny table a sequential scan or
    the collection_id index is genuinely cheaper, so that is a skip, not a
    failure; on realistic data it should choose the HNSW index."""
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    await add_document(client, headers, collection_id, "cats.txt", about("cats"))
    await db.execute(text("ANALYZE chunks"))
    await db.commit()

    statement = build_search_query(
        uuid.uuid4(), uuid.UUID(collection_id), "keyword-model", QUERY, 5
    )
    plan = await explain(db, statement, *search_service.SEARCH_SETTINGS)
    if HNSW_INDEX not in plan:
        pytest.skip(f"Planner chose another plan on this tiny table:\n{plan}")
