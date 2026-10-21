"""Hybrid retrieval: vector + keyword, fused with RRF."""

import uuid

import pytest

from app.api.deps import embedding_provider
from app.core.config import get_settings
from app.main import app
from app.models import Collection
from app.services.retrieval import hybrid_search, search_by_vector
from tests.embedding_fakes import KeywordProvider
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload

# The fake embedder only "understands" these words; "ZX-9" means nothing to it.
KEYWORDS = ["battery", "charging", "screen"]


@pytest.fixture(autouse=True)
def short_documents_allowed(monkeypatch):
    monkeypatch.setattr(get_settings(), "MIN_EXTRACTED_CHARS", 1)


@pytest.fixture
def embedder(client):
    provider = KeywordProvider(KEYWORDS, model="keyword-model")
    app.dependency_overrides[embedding_provider] = lambda: provider
    return provider


async def collection_with(client, db, docs, email="owner@example.com"):
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers, f"c-{uuid.uuid4().hex[:6]}")
    for name, body in docs.items():
        assert (
            await upload(client, headers, collection_id, name, body.encode())
        ).status_code == 202
    assert all(r.outcome == "ready" for r in await run_queued_jobs())
    return await db.get(Collection, uuid.UUID(collection_id))


# Five documents about battery charging (close in meaning to a battery
# question), and one part note that is the only one naming "ZX-9": it mentions
# the battery once but is mostly about the screen, so by meaning it is far
# from "ZX-9 battery" (distance ~0.68 against ~0.11 for the guides).
DOCS = {
    f"charging-{n}.md": f"Battery charging guide {n}: charge the battery fully." for n in range(5)
}
DOCS["zz-part-note.md"] = (
    "Part ZX-9 fits the screen frame: screen hinge, screen glass, battery clip."
)
NOTE_DISTANCE = 1 - 1 / 10**0.5  # battery 1, screen 3 vs. a pure "battery" query


async def hybrid(db, embedder, collection, query, match="strict", candidates=20, top_k=3):
    vector = await embedder.embed_query(query)
    return await hybrid_search(
        db,
        collection.owner_id,
        collection.id,
        embedder.model,
        vector,
        query,
        candidates,
        top_k,
        60,
        "english",
        match,
    )


async def test_rare_exact_term_ranked_low_by_vectors_surfaces_in_hybrid(client, db, embedder):
    collection = await collection_with(client, db, DOCS)
    query = "ZX-9 battery"

    vector = await embedder.embed_query(query)
    by_vector = await search_by_vector(
        db, collection.owner_id, collection.id, embedder.model, vector, 20
    )
    vector_rank = [c.document_name for c in by_vector].index("zz-part-note.md") + 1
    assert vector_rank == len(DOCS)  # dead last by meaning alone

    results = await hybrid(db, embedder, collection, query, top_k=3)
    assert results[0].document_name == "zz-part-note.md"  # first once keywords count
    assert results[0].keyword_rank == 1 and results[0].vector_rank == vector_rank


async def test_every_result_has_a_vector_distance_and_appears_once(client, db, embedder):
    collection = await collection_with(client, db, DOCS)
    results = await hybrid(db, embedder, collection, "battery charging ZX-9", "loose", top_k=10)

    assert len({r.chunk_id for r in results}) == len(results)
    assert all(r.distance is not None for r in results)
    assert all(r.rrf_score is not None for r in results)
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_keyword_only_hit_beyond_the_vector_candidates_still_gets_a_distance(
    client, db, embedder
):
    collection = await collection_with(client, db, DOCS)
    # Only 2 vector candidates: the part note isn't among them, only keyword finds it.
    results = await hybrid(db, embedder, collection, "ZX-9 battery", candidates=2, top_k=5)
    [note] = [r for r in results if r.document_name == "zz-part-note.md"]
    assert note.vector_rank is None and note.keyword_rank == 1
    assert note.distance == pytest.approx(NOTE_DISTANCE, abs=0.01)  # computed anyway


async def test_top_k_limits_the_fused_list(client, db, embedder):
    collection = await collection_with(client, db, DOCS)
    assert len(await hybrid(db, embedder, collection, "battery", "loose", top_k=2)) == 2


async def test_hybrid_never_returns_another_users_chunks(client, db, embedder):
    mine = await collection_with(client, db, {"mine.md": "Battery charging notes for my team."})
    await collection_with(
        client, db, {"theirs.md": "Part ZX-9 ZX-9 ZX-9 cable."}, "bob@example.com"
    )
    results = await hybrid(db, embedder, mine, "ZX-9 cable", "loose", top_k=10)
    assert {r.document_name for r in results} <= {"mine.md"}


async def test_same_query_gives_the_same_order(client, db, embedder):
    collection = await collection_with(client, db, DOCS)
    first = [
        r.document_name for r in await hybrid(db, embedder, collection, "battery", "loose", top_k=6)
    ]
    again = [
        r.document_name for r in await hybrid(db, embedder, collection, "battery", "loose", top_k=6)
    ]
    assert first == again
