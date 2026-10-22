"""Keyword (full-text) search: matching, ranking, isolation, staying in sync."""

import uuid

import pytest

from app.api.deps import embedding_provider
from app.core.config import get_settings
from app.main import app
from app.models import Collection
from app.services import processing
from app.services.chunking import ChunkData
from app.services.retrieval import keyword_search
from tests.embedding_fakes import KeywordProvider
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload

KEYWORDS = ["refund", "shipping", "warranty", "error"]


@pytest.fixture(autouse=True)
def short_documents_allowed(monkeypatch):
    """These test documents are deliberately tiny; the scanned-PDF guard
    (MIN_EXTRACTED_CHARS) would otherwise reject them."""
    monkeypatch.setattr(get_settings(), "MIN_EXTRACTED_CHARS", 1)


@pytest.fixture
def embedder(client):
    provider = KeywordProvider(KEYWORDS, model="keyword-model")
    app.dependency_overrides[embedding_provider] = lambda: provider
    return provider


async def collection_with(client, db, docs: dict[str, str], email="owner@example.com"):
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers, f"c-{uuid.uuid4().hex[:6]}")
    for name, body in docs.items():
        assert (
            await upload(client, headers, collection_id, name, body.encode())
        ).status_code == 202
    results = await run_queued_jobs()
    assert all(r.outcome == "ready" for r in results), results
    return headers, await db.get(Collection, uuid.UUID(collection_id))


async def search(db, embedder, collection, query, match="strict", config="english", limit=20):
    vector = await embedder.embed_query(query)
    return await keyword_search(
        db, collection.owner_id, collection.id, embedder.model, vector, query, limit, config, match
    )


DOCS = {
    "refunds.md": "# Refunds\n\nA refund is paid within 14 days of the return arriving.",
    "shipping.md": "# Shipping\n\nOrders ship within two business days by courier.",
    "runbook.md": "# Runbook\n\nIf the scanner shows ERR-4471, reseat the battery and restart.",
}


async def test_strict_needs_every_term_loose_needs_any(client, db, embedder):
    _, collection = await collection_with(client, db, DOCS)

    strict = await search(db, embedder, collection, "refund courier", "strict")
    loose = await search(db, embedder, collection, "refund courier", "loose")

    assert strict == []  # no chunk has both words
    assert {hit.document_name for hit in loose} == {"refunds.md", "shipping.md"}


async def test_rare_exact_term_is_found(client, db, embedder):
    _, collection = await collection_with(client, db, DOCS)
    [hit] = await search(db, embedder, collection, "ERR-4471")
    assert hit.document_name == "runbook.md"
    assert hit.keyword_rank == 1 and hit.keyword_score > 0
    assert hit.distance is not None  # vector distance computed for the gate


async def test_better_matches_rank_higher(client, db, embedder):
    docs = {
        "once.md": "The refund rule. Other unrelated text about the office kitchen.",
        "often.md": "Refund policy: every refund request gets a refund decision quickly.",
    }
    _, collection = await collection_with(client, db, docs)
    hits = await search(db, embedder, collection, "refund", "loose")
    assert [h.document_name for h in hits] == ["often.md", "once.md"]
    assert [h.keyword_rank for h in hits] == [1, 2]


async def test_english_stems_words_and_simple_does_not(client, db, embedder):
    _, collection = await collection_with(client, db, DOCS)
    # "refunded" and "refund" share the stem "refund" in English.
    assert [h.document_name for h in await search(db, embedder, collection, "refunded")] == [
        "refunds.md"
    ]
    assert await search(db, embedder, collection, "refunded", config="simple") == []
    assert len(await search(db, embedder, collection, "refund", config="simple")) == 1


async def test_section_titles_are_searchable(client, db, embedder):
    docs = {"guide.md": "# Troubleshooting\n\nRestart the device and wait a minute."}
    _, collection = await collection_with(client, db, docs)
    [hit] = await search(db, embedder, collection, "troubleshooting")
    assert hit.section_title == "Troubleshooting"


async def test_query_with_nothing_searchable_returns_nothing(client, db, embedder):
    _, collection = await collection_with(client, db, DOCS)
    assert await search(db, embedder, collection, "!!! ???", "loose") == []
    assert await search(db, embedder, collection, "the and of", "strict") == []  # stop words


async def test_never_returns_another_users_or_collections_chunks(client, db, embedder):
    headers, mine = await collection_with(client, db, {"mine.md": "Nothing relevant here."})
    second_id = await create_collection(client, headers, "Second")
    await upload(
        client, headers, second_id, "other.md", b"ERR-4471 appears in my other collection."
    )
    await run_queued_jobs()
    await collection_with(
        client, db, {"theirs.md": "ERR-4471 ERR-4471 in someone else's runbook."}, "bob@example.com"
    )

    for match in ("strict", "loose"):
        assert await search(db, embedder, mine, "ERR-4471", match) == []
    second = await db.get(Collection, uuid.UUID(second_id))
    [hit] = await search(db, embedder, second, "ERR-4471")
    assert hit.document_name == "other.md"  # not Bob's, though Bob's matches more often


async def test_keyword_column_follows_reprocessing(client, db, embedder, monkeypatch):
    headers, collection = await collection_with(
        client, db, {"notes.md": "The original wording mentions zeppelins."}
    )
    assert len(await search(db, embedder, collection, "zeppelins")) == 1
    [document_id] = {h.document_id for h in await search(db, embedder, collection, "zeppelins")}

    new_text = [ChunkData(0, "Rewritten text about submarines instead.", None, None, 6)]
    monkeypatch.setattr(processing, "chunk_segments", lambda *args: new_text)
    assert (
        await client.post(f"/documents/{document_id}/reprocess", headers=headers)
    ).status_code == 202
    await run_queued_jobs()

    assert await search(db, embedder, collection, "zeppelins") == []
    assert len(await search(db, embedder, collection, "submarines")) == 1
