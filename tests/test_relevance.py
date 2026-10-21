"""The relevance gate, on its own and through /ask and /search in hybrid mode."""

import uuid

import pytest

from app.api.deps import embedding_provider, llm_provider
from app.core.config import get_settings
from app.main import app
from app.services.answering import NOT_FOUND_ANSWER
from app.services.llm import FakeLLMProvider
from app.services.relevance import passes, relevant
from app.services.retrieval import RetrievedChunk
from tests.embedding_fakes import KeywordProvider, RecordingSleep
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload


def chunk(distance, keyword_rank=None, all_terms=None) -> RetrievedChunk:
    return RetrievedChunk(
        uuid.uuid4(),
        uuid.uuid4(),
        "doc.md",
        "text",
        None,
        None,
        distance,
        keyword_rank=keyword_rank,
        keyword_all_terms=all_terms,
    )


# --- The rule ---


def test_close_enough_by_meaning_passes_in_any_mode():
    for hybrid in (False, True):
        assert passes(chunk(0.70), 0.72, 0.0, hybrid)
        assert passes(chunk(0.72), 0.72, 0.0, hybrid)  # the threshold itself passes
        assert not passes(chunk(0.73), 0.72, 0.0, hybrid)


def test_strong_keyword_match_gets_the_slack_in_hybrid_mode_only():
    strong = chunk(0.78, keyword_rank=1, all_terms=True)
    assert passes(strong, 0.72, 0.08, hybrid=True)
    assert not passes(strong, 0.72, 0.08, hybrid=False)
    assert not passes(strong, 0.72, 0.0, hybrid=True)  # slack 0: the path is off


@pytest.mark.parametrize(
    ("candidate", "why"),
    [
        (chunk(0.78, keyword_rank=1, all_terms=False), "only some terms matched"),
        (chunk(0.78, keyword_rank=4, all_terms=True), "not in the keyword top 3"),
        (chunk(0.81, keyword_rank=1, all_terms=True), "beyond threshold + slack"),
        (chunk(0.78), "no keyword match at all"),
    ],
)
def test_weak_keyword_evidence_gets_no_slack(candidate, why):
    assert not passes(candidate, 0.72, 0.08, hybrid=True), why


def test_relevant_keeps_order_and_drops_the_rest():
    a, b, c = chunk(0.3), chunk(0.9), chunk(0.5)
    assert relevant([a, b, c], 0.72, 0.0, False) == [a, c]


# --- Through the API, in hybrid mode ---

KEYWORDS = ["refund", "shipping", "volcano"]


@pytest.fixture
def hybrid(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "RAG_SEARCH_MODE", "hybrid")
    monkeypatch.setattr(settings, "RAG_KEYWORD_GATE_SLACK", 0.1)
    embedder = KeywordProvider(KEYWORDS, model="keyword-model")
    llm = FakeLLMProvider(answer="Refunds take 14 days [1].", sleep=RecordingSleep())
    app.dependency_overrides[embedding_provider] = lambda: embedder
    app.dependency_overrides[llm_provider] = lambda: llm
    return llm


async def setup(client):
    headers = await auth_headers(client, "owner@example.com")
    collection_id = await create_collection(client, headers)
    docs = {
        "refunds.md": "Refund policy. A refund is paid within 14 days. " * 3,
        "shipping.md": "Shipping policy. Orders ship within two days. " * 3,
    }
    for name, body in docs.items():
        await upload(client, headers, collection_id, name, body.encode())
    await run_queued_jobs()
    return headers, collection_id


async def ask(client, headers, collection_id, question):
    return await client.post(
        f"/collections/{collection_id}/ask", json={"question": question}, headers=headers
    )


async def test_gate_rejects_an_unrelated_question_in_hybrid_mode(client, hybrid):
    headers, collection_id = await setup(client)
    # Shares the word "policy" with both documents, but means something else.
    body = (
        await ask(client, headers, collection_id, "What is the volcano evacuation policy?")
    ).json()
    assert (body["answer"], body["grounded"]) == (NOT_FOUND_ANSWER, False)
    assert hybrid.calls == []  # the model was never asked


async def test_related_question_is_answered_in_hybrid_mode(client, hybrid):
    headers, collection_id = await setup(client)
    body = (await ask(client, headers, collection_id, "How long does a refund take?")).json()
    assert body["grounded"] is True
    assert body["citations"][0]["document_name"] == "refunds.md"


async def test_search_shows_both_rankings_in_hybrid_mode(client, hybrid):
    headers, collection_id = await setup(client)
    results = (
        await client.post(
            f"/collections/{collection_id}/search",
            json={"query": "refund", "limit": 5},
            headers=headers,
        )
    ).json()
    top = results[0]
    assert top["document_name"] == "refunds.md"
    assert top["vector_rank"] == 1 and top["keyword_rank"] == 1
    assert top["keyword_all_terms"] is True and top["rrf_score"] > 0
    assert top["distance"] == pytest.approx(0, abs=0.01)
