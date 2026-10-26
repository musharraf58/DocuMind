"""POST /collections/{id}/ask with conversations: saving turns, follow-up
rewriting, citation snapshots, the per-conversation lock."""

import logging
import uuid

import pytest
from sqlalchemy import func, select

from app.api.deps import embedding_provider, llm_provider
from app.main import app
from app.models import Chunk, Collection, Message, MessageStatus
from app.services import conversations as conversation_service
from app.services.llm import FakeLLMProvider, LLMError, RetryableLLMError
from app.services.rewrite import REWRITE_SYSTEM
from tests.embedding_fakes import KeywordProvider, RecordingSleep
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload

KEYWORDS = ["cats", "databases", "cooking", "rockets"]


def about(topic: str) -> str:
    return f"This document is all about {topic}. " * 5


@pytest.fixture
def embedder(client):
    """Keyword embeddings that record every query embedded."""
    provider = KeywordProvider(KEYWORDS)
    provider.embedded_queries = []
    real = provider.embed_query

    async def recording(text):
        provider.embedded_queries.append(text)
        return await real(text)

    provider.embed_query = recording
    app.dependency_overrides[embedding_provider] = lambda: provider
    return provider


@pytest.fixture
def llm(client):
    """Rewrites follow-ups to a fixed standalone question; answers citing [1]."""

    def reply(system, user):
        if system == REWRITE_SYSTEM:
            return "What do cats do?"
        return "Cats purr [1] and fly [9]."

    fake = FakeLLMProvider(answer=reply, sleep=RecordingSleep())
    app.dependency_overrides[llm_provider] = lambda: fake
    return fake


async def setup(client, docs=None, email="owner@example.com"):
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    for name, text in (docs if docs is not None else {"cats.txt": about("cats")}).items():
        assert (
            await upload(client, headers, collection_id, name, text.encode())
        ).status_code == 202
    await run_queued_jobs()
    return headers, collection_id


async def ask(client, headers, collection_id, question, conversation_id=None):
    body = {"question": question}
    if conversation_id:
        body["conversation_id"] = conversation_id
    return await client.post(f"/collections/{collection_id}/ask", json=body, headers=headers)


def rewrite_calls(llm):
    return [call for call in llm.calls if call[0] == REWRITE_SYSTEM]


async def test_first_question_starts_a_conversation_and_saves_both_messages(client, embedder, llm):
    headers, collection_id = await setup(client)
    body = (await ask(client, headers, collection_id, "What do cats do?")).json()

    conversation = (
        await client.get(f"/conversations/{body['conversation_id']}", headers=headers)
    ).json()
    assert conversation["title"] == "What do cats do?"
    question, answer = conversation["messages"]
    assert (question["role"], question["content"], question["status"]) == (
        "user",
        "What do cats do?",
        "complete",
    )
    assert answer["id"] == body["message_id"]
    assert (answer["status"], answer["grounded"]) == ("complete", True)
    # What was returned is what was saved, invalid [9] removed from both.
    assert answer["content"] == body["answer"] == "Cats purr [1] and fly."
    assert [c["number"] for c in answer["citations"]] == [c["number"] for c in body["citations"]]
    assert answer["prompt_tokens"] == body["usage"]["prompt_tokens"] > 0


async def test_first_question_never_triggers_a_rewrite(client, embedder, llm):
    headers, collection_id = await setup(client)
    await ask(client, headers, collection_id, "What about cats?")
    assert rewrite_calls(llm) == []
    assert embedder.embedded_queries == ["What about cats?"]  # one Voyage request


async def test_follow_up_is_rewritten_and_the_rewrite_is_what_gets_embedded(
    client, embedder, llm, caplog
):
    caplog.set_level(logging.INFO, logger="app")
    headers, collection_id = await setup(client)
    first = (await ask(client, headers, collection_id, "Tell me about cats")).json()
    embedder.embedded_queries.clear()
    llm.calls.clear()

    response = await ask(
        client, headers, collection_id, "And what do they do?", first["conversation_id"]
    )

    assert response.status_code == 200
    [(_, rewrite_prompt, _)] = rewrite_calls(llm)
    assert "Tell me about cats" in rewrite_prompt and "And what do they do?" in rewrite_prompt
    assert embedder.embedded_queries == ["What do cats do?"]  # the REWRITTEN question, once
    # The answer prompt keeps the user's own words, plus the history.
    _, answer_prompt, _ = llm.calls[-1]
    assert "<question>\nAnd what do they do?\n</question>" in answer_prompt
    assert "<history>" in answer_prompt and "Tell me about cats" in answer_prompt
    # Logged with ids and token counts only.
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("rewrite:")]
    assert first["conversation_id"] in line and "rewritten=True" in line
    assert "completion_tokens=" in line
    assert "And what do they do?" not in caplog.text and "What do cats do?" not in caplog.text


async def test_failed_rewrite_falls_back_to_the_original_question(client, embedder, llm):
    headers, collection_id = await setup(client)
    first = (await ask(client, headers, collection_id, "Tell me about cats")).json()
    embedder.embedded_queries.clear()
    llm.failures = [LLMError("The answer service rejected the request")]  # hits the rewrite

    response = await ask(
        client, headers, collection_id, "What about cats then?", first["conversation_id"]
    )

    assert response.status_code == 200
    assert embedder.embedded_queries == ["What about cats then?"]
    assert response.json()["grounded"] is True


async def test_nothing_relevant_is_saved_without_an_llm_call(client, embedder, llm):
    headers, collection_id = await setup(client)
    body = (await ask(client, headers, collection_id, "Tell me about rockets")).json()
    assert llm.calls == []
    messages = (
        await client.get(f"/conversations/{body['conversation_id']}", headers=headers)
    ).json()["messages"]
    assert (messages[1]["content"], messages[1]["grounded"]) == (
        "I could not find this in your documents.",
        False,
    )


async def test_llm_failure_saves_failed_keeps_the_question_and_releases_the_lock(
    client, embedder, llm
):
    headers, collection_id = await setup(client)
    llm.failures = [RetryableLLMError("The answer service timed out")] * 3
    response = await ask(client, headers, collection_id, "What do cats do?")
    assert response.status_code == 503

    [conversation] = (await client.get("/conversations", headers=headers)).json()
    messages = (await client.get(f"/conversations/{conversation['id']}", headers=headers)).json()[
        "messages"
    ]
    assert messages[0]["content"] == "What do cats do?"  # the question is kept
    assert (messages[1]["status"], messages[1]["error_message"]) == (
        "failed",
        "The answer service timed out",
    )
    # The conversation isn't stuck: the next question goes through.
    again = await ask(client, headers, collection_id, "What do cats do?", conversation["id"])
    assert again.status_code == 200


async def test_second_answer_while_one_is_in_progress_is_409(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    first = (await ask(client, headers, collection_id, "What do cats do?")).json()
    collection = await db.get(Collection, uuid.UUID(collection_id))
    in_flight = await conversation_service.start_turn(  # e.g. a stream still running
        db, collection, "Another?", uuid.UUID(first["conversation_id"]), 6
    )

    response = await ask(client, headers, collection_id, "Third?", first["conversation_id"])
    assert response.status_code == 409
    assert response.json()["detail"] == "An answer is already being generated in this conversation"
    assert response.json()["code"] == "conflict"
    assert embedder.embedded_queries == ["What do cats do?"]  # nothing done for the refused one

    await conversation_service.finish_turn(db, in_flight, MessageStatus.complete, "done")
    assert (
        await ask(client, headers, collection_id, "Third?", first["conversation_id"])
    ).status_code == 200


async def test_conversation_of_another_collection_is_404(client, embedder, llm):
    headers, collection_id = await setup(client)
    other_collection = await create_collection(client, headers, "Other")
    first = (await ask(client, headers, collection_id, "What do cats do?")).json()

    response = await ask(client, headers, other_collection, "And?", first["conversation_id"])
    assert response.status_code == 404
    assert embedder.embedded_queries == ["What do cats do?"]


async def test_asking_into_another_users_conversation_is_404(client, embedder, llm):
    alice, alice_collection = await setup(client, email="alice@example.com")
    first = (await ask(client, alice, alice_collection, "What do cats do?")).json()
    bob, bob_collection = await setup(client, docs={}, email="bob@example.com")

    into_alices = await ask(client, bob, alice_collection, "Hi", first["conversation_id"])
    via_own_collection = await ask(client, bob, bob_collection, "Hi", first["conversation_id"])
    assert into_alices.status_code == via_own_collection.status_code == 404


async def test_citation_snapshots_survive_reprocessing_the_document(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    body = (await ask(client, headers, collection_id, "What do cats do?")).json()
    url = f"/conversations/{body['conversation_id']}"
    [before] = (await client.get(url, headers=headers)).json()["messages"][1]["citations"]
    document_id = before["document_id"]

    assert (
        await client.post(f"/documents/{document_id}/reprocess", headers=headers)
    ).status_code == 202
    await run_queued_jobs()  # the document's chunks are replaced: new ids

    old_chunk_still_there = await db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.id == uuid.UUID(before["chunk_id"]))
    )
    assert old_chunk_still_there == 0
    [after] = (await client.get(url, headers=headers)).json()["messages"][1]["citations"]
    assert after == before  # the snapshot is unchanged and still readable
    assert after["document_name"] == "cats.txt" and after["snippet"].startswith("This document")


async def test_deleting_the_collection_removes_its_conversations(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    await ask(client, headers, collection_id, "What do cats do?")
    assert (
        await client.delete(f"/collections/{collection_id}", headers=headers)
    ).status_code == 204
    assert (await client.get("/conversations", headers=headers)).json() == []
    assert await db.scalar(select(func.count()).select_from(Message)) == 0
