"""POST /collections/{id}/ask/stream: events, saving, disconnects, failures,
heartbeats, and no database connection held while streaming."""

import asyncio
import logging
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import embedding_provider, llm_provider
from app.core.config import get_settings
from app.main import app
from app.models import Collection, Message, MessageStatus
from app.services import conversations as conversation_service
from app.services.answering import NOT_FOUND_ANSWER
from app.services.embeddings import EmbeddingBusyError, RetryableEmbeddingError
from app.services.llm import FakeLLMProvider, LLMBusyError, RetryableLLMError
from app.services.rewrite import REWRITE_SYSTEM
from tests.asgi_client import StreamResult, post_stream
from tests.conftest import TEST_DB_URL
from tests.embedding_fakes import KeywordProvider, RecordingSleep
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload

KEYWORDS = ["cats", "databases", "cooking", "rockets"]
STREAMED = ["Cats ", "purr ", "[1] ", "and fly ", "[9]."]


def about(topic: str) -> str:
    return f"This document is all about {topic}. " * 5


@pytest.fixture
def embedder(client):
    provider = KeywordProvider(KEYWORDS)
    provider.embedded_queries = []
    real = provider.embed_query

    async def recording(text_):
        provider.embedded_queries.append(text_)
        return await real(text_)

    provider.embed_query = recording
    app.dependency_overrides[embedding_provider] = lambda: provider
    return provider


@pytest.fixture
def llm(client):
    def reply(system, user):
        return "What do cats do?" if system == REWRITE_SYSTEM else "".join(STREAMED)

    fake = FakeLLMProvider(answer=reply, chunks=STREAMED, sleep=RecordingSleep())
    app.dependency_overrides[llm_provider] = lambda: fake
    return fake


async def setup(client, docs=None, email="owner@example.com"):
    headers = await auth_headers(client, email)
    collection_id = await create_collection(client, headers)
    for name, body in (docs if docs is not None else {"cats.txt": about("cats")}).items():
        assert (
            await upload(client, headers, collection_id, name, body.encode())
        ).status_code == 202
    await run_queued_jobs()
    return headers, collection_id


async def stream(headers, collection_id, question, conversation_id=None, **kwargs) -> StreamResult:
    body = {"question": question}
    if conversation_id:
        body["conversation_id"] = str(conversation_id)
    return await post_stream(
        app, f"/collections/{collection_id}/ask/stream", body, headers, **kwargs
    )


async def saved(db, message_id) -> Message:
    return await db.get(Message, uuid.UUID(message_id), populate_existing=True)


def names(result: StreamResult) -> list[str]:
    return [name for name, _ in result.events]


async def test_events_arrive_in_order_and_done_equals_what_is_saved(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    result = await stream(headers, collection_id, "What do cats do?")

    assert result.status == 200
    assert result.headers["content-type"].startswith("text/event-stream")
    # An authenticated response: never stored (stricter than SSE's usual no-cache).
    assert result.headers["cache-control"] == "no-store"
    assert result.headers["x-accel-buffering"] == "no"
    assert names(result) == ["meta"] + ["token"] * len(STREAMED) + ["done"]

    meta = result.events[0][1]
    tokens = "".join(data["text"] for name, data in result.events if name == "token")
    done = result.events[-1][1]
    assert tokens == "Cats purr [1] and fly [9]."  # the raw stream, [9] included
    assert done["answer"] == "Cats purr [1] and fly."  # the checked answer: [9] removed
    assert (done["conversation_id"], done["message_id"]) == (
        meta["conversation_id"],
        meta["message_id"],
    )

    message = await saved(db, done["message_id"])
    assert message.status == MessageStatus.complete
    assert message.content == done["answer"]
    assert message.grounded is done["grounded"] is True
    assert message.citations == done["citations"]
    assert [c["number"] for c in done["citations"]] == [1]
    assert (message.prompt_tokens, message.completion_tokens) == (
        done["usage"]["prompt_tokens"],
        done["usage"]["completion_tokens"],
    )
    assert done["usage"]["prompt_tokens"] > 0


async def test_nothing_relevant_streams_the_fixed_answer_without_the_llm(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    result = await stream(headers, collection_id, "Tell me about rockets")

    assert names(result) == ["meta", "token", "done"]
    assert result.events[1][1] == {"text": NOT_FOUND_ANSWER}
    done = result.events[2][1]
    assert (done["answer"], done["grounded"], done["citations"]) == (NOT_FOUND_ANSWER, False, [])
    assert llm.calls == []
    message = await saved(db, done["message_id"])
    assert (message.status, message.content) == (MessageStatus.complete, NOT_FOUND_ANSWER)


async def test_disconnect_mid_stream_cancels_the_llm_and_saves_cancelled(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    llm.pause_after = 2  # stop producing after "Cats purr "

    def two_tokens_received(result):
        return names(result).count("token") == 2

    result = await stream(
        headers, collection_id, "What do cats do?", disconnect_when=two_tokens_received
    )

    assert "done" not in names(result)
    assert llm.streams_cancelled == 1 and llm.streams_completed == 0  # upstream cancelled
    message = await saved(db, result.events[0][1]["message_id"])
    assert message.status == MessageStatus.cancelled
    assert message.content == "Cats purr "  # the partial text
    assert message.citations == []
    # The conversation isn't stuck.
    llm.pause_after = None
    again = await stream(headers, collection_id, "Again?", result.events[0][1]["conversation_id"])
    assert names(again)[-1] == "done"


async def test_llm_failure_mid_stream_sends_error_and_saves_failed(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    llm.fail_after = 2
    llm.fail_with = RetryableLLMError("The answer service timed out")

    result = await stream(headers, collection_id, "What do cats do?")

    assert names(result) == ["meta", "token", "token", "error"]
    assert result.events[-1][1] == {
        "message": "The answer service timed out",
        "status": 503,
        "retry_after": None,
    }
    assert "Traceback" not in result.text
    meta = result.events[0][1]
    message = await saved(db, meta["message_id"])
    assert (message.status, message.error_message) == (
        MessageStatus.failed,
        "The answer service timed out",
    )
    detail = (await client.get(f"/conversations/{meta['conversation_id']}", headers=headers)).json()
    assert detail["messages"][0]["content"] == "What do cats do?"  # the question is kept
    llm.fail_after = None
    again = await stream(headers, collection_id, "Again?", meta["conversation_id"])
    assert names(again)[-1] == "done"  # the lock was released


async def test_rate_limit_before_any_text_is_an_error_event_with_retry_hint(client, embedder, llm):
    headers, collection_id = await setup(client)
    llm.failures = [LLMBusyError(retry_after=42)]
    result = await stream(headers, collection_id, "What do cats do?")
    assert names(result) == ["meta", "error"]
    assert result.events[-1][1]["status"] == 429
    assert result.events[-1][1]["retry_after"] == 42


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (EmbeddingBusyError(retry_after=12), 429),
        (RetryableEmbeddingError("Embedding service is unavailable; try again later"), 503),
    ],
)
async def test_retrieval_failure_is_an_error_event(client, db, embedder, llm, error, status):
    headers, collection_id = await setup(client)

    async def fail(_):
        raise error

    embedder.embed_query = fail
    result = await stream(headers, collection_id, "What do cats do?")
    assert names(result) == ["meta", "error"]
    assert result.events[-1][1]["status"] == status
    assert result.events[-1][1]["message"] == str(error)
    message = await saved(db, result.events[0][1]["message_id"])
    assert message.status == MessageStatus.failed
    assert llm.calls == []


async def test_no_database_connection_is_held_while_streaming(client, embedder, llm):
    headers, collection_id = await setup(client)
    llm.pause_after = 1
    task = asyncio.create_task(stream(headers, collection_id, "What do cats do?"))
    await asyncio.wait_for(llm.paused.wait(), 10)  # mid-stream: a token sent, more to come

    observer = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with observer.connect() as conn:
        others = await conn.scalar(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            )
        )
    await observer.dispose()
    assert others == 0  # no connection open from the app while it streams

    llm.resume.set()
    result = await task
    assert names(result)[-1] == "done"


async def test_follow_up_is_rewritten_before_retrieval(client, embedder, llm):
    headers, collection_id = await setup(client)
    first = await stream(headers, collection_id, "Tell me about cats")
    embedder.embedded_queries.clear()

    second = await stream(
        headers, collection_id, "And what do they do?", first.events[0][1]["conversation_id"]
    )
    assert names(second)[-1] == "done"
    assert embedder.embedded_queries == ["What do cats do?"]  # the rewritten question, once
    assert len([c for c in llm.calls if c[0] == REWRITE_SYSTEM]) == 1


async def test_first_question_never_triggers_a_rewrite(client, embedder, llm):
    headers, collection_id = await setup(client)
    await stream(headers, collection_id, "Tell me about cats")
    assert [c for c in llm.calls if c[0] == REWRITE_SYSTEM] == []
    assert embedder.embedded_queries == ["Tell me about cats"]


async def test_concurrent_request_on_the_same_conversation_is_409(client, db, embedder, llm):
    headers, collection_id = await setup(client)
    first = await stream(headers, collection_id, "What do cats do?")
    conversation_id = first.events[0][1]["conversation_id"]
    collection = await db.get(Collection, uuid.UUID(collection_id))
    in_flight = await conversation_service.start_turn(
        db, collection, "Still answering?", uuid.UUID(conversation_id), 6
    )

    refused = await stream(headers, collection_id, "Another?", conversation_id)
    assert refused.status == 409
    assert "already being generated" in refused.text

    await conversation_service.finish_turn(db, in_flight, MessageStatus.complete, "ok")
    assert (await stream(headers, collection_id, "Another?", conversation_id)).status == 200


async def test_other_users_conversation_and_collection_are_404(client, embedder, llm):
    alice, alice_collection = await setup(client, email="alice@example.com")
    first = await stream(alice, alice_collection, "What do cats do?")
    conversation_id = first.events[0][1]["conversation_id"]
    bob, bob_collection = await setup(client, docs={}, email="bob@example.com")

    into_alices_collection = await stream(bob, alice_collection, "Hi")
    into_alices_conversation = await stream(bob, bob_collection, "Hi", conversation_id)
    assert into_alices_collection.status == into_alices_conversation.status == 404
    assert len(embedder.embedded_queries) == 1  # only Alice's question was ever embedded


async def test_invalid_question_is_rejected_before_anything(client, embedder, llm):
    headers, collection_id = await setup(client)
    result = await stream(headers, collection_id, "   ")
    assert result.status == 422
    assert embedder.embedded_queries == [] and llm.calls == []


async def test_heartbeats_are_sent_while_waiting(client, embedder, llm, monkeypatch):
    monkeypatch.setattr(get_settings(), "STREAM_HEARTBEAT_SECONDS", 0.05)
    headers, collection_id = await setup(client)
    llm.chunk_delay = 0.2  # a slow model: real short sleeps between pieces

    result = await stream(headers, collection_id, "What do cats do?")

    assert result.heartbeats >= 3
    assert names(result)[-1] == "done"  # heartbeats are comments, not events


async def test_logs_hold_ids_and_counts_only(client, embedder, llm, caplog):
    caplog.set_level(logging.INFO, logger="app")
    headers, collection_id = await setup(client)
    question = "What exactly do cats do on Tuesdays?"
    result = await stream(headers, collection_id, question)
    first_id = result.events[0][1]["conversation_id"]
    await stream(headers, collection_id, "And on Wednesdays?", first_id)

    assert any(r.getMessage().startswith("answer: ") for r in caplog.records)
    assert any(r.getMessage().startswith("rewrite: ") for r in caplog.records)
    for leak in (question, "And on Wednesdays?", "Cats purr", "all about cats"):
        assert leak not in caplog.text
