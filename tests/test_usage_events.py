"""Usage events: one row per successful provider request, for the right user,
with counts and ids only (never any question, answer or document text)."""

# The embedder/llm fixtures are imported from test_ask_api and then requested
# by name, which ruff reads as redefinitions.
# ruff: noqa: F811

import logging
import uuid

from sqlalchemy import select

from app.models import Conversation, UsageEvent, UsageKind
from app.services import usage
from tests.helpers import auth_headers, create_collection, upload
from tests.test_ask_api import about, ask, embedder, llm, setup  # noqa: F401 (fixtures)
from tests.test_ask_stream import stream

SECRET_QUESTION = "What do cats do on secret project nightingale?"
SECRET_ANSWER = "Cats purr when content [1]."


async def events(db) -> list[UsageEvent]:
    return list((await db.scalars(select(UsageEvent).order_by(UsageEvent.id))).all())


def kinds(rows) -> list[UsageKind]:
    return [row.kind for row in rows]


async def user_id(client, headers) -> uuid.UUID:
    return uuid.UUID((await client.get("/auth/me", headers=headers)).json()["id"])


async def test_ingestion_records_embed_document(client, db, embedder, llm):
    headers, _ = await setup(client, {"cats.txt": about("cats")})
    rows = await events(db)
    assert kinds(rows) == [UsageKind.embed_document]
    row = rows[0]
    assert row.user_id == await user_id(client, headers)
    assert row.document_id is not None
    assert row.conversation_id is None
    assert row.tokens_in > 0
    assert row.model == embedder.model


async def test_ask_records_query_and_answer(client, db, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    response = await ask(client, headers, collection_id, SECRET_QUESTION)
    assert response.status_code == 200

    rows = [r for r in await events(db) if r.kind != UsageKind.embed_document]
    assert kinds(rows) == [UsageKind.embed_query, UsageKind.llm_answer]
    conversation_id = uuid.UUID(response.json()["conversation_id"])
    uid = await user_id(client, headers)
    for row in rows:
        assert row.user_id == uid
        assert row.conversation_id == conversation_id
    answer = rows[1]
    assert answer.model == llm.model
    assert (answer.tokens_in, answer.tokens_out) == (
        response.json()["usage"]["prompt_tokens"],
        response.json()["usage"]["completion_tokens"],
    )


async def test_follow_up_records_the_rewrite(client, db, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    first = await ask(client, headers, collection_id, "What do cats do?")
    await client.post(
        f"/collections/{collection_id}/ask",
        json={"question": "And then?", "conversation_id": first.json()["conversation_id"]},
        headers=headers,
    )
    asked = [r.kind for r in await events(db) if r.kind != UsageKind.embed_document]
    assert asked == [
        UsageKind.embed_query,
        UsageKind.llm_answer,
        UsageKind.llm_rewrite,
        UsageKind.embed_query,
        UsageKind.llm_answer,
    ]


async def test_not_found_records_no_llm_call(client, db, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    await ask(client, headers, collection_id, "How do rockets work?")
    asked = [r.kind for r in await events(db) if r.kind != UsageKind.embed_document]
    assert asked == [UsageKind.embed_query]


async def test_stream_records_the_answer(client, db, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    result = await stream(headers, collection_id, "What do cats do?")
    assert result.status == 200
    rows = [r for r in await events(db) if r.kind != UsageKind.embed_document]
    assert kinds(rows) == [UsageKind.embed_query, UsageKind.llm_answer]
    conversation = await db.scalar(select(Conversation))
    assert {r.conversation_id for r in rows} == {conversation.id}
    assert rows[1].tokens_out > 0


async def test_search_records_embed_query(client, db, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    response = await client.post(
        f"/collections/{collection_id}/search", json={"query": "cats"}, headers=headers
    )
    assert response.status_code == 200
    rows = [r for r in await events(db) if r.kind != UsageKind.embed_document]
    assert kinds(rows) == [UsageKind.embed_query]
    assert rows[0].user_id == await user_id(client, headers)


async def test_no_content_is_ever_stored(client, db, embedder, llm):
    secret_doc = "Nightingale launch codes are kept in the blue binder. " * 10
    headers, collection_id = await setup(client, {"cats.txt": about("cats") + secret_doc})
    await ask(client, headers, collection_id, SECRET_QUESTION)
    for row in await events(db):
        stored = " ".join(str(getattr(row, column.name)) for column in UsageEvent.__table__.columns)
        for secret in ["nightingale", "binder", "purr", "cats"]:
            assert secret not in stored.lower()
    # And the table has no column that could hold text beyond the model name.
    text_columns = {c.name for c in UsageEvent.__table__.columns if c.type.python_type is str} - {
        "kind"
    }
    assert text_columns == {"model"}


async def test_each_user_gets_their_own_events(client, db, embedder, llm):
    alice, alice_collection = await setup(client, {"cats.txt": about("cats")}, "a@example.com")
    bob, bob_collection = await setup(client, {"db.txt": about("databases")}, "b@example.com")
    await ask(client, alice, alice_collection, "What do cats do?")
    alice_id, bob_id = await user_id(client, alice), await user_id(client, bob)
    rows = await events(db)
    assert {r.user_id for r in rows if r.kind == UsageKind.llm_answer} == {alice_id}
    assert [r.kind for r in rows if r.user_id == bob_id] == [UsageKind.embed_document]


async def test_a_failed_insert_never_breaks_the_request(
    client, db, embedder, llm, monkeypatch, caplog
):
    async def broken_save(self, scope, event):
        raise RuntimeError("database is gone")

    monkeypatch.setattr(usage.UsageRecorder, "save", broken_save)
    headers = await auth_headers(client, "resilient@example.com")
    collection_id = await create_collection(client, headers)
    with caplog.at_level(logging.WARNING, logger="app.services.usage"):
        response = await client.post(
            f"/collections/{collection_id}/search", json={"query": "cats"}, headers=headers
        )
    assert response.status_code == 200
    assert "Could not record usage" in caplog.text


async def test_nothing_is_recorded_outside_a_scope(db):
    await usage.record(UsageKind.embed_query, "some-model", 10)
    assert await events(db) == []


async def test_scope_ends_with_the_block(db, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.models import User

    user = User(email="scoped@example.com", hashed_password="x")
    db.add(user)
    await db.commit()
    recorder = usage.UsageRecorder(async_sessionmaker(test_engine, expire_on_commit=False), None)
    with recorder.scope(user.id):
        await usage.record(UsageKind.embed_query, "m", 5)
    await usage.record(UsageKind.embed_query, "m", 7)
    assert [r.tokens_in for r in await events(db)] == [5]


async def test_uploads_alone_record_nothing(client, db):
    """Uploading stores a file; embedding happens in the worker."""
    headers = await auth_headers(client, "up@example.com")
    collection_id = await create_collection(client, headers)
    await upload(client, headers, collection_id, "a.txt", b"some text here, long enough")
    assert await events(db) == []
