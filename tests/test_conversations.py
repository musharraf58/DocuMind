"""The conversation service (lock, history, titles) and the history endpoints."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from app.models import Collection, Message, MessageStatus
from app.services import conversations as service
from tests.helpers import auth_headers, create_collection, error_body


async def owner(client, db, email="alice@example.com"):
    headers = await auth_headers(client, email)
    collection = await db.get(Collection, uuid.UUID(await create_collection(client, headers)))
    return headers, collection


async def answered(db, collection, question, answer="An answer [1].", conversation_id=None):
    """A complete turn, as the ask endpoints will produce it."""
    turn = await service.start_turn(db, collection, question, conversation_id, history_turns=6)
    await service.finish_turn(db, turn, MessageStatus.complete, answer, grounded=True)
    return turn


# --- Titles ---


def test_title_is_the_first_question_on_one_line():
    assert service.make_title("  What   is\nthe refund policy? ") == "What is the refund policy?"


def test_long_title_is_cut_at_a_word_boundary():
    title = service.make_title("Tell me everything about " + "refunds and returns " * 10)
    assert len(title) <= service.TITLE_CHARS + 1
    assert title.endswith("…") and not title[:-1].endswith(" ")


# --- Turns and the lock ---


async def test_first_question_creates_a_conversation_and_saves_the_question(client, db):
    _, collection = await owner(client, db)
    turn = await service.start_turn(db, collection, "What is the policy?", None, 6)

    assert not turn.is_follow_up
    messages = (
        await db.scalars(select(Message).where(Message.conversation_id == turn.conversation_id))
    ).all()
    by_id = {m.id: m for m in messages}
    assert by_id[turn.user_message_id].content == "What is the policy?"
    assert by_id[turn.user_message_id].status == MessageStatus.complete
    assert by_id[turn.assistant_message_id].status == MessageStatus.in_progress


async def test_second_concurrent_answer_is_refused_until_the_first_finishes(client, db):
    _, collection = await owner(client, db)
    first = await service.start_turn(db, collection, "First?", None, 6)

    with pytest.raises(service.ConversationBusyError):
        await service.start_turn(db, collection, "Second?", first.conversation_id, 6)

    await service.finish_turn(db, first, MessageStatus.failed, "", error_message="boom")
    # A refused turn rolls the session back; a new request would load afresh.
    collection = await db.get(Collection, first.collection_id)
    second = await service.start_turn(db, collection, "Second?", first.conversation_id, 6)
    assert second.conversation_id == first.conversation_id


@pytest.mark.parametrize(
    "status", [MessageStatus.complete, MessageStatus.failed, MessageStatus.cancelled]
)
async def test_every_way_of_finishing_releases_the_lock(client, db, status):
    _, collection = await owner(client, db)
    turn = await service.start_turn(db, collection, "Q?", None, 6)
    await service.finish_turn(db, turn, status, "partial")
    await service.start_turn(db, collection, "Next?", turn.conversation_id, 6)


async def test_refused_question_is_not_saved(client, db):
    _, collection = await owner(client, db)
    first = await service.start_turn(db, collection, "First?", None, 6)
    with pytest.raises(service.ConversationBusyError):
        await service.start_turn(db, collection, "Second?", first.conversation_id, 6)
    count = await db.scalar(
        select(func.count()).select_from(Message).where(Message.content == "Second?")
    )
    assert count == 0


async def test_abandoned_answer_is_freed_after_a_while(client, db):
    _, collection = await owner(client, db)
    stuck = await service.start_turn(db, collection, "First?", None, 6)
    await db.execute(
        update(Message)
        .where(Message.id == stuck.assistant_message_id)
        .values(created_at=datetime.now(UTC) - timedelta(minutes=6))
    )
    await db.commit()

    await service.start_turn(db, collection, "Again?", stuck.conversation_id, 6)
    abandoned = await db.get(Message, stuck.assistant_message_id, populate_existing=True)
    assert abandoned.status == MessageStatus.failed
    assert abandoned.error_message == service.INTERRUPTED


async def test_finish_happens_only_once(client, db):
    _, collection = await owner(client, db)
    turn = await service.start_turn(db, collection, "Q?", None, 6)
    assert await service.finish_turn(db, turn, MessageStatus.complete, "A") is True
    assert await service.finish_turn(db, turn, MessageStatus.failed, "") is False
    message = await db.get(Message, turn.assistant_message_id, populate_existing=True)
    assert (message.status, message.content) == (MessageStatus.complete, "A")


async def test_conversation_of_another_collection_or_user_is_404(client, db):
    _, collection = await owner(client, db)
    _, other_collection = await owner(client, db, "same-user-other@example.com")
    turn = await answered(db, collection, "Q?")

    with pytest.raises(HTTPException) as info:  # another user's collection
        await service.start_turn(db, other_collection, "Q?", turn.conversation_id, 6)
    assert info.value.status_code == 404

    headers, _ = await owner(client, db, "bob@example.com")
    second_collection_of_alice = await db.get(
        Collection, uuid.UUID(await create_collection(client, headers, "Bob docs"))
    )
    with pytest.raises(HTTPException) as info:
        await service.start_turn(db, second_collection_of_alice, "Q?", turn.conversation_id, 6)
    assert info.value.status_code == 404


async def test_history_is_completed_turns_oldest_first(client, db):
    _, collection = await owner(client, db)
    first = await answered(db, collection, "Q1", "A1")
    failed = await service.start_turn(db, collection, "Q2", first.conversation_id, 6)
    await service.finish_turn(db, failed, MessageStatus.failed, "")
    await answered(db, collection, "Q3", "A3", first.conversation_id)

    history = await service.load_history(db, first.conversation_id, 6)
    assert history == [service.HistoryTurn("Q1", "A1"), service.HistoryTurn("Q3", "A3")]


async def test_history_keeps_the_most_recent_turns(client, db):
    _, collection = await owner(client, db)
    turn = await answered(db, collection, "Q0", "A0")
    for n in range(1, 5):
        await answered(db, collection, f"Q{n}", f"A{n}", turn.conversation_id)
    history = await service.load_history(db, turn.conversation_id, 2)
    assert [h.question for h in history] == ["Q3", "Q4"]


async def test_follow_up_turn_carries_the_history(client, db):
    _, collection = await owner(client, db)
    first = await answered(db, collection, "Q1", "A1")
    follow_up = await service.start_turn(db, collection, "And then?", first.conversation_id, 6)
    assert follow_up.is_follow_up
    assert follow_up.history == [service.HistoryTurn("Q1", "A1")]


# --- Endpoints ---


async def test_list_conversations_newest_first_with_collection_filter(client, db):
    headers, collection = await owner(client, db)
    second_collection = await db.get(
        Collection, uuid.UUID(await create_collection(client, headers, "Other"))
    )
    older = await answered(db, collection, "Older question")
    other = await answered(db, second_collection, "In the other collection")
    newer = await answered(db, collection, "Newer question")

    everything = (await client.get("/conversations", headers=headers)).json()
    assert [c["id"] for c in everything] == [
        str(newer.conversation_id),
        str(other.conversation_id),
        str(older.conversation_id),
    ]
    filtered = await client.get(
        "/conversations", params={"collection_id": str(collection.id)}, headers=headers
    )
    assert [c["title"] for c in filtered.json()] == ["Newer question", "Older question"]
    page = await client.get("/conversations", params={"limit": 1, "offset": 1}, headers=headers)
    assert [c["id"] for c in page.json()] == [str(other.conversation_id)]


async def test_get_conversation_shows_messages_oldest_first(client, db):
    headers, collection = await owner(client, db)
    turn = await answered(db, collection, "Q1", "A1")
    await answered(db, collection, "Q2", "A2", turn.conversation_id)

    body = (await client.get(f"/conversations/{turn.conversation_id}", headers=headers)).json()
    assert body["title"] == "Q1"
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "Q1"),
        ("assistant", "A1"),
        ("user", "Q2"),
        ("assistant", "A2"),
    ]
    assert body["messages"][1]["status"] == "complete"
    page = await client.get(
        f"/conversations/{turn.conversation_id}",
        params={"limit": 2, "offset": 2},
        headers=headers,
    )
    assert [m["content"] for m in page.json()["messages"]] == ["Q2", "A2"]


async def test_rename_conversation(client, db):
    headers, collection = await owner(client, db)
    turn = await answered(db, collection, "Q1")
    url = f"/conversations/{turn.conversation_id}"

    response = await client.patch(url, json={"title": "  Refund   questions "}, headers=headers)
    assert response.status_code == 200
    assert response.json()["title"] == "Refund questions"
    assert (await client.patch(url, json={"title": "   "}, headers=headers)).status_code == 422
    assert (await client.patch(url, json={"title": "x" * 201}, headers=headers)).status_code == 422


async def test_delete_conversation_removes_its_messages(client, db):
    headers, collection = await owner(client, db)
    turn = await answered(db, collection, "Q1")
    url = f"/conversations/{turn.conversation_id}"

    assert (await client.delete(url, headers=headers)).status_code == 204
    assert (await client.get(url, headers=headers)).status_code == 404
    remaining = await db.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == turn.conversation_id)
    )
    assert remaining == 0


async def test_another_user_gets_404_on_every_conversation_endpoint(client, db):
    _, collection = await owner(client, db)
    turn = await answered(db, collection, "Alice's question")
    bob = await auth_headers(client, "bob@example.com")
    url = f"/conversations/{turn.conversation_id}"

    responses = [
        await client.get(url, headers=bob),
        await client.patch(url, json={"title": "mine now"}, headers=bob),
        await client.delete(url, headers=bob),
        await client.get(f"/conversations/{uuid.uuid4()}", headers=bob),
    ]
    assert [r.status_code for r in responses] == [404, 404, 404, 404]
    # Indistinguishable from "doesn't exist" (apart from the per-request id).
    assert len({str(error_body(r)) for r in responses}) == 1
    assert (await client.get("/conversations", headers=bob)).json() == []
    # Nothing changed for Alice.
    still = await service.get_owned_conversation(db, turn.conversation_id, collection.owner_id)
    assert still.title == "Alice's question"


async def test_conversation_endpoints_require_authentication(client):
    assert (await client.get("/conversations")).status_code == 401
    assert (await client.get(f"/conversations/{uuid.uuid4()}")).status_code == 401
