"""Database-level guarantees of the conversation tables."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.models import (
    Collection,
    Conversation,
    Message,
    MessageRole,
    MessageStatus,
    User,
)


async def conversation(db) -> Conversation:
    user = User(email=f"{uuid.uuid4().hex[:8]}@example.com", hashed_password="x")
    db.add(user)
    await db.flush()
    collection = Collection(owner_id=user.id, name="Docs")
    db.add(collection)
    await db.flush()
    conv = Conversation(owner_id=user.id, collection_id=collection.id, title="t")
    db.add(conv)
    await db.commit()
    return conv


def message(conv: Conversation, status: MessageStatus, role=MessageRole.assistant) -> Message:
    return Message(conversation_id=conv.id, owner_id=conv.owner_id, role=role, status=status)


async def test_only_one_answer_in_progress_per_conversation(db):
    conv = await conversation(db)
    conv_id = conv.id
    db.add(message(conv, MessageStatus.in_progress))
    await db.commit()

    db.add(message(conv, MessageStatus.in_progress))
    with pytest.raises(IntegrityError, match="uq_messages_one_in_progress"):
        await db.commit()
    await db.rollback()
    conv = await db.get(Conversation, conv_id)  # a rollback forgets loaded objects

    # Finished messages don't count: any number may exist.
    for status in (MessageStatus.complete, MessageStatus.failed, MessageStatus.cancelled):
        db.add(message(conv, status))
    await db.commit()


async def test_other_conversations_are_not_blocked(db):
    first, second = await conversation(db), await conversation(db)
    db.add_all(
        [message(first, MessageStatus.in_progress), message(second, MessageStatus.in_progress)]
    )
    await db.commit()


async def test_messages_in_one_transaction_get_distinct_ordered_times(db):
    conv = await conversation(db)
    question = message(conv, MessageStatus.complete, MessageRole.user)
    answer = message(conv, MessageStatus.in_progress)
    db.add(question)
    await db.flush()
    db.add(answer)
    await db.commit()
    await db.refresh(question)
    await db.refresh(answer)
    assert question.created_at < answer.created_at


async def test_deleting_a_collection_deletes_its_conversations_and_messages(db):
    conv = await conversation(db)
    db.add(message(conv, MessageStatus.complete))
    await db.commit()

    await db.delete(await db.get(Collection, conv.collection_id))
    await db.commit()
    assert await db.scalar(select(func.count()).select_from(Conversation)) == 0
    assert await db.scalar(select(func.count()).select_from(Message)) == 0


async def test_citations_default_to_an_empty_list(db):
    conv = await conversation(db)
    msg = message(conv, MessageStatus.complete)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    assert msg.citations == []
