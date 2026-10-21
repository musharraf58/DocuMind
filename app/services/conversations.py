"""Conversations: threads of questions and answers about one collection.

The one-answer-at-a-time rule lives in the database: an assistant message
is created as "in_progress" before generating, and a partial unique index
allows only one such message per conversation. A second concurrent request
fails to insert its own and gets 409. Finishing the answer (complete,
failed or cancelled) releases it. If a process dies mid-answer, the next
request treats an in-progress message older than STALE_IN_PROGRESS as
abandoned.
"""

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import clock
from app.models import Collection, Conversation, Message, MessageRole, MessageStatus

logger = logging.getLogger(__name__)

STALE_IN_PROGRESS = timedelta(minutes=5)
TITLE_CHARS = 60
INTERRUPTED = "The answer was interrupted"
_WHITESPACE = re.compile(r"\s+")


class ConversationBusyError(Exception):
    """Another answer is already being generated in this conversation."""


@dataclass(frozen=True)
class HistoryTurn:
    question: str
    answer: str


@dataclass(frozen=True)
class Turn:
    """A question accepted into a conversation, its answer not yet generated."""

    conversation_id: uuid.UUID
    collection_id: uuid.UUID
    owner_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    history: list[HistoryTurn]  # earlier completed turns, oldest first

    @property
    def is_follow_up(self) -> bool:
        return bool(self.history)


def make_title(question: str) -> str:
    """The first question, on one line, cut at a word boundary (no LLM call)."""
    title = _WHITESPACE.sub(" ", question).strip()
    if len(title) <= TITLE_CHARS:
        return title
    return title[:TITLE_CHARS].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"


async def get_owned_conversation(
    db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation:
    """The conversation if it belongs to user_id, else 404. Filters by id AND
    owner in one query: "not yours" and "doesn't exist" look the same."""
    conversation = await db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == user_id
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


async def start_turn(
    db: AsyncSession,
    collection: Collection,
    question: str,
    conversation_id: uuid.UUID | None,
    history_turns: int,
) -> Turn:
    """Accept a question: find or create the conversation, save the question,
    and create the in-progress answer (the lock), all in one transaction.

    Raises 404 if the conversation isn't the caller's or belongs to another
    collection, and ConversationBusyError if an answer is already in progress.
    The question is saved BEFORE anything is generated, so a failed answer
    never loses it. On ConversationBusyError the session has been rolled back
    (so objects loaded in it are expired).
    """
    owner_id = collection.owner_id
    if conversation_id is not None:
        conversation = await get_owned_conversation(db, conversation_id, owner_id)
        if conversation.collection_id != collection.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = await load_history(db, conversation.id, history_turns)
        # An answer still "in progress" long after it started was abandoned
        # (the process died): free the conversation.
        await db.execute(
            update(Message)
            .where(
                Message.conversation_id == conversation.id,
                Message.status == MessageStatus.in_progress,
                Message.created_at < clock.utcnow() - STALE_IN_PROGRESS,
            )
            .values(status=MessageStatus.failed, error_message=INTERRUPTED)
        )
    else:
        conversation = Conversation(
            owner_id=owner_id, collection_id=collection.id, title=make_title(question)
        )
        db.add(conversation)
        await db.flush()
        history = []

    user_message = Message(
        conversation_id=conversation.id,
        owner_id=owner_id,
        role=MessageRole.user,
        content=question,
        status=MessageStatus.complete,
    )
    db.add(user_message)
    await db.flush()  # before the answer, so it is earlier in the thread
    answer = Message(
        conversation_id=conversation.id,
        owner_id=owner_id,
        role=MessageRole.assistant,
        status=MessageStatus.in_progress,
    )
    db.add(answer)
    conversation.updated_at = clock.utcnow()
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        if "uq_messages_one_in_progress" in str(exc.orig):
            raise ConversationBusyError from exc
        raise
    return Turn(
        conversation_id=conversation.id,
        collection_id=collection.id,
        owner_id=owner_id,
        user_message_id=user_message.id,
        assistant_message_id=answer.id,
        history=history,
    )


async def load_history(
    db: AsyncSession, conversation_id: uuid.UUID, max_turns: int
) -> list[HistoryTurn]:
    """The last `max_turns` completed turns (question + complete answer),
    oldest first. Questions whose answer failed or was cancelled are left out:
    there's nothing reliable to build on."""
    if max_turns <= 0:
        return []
    rows = (
        await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(max_turns * 2 + 4)  # a few spare for incomplete turns
        )
    ).all()
    turns: list[HistoryTurn] = []
    pending_answer: Message | None = None
    for message in rows:  # newest first
        if message.role == MessageRole.assistant:
            pending_answer = message
        elif pending_answer is not None:
            if pending_answer.status == MessageStatus.complete:
                turns.append(HistoryTurn(message.content, pending_answer.content))
            pending_answer = None
    return list(reversed(turns[:max_turns]))


async def finish_turn(
    db: AsyncSession,
    turn: Turn,
    status: MessageStatus,
    content: str,
    grounded: bool | None = None,
    citations: Sequence[dict] = (),
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error_message: str | None = None,
) -> bool:
    """Save the outcome of the answer and so release the conversation. Only an
    answer still in progress is changed (False if it was already finished,
    e.g. marked abandoned in the meantime)."""
    now = clock.utcnow()
    finished = await db.scalar(
        update(Message)
        .where(Message.id == turn.assistant_message_id, Message.status == MessageStatus.in_progress)
        .values(
            status=status,
            content=content,
            grounded=grounded,
            citations=list(citations),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_message=error_message,
        )
        .returning(Message.id)
        .execution_options(synchronize_session=False)
    )
    await db.execute(
        update(Conversation)
        .where(Conversation.id == turn.conversation_id)
        .values(updated_at=now)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return finished is not None


async def list_conversations(
    db: AsyncSession,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID | None,
    limit: int,
    offset: int,
) -> Sequence[Conversation]:
    query = select(Conversation).where(Conversation.owner_id == owner_id)
    if collection_id is not None:
        query = query.where(Conversation.collection_id == collection_id)
    result = await db.scalars(
        query.order_by(Conversation.updated_at.desc(), Conversation.id).limit(limit).offset(offset)
    )
    return result.all()


async def list_messages(
    db: AsyncSession, conversation: Conversation, limit: int, offset: int
) -> Sequence[Message]:
    result = await db.scalars(
        select(Message)
        .where(
            Message.conversation_id == conversation.id, Message.owner_id == conversation.owner_id
        )
        .order_by(Message.created_at, Message.id)
        .limit(limit)
        .offset(offset)
    )
    return result.all()


async def rename(db: AsyncSession, conversation: Conversation, title: str) -> Conversation:
    conversation.title = title
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def delete(db: AsyncSession, conversation: Conversation) -> None:
    await db.delete(conversation)  # messages go with it (ON DELETE CASCADE)
    await db.commit()
