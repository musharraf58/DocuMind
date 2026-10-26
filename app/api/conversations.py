import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from app.api.demo import not_demo
from app.api.deps import CurrentUser, DbSession
from app.api.limits import limit_default
from app.schemas.conversation import (
    ConversationDetail,
    ConversationRead,
    ConversationRename,
    MessageRead,
)
from app.services import conversations as conversation_service

router = APIRouter(tags=["conversations"], dependencies=[limit_default])


@router.get("/conversations", response_model=list[ConversationRead])
async def list_conversations(
    db: DbSession,
    user: CurrentUser,
    collection_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Your conversations, most recently active first."""
    return await conversation_service.list_conversations(db, user.id, collection_id, limit, offset)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """A conversation with its messages, oldest first."""
    conversation = await conversation_service.get_owned_conversation(db, conversation_id, user.id)
    messages = await conversation_service.list_messages(db, conversation, limit, offset)
    return ConversationDetail(
        **ConversationRead.model_validate(conversation).model_dump(),
        messages=[MessageRead.model_validate(m) for m in messages],
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationRead,
    dependencies=[not_demo],
)
async def rename_conversation(
    conversation_id: uuid.UUID, body: ConversationRename, db: DbSession, user: CurrentUser
):
    conversation = await conversation_service.get_owned_conversation(db, conversation_id, user.id)
    return await conversation_service.rename(db, conversation, body.title)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[not_demo],
)
async def delete_conversation(conversation_id: uuid.UUID, db: DbSession, user: CurrentUser):
    conversation = await conversation_service.get_owned_conversation(db, conversation_id, user.id)
    await conversation_service.delete(db, conversation)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
