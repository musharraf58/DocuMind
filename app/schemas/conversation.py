import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import MessageRole, MessageStatus
from app.models.conversation import TITLE_MAX_LENGTH


class CitationSnapshot(BaseModel):
    """A source as it was when the answer was given. chunk_id may no longer
    exist (re-processing a document replaces its chunks); the rest is a copy."""

    number: int
    document_id: uuid.UUID
    document_name: str
    page_number: int | None
    section_title: str | None
    snippet: str
    distance: float
    chunk_id: uuid.UUID | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    status: MessageStatus
    grounded: bool | None
    citations: list[CitationSnapshot]
    prompt_tokens: int
    completion_tokens: int
    error_message: str | None
    created_at: datetime


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]  # oldest first, paginated


class ConversationRename(BaseModel):
    title: str = Field(max_length=TITLE_MAX_LENGTH)

    @field_validator("title")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Title must not be empty")
        return value
