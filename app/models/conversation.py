import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

TITLE_MAX_LENGTH = 200


class MessageRole(enum.StrEnum):
    user = "user"
    assistant = "assistant"


class MessageStatus(enum.StrEnum):
    in_progress = "in_progress"  # an answer being generated right now
    complete = "complete"
    failed = "failed"
    cancelled = "cancelled"  # the client went away mid-answer


class Conversation(Base):
    """A thread of questions and answers about one collection."""

    __tablename__ = "conversations"
    __table_args__ = (
        # "My conversations, most recently active first".
        Index("ix_conversations_owner_updated", "owner_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # Deleting the collection deletes its conversations (and, through them, messages).
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(TITLE_MAX_LENGTH))
    # clock_timestamp(), not now(): now() is the start of the transaction, so
    # rows written in one transaction would all get the same time.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        # A conversation's messages in order.
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        # At most ONE answer being generated per conversation: a second
        # concurrent request can't insert its in-progress message (409).
        Index(
            "uq_messages_one_in_progress",
            "conversation_id",
            unique=True,
            postgresql_where=text("status = 'in_progress'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"))
    content: Mapped[str] = mapped_column(Text, default="", server_default="")
    status: Mapped[MessageStatus] = mapped_column(Enum(MessageStatus, name="message_status"))
    grounded: Mapped[bool | None] = mapped_column(Boolean, default=None)
    # Snapshots of what was cited, copied at answer time: document_id,
    # document_name, page_number, section_title, snippet, distance, number,
    # and chunk_id WITHOUT a foreign key. Re-processing a document replaces
    # its chunks (new ids), and old answers must still show their sources.
    citations: Mapped[list[dict]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Why an answer failed: a short, safe message (never provider text).
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )
