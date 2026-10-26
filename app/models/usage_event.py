import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class UsageKind(enum.StrEnum):
    embed_document = "embed_document"
    embed_query = "embed_query"
    llm_answer = "llm_answer"
    llm_rewrite = "llm_rewrite"


class UsageEvent(Base):
    """One provider request that succeeded: who, what kind, how many tokens,
    which model, when. The historical record behind GET /usage.

    Deliberately NO content: no question, answer, prompt or document text,
    only counts and ids. document_id / conversation_id say what it was for,
    without foreign keys: deleting a document must not erase (or be blocked
    by) the record of what embedding it cost.
    """

    __tablename__ = "usage_events"
    # The hot query is "this user's events since <time>" (GET /usage).
    __table_args__ = (Index("ix_usage_events_user_id_created_at", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[UsageKind] = mapped_column(Enum(UsageKind, name="usage_kind"))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    document_id: Mapped[uuid.UUID | None]
    conversation_id: Mapped[uuid.UUID | None]
