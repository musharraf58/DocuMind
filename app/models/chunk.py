import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Chunk(Base):
    """A piece of a document's extracted text, small enough to embed and retrieve."""

    __tablename__ = "chunks"
    __table_args__ = (
        # Also serves as the index for "chunks of document X" lookups, since
        # document_id is its first column; a separate document_id index would
        # be redundant.
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # All three cascade, so deleting a document, collection or user removes
    # its chunks in the database without the app loading them.
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Indexed so a collection delete can find chunks to cascade without a full
    # table scan; Phase 4 search filters by collection too.
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, default=None)
    section_title: Mapped[str | None] = mapped_column(Text, default=None)
    token_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
