import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class DocumentStatus(enum.StrEnum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # The same bytes can't be uploaded twice into one collection. Another
        # collection (or another user) may hold the same file.
        UniqueConstraint("collection_id", "content_hash", name="uq_documents_collection_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    # Also stored directly (not just via the collection) so "my documents" queries
    # and ownership checks need no join.
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 hex
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        default=DocumentStatus.pending,
        server_default=DocumentStatus.pending.value,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    # Set when processing is claimed. A "processing" document whose claim is
    # old was abandoned (e.g. the server restarted) and may be claimed again.
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    # Identifies the run that holds the "processing" claim. Every write a run
    # makes (heartbeat, final save, failure) requires its own id to still be
    # here, so a run whose claim was taken over can never overwrite the newer
    # run's work; it notices at its next heartbeat and stops.
    processing_run_id: Mapped[uuid.UUID | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
