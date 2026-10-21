from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.chunk import EMBEDDING_COLUMN_DIMENSIONS


class EmbeddingCacheEntry(Base):
    """A stored embedding, so the same text is never sent to the provider twice.

    Keyed by everything that changes the vector: the model, whether the text
    was embedded as a query or a document (Voyage adds a different prompt
    for each), the output size, and the text itself (as a SHA-256 hash).
    """

    __tablename__ = "embedding_cache"

    model: Mapped[str] = mapped_column(Text, primary_key=True)
    input_type: Mapped[str] = mapped_column(String(16), primary_key=True)
    dimensions: Mapped[int] = mapped_column(Integer, primary_key=True)
    text_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_COLUMN_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
