import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models import DocumentStatus


class DocumentRead(BaseModel):
    """Output only. storage_key and owner_id are deliberately not exposed:
    clients identify documents by id, never by where bytes live on disk."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    collection_id: uuid.UUID
    original_filename: str
    content_type: str
    size_bytes: int
    content_hash: str
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
