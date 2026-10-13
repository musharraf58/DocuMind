import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChunkRead(BaseModel):
    """Output only. owner_id and collection_id are internal bookkeeping (the
    caller already knows the document), so they are not exposed."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    page_number: int | None
    section_title: str | None
    token_count: int
    created_at: datetime
