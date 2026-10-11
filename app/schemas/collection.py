import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)


class CollectionUpdate(BaseModel):
    """PATCH input: only fields the client actually sent are changed."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def name_cannot_be_null(self) -> "CollectionUpdate":
        # Omitting "name" is fine; sending "name": null would try to NULL a
        # NOT NULL column. Reject it as a client error instead of a 500.
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name cannot be null")
        return self


class CollectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
