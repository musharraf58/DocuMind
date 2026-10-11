import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    """Input for registration."""

    email: EmailStr
    # 128 cap: argon2 on huge inputs is a cheap way to burn server CPU.
    password: str = Field(min_length=8, max_length=128)


class UserRead(BaseModel):
    """Output. There is deliberately no password field of any kind here."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    created_at: datetime
