import uuid
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
)

from app.core.passwords import MAX_LENGTH, password_problem


class UserCreate(BaseModel):
    """Input for registration."""

    email: EmailStr
    # Length limits are checked by password_problem (with clearer messages);
    # the max_length here only stops absurd inputs early. 128 cap: argon2 on
    # huge inputs is a cheap way to burn server CPU.
    password: str = Field(max_length=MAX_LENGTH)
    # Needed when REGISTRATION_MODE is "invite". SecretStr: never shows up in a
    # log line or repr.
    invite_code: SecretStr | None = Field(default=None, max_length=200)

    @field_validator("password")
    @classmethod
    def check_password(cls, password: str, info: ValidationInfo) -> str:
        # Fields are validated in order, so a valid email is already in info.data.
        problem = password_problem(password, info.data.get("email"))
        if problem:
            raise ValueError(problem)
        return password


class ChangePassword(BaseModel):
    current_password: str = Field(max_length=MAX_LENGTH)
    new_password: str = Field(max_length=MAX_LENGTH)


class DeleteAccount(BaseModel):
    password: str = Field(max_length=MAX_LENGTH)


class UserRead(BaseModel):
    """Output. There is deliberately no password field of any kind here."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    created_at: datetime
    # The shared read-only demo account (a client can show a banner).
    is_demo: bool = False
