import uuid

from pydantic import BaseModel, field_validator

from app.core.config import get_settings


class AskRequest(BaseModel):
    question: str
    # Continue this conversation; without it a new conversation is started.
    conversation_id: uuid.UUID | None = None

    @field_validator("question")
    @classmethod
    def sensible_length(cls, value: str) -> str:
        # Checked here, while the request is parsed: a bad question is
        # rejected (422) before anything is embedded or sent anywhere.
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty")
        limit = get_settings().ASK_MAX_QUESTION_CHARS
        if len(value) > limit:
            raise ValueError(f"Question must be at most {limit} characters")
        return value


class Citation(BaseModel):
    number: int  # the [n] used in the answer
    document_id: uuid.UUID
    document_name: str
    page_number: int | None
    section_title: str | None
    snippet: str  # the start of the cited chunk
    distance: float


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class AskResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID  # the saved answer
    answer: str
    # True only if the answer cites at least one of the supplied sources.
    grounded: bool
    citations: list[Citation]
    usage: Usage
