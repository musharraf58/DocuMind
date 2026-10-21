import uuid

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank")
        return value


class SearchResult(BaseModel):
    """One matching chunk, with what each search said about it (for
    debugging retrieval). distance is the vector (cosine) distance: smaller
    = more similar in meaning. Ranks are 1-based; null = not in that list."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    content: str
    page_number: int | None
    section_title: str | None
    distance: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    keyword_all_terms: bool | None = None
    rrf_score: float | None = None
