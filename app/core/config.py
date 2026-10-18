import math
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Share of the token budget held back from document embedding so that search
# queries can still get through while a long document is being embedded.
QUERY_RESERVE_FRACTION = 0.05


class Settings(BaseSettings):
    """App configuration. Values come from environment variables, then .env."""

    # env_ignore_empty: "NAME=" with nothing after it means "use the default",
    # rather than an empty string that may not even parse (e.g. as a number).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    DATABASE_URL: str
    REDIS_URL: str
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ANTHROPIC_API_KEY: str = ""
    # Where uploaded files live. In Docker this is a mounted named volume.
    STORAGE_PATH: str = "./storage"
    MAX_UPLOAD_MB: int = 20

    # Chunking. Tokens are counted with tiktoken's cl100k_base as an approximation.
    CHUNK_TOKENS: int = Field(default=500, ge=50)
    CHUNK_OVERLAP_TOKENS: int = Field(default=60, ge=0)

    # Extraction limits. Going over any of them marks the document failed.
    MAX_EXTRACT_PAGES: int = Field(default=500, ge=1)
    MAX_EXTRACTED_CHARS: int = Field(default=2_000_000, ge=1)
    # Less text than this (e.g. a scanned PDF with no text layer) counts as failed.
    MIN_EXTRACTED_CHARS: int = Field(default=50, ge=1)
    # Zip-bomb guard for DOCX (a DOCX file is a ZIP archive).
    MAX_DOCX_UNCOMPRESSED_MB: int = Field(default=100, ge=1)
    MAX_DOCX_ENTRIES: int = Field(default=1000, ge=1)

    # Embeddings. "fake" gives deterministic vectors with no network calls (for
    # local dev and tests). The default is the real provider on purpose: a
    # deployment that forgot this setting must fail loudly, not silently store
    # meaningless vectors.
    EMBEDDING_PROVIDER: Literal["voyage", "fake"] = "voyage"
    EMBEDDING_MODEL: str = "voyage-4-lite"
    # Must match the chunks.embedding column (see migration 0005).
    EMBEDDING_DIMENSIONS: int = Field(default=1024, ge=1)
    # SecretStr prints as "**********", so the key can't leak through a log or repr.
    EMBEDDING_API_KEY: SecretStr = SecretStr("")
    EMBEDDING_BATCH_SIZE: int = Field(default=64, ge=1)
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=30, gt=0)

    # The provider account's real limits, per rolling 60 seconds.
    EMBEDDING_TPM_LIMIT: int = Field(default=10_000, ge=1)
    EMBEDDING_RPM_LIMIT: int = Field(default=3, ge=1)
    # Fraction of the TOKEN limit we plan to use. Our tiktoken count can be well
    # below the provider's own count (different tokenizer, plus the retrieval
    # prompt the provider adds to each text), so leave plenty of room.
    # Requests are counted exactly and don't need this (see embedding_request_budget).
    EMBEDDING_RATE_SAFETY: float = Field(default=0.5, gt=0, le=1)

    # Retries per batch (each batch starts with a fresh count).
    EMBEDDING_MAX_RETRIES: int = Field(default=6, ge=0, le=10)
    # Waits between retries: exponential from BASE, capped at MAX, with jitter.
    # A Retry-After header from the provider takes precedence when present.
    EMBEDDING_RETRY_BASE_SECONDS: float = Field(default=2, gt=0)
    EMBEDDING_RETRY_MAX_SECONDS: float = Field(default=60, gt=0)
    # Shortest wait after a 429 ("too many requests"). Unset means 60 /
    # EMBEDDING_RPM_LIMIT: the spacing at which requests stay under the limit.
    EMBEDDING_MIN_RETRY_WAIT_SECONDS: float | None = Field(default=None, gt=0)
    # A search query never waits longer than this for embedding capacity; past
    # it the API answers "busy, retry in N seconds" instead of hanging.
    EMBEDDING_QUERY_MAX_WAIT_SECONDS: float = Field(default=20, gt=0)

    @property
    def embedding_token_budget(self) -> int:
        """Tokens we allow ourselves per 60 seconds."""
        return math.floor(self.EMBEDDING_TPM_LIMIT * self.EMBEDDING_RATE_SAFETY + 1e-9)

    @property
    def embedding_request_budget(self) -> int:
        """Requests we allow ourselves per 60 seconds: one below the limit (at
        least 1). Requests are counted exactly; the one-request margin covers
        our clock and the provider's not lining up perfectly."""
        return max(1, self.EMBEDDING_RPM_LIMIT - 1)

    @property
    def embedding_min_retry_wait(self) -> float:
        if self.EMBEDDING_MIN_RETRY_WAIT_SECONDS is not None:
            return self.EMBEDDING_MIN_RETRY_WAIT_SECONDS
        return 60 / self.EMBEDDING_RPM_LIMIT

    @property
    def embedding_query_reserve_tokens(self) -> int:
        return math.ceil(self.embedding_token_budget * QUERY_RESERVE_FRACTION)

    @property
    def embedding_query_reserve_requests(self) -> int:
        """One request slot per window stays free for queries (when there are
        at least two; with a single slot, documents and queries share it)."""
        return 1 if self.embedding_request_budget >= 2 else 0

    @property
    def embedding_document_token_budget(self) -> int:
        """Largest batch of document text: the budget minus the queries' share."""
        return self.embedding_token_budget - self.embedding_query_reserve_tokens

    @model_validator(mode="after")
    def check_embedding_settings(self) -> Self:
        if self.EMBEDDING_RETRY_MAX_SECONDS < self.EMBEDDING_RETRY_BASE_SECONDS:
            raise ValueError("EMBEDDING_RETRY_MAX_SECONDS must be at least the base")
        if self.CHUNK_TOKENS > self.embedding_document_token_budget:
            # One request can never carry more than the per-minute allowance, so
            # a chunk bigger than a batch could never be embedded at all.
            raise ValueError(
                f"CHUNK_TOKENS ({self.CHUNK_TOKENS}) must not exceed the embedding token "
                f"budget per batch ({self.embedding_document_token_budget}, from "
                "EMBEDDING_TPM_LIMIT and EMBEDDING_RATE_SAFETY)"
            )
        return self

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_TOKENS // 2:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be less than half of CHUNK_TOKENS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
