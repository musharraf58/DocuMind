from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration. Values come from environment variables, then .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
    EMBEDDING_MAX_RETRIES: int = Field(default=4, ge=0, le=10)
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_TOKENS // 2:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be less than half of CHUNK_TOKENS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
