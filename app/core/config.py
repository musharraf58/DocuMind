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
    # Requests are counted exactly and don't need this; how they are shared
    # between ingestion and searches is EMBEDDING_INGEST_RPM_SHARE.
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
    # Share of the requests per minute that background ingestion may use; the
    # rest stays free for interactive searches (0.67 of 3 = 2 requests).
    EMBEDDING_INGEST_RPM_SHARE: float = Field(default=0.67, gt=0, le=1)

    # Answer generation (LLM). "fake" gives scripted answers with no network
    # calls (local dev, tests). The default is the real provider so that a
    # deployment that forgot this setting fails loudly instead of answering
    # with canned text.
    LLM_PROVIDER: Literal["gemini", "fake"] = "gemini"
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    LLM_API_KEY: SecretStr = SecretStr("")
    # Includes the model's "thinking" tokens, which count toward this limit.
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=600, ge=50)
    LLM_TIMEOUT_SECONDS: float = Field(default=30, gt=0)
    # Google publishes no free-tier figures any more (see AI Studio for your
    # project); 15/minute was measured independently, so we stay below it.
    LLM_RPM_LIMIT: int = Field(default=10, ge=1)
    # Google strongly recommends the default 1.0 for Gemini 3 models: lower
    # values "may lead to unexpected behavior, such as looping".
    LLM_TEMPERATURE: float = Field(default=1.0, ge=0, le=2)

    # Retrieval-augmented answers (POST /collections/{id}/ask).
    RAG_TOP_K: int = Field(default=8, ge=1, le=50)
    # Chunks farther than this (cosine distance) count as not relevant.
    # Chosen with python -m eval.sweep for voyage-4-lite (docs/EVALUATION.md);
    # re-measure after changing the embedding model.
    RAG_MAX_DISTANCE: float = Field(default=0.70, gt=0, le=2)
    # How much source text (approximate tokens) goes into the prompt.
    RAG_CONTEXT_TOKEN_BUDGET: int = Field(default=3000, ge=100)
    ASK_MAX_QUESTION_CHARS: int = Field(default=1000, ge=1, le=10_000)

    # Hybrid search. "vector": meaning only. "hybrid": meaning plus keywords
    # (Postgres full-text search), merged by Reciprocal Rank Fusion. Chosen
    # from evaluation results (docs/EVALUATION.md).
    RAG_SEARCH_MODE: Literal["vector", "hybrid"] = "hybrid"
    # How many candidates each list (vector, keyword) contributes to the fusion.
    HYBRID_CANDIDATES: int = Field(default=20, ge=1, le=200)
    # The k in RRF: score = sum of 1 / (k + rank). 60 is the usual choice.
    RRF_K: int = Field(default=60, ge=1)
    # "strict": a keyword match needs every term; "loose": any term.
    RAG_KEYWORD_MATCH: Literal["strict", "loose"] = "strict"
    # Postgres text search configuration ('english' stems words and drops
    # stop words; 'simple' doesn't). The indexed column uses 'english' (see
    # migration 0009); another value is computed on the fly, without an index.
    TEXT_SEARCH_CONFIG: str = Field(default="english", pattern=r"^[a-z_]+$")
    # Hybrid gate: a strong keyword match (all terms, keyword top 3) is kept
    # up to RAG_MAX_DISTANCE + this. 0 disables that looser path.
    RAG_KEYWORD_GATE_SLACK: float = Field(default=0.10, ge=0, le=1)

    # Conversations. How much earlier conversation goes into the prompt: at most
    # this many turns (question + answer), newest kept first, within the budget.
    RAG_HISTORY_MAX_TURNS: int = Field(default=6, ge=0, le=50)
    RAG_HISTORY_TOKEN_BUDGET: int = Field(default=1000, ge=0)
    # A follow-up is first rewritten into a standalone question (one short LLM
    # call); past this, the original question is used instead.
    RAG_REWRITE_TIMEOUT_SECONDS: float = Field(default=10, gt=0)
    # While a streamed answer is waiting, send an SSE comment this often so
    # proxies and browsers don't close an idle-looking connection.
    STREAM_HEARTBEAT_SECONDS: float = Field(default=15, gt=0)

    # Background jobs (Celery worker).
    # Tries per document for transient failures (rate limits, timeouts, 5xx).
    MAX_JOB_ATTEMPTS: int = Field(default=4, ge=1)
    # Wait before retrying a transient failure: BASE, then 2x, then 4x ...
    JOB_RETRY_BASE_SECONDS: float = Field(default=30, gt=0)
    # Longest one job may run. At 10,000 tokens/minute the largest allowed
    # document takes about 105 minutes to embed.
    JOB_TIMEOUT_MINUTES: float = Field(default=120, gt=0)
    # Sweeper: re-enqueue documents "pending" for longer than this...
    SWEEP_PENDING_MINUTES: float = Field(default=3, gt=0)
    # ...and treat "processing" documents with no heartbeat for this long as stuck.
    STALE_PROCESSING_MINUTES: float = Field(default=10, gt=0)

    @property
    def embedding_token_budget(self) -> int:
        """Tokens we allow ourselves per 60 seconds."""
        return math.floor(self.EMBEDDING_TPM_LIMIT * self.EMBEDDING_RATE_SAFETY + 1e-9)

    @property
    def embedding_request_budget(self) -> int:
        """Requests per 60 seconds for everyone together: the account's limit.
        (Small clock differences with the provider are absorbed by the 429
        handling: Retry-After and the minimum wait.)"""
        return self.EMBEDDING_RPM_LIMIT

    @property
    def embedding_ingest_request_budget(self) -> int:
        """Requests per 60 seconds background ingestion may use (at least 1)."""
        share = self.EMBEDDING_RPM_LIMIT * self.EMBEDDING_INGEST_RPM_SHARE
        return max(1, math.floor(share + 1e-9))

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
        """Request slots per window that only searches may use."""
        return self.embedding_request_budget - self.embedding_ingest_request_budget

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
