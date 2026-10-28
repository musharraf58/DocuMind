import math
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

from app.core.db_url import to_asyncpg_url

# Share of the token budget held back from document embedding so that search
# queries can still get through while a long document is being embedded.
QUERY_RESERVE_FRACTION = 0.05

# SECRET_KEY signs every login token: whoever knows it can sign a token for
# any user. These are values from examples and tutorials that people forget
# to change (compared case-insensitively).
KNOWN_WEAK_SECRETS = {
    "change-me",
    "changeme",
    "change_me",
    "please-change-me",
    "secret",
    "secret-key",
    "secret_key",
    "your-secret-key",
    "your_secret_key",
    "supersecret",
    "dev",
    "development",
    "test",
    "password",
    "documind",
}
MIN_SECRET_LENGTH = 32
MIN_INVITE_CODE_LENGTH = 12


class Settings(BaseSettings):
    """App configuration. Values come from environment variables, then .env."""

    # env_ignore_empty: "NAME=" with nothing after it means "use the default",
    # rather than an empty string that may not even parse (e.g. as a number).
    # hide_input_in_errors: a validation error would otherwise print every input
    # value, API keys and SECRET_KEY included, into the startup log.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True, hide_input_in_errors=True
    )

    # Any Postgres URL: "postgresql+asyncpg://..." as Compose uses, or a host's
    # libpq-style "postgresql://...?sslmode=require" (translated for asyncpg,
    # see app/core/db_url.py).
    DATABASE_URL: str
    # For migrations only: a direct (non-pooled) connection. Neon's pooled
    # URL goes through PgBouncer, which migrations shouldn't use (they rely on
    # session state that transaction pooling doesn't keep). Empty: DATABASE_URL.
    DATABASE_URL_DIRECT: str = ""
    # DATABASE_URL goes through PgBouncer in transaction mode (Neon's
    # "-pooler" host): no prepared-statement caches, timeouts set per
    # transaction. See app/core/db_url.py.
    DB_PGBOUNCER: bool = False
    REDIS_URL: str
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ANTHROPIC_API_KEY: str = ""
    # Where uploaded files live. "local": the directory STORAGE_PATH (in
    # Docker Compose a named volume). "s3": an S3-compatible bucket (R2, B2,
    # MinIO, AWS); required where the disk is ephemeral, e.g. Render's free tier.
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    STORAGE_PATH: str = "./storage"
    # Empty endpoint: AWS itself. R2: https://<account id>.r2.cloudflarestorage.com
    # B2: https://s3.<region>.backblazeb2.com
    S3_ENDPOINT_URL: str = ""
    S3_BUCKET: str = ""
    S3_ACCESS_KEY_ID: SecretStr = SecretStr("")
    S3_SECRET_ACCESS_KEY: SecretStr = SecretStr("")
    # R2: "auto"; B2: the region in the endpoint, e.g. "us-west-004".
    S3_REGION: str = "us-east-1"
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

    # Where document jobs run. "celery": a separate Celery worker process
    # (Docker Compose). "inline": inside the API process, one at a time, from
    # a bounded in-memory queue; for hosts without a free background worker
    # (Render's free tier). See app/services/inline_worker.py.
    INGESTION_MODE: Literal["celery", "inline"] = "celery"
    # Inline mode: documents waiting at most; uploads beyond it get 503.
    INLINE_QUEUE_CAPACITY: int = Field(default=20, ge=1, le=1000)
    # Inline mode: on shutdown, let the running job finish for this long, then
    # stop it and put its document back to "pending".
    INLINE_SHUTDOWN_GRACE_SECONDS: float = Field(default=15, ge=0)
    # Inline mode: the sweeper (and retention) run this often, in-process.
    MAINTENANCE_INTERVAL_SECONDS: float = Field(default=120, ge=10)

    # Background jobs (both modes).
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

    # Capacity guard: refuse new uploads once the database is this big (MB;
    # 0 = no limit). Ask, search and reading keep working. Neon's free tier
    # caps storage; see app/services/capacity.py.
    MAX_DB_SIZE_MB: int = Field(default=400, ge=0)
    # Delete ordinary (non-demo) accounts older than this many days, with
    # everything they own (0 = never). Runs with the periodic maintenance.
    RETENTION_DAYS: int = Field(default=0, ge=0)

    # Database connections (API process). See app/core/database.py.
    DB_POOL_SIZE: int = Field(default=5, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0)
    DB_POOL_TIMEOUT_SECONDS: float = Field(default=10, gt=0)
    # Replace pooled connections older than this (seconds).
    DB_POOL_RECYCLE_SECONDS: int = Field(default=1800, ge=30)
    # Postgres cancels any single statement running longer than this. The
    # slowest measured API query (vector search, 10,000 chunks) takes ~0.1 s.
    DB_STATEMENT_TIMEOUT_MS: int = Field(default=15_000, ge=100)
    # The worker writes a whole document's chunks and vectors in one go.
    DB_WORKER_STATEMENT_TIMEOUT_MS: int = Field(default=300_000, ge=100)

    # Logging. "json": one JSON object per line, for log collectors.
    LOG_FORMAT: Literal["text", "json"] = "text"

    # "production" turns on strict startup checks (see check_production) and HSTS.
    APP_ENV: Literal["development", "production"] = "development"
    # Browser origins allowed to call the API, comma-separated, e.g.
    # "https://app.example.com". Empty: no cross-origin browser access. "*" is refused.
    CORS_ORIGINS: str = ""
    # Largest request body for everything except uploads (JSON bodies are tiny).
    MAX_REQUEST_BODY_KB: int = Field(default=64, ge=1)
    # How many reverse proxies WE run in front of the API. 0: use the connection
    # address and ignore X-Forwarded-For (which any client can forge). N: the
    # client is the Nth address from the right in X-Forwarded-For.
    TRUSTED_PROXY_COUNT: int = Field(default=0, ge=0, le=10)
    # Inbound rate limits ("N/second|minute|hour|day"), per endpoint class. See
    # docs/SECURITY.md for why each is what it is.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_LOGIN_PER_IP: str = "10/minute"
    RATE_LIMIT_LOGIN_PER_EMAIL: str = "5/minute"
    RATE_LIMIT_REGISTER_PER_IP: str = "5/hour"
    # Changing the password, deleting the account (per user).
    RATE_LIMIT_ACCOUNT: str = "5/minute"
    RATE_LIMIT_ASK: str = "10/minute"
    RATE_LIMIT_UPLOAD: str = "10/hour"
    # Every other authenticated endpoint.
    RATE_LIMIT_DEFAULT: str = "120/minute"
    # Progressive delay after failed logins (per account + IP): this many are
    # free, then each failure blocks the next try for BASE, 2xBASE, 4xBASE ...
    # seconds, up to MAX. Forgotten after WINDOW minutes without a failure.
    LOGIN_FREE_ATTEMPTS: int = Field(default=3, ge=0)
    LOGIN_DELAY_BASE_SECONDS: float = Field(default=2, gt=0)
    LOGIN_DELAY_MAX_SECONDS: float = Field(default=900, gt=0)
    LOGIN_FAILURE_WINDOW_MINUTES: float = Field(default=15, gt=0)
    # Who may create an account. "open": anyone; "invite": anyone with
    # INVITE_CODE (an empty code means nobody); "closed": nobody.
    REGISTRATION_MODE: Literal["open", "invite", "closed"] = "invite"
    # Never logged or printed (SecretStr). Share it only with people you invite.
    INVITE_CODE: SecretStr = SecretStr("")
    # Upload caps per user: total stored, number of documents, and documents
    # waiting or being processed at once.
    MAX_STORAGE_MB_PER_USER: int = Field(default=100, ge=1)
    MAX_DOCUMENTS_PER_USER: int = Field(default=50, ge=1)
    MAX_PENDING_DOCUMENTS_PER_USER: int = Field(default=5, ge=1)
    # Daily quotas, per user (Redis counters; the day starts at QUOTA_RESET_HOUR_UTC).
    DAILY_QUESTIONS_PER_USER: int = Field(default=50, ge=0)
    # About a 300-page book; queries count too (a few tokens each).
    DAILY_EMBED_TOKENS_PER_USER: int = Field(default=200_000, ge=0)
    DAILY_UPLOADS_PER_USER: int = Field(default=20, ge=0)
    # LLM requests per day for ALL users together (answers, rewrites, retries).
    # Gemini's free tier allowed about 500 a day when measured; this leaves room.
    GLOBAL_DAILY_LLM_BUDGET: int = Field(default=350, ge=0)
    QUOTA_RESET_HOUR_UTC: int = Field(default=0, ge=0, le=23)
    # The shared read-only demo account, for `python -m scripts.seed_demo`
    # only (the app itself never reads these). Never commit real values.
    DEMO_EMAIL: str = ""
    DEMO_PASSWORD: SecretStr = SecretStr("")
    # The demo page (/demo, app/static/demo/; / redirects to it when enabled).
    DEMO_UI_ENABLED: bool = False
    # One line shown under the demo page's login form, e.g. how to log in to
    # the demo account. Shown to every visitor; empty: no hint. Kept out of
    # the page source so no credential is ever committed.
    DEMO_UI_HINT: str = Field(default="", max_length=300)
    # Interactive API docs (/docs, /redoc, /openapi.json).
    DOCS_ENABLED: bool = True
    # In production, docs stay refused unless this explicit opt-in is set too.
    ALLOW_DOCS_IN_PRODUCTION: bool = False
    # Production refuses the fake embedding/LLM providers unless this is set.
    # Only for running the production configuration locally without paid keys
    # (docker-compose.lite.yml); never set it on a real deployment.
    ALLOW_FAKE_PROVIDERS_IN_PRODUCTION: bool = False

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

    @property
    def database_url(self) -> URL:
        """DATABASE_URL in the form asyncpg accepts."""
        return to_asyncpg_url(self.DATABASE_URL)

    @property
    def migration_database_url(self) -> URL:
        return to_asyncpg_url(self.DATABASE_URL_DIRECT or self.DATABASE_URL)

    @field_validator("DATABASE_URL", "DATABASE_URL_DIRECT")
    @classmethod
    def check_database_url(cls, value: str) -> str:
        if value:
            to_asyncpg_url(value)  # fail at startup, with a message that hides the URL
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @field_validator(
        "RATE_LIMIT_LOGIN_PER_IP",
        "RATE_LIMIT_LOGIN_PER_EMAIL",
        "RATE_LIMIT_REGISTER_PER_IP",
        "RATE_LIMIT_ACCOUNT",
        "RATE_LIMIT_ASK",
        "RATE_LIMIT_UPLOAD",
        "RATE_LIMIT_DEFAULT",
    )
    @classmethod
    def check_rate(cls, value: str) -> str:
        from app.core.rate_limit import Rate

        Rate.parse(value)  # a typo fails at startup, not on the first request
        return value

    @model_validator(mode="after")
    def check_cors(self) -> Self:
        # "*" lets any website call the API from its visitors' browsers. Never
        # wanted here, in any environment: list the real origins instead.
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS must list origins explicitly; '*' is not allowed")
        return self

    def secret_key_problem(self) -> str | None:
        key = self.SECRET_KEY
        if key.strip().lower() in KNOWN_WEAK_SECRETS:
            return "SECRET_KEY is a well-known example value"
        if len(key) < MIN_SECRET_LENGTH:
            return f"SECRET_KEY must be at least {MIN_SECRET_LENGTH} characters"
        if len(set(key)) < 10:
            return "SECRET_KEY is too repetitive to be random"
        return None

    @model_validator(mode="after")
    def check_production(self) -> Self:
        """In production, refuse to start with a configuration that would be
        unsafe on the public internet. All problems are listed at once."""
        if self.APP_ENV != "production":
            return self
        problems = []
        if secret := self.secret_key_problem():
            problems.append(
                f"{secret}. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if self.DOCS_ENABLED and not self.ALLOW_DOCS_IN_PRODUCTION:
            problems.append(
                "API docs are enabled: set DOCS_ENABLED=false, or ALLOW_DOCS_IN_PRODUCTION=true "
                "if you really want them public"
            )
        if not self.RATE_LIMIT_ENABLED:
            problems.append("RATE_LIMIT_ENABLED must be true")
        fake = self.EMBEDDING_PROVIDER == "fake" or self.LLM_PROVIDER == "fake"
        if fake and not self.ALLOW_FAKE_PROVIDERS_IN_PRODUCTION:
            problems.append("the fake embedding/LLM providers are for development only")
        code = self.INVITE_CODE.get_secret_value()
        if self.REGISTRATION_MODE == "invite" and 0 < len(code) < MIN_INVITE_CODE_LENGTH:
            problems.append(f"INVITE_CODE must be at least {MIN_INVITE_CODE_LENGTH} characters")
        if problems:
            raise ValueError("Refusing to start with APP_ENV=production: " + "; ".join(problems))
        return self

    @model_validator(mode="after")
    def check_storage(self) -> Self:
        if self.STORAGE_BACKEND == "s3":
            missing = [
                name
                for name, value in (
                    ("S3_BUCKET", self.S3_BUCKET),
                    ("S3_ACCESS_KEY_ID", self.S3_ACCESS_KEY_ID.get_secret_value()),
                    ("S3_SECRET_ACCESS_KEY", self.S3_SECRET_ACCESS_KEY.get_secret_value()),
                )
                if not value
            ]
            if missing:
                raise ValueError("STORAGE_BACKEND=s3 needs " + ", ".join(missing))
        return self

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
