from functools import lru_cache

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
    EMBEDDING_API_KEY: str = ""
    # Where uploaded files live. In Docker this is a mounted named volume.
    STORAGE_PATH: str = "./storage"
    MAX_UPLOAD_MB: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
