from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration. Values come from environment variables, then .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    REDIS_URL: str
    SECRET_KEY: str
    ANTHROPIC_API_KEY: str = ""
    EMBEDDING_API_KEY: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
