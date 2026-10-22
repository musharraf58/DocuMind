from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal, get_db
from app.core.redis import get_redis
from app.core.security import decode_access_token
from app.models import User
from app.services.embeddings import EmbeddingError, EmbeddingProvider, get_embedding_provider
from app.services.llm import LLMProvider, get_llm_provider
from app.services.queue import JobQueue, get_queue
from app.services.quotas import Quotas
from app.services.storage import StorageBackend, get_storage
from app.services.usage import UsageRecorder

# HTTPBearer makes Swagger's "Authorize" dialog a single "Value" box for a token.
# auto_error=False so a missing/malformed header reaches our code and we return
# a consistent 401 (FastAPI versions differ: some answer 403 on their own).
bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """For code that opens its own short sessions (the ask endpoints), instead
    of holding one session for the whole request. Tests override this."""
    return SessionLocal


SessionFactory = Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)]
StorageDep = Annotated[StorageBackend, Depends(get_storage)]
QueueDep = Annotated[JobQueue, Depends(get_queue)]
RedisDep = Annotated[Redis, Depends(get_redis)]


def get_usage_recorder(sessions: SessionFactory, redis: RedisDep) -> UsageRecorder:
    """Where usage events and daily counters go (the test database and fake
    Redis in tests, through the overrides of the two dependencies)."""
    return UsageRecorder(sessions, redis)


RecorderDep = Annotated[UsageRecorder, Depends(get_usage_recorder)]


def get_quotas(redis: RedisDep) -> Quotas:
    return Quotas(redis)


QuotasDep = Annotated[Quotas, Depends(get_quotas)]


def embedding_provider() -> EmbeddingProvider:
    """FastAPI dependency. A misconfiguration (e.g. dimension mismatch) is a
    503 with the safe explanation rather than a bare 500."""
    try:
        return get_embedding_provider()
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


EmbedderDep = Annotated[EmbeddingProvider, Depends(embedding_provider)]


def llm_provider() -> LLMProvider:
    """FastAPI dependency for the answer-generating LLM."""
    return get_llm_provider()


LLMDep = Annotated[LLMProvider, Depends(llm_provider)]


async def get_current_user(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    claims = decode_access_token(credentials.credentials)
    if claims is None:
        raise unauthorized
    # Look the user up on every request: a deleted user's old token stops working.
    user = await db.get(User, claims.user_id)
    # A token from before the last password change carries an older version.
    if user is None or user.token_version != claims.token_version:
        raise unauthorized
    # End the read transaction: it would otherwise keep a database connection
    # checked out for the rest of the request, including a minutes-long
    # streamed answer. (Sessions don't expire objects on commit, so `user`
    # stays usable.)
    await db.commit()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
