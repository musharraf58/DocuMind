"""Recording usage: one usage_events row per successful provider request.

The providers (app/services/embeddings/base.py, app/services/llm/base.py)
call record() after each request that succeeded. They don't know which user
the request is for, so the code that does know opens a scope first:

    with recorder.scope(user_id, conversation_id=...):
        ... anything here that embeds or calls the LLM is recorded for user_id

The scope lives in a contextvar (like the request id): it follows the
current task through every await, without being threaded through every
function between the endpoint and the provider. Outside any scope (the
evaluation runner, scripts) nothing is recorded.

Recording never breaks the user's request: a failed insert is logged and
dropped. The record is history, not enforcement (enforcement is the Redis
counters in app/services/quotas.py).
"""

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import UsageEvent, UsageKind
from app.services.quotas import EMBED_TOKENS, Quotas

logger = logging.getLogger(__name__)

EMBED_KINDS = {UsageKind.embed_document, UsageKind.embed_query}


@dataclass(frozen=True)
class _Scope:
    recorder: "UsageRecorder"
    user_id: uuid.UUID
    document_id: uuid.UUID | None
    conversation_id: uuid.UUID | None


_current: ContextVar[_Scope | None] = ContextVar("usage_scope", default=None)


class UsageRecorder:
    """Where events go: a session factory for the Postgres row, and the
    Redis client for the daily counters (see app/services/quotas.py)."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], redis: Redis | None) -> None:
        self.sessions = sessions
        self.redis = redis

    @contextmanager
    def scope(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
    ) -> Iterator[None]:
        token = _current.set(_Scope(self, user_id, document_id, conversation_id))
        try:
            yield
        finally:
            _current.reset(token)

    async def save(self, scope: _Scope, event: UsageEvent) -> None:
        async with self.sessions() as db:
            db.add(event)
            await db.commit()


def scoped(
    recorder: "UsageRecorder | None",
    user_id: uuid.UUID,
    document_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
):
    """recorder.scope(...), or a scope that records nothing if there is no
    recorder (callers such as the evaluation runner pass none)."""
    if recorder is None:
        return nullcontext()
    return recorder.scope(user_id, document_id, conversation_id)


def current_user_id() -> uuid.UUID | None:
    scope = _current.get()
    return scope.user_id if scope else None


def current_quotas() -> tuple[uuid.UUID, Quotas] | None:
    """The scope's user and their quotas, if a scope with Redis is open."""
    scope = _current.get()
    if scope is None or scope.recorder.redis is None:
        return None
    return scope.user_id, Quotas(scope.recorder.redis)


async def record(kind: UsageKind, model: str, tokens_in: int, tokens_out: int = 0) -> None:
    """Called by the providers after each successful request."""
    scope = _current.get()
    if scope is None:
        return
    event = UsageEvent(
        user_id=scope.user_id,
        kind=kind,
        tokens_in=max(0, int(tokens_in)),
        tokens_out=max(0, int(tokens_out)),
        model=model[:100],
        document_id=scope.document_id,
        conversation_id=scope.conversation_id,
    )
    if kind in EMBED_KINDS and scope.recorder.redis is not None:
        # The daily quota counts what the provider actually reported.
        await Quotas(scope.recorder.redis).add(scope.user_id, EMBED_TOKENS, event.tokens_in)
    try:
        await scope.recorder.save(scope, event)
    except Exception:
        # Ids and counts only, like the event itself.
        logger.warning(
            "Could not record usage (user=%s kind=%s tokens_in=%d tokens_out=%d)",
            scope.user_id,
            kind,
            event.tokens_in,
            event.tokens_out,
            exc_info=True,
        )
