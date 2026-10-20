"""POST /collections/{id}/ask/stream: the answer as server-sent events.

Events, in order: meta (ids), token (text pieces as they are generated),
then done (the final, checked answer: the source of truth) or error.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from app.api.errors import provider_error
from app.core.config import get_settings
from app.models import MessageStatus
from app.services import answering
from app.services.answering import DECLINED_ANSWER, NOT_FOUND_ANSWER, Sessions
from app.services.conversations import Turn
from app.services.embeddings import EmbeddingError, EmbeddingProvider
from app.services.llm import LLMError, LLMProvider, LLMRefusalError, LLMUsage
from app.services.sse import event

logger = logging.getLogger(__name__)

SOMETHING_WENT_WRONG = "Something went wrong while answering"


async def stream_answer(
    sessions: Sessions,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    turn: Turn,
    question: str,
) -> AsyncIterator[str]:
    """Runs after begin() (the question is saved, the answer is in progress).
    No database session is open here except the short ones inside
    find_sources() and finish()."""
    settings = get_settings()
    yield event(
        "meta",
        {
            "conversation_id": str(turn.conversation_id),
            "message_id": str(turn.assistant_message_id),
        },
    )
    partial = ""
    try:
        found = await answering.find_sources(sessions, embedder, llm, turn, question)
        if not found.sources:
            # Nothing relevant: the fixed answer, and no LLM call at all.
            async for chunk in _fixed_answer(sessions, turn, NOT_FOUND_ANSWER):
                yield chunk
            return

        prompt = answering.answer_prompt(turn, question, found.sources)
        usage = LLMUsage(0, 0)
        try:
            # aclosing: if we stop early (client gone), the provider stream is
            # closed at once, which hangs up on the provider.
            async with contextlib.aclosing(
                llm.stream(prompt.system, prompt.user, settings.LLM_MAX_OUTPUT_TOKENS)
            ) as pieces:
                async for piece in pieces:
                    if isinstance(piece, LLMUsage):
                        usage = piece
                    else:
                        partial += piece
                        yield event("token", {"text": piece})
        except LLMRefusalError:
            async for chunk in _fixed_answer(sessions, turn, DECLINED_ANSWER, streamed=partial):
                yield chunk
            return

        # The streamed text is raw: invalid citation markers are only removed
        # now, so `done` (not the tokens) is the answer to keep.
        checked = answering.check_answer(partial, found.sources)
        citations = answering.citation_snapshots(checked.cited)
        await answering.finish(
            sessions,
            turn,
            MessageStatus.complete,
            checked.text,
            checked.grounded,
            citations,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        yield event(
            "done",
            answering.done_payload(
                turn,
                checked.text,
                checked.grounded,
                citations,
                (usage.prompt_tokens, usage.completion_tokens),
            ),
        )
    except (EmbeddingError, LLMError) as exc:
        await answering.finish(
            sessions, turn, MessageStatus.failed, partial, error_message=str(exc)
        )
        http = provider_error(exc)
        retry_after = (http.headers or {}).get("Retry-After")
        yield event(
            "error",
            {
                "message": str(exc),  # our own fixed messages, never provider text
                "status": http.status_code,
                "retry_after": int(retry_after) if retry_after else None,
            },
        )
    except (asyncio.CancelledError, GeneratorExit):
        # The client went away: keep what was generated so far, no citations.
        await answering.finish(sessions, turn, MessageStatus.cancelled, partial)
        raise
    except Exception:
        logger.exception("Streaming answer failed: message=%s", turn.assistant_message_id)
        await answering.finish(
            sessions, turn, MessageStatus.failed, partial, error_message=SOMETHING_WENT_WRONG
        )
        yield event("error", {"message": SOMETHING_WENT_WRONG, "status": 500, "retry_after": None})


async def _fixed_answer(
    sessions: Sessions, turn: Turn, text: str, streamed: str = ""
) -> AsyncIterator[str]:
    if not streamed:
        yield event("token", {"text": text})
    await answering.finish(sessions, turn, MessageStatus.complete, text, False)
    yield event("done", answering.done_payload(turn, text, False, [], (0, 0)))
