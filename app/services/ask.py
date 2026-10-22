"""POST /collections/{id}/ask: answer in one response (no streaming), saving
the question and the answer to the conversation."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.models import MessageStatus
from app.services import answering, usage
from app.services.answering import (  # noqa: F401 (re-exported for callers)
    DECLINED_ANSWER,
    NOT_FOUND_ANSWER,
    Sessions,
    snippet,
)
from app.services.context import Source
from app.services.conversations import Turn
from app.services.embeddings import EmbeddingError, EmbeddingProvider
from app.services.llm import LLMError, LLMProvider, LLMRefusalError
from app.services.usage import UsageRecorder

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    text: str
    grounded: bool
    citations: list[Source] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def answer_question(
    sessions: Sessions,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID,
    question: str,
    conversation_id: uuid.UUID | None = None,
    recorder: UsageRecorder | None = None,
) -> Answer:
    """Raises 404 / ConversationBusyError before anything is saved, and
    EmbeddingError / LLMError (after saving the answer as failed)."""
    started = time.monotonic()
    turn = await answering.begin(sessions, owner_id, collection_id, question, conversation_id)
    with usage.scoped(recorder, owner_id, conversation_id=turn.conversation_id):
        return await _answer(sessions, embedder, llm, turn, question, started)


async def _answer(
    sessions: Sessions,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    turn: Turn,
    question: str,
    started: float,
) -> Answer:
    settings = get_settings()
    try:
        found = await answering.find_sources(sessions, embedder, llm, turn, question)
        if not found.sources:
            # Nothing relevant: no LLM call, no cost, no invented answer.
            answer = Answer(
                turn.conversation_id, turn.assistant_message_id, NOT_FOUND_ANSWER, False
            )
            await answering.finish(sessions, turn, MessageStatus.complete, answer.text, False)
            _log(turn, found.retrieved, 0, answer, llm_called=False, started=started)
            return answer

        prompt = answering.answer_prompt(turn, question, found.sources)
        try:
            result = await llm.generate(prompt.system, prompt.user, settings.LLM_MAX_OUTPUT_TOKENS)
        except LLMRefusalError:
            answer = Answer(turn.conversation_id, turn.assistant_message_id, DECLINED_ANSWER, False)
            await answering.finish(sessions, turn, MessageStatus.complete, answer.text, False)
            _log(turn, found.retrieved, len(found.sources), answer, True, started)
            return answer

        checked = answering.check_answer(result.text, found.sources)
        answer = Answer(
            turn.conversation_id,
            turn.assistant_message_id,
            checked.text,
            checked.grounded,
            checked.cited,
            result.prompt_tokens,
            result.completion_tokens,
        )
        await answering.finish(
            sessions,
            turn,
            MessageStatus.complete,
            answer.text,
            answer.grounded,
            answering.citation_snapshots(answer.citations),
            answer.prompt_tokens,
            answer.completion_tokens,
        )
        _log(turn, found.retrieved, len(found.sources), answer, True, started)
        return answer
    except (EmbeddingError, LLMError) as exc:
        await answering.finish(sessions, turn, MessageStatus.failed, error_message=str(exc))
        raise
    except asyncio.CancelledError:
        await answering.finish(sessions, turn, MessageStatus.cancelled)
        raise
    except Exception:
        await answering.finish(
            sessions, turn, MessageStatus.failed, error_message="Something went wrong"
        )
        raise


def _log(
    turn: Turn, retrieved: int, used: int, answer: Answer, llm_called: bool, started: float
) -> None:
    # Ids, counts and token usage only: never the question or any document text.
    logger.info(
        "ask: conversation=%s message=%s retrieved=%d sources=%d cited=%d grounded=%s "
        "llm_called=%s prompt_tokens=%d completion_tokens=%d duration_ms=%d",
        turn.conversation_id,
        turn.assistant_message_id,
        retrieved,
        used,
        len(answer.citations),
        answer.grounded,
        llm_called,
        answer.prompt_tokens,
        answer.completion_tokens,
        (time.monotonic() - started) * 1000,
    )
