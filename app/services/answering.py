"""The answer pipeline shared by POST /ask and POST /ask/stream.

Database work happens in short sessions of its own, and NO session is open
while we wait on the network (the rewrite, the embedding, the answer):

  begin()        session A: ownership, conversation, save the question,
                 create the in-progress answer (the lock), load history
  find_sources() no session: rewrite a follow-up, embed it (one Voyage call)
                 session B: the vector query
  ...generate or stream the answer: no session...
  finish()       session C: save the outcome, which releases the conversation
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.models import MessageStatus
from app.services import collections as collection_service
from app.services import retrieval
from app.services.citations import CheckedAnswer, check_citations
from app.services.context import Source, assemble_context
from app.services.conversations import Turn, finish_turn, start_turn
from app.services.embeddings import EmbeddingProvider
from app.services.llm import LLMProvider
from app.services.prompts import Prompt, build_prompt, select_history
from app.services.rewrite import Rewrite, rewrite_question

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "I could not find this in your documents."
DECLINED_ANSWER = "I can't answer this question from your documents."
SNIPPET_CHARS = 200

Sessions = async_sessionmaker[AsyncSession]


@dataclass(frozen=True)
class Sources:
    sources: list[Source]  # empty: nothing relevant, so no LLM call
    rewrite: Rewrite
    retrieved: int


async def begin(
    sessions: Sessions,
    owner_id: uuid.UUID,
    collection_id: uuid.UUID,
    question: str,
    conversation_id: uuid.UUID | None,
    settings: Settings | None = None,
) -> Turn:
    """Session A. Raises 404 (collection or conversation not the caller's, or a
    conversation of another collection) and ConversationBusyError (409)."""
    settings = settings or get_settings()
    async with sessions() as db:
        collection = await collection_service.get_owned_collection(db, collection_id, owner_id)
        return await start_turn(
            db, collection, question, conversation_id, settings.RAG_HISTORY_MAX_TURNS
        )


async def find_sources(
    sessions: Sessions,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    turn: Turn,
    question: str,
    settings: Settings | None = None,
) -> Sources:
    """Rewrite (follow-ups only), embed once, search, keep what's relevant,
    and number it within the context budget."""
    settings = settings or get_settings()
    rewrite = await rewrite_question(
        llm,
        question,
        turn.history,
        max_seconds=settings.RAG_REWRITE_TIMEOUT_SECONDS,
        max_chars=settings.ASK_MAX_QUESTION_CHARS,
    )
    if turn.is_follow_up:
        # Ids, outcome and token counts only: never the question text.
        logger.info(
            "rewrite: conversation=%s message=%s rewritten=%s reason=%s "
            "prompt_tokens=%d completion_tokens=%d",
            turn.conversation_id,
            turn.assistant_message_id,
            rewrite.rewritten,
            rewrite.reason,
            rewrite.prompt_tokens,
            rewrite.completion_tokens,
        )
    vector = await embedder.embed_query(rewrite.question)  # one Voyage request
    async with sessions() as db:
        chunks = await retrieval.search_by_vector(
            db, turn.owner_id, turn.collection_id, embedder.model, vector, settings.RAG_TOP_K
        )
    relevant = retrieval.filter_relevant(chunks, settings.RAG_MAX_DISTANCE)
    sources = assemble_context(relevant, settings.RAG_CONTEXT_TOKEN_BUDGET) if relevant else []
    return Sources(sources, rewrite, len(chunks))


def answer_prompt(
    turn: Turn, question: str, sources: Sequence[Source], settings: Settings | None = None
) -> Prompt:
    """The user's own question (not the rewrite), recent history, the sources."""
    settings = settings or get_settings()
    history = select_history(
        turn.history, settings.RAG_HISTORY_MAX_TURNS, settings.RAG_HISTORY_TOKEN_BUDGET
    )
    return build_prompt(question, [source.render() for source in sources], history)


def check_answer(text: str, sources: Sequence[Source]) -> CheckedAnswer:
    return check_citations(text, sources)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The start of a chunk, cut at a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def citation_snapshots(cited: Sequence[Source]) -> list[dict]:
    """What is saved with the answer: copies, not references. Re-processing a
    document replaces its chunks (new ids), and old answers must still be
    able to show what they cited."""
    return [
        {
            "number": source.number,
            "chunk_id": str(source.chunk.chunk_id),
            "document_id": str(source.chunk.document_id),
            "document_name": source.chunk.document_name,
            "page_number": source.chunk.page_number,
            "section_title": source.chunk.section_title,
            "snippet": snippet(source.chunk.content),
            "distance": source.chunk.distance,
        }
        for source in cited
    ]


async def finish(
    sessions: Sessions,
    turn: Turn,
    status: MessageStatus,
    content: str = "",
    grounded: bool | None = None,
    citations: Sequence[dict] = (),
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error_message: str | None = None,
) -> None:
    """Session C: save the outcome, releasing the conversation. Shielded from
    cancellation, because it often runs exactly when the request is being
    cancelled (the client went away) and must still happen."""

    async def save() -> None:
        async with sessions() as db:
            await finish_turn(
                db,
                turn,
                status,
                content,
                grounded,
                citations,
                prompt_tokens,
                completion_tokens,
                error_message,
            )

    task = asyncio.ensure_future(save())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep waiting: the save must finish. (Under anyio, which Starlette
            # uses, cancellation is re-delivered at every await until we leave.)
            cancelled = True
    task.result()  # surface a failed save
    if cancelled:
        raise asyncio.CancelledError
    logger.info(
        "answer: conversation=%s message=%s status=%s grounded=%s citations=%d "
        "prompt_tokens=%d completion_tokens=%d",
        turn.conversation_id,
        turn.assistant_message_id,
        status,
        grounded,
        len(citations),
        prompt_tokens,
        completion_tokens,
    )


def done_payload(turn: Turn, answer: str, grounded: bool, citations: list[dict], usage) -> dict:
    """The final event: exactly what is saved, so a client can replace the
    streamed text with it."""
    return {
        "conversation_id": str(turn.conversation_id),
        "message_id": str(turn.assistant_message_id),
        "answer": answer,
        "grounded": grounded,
        "citations": citations,
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }
