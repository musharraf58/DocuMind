"""Answering a question from a collection's documents (retrieval-augmented
generation): retrieve, keep what's relevant, build the prompt, generate,
check the citations."""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models import Collection
from app.services import retrieval
from app.services.citations import check_citations
from app.services.context import Source, assemble_context
from app.services.embeddings import EmbeddingProvider
from app.services.llm import LLMProvider, LLMRefusalError
from app.services.prompts import build_prompt

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "I could not find this in your documents."
DECLINED_ANSWER = "I can't answer this question from your documents."
SNIPPET_CHARS = 200


@dataclass
class Answer:
    text: str
    grounded: bool
    citations: list[Source] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def answer_question(
    db: AsyncSession,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    collection: Collection,
    question: str,
    settings: Settings | None = None,
) -> Answer:
    """Answer `question` from a collection the caller has already proven they own.
    Raises EmbeddingError / LLMError for provider failures (the API maps them)."""
    settings = settings or get_settings()
    started = time.monotonic()
    chunks = await retrieval.retrieve(db, embedder, collection, question, settings.RAG_TOP_K)
    relevant = retrieval.filter_relevant(chunks, settings.RAG_MAX_DISTANCE)

    if not relevant:
        # Nothing close enough: don't ask the model to answer from nothing,
        # it would only guess. No LLM call, no cost, no invented answer.
        answer = Answer(NOT_FOUND_ANSWER, grounded=False)
        _log(collection, len(chunks), 0, 0, answer, llm_called=False, started=started)
        return answer

    sources = assemble_context(relevant, settings.RAG_CONTEXT_TOKEN_BUDGET)
    prompt = build_prompt(question, [source.render() for source in sources])
    try:
        result = await llm.generate(prompt.system, prompt.user, settings.LLM_MAX_OUTPUT_TOKENS)
    except LLMRefusalError:
        answer = Answer(DECLINED_ANSWER, grounded=False)
        _log(collection, len(chunks), len(relevant), len(sources), answer, True, started)
        return answer

    checked = check_citations(result.text, sources)
    answer = Answer(
        checked.text,
        checked.grounded,
        checked.cited,
        result.prompt_tokens,
        result.completion_tokens,
    )
    _log(collection, len(chunks), len(relevant), len(sources), answer, True, started)
    return answer


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The start of a chunk, cut at a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def _log(
    collection: Collection,
    retrieved: int,
    relevant: int,
    used: int,
    answer: Answer,
    llm_called: bool,
    started: float,
) -> None:
    # Counts and token usage only: never the question or any document text.
    logger.info(
        "ask: collection=%s retrieved=%d relevant=%d sources=%d cited=%d grounded=%s "
        "llm_called=%s prompt_tokens=%d completion_tokens=%d duration_ms=%d",
        collection.id,
        retrieved,
        relevant,
        used,
        len(answer.citations),
        answer.grounded,
        llm_called,
        answer.prompt_tokens,
        answer.completion_tokens,
        (time.monotonic() - started) * 1000,
    )
