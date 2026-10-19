import uuid

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession, EmbedderDep, LLMDep
from app.api.errors import provider_error
from app.schemas.ask import AskRequest, AskResponse, Citation, Usage
from app.services import ask as ask_service
from app.services import collections as collection_service
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError

router = APIRouter(tags=["ask"])


@router.post("/collections/{collection_id}/ask", response_model=AskResponse)
async def ask(
    collection_id: uuid.UUID,
    body: AskRequest,
    db: DbSession,
    user: CurrentUser,
    embedder: EmbedderDep,
    llm: LLMDep,
):
    """Answer a question from the documents in one of your collections, citing
    the sources it used. grounded=false means no supplied source backs the
    answer (including "I could not find this in your documents")."""
    # Ownership first: 404 for other users' collections, before anything is
    # embedded or retrieved.
    collection = await collection_service.get_owned_collection(db, collection_id, user.id)
    try:
        answer = await ask_service.answer_question(db, embedder, llm, collection, body.question)
    except (EmbeddingError, LLMError) as exc:
        raise provider_error(exc) from exc
    return AskResponse(
        answer=answer.text,
        grounded=answer.grounded,
        citations=[
            Citation(
                number=source.number,
                document_id=source.chunk.document_id,
                document_name=source.chunk.document_name,
                page_number=source.chunk.page_number,
                section_title=source.chunk.section_title,
                snippet=ask_service.snippet(source.chunk.content),
                distance=source.chunk.distance,
            )
            for source in answer.citations
        ],
        usage=Usage(prompt_tokens=answer.prompt_tokens, completion_tokens=answer.completion_tokens),
    )
