import uuid

from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser, EmbedderDep, LLMDep, SessionFactory
from app.api.errors import provider_error
from app.schemas.ask import AskRequest, AskResponse, Citation, Usage
from app.services import ask as ask_service
from app.services.conversations import ConversationBusyError
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError

router = APIRouter(tags=["ask"])

BUSY = "An answer is already being generated in this conversation"


@router.post("/collections/{collection_id}/ask", response_model=AskResponse)
async def ask(
    collection_id: uuid.UUID,
    body: AskRequest,
    user: CurrentUser,
    sessions: SessionFactory,
    embedder: EmbedderDep,
    llm: LLMDep,
):
    """Answer a question from the documents in one of your collections, citing
    the sources it used, and save both to a conversation (a new one unless
    conversation_id is given). grounded=false means no supplied source backs
    the answer (including "I could not find this in your documents")."""
    try:
        answer = await ask_service.answer_question(
            sessions, embedder, llm, user.id, collection_id, body.question, body.conversation_id
        )
    except ConversationBusyError as exc:
        raise HTTPException(status_code=409, detail=BUSY) from exc
    except (EmbeddingError, LLMError) as exc:
        raise provider_error(exc) from exc
    return AskResponse(
        conversation_id=answer.conversation_id,
        message_id=answer.message_id,
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
