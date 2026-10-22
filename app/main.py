from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ask, auth, collections, conversations, documents, health, search
from app.core.database import SessionLocal
from app.core.limits import MaxBodySizeMiddleware
from app.core.logging import configure_logging
from app.core.request_id import RequestIdMiddleware
from app.services.integrity import report_missing_embeddings

# Uvicorn sets up only its own loggers. Without this, the app's INFO messages
# (e.g. "embedded N chunks") are silently dropped.
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await report_missing_embeddings(SessionLocal)
    yield


app = FastAPI(title="DocuMind", lifespan=lifespan)
app.add_middleware(MaxBodySizeMiddleware)
# Added last = outermost: every response, even a rejection by the middleware
# above, carries the id.
app.add_middleware(RequestIdMiddleware)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(conversations.router)
