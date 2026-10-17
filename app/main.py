import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, collections, documents, health, search
from app.core.database import SessionLocal
from app.core.limits import MaxBodySizeMiddleware
from app.services.integrity import report_missing_embeddings

# Uvicorn sets up only its own loggers. Without this, the app's INFO messages
# (e.g. "embedded N chunks") are silently dropped; libraries stay at WARNING.
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logging.getLogger("app").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await report_missing_embeddings(SessionLocal)
    yield


app = FastAPI(title="DocuMind", lifespan=lifespan)
app.add_middleware(MaxBodySizeMiddleware)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(search.router)
