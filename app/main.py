import logging

from fastapi import FastAPI

from app.api import auth, collections, documents, health, search
from app.core.limits import MaxBodySizeMiddleware

# Uvicorn sets up only its own loggers. Without this, the app's INFO messages
# (e.g. "embedded N chunks") are silently dropped; libraries stay at WARNING.
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logging.getLogger("app").setLevel(logging.INFO)

app = FastAPI(title="DocuMind")
app.add_middleware(MaxBodySizeMiddleware)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(search.router)
