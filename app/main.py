from fastapi import FastAPI

from app.api import auth, collections, documents, health, search
from app.core.limits import MaxBodySizeMiddleware

app = FastAPI(title="DocuMind")
app.add_middleware(MaxBodySizeMiddleware)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(search.router)
