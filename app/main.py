from fastapi import FastAPI

from app.api import auth, collections, health

app = FastAPI(title="DocuMind")
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(collections.router)
