from fastapi import FastAPI

from app.api import health

app = FastAPI(title="DocuMind")
app.include_router(health.router)
