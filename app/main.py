from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, auth, collections, conversations, documents, health, search, usage
from app.core.config import Settings, get_settings
from app.core.database import SessionLocal
from app.core.errors import CatchAllMiddleware, install_error_handlers
from app.core.headers import SecurityHeadersMiddleware
from app.core.limits import MaxBodySizeMiddleware
from app.core.logging import configure_logging
from app.core.request_id import HEADER as REQUEST_ID_HEADER
from app.core.request_id import RequestIdMiddleware
from app.services.integrity import report_missing_embeddings

# Uvicorn sets up only its own loggers. Without this, the app's INFO messages
# (e.g. "embedded N chunks") are silently dropped.
configure_logging()

# Response headers a browser script on an allowed origin may read.
EXPOSED_HEADERS = [
    "Retry-After",
    "RateLimit-Limit",
    "RateLimit-Remaining",
    "RateLimit-Reset",
    REQUEST_ID_HEADER,
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await report_missing_embeddings(SessionLocal)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    docs = settings.DOCS_ENABLED
    app = FastAPI(
        title="DocuMind",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    install_error_handlers(app)

    # Middleware: the LAST one added is the OUTERMOST, so a request passes
    # through them bottom to top, and the response top to bottom.
    app.add_middleware(MaxBodySizeMiddleware)
    # Unhandled exceptions become a generic 500 here, inside the middleware
    # below, so the 500 still gets security headers and the request id.
    app.add_middleware(CatchAllMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    if settings.cors_origins:
        # Outside the error handling, so a browser can read error responses too.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            # Bearer tokens in a header, no cookies: credentials mode isn't needed.
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
            expose_headers=EXPOSED_HEADERS,
            max_age=600,
        )
    # Outermost: every response, even a rejection by the middleware above,
    # carries the id.
    app.add_middleware(RequestIdMiddleware)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(conversations.router)
    app.include_router(usage.router)
    return app


app = create_app()
