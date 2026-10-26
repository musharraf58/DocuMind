"""One shape for every error response:

    {"detail": "human-readable message", "code": "machine_readable", "request_id": "..."}

plus extra fields where useful (e.g. "errors" for validation, "limit" for a
limit that was hit). "detail" is kept from FastAPI's own shape, so existing
clients that read it keep working; "code" lets a client react without
parsing the message; "request_id" is what a user can quote in a bug report.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_id import get_request_id

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "Internal server error"

DEFAULT_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    501: "not_implemented",
    503: "unavailable",
}


class ApiError(HTTPException):
    """An HTTPException with a specific code and optional extra fields, e.g.
    ApiError(403, "Storage limit reached", code="storage_limit_reached",
    extra={"limit_bytes": ...})."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        code: str | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code or code_for(status_code)
        self.extra = extra or {}


def code_for(status_code: int) -> str:
    return DEFAULT_CODES.get(status_code, "error")


def error_body(status_code: int, detail: Any, code: str | None = None, **extra: Any) -> dict:
    return {
        "detail": detail,
        "code": code or code_for(status_code),
        "request_id": get_request_id(),
        **extra,
    }


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = getattr(exc, "code", None)
    extra = getattr(exc, "extra", {})
    return JSONResponse(
        error_body(exc.status_code, exc.detail, code, **extra),
        status_code=exc.status_code,
        headers=exc.headers,
    )


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default response echoes each rejected value ("input"), which
    # would send a rejected password straight back (and into any log of the
    # response). Keep only where the problem is and what it is.
    errors = [
        {"loc": list(error.get("loc", ())), "msg": error.get("msg", ""), "type": error.get("type")}
        for error in exc.errors()
    ]
    return JSONResponse(
        error_body(422, "Request validation failed", errors=errors), status_code=422
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)


class CatchAllMiddleware:
    """Turns any unhandled exception into a generic 500 with the request id.

    The stack trace goes to the log (with the same request id), never into the
    response: it can reveal file paths, library versions, SQL or data. If the
    response had already started (a stream that failed halfway), it's too
    late to send a 500; the error is logged and the connection closed.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("Unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if started:
                raise
            response = JSONResponse(error_body(500, INTERNAL_ERROR), status_code=500)
            await response(scope, receive, send)
