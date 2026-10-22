"""A request id for every request, so one request's log lines can be found
together (and matched to what a user reports: the id is in every response).

The id lives in a contextvar: a value that follows the current task through
every await, without being passed around by hand. The logging filter in
app.core.logging reads it, so every log line written while handling a
request carries its id. The Celery queue sends it along with the job, and the
worker sets it again before running the job (app.workers.celery_app).
"""

import re
import uuid
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HEADER = "X-Request-ID"
# Accept the caller's id only if it looks harmless: it ends up in logs and in
# response headers, so no spaces, quotes or line breaks (log injection).
VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,128}$")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def new_request_id() -> str:
    return uuid.uuid4().hex


def accept_or_generate(incoming: str | None) -> str:
    if incoming and VALID_ID.fullmatch(incoming):
        return incoming
    return new_request_id()


class RequestIdMiddleware:
    """Take a well-formed X-Request-ID from the request, or make one; return
    it in the response and keep it in the contextvar while the request runs.
    Also stored in scope["state"], where error handlers can read it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = None
        for name, value in scope["headers"]:
            if name == b"x-request-id":
                incoming = value.decode("latin-1")
                break
        request_id = accept_or_generate(incoming)
        scope.setdefault("state", {})["request_id"] = request_id
        token = _request_id.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _request_id.reset(token)
