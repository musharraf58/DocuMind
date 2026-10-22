import re

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings
from app.core.errors import error_body

# Multipart framing (boundaries, headers) adds a little on top of the file itself.
MULTIPART_OVERHEAD = 64 * 1024
# The one route that accepts files; everything else takes small JSON or form bodies.
UPLOAD_PATH = re.compile(r"^/collections/[^/]+/documents/?$")


class BodyTooLargeError(Exception):
    pass


def body_limit(method: str, path: str) -> int:
    """Largest body accepted for this request, in bytes."""
    settings = get_settings()
    if method == "POST" and UPLOAD_PATH.match(path):
        return settings.MAX_UPLOAD_MB * 1024 * 1024 + MULTIPART_OVERHEAD
    return settings.MAX_REQUEST_BODY_KB * 1024


class MaxBodySizeMiddleware:
    """Reject request bodies over the limit before the app reads them.

    Starlette would otherwise receive a whole body (even an unauthenticated
    one) before our route runs. Two checks:
    - a declared Content-Length over the limit is refused up front;
    - a body without one (chunked encoding) is counted as it arrives, and
      refused as soon as it goes over.
    Uploads get MAX_UPLOAD_MB (the upload code still checks the exact file
    size); every other request gets MAX_REQUEST_BODY_KB.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = body_limit(scope["method"], scope["path"])
        for name, value in scope["headers"]:
            if name == b"content-length" and value.isdigit() and int(value) > limit:
                await _too_large(scope, receive, send)
                return

        received = 0
        exceeded = False
        replaced = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            # FastAPI turns any error while reading the body into its own
            # "400 error parsing the body". Once we know the real reason,
            # send our 413 instead of whatever the app answers.
            nonlocal replaced
            if exceeded:
                if message["type"] == "http.response.start" and not replaced:
                    replaced = True
                    await _too_large(scope, receive, send)
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except BodyTooLargeError:
            if not replaced:
                replaced = True
                await _too_large(scope, receive, send)


async def _too_large(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse(error_body(413, "Request body too large"), status_code=413)
    await response(scope, receive, send)
