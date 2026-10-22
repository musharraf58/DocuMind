from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import get_settings
from app.core.errors import error_body

# Multipart framing (boundaries, headers) adds a little on top of the file itself.
MULTIPART_OVERHEAD = 64 * 1024


class MaxBodySizeMiddleware:
    """Reject requests whose declared Content-Length is clearly over the upload limit.

    Starlette receives the whole multipart body before our route runs, so without
    this an oversize upload (even an unauthenticated one) would be fully received
    first. This refuses it up front. It is a cheap first line of defense only: a
    client can omit Content-Length (chunked encoding), so the upload code still
    enforces the limit exactly while streaming.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            limit = get_settings().MAX_UPLOAD_MB * 1024 * 1024 + MULTIPART_OVERHEAD
            for name, value in scope["headers"]:
                if name == b"content-length" and value.isdigit() and int(value) > limit:
                    response = JSONResponse(
                        error_body(413, "Request body too large"), status_code=413
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)
