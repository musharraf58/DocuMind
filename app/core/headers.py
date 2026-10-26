"""Security headers on every response.

This is a JSON API: nothing it returns should ever be rendered as a page,
framed, sniffed into another type, or cached when it holds a user's data.
"""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings

ALWAYS = {
    # Never guess a type other than the declared Content-Type (e.g. run a
    # "text/plain" upload as HTML/JS).
    "X-Content-Type-Options": "nosniff",
    # No other site may show our responses in a frame (clickjacking).
    "X-Frame-Options": "DENY",
    # Don't leak our URLs (they contain ids) to other sites in the Referer.
    "Referrer-Policy": "no-referrer",
    # Other origins may not embed our responses as images, scripts, etc.
    "Cross-Origin-Resource-Policy": "same-origin",
}
# JSON responses load nothing; if one is ever opened as a page, it runs nothing.
API_CSP = "default-src 'none'; frame-ancestors 'none'"
# The docs pages load Swagger UI / ReDoc from a CDN, so they skip the strict CSP.
DOCS_PATHS = ("/docs", "/redoc")
HSTS = "max-age=31536000; includeSubDomains"


class SecurityHeadersMiddleware:
    """Adds the headers above. Cache-Control: no-store on anything a user
    authenticated for (the request carried a token) and on /auth/* (a login
    response carries a token): proxies and browsers must not keep a copy.
    HSTS (always use HTTPS) only in production, which runs behind HTTPS."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        private = path.startswith("/auth") or any(
            name == b"authorization" for name, _ in scope["headers"]
        )
        production = get_settings().APP_ENV == "production"

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in ALWAYS.items():
                    headers.setdefault(name, value)
                if not path.startswith(DOCS_PATHS):
                    headers.setdefault("Content-Security-Policy", API_CSP)
                if private:
                    headers["Cache-Control"] = "no-store"
                if production:
                    headers.setdefault("Strict-Transport-Security", HSTS)
            await send(message)

        await self.app(scope, receive, send_with_headers)
