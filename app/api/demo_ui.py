"""The demo page: a small browser client for the API, served at /demo.

Only included when DEMO_UI_ENABLED is true (app/main.py); otherwise none of
these routes exist and they answer 404 like any unknown path. The page is
plain HTML, CSS and JavaScript modules from app/static/demo/: no build step,
nothing loaded from another origin. Its Content-Security-Policy is
app/core/headers.py's DEMO_CSP.

The login hint comes from the DEMO_UI_HINT setting through
/demo/config.json, so no credential is ever written into the page source.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

from app.core.config import Settings
from app.core.errors import ApiError

STATIC = Path(__file__).resolve().parent.parent / "static" / "demo"
MEDIA_TYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
# Served by name from this fixed list: a request can never reach any other
# file (no paths, no "..").
FILES = {
    path.name: MEDIA_TYPES[path.suffix]
    for path in STATIC.iterdir()
    if path.is_file() and path.suffix in MEDIA_TYPES
}


def make_router(settings: Settings) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse("/demo", status_code=307)

    @router.get("/demo")
    async def page() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @router.get("/demo/config.json")
    async def config() -> dict:
        return {"hint": settings.DEMO_UI_HINT}

    @router.get("/demo/static/{name}")
    async def static_file(name: str) -> FileResponse:
        if name not in FILES:
            raise ApiError(404, "Not Found")
        # no-cache: the browser revalidates (cheap), so a deploy is seen at once.
        return FileResponse(
            STATIC / name, media_type=FILES[name], headers={"Cache-Control": "no-cache"}
        )

    return router
