"""The demo page (/demo): served only when DEMO_UI_ENABLED, under a strict
Content-Security-Policy, and its JavaScript never turns text into HTML."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.demo_ui import STATIC
from app.core.config import Settings, get_settings
from app.core.headers import API_CSP, DEMO_CSP
from app.main import create_app

PAGES = ["/", "/demo", "/demo/config.json", "/demo/static/app.js", "/demo/static/demo.css"]


@pytest.fixture
async def build_client():
    clients = []

    async def build(**changes) -> AsyncClient:
        app = create_app(get_settings().model_copy(update=changes))
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client

    yield build
    for client in clients:
        await client.aclose()


async def test_off_by_default_every_demo_path_is_404(build_client):
    assert Settings.model_fields["DEMO_UI_ENABLED"].default is False
    client = await build_client(DEMO_UI_ENABLED=False)
    for path in PAGES:
        assert (await client.get(path)).status_code == 404, path


async def test_enabled_serves_the_page_with_the_strict_csp(build_client):
    client = await build_client(DEMO_UI_ENABLED=True)
    response = await client.get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["Content-Security-Policy"] == DEMO_CSP
    # The existing security headers still apply.
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert '<script type="module" src="/demo/static/app.js">' in response.text


def test_the_csp_allows_only_this_origin_and_no_inline_code():
    directives = dict(part.strip().split(" ", 1) for part in DEMO_CSP.split(";"))
    assert directives["default-src"] == "'self'"
    assert directives["script-src"] == "'self'"
    assert directives["style-src"] == "'self'"
    assert directives["connect-src"] == "'self'"
    assert directives["object-src"] == "'none'"
    assert directives["frame-ancestors"] == "'none'"
    assert "unsafe" not in DEMO_CSP  # no 'unsafe-inline', no 'unsafe-eval'


async def test_static_files_types_and_csp(build_client):
    client = await build_client(DEMO_UI_ENABLED=True)
    for name, media_type in [("app.js", "text/javascript"), ("demo.css", "text/css")]:
        response = await client.get(f"/demo/static/{name}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)
        assert response.headers["Content-Security-Policy"] == DEMO_CSP


@pytest.mark.parametrize(
    "name", ["missing.js", "index.html", "..%2Fdemo_ui.py", "%2E%2E%2F%2E%2E%2Fmain.py"]
)
async def test_only_the_listed_static_files_are_served(build_client, name):
    client = await build_client(DEMO_UI_ENABLED=True)
    assert (await client.get(f"/demo/static/{name}")).status_code == 404


async def test_root_redirects_to_the_demo_when_enabled(build_client):
    client = await build_client(DEMO_UI_ENABLED=True)
    response = await client.get("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/demo"


async def test_the_hint_comes_from_the_setting(build_client):
    client = await build_client(DEMO_UI_ENABLED=True, DEMO_UI_HINT="Ask the organiser for access")
    assert (await client.get("/demo/config.json")).json() == {
        "hint": "Ask the organiser for access"
    }
    client = await build_client(DEMO_UI_ENABLED=True)
    assert (await client.get("/demo/config.json")).json() == {"hint": ""}


async def test_the_api_keeps_its_own_csp(build_client):
    client = await build_client(DEMO_UI_ENABLED=True)
    response = await client.get("/health/live")
    assert response.headers["Content-Security-Policy"] == API_CSP


# --- The page's source ------------------------------------------------------

SOURCES = sorted(STATIC.iterdir())
SCRIPTS = [path for path in SOURCES if path.suffix == ".js"]


def code(path: Path) -> str:
    """The file with its comments removed (so comments may name things the
    code must not use)."""
    text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
    return re.sub(r"(^|\s)//.*", r"\1", text)


def test_there_are_scripts_to_check():
    assert {path.name for path in SCRIPTS} >= {"app.js", "api.js", "sse.js", "ui.js"}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_no_api_that_parses_text_as_html_or_code(path):
    """Model output, document snippets and server messages must reach the page
    as text (textContent), never as markup or code."""
    source = path.read_text()  # comments included: the names must not appear at all
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert forbidden not in source, f"{path.name} uses {forbidden}"
    assert not re.search(r"\beval\s*\(", source), f"{path.name} uses eval"
    assert not re.search(r"\bnew\s+Function\b", source), f"{path.name} uses new Function"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_the_token_is_never_stored(path):
    source = code(path)
    for storage in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
        assert storage not in source, f"{path.name} uses {storage}"


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict]] = []
        self.inline: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        if self.tags and self.tags[-1][0] in ("script", "style") and data.strip():
            self.inline.append(self.tags[-1][0])


def test_the_html_has_no_inline_script_style_or_handlers():
    parser = _Tags()
    parser.feed((STATIC / "index.html").read_text())
    assert parser.inline == []
    for tag, attrs in parser.tags:
        assert tag != "style"
        assert "style" not in attrs, tag
        assert not [name for name in attrs if name.startswith("on")], tag
        if tag == "script":
            assert attrs.get("src", "").startswith("/demo/static/")


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_nothing_is_loaded_from_another_origin(path):
    # An XML namespace (xmlns="http://www.w3.org/...") is a name, not a download.
    source = re.sub(r'xmlns(:\w+)?="[^"]*"', "", code(path))
    assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", source), path.name


def test_form_fields_have_labels():
    parser = _Tags()
    parser.feed((STATIC / "index.html").read_text())
    labelled = {attrs["for"] for tag, attrs in parser.tags if tag == "label"}
    fields = [attrs["id"] for tag, attrs in parser.tags if tag in ("input", "textarea")]
    assert fields and set(fields) <= labelled


def test_the_streaming_answer_is_announced():
    parser = _Tags()
    parser.feed((STATIC / "index.html").read_text())
    (messages,) = [attrs for tag, attrs in parser.tags if attrs.get("id") == "messages"]
    assert messages["aria-live"] == "polite"
