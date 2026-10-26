"""Every log record emitted while a request is handled carries its request id:
the application's own lines AND uvicorn's access log line."""

# The embedder/llm fixtures are imported from test_ask_api and then requested
# by name, which ruff reads as redefinitions.
# ruff: noqa: F811

import asyncio
import contextlib
import logging
import socket

import pytest
import uvicorn

from app.core.logging import configure_logging, make_formatter
from app.main import app
from tests.test_ask_api import about, embedder, llm, setup  # noqa: F401 (fixtures)
from tests.test_ask_stream import stream

REQUEST_ID = "test-request-id-0001"


# The test's own HTTP client logs a line after each response arrives. That is
# the test talking, not the server handling a request, so it is not captured.
CLIENT_LOGGERS = ("httpx", "httpcore")


class Capture(logging.Handler):
    """Collects records without formatting them (like any handler would)."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith(CLIENT_LOGGERS):
            self.records.append(record)


@contextlib.contextmanager
def capture_everything():
    """A handler on the root logger and on every logger that has handlers of
    its own and doesn't pass records up (uvicorn's), at DEBUG for the app."""
    handler = Capture()
    loggers = [
        logging.getLogger(name) for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access")
    ]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        for logger, level in zip(loggers, previous, strict=True):
            logger.removeHandler(handler)
            logger.setLevel(level)


def assert_all_carry(records: list[logging.LogRecord], request_id: str = REQUEST_ID) -> None:
    assert records, "the request logged nothing, so the test proves nothing"
    # What would be printed, not just the attribute: the id is in the text.
    formatter = make_formatter("text")
    unprinted = [r.name for r in records if f"[{request_id}]" not in formatter.format(r)]
    assert unprinted == []
    missing = [
        f"{r.name}: {r.getMessage()[:60]} (request_id={getattr(r, 'request_id', 'MISSING')})"
        for r in records
        if getattr(r, "request_id", None) != request_id
    ]
    assert missing == []


# --- the application's own log lines, through real endpoints -------------------------


async def test_ask_logs_carry_the_request_id(client, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    with capture_everything() as records:
        response = await client.post(
            f"/collections/{collection_id}/ask",
            json={"question": "What do cats do?"},
            headers={**headers, "X-Request-ID": REQUEST_ID},
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == REQUEST_ID
    assert any(r.name == "app.services.ask" for r in records)  # the app logged something
    assert_all_carry(records)


async def test_streamed_answer_logs_carry_the_request_id(client, embedder, llm):
    """The stream runs after the endpoint returned, in tasks created later."""
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    with capture_everything() as records:
        result = await stream(
            {**headers, "X-Request-ID": REQUEST_ID}, collection_id, "What do cats do?"
        )
    assert result.status == 200
    assert any(r.name == "app.services.answering" for r in records)
    assert_all_carry(records)


async def test_error_logs_carry_the_request_id(client):
    """A 500's stack trace is logged under the id the response shows."""
    from app.core.database import get_db

    async def explode():
        raise RuntimeError("boom")
        yield  # pragma: no cover

    app.dependency_overrides[get_db] = explode
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as ac:
        with capture_everything() as records:
            response = await ac.get("/health", headers={"X-Request-ID": REQUEST_ID})
    assert response.status_code == 500
    assert any(r.exc_info for r in records)
    assert_all_carry(records)


async def test_records_outside_a_request_have_no_id(client):
    with capture_everything() as records:
        logging.getLogger("app.test").warning("not in a request")
    assert [r.request_id for r in records] == ["-"]


# --- uvicorn's access log, through a real uvicorn server -----------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def live_server(client):
    """The real app behind a real uvicorn server (in this process, so the
    test fixtures' database and fake Redis apply). uvicorn sets up its own log
    handlers when it is configured, exactly as in production."""
    # Routes that log from a worker thread and from a task, added to the real app.

    @app.get("/_test/log")
    def sync_endpoint():  # a sync endpoint runs in a worker thread
        logging.getLogger("app.test").info("from the endpoint thread")
        return {"ok": True}

    @app.get("/_test/log-async")
    async def async_endpoint():
        logging.getLogger("app.test").info("from the endpoint")
        await asyncio.create_task(_log_in_task())
        return {"ok": True}

    async def _log_in_task():
        logging.getLogger("app.test").info("from a task the endpoint started")

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off")
    server = uvicorn.Server(config)
    # Importing app.main (as `uvicorn app.main:app` does after configuring
    # uvicorn's logging) runs configure_logging(): do it again, in that order.
    configure_logging()
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started
    try:
        yield port
    finally:
        server.should_exit = True
        await task
        app.router.routes = [
            r for r in app.router.routes if not getattr(r, "path", "").startswith("/_test")
        ]


async def raw_get(port: int, path: str, request_id: str) -> str:
    """A bare HTTP request (httpx is blocked in tests, and this is all we need)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: test\r\nX-Request-ID: {request_id}\r\n"
        "Connection: close\r\n\r\n".encode()
    )
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data.decode()


async def test_uvicorn_access_line_carries_the_request_id(live_server):
    with capture_everything() as records:
        response = await raw_get(live_server, "/_test/log", REQUEST_ID)
        await asyncio.sleep(0.1)  # the access line is written as the response starts
    assert "200 OK" in response and f"x-request-id: {REQUEST_ID}" in response.lower()
    access = [r for r in records if r.name == "uvicorn.access"]
    assert len(access) == 1, [r.name for r in records]
    assert_all_carry(records)


async def test_uvicorn_access_line_is_formatted_with_the_id(live_server):
    """Not just present on the record: it is in the text that gets printed."""
    handler_texts = []

    class Spy(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            handler_texts.append(self.format(record))

    # The real handlers' formatter, as configure_logging left it.
    access = logging.getLogger("uvicorn.access")
    formatter = access.handlers[0].formatter
    spy = Spy()
    spy.setFormatter(formatter)
    access.addHandler(spy)
    try:
        await raw_get(live_server, "/_test/log", REQUEST_ID)
        await asyncio.sleep(0.1)
    finally:
        access.removeHandler(spy)
    assert len(handler_texts) == 1
    assert f"[{REQUEST_ID}]" in handler_texts[0]
    assert "GET /_test/log" in handler_texts[0]


async def test_lines_from_threads_and_tasks_carry_the_id(live_server):
    with capture_everything() as records:
        await raw_get(live_server, "/_test/log", REQUEST_ID)
        await raw_get(live_server, "/_test/log-async", REQUEST_ID)
        await asyncio.sleep(0.1)
    app_lines = {r.getMessage() for r in records if r.name == "app.test"}
    assert app_lines == {
        "from the endpoint thread",
        "from the endpoint",
        "from a task the endpoint started",
    }
    assert_all_carry(records)


async def test_concurrent_requests_do_not_mix_ids(live_server):
    ids = [f"concurrent-request-{i:04d}" for i in range(5)]
    with capture_everything() as records:
        await asyncio.gather(*(raw_get(live_server, "/_test/log-async", rid) for rid in ids))
        await asyncio.sleep(0.1)
    per_id = {rid: [r for r in records if r.request_id == rid] for rid in ids}
    for rid, mine in per_id.items():
        # 2 app lines + 1 access line each, every one under its own id only.
        assert len(mine) == 3, (rid, [r.getMessage() for r in mine])
    assert sum(len(v) for v in per_id.values()) == len(records)
