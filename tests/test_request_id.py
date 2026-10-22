import io
import json
import logging
from types import SimpleNamespace

from app.core.logging import JsonFormatter, RequestIdFilter
from app.core.request_id import accept_or_generate, get_request_id, set_request_id
from app.workers.celery_app import REQUEST_ID_HEADER, clear_request_id, use_request_id


async def test_generated_when_missing(client):
    response = await client.get("/health")
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32


async def test_well_formed_id_is_returned(client):
    response = await client.get("/health", headers={"X-Request-ID": "abc-123.DEF_456"})
    assert response.headers["X-Request-ID"] == "abc-123.DEF_456"


async def test_malformed_id_is_replaced(client):
    for bad in ["short", "has space in it", "x" * 129, "eviléid12345".encode("latin-1")]:
        response = await client.get("/health", headers={"X-Request-ID": bad})
        sent = bad.decode("latin-1") if isinstance(bad, bytes) else bad
        assert len(response.headers["X-Request-ID"]) == 32
        assert response.headers["X-Request-ID"] != sent


async def test_each_request_gets_its_own(client):
    first = (await client.get("/health")).headers["X-Request-ID"]
    second = (await client.get("/health")).headers["X-Request-ID"]
    assert first != second


async def test_error_responses_carry_it_too(client):
    response = await client.get("/auth/me")
    assert response.status_code == 401
    assert "X-Request-ID" in response.headers


def test_accept_or_generate():
    assert accept_or_generate("abcdefgh") == "abcdefgh"
    assert accept_or_generate(None) != accept_or_generate(None)
    assert accept_or_generate("a\nb-injected-line") != "a\nb-injected-line"


def _log_line(formatter: logging.Formatter) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    logger = logging.getLogger("test.request_id")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("hello %s", "world")
    finally:
        logger.removeHandler(handler)
    return stream.getvalue()


def test_log_lines_include_the_request_id():
    set_request_id("req-12345678")
    try:
        line = _log_line(logging.Formatter("[%(request_id)s] %(message)s"))
    finally:
        set_request_id(None)
    assert line.strip() == "[req-12345678] hello world"


def test_outside_a_request_the_id_is_a_dash():
    line = _log_line(logging.Formatter("[%(request_id)s] %(message)s"))
    assert line.strip() == "[-] hello world"


def test_json_format():
    set_request_id("req-12345678")
    try:
        entry = json.loads(_log_line(JsonFormatter()))
    finally:
        set_request_id(None)
    assert entry["request_id"] == "req-12345678"
    assert entry["message"] == "hello world"
    assert entry["level"] == "WARNING"


def test_worker_uses_the_queued_request_id():
    for request in [
        SimpleNamespace(**{REQUEST_ID_HEADER: "req-from-api-1"}),
        SimpleNamespace(headers={REQUEST_ID_HEADER: "req-from-api-1"}),
    ]:
        use_request_id(task=SimpleNamespace(request=request))
        assert get_request_id() == "req-from-api-1"
        clear_request_id()
        assert get_request_id() is None


def test_worker_makes_a_fresh_id_without_a_valid_one():
    for request in [SimpleNamespace(), SimpleNamespace(**{REQUEST_ID_HEADER: "bad id"})]:
        use_request_id(task=SimpleNamespace(request=request))
        assert get_request_id() not in (None, "bad id")
        clear_request_id()


async def test_queue_sends_the_request_id(monkeypatch):
    from app.services.queue import CeleryJobQueue
    from app.workers import celery_app as worker_app

    sent = {}
    monkeypatch.setattr(worker_app.celery_app, "send_task", lambda *a, **kw: sent.update(kw))
    set_request_id("req-queued-123")
    try:
        import uuid

        assert await CeleryJobQueue().enqueue(uuid.uuid4())
    finally:
        set_request_id(None)
    assert sent["headers"] == {REQUEST_ID_HEADER: "req-queued-123"}
