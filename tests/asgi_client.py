"""A minimal ASGI driver for streaming responses.

httpx's ASGITransport collects the whole response before returning it, so it
can't observe a stream as it happens or disconnect in the middle. This talks
to the app directly: it records each body chunk as it is sent and can send
"http.disconnect" when a condition is met, exactly as uvicorn does when a
browser goes away (uvicorn announces ASGI spec 2.3, so Starlette listens
for that message and cancels the response).
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class StreamResult:
    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    @property
    def events(self) -> list[tuple[str, dict]]:
        """(name, data) for each SSE event, in order (comments left out)."""
        parsed = []
        for block in self.text.split("\n\n"):
            lines = [line for line in block.split("\n") if line and not line.startswith(":")]
            if not lines:
                continue
            name = next(line[7:] for line in lines if line.startswith("event: "))
            data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
            parsed.append((name, data))
        return parsed

    @property
    def heartbeats(self) -> int:
        return self.text.count(": heartbeat\n\n")


async def post_stream(
    app,
    path: str,
    body: dict,
    headers: dict[str, str],
    disconnect_when: Callable[[StreamResult], bool] | None = None,
    max_seconds: float = 10,
) -> StreamResult:
    result = StreamResult()
    disconnect = asyncio.Event()
    payload = json.dumps(body).encode()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            result.status = message["status"]
            result.headers = {k.decode(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            if message.get("body"):
                result.chunks.append(message["body"].decode())
            if disconnect_when is not None and disconnect_when(result):
                disconnect.set()
            if not message.get("more_body"):
                disconnect.set()  # finished: a real client would now close too

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")]
        + [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("test", 1234),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), max_seconds)
    return result
