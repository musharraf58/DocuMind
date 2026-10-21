"""Server-sent events: formatting, and keeping an idle stream alive.

Format (text/event-stream): each event is "event: <name>" and "data: <json>"
lines followed by a blank line. Lines starting with ":" are comments, which
clients ignore; we send one as a heartbeat so proxies and browsers don't
close a connection that looks idle while we wait for the model.
"""

import asyncio
import json
from collections.abc import AsyncIterator

HEARTBEAT = ": heartbeat\n\n"
_DONE = object()


def event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


async def with_heartbeats(events: AsyncIterator[str], interval: float) -> AsyncIterator[str]:
    """Pass `events` through, adding a heartbeat whenever none arrives for
    `interval` seconds.

    The events are produced by a separate task, so waiting for the next one
    can time out (heartbeat) without disturbing the producer. If the client
    goes away, this generator is cancelled or closed; it then cancels the
    producer and waits until the producer has finished cleaning up (saving
    the answer as cancelled), however many times it is cancelled meanwhile.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        try:
            async for item in events:
                await queue.put(item)
        except BaseException as exc:  # noqa: BLE001 (handed to the consumer below)
            await queue.put(exc)
            raise
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), interval)
            except TimeoutError:
                yield HEARTBEAT
                continue
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                if isinstance(item, asyncio.CancelledError):
                    return
                raise item
            yield item
    finally:
        if not producer.done():
            producer.cancel()
        while not producer.done():
            try:
                # wait() never cancels the producer itself; if WE are
                # cancelled again (anyio re-delivers), just keep waiting.
                await asyncio.wait({producer})
            except asyncio.CancelledError:
                continue
        if not producer.cancelled():
            producer.exception()  # retrieved, so asyncio doesn't warn about it
