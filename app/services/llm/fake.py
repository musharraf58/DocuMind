import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from app.services.chunking import count_tokens
from app.services.llm.base import LLMProvider, LLMResult, LLMUsage

DEFAULT_ANSWER = "Here is what the sources say [1]."


class FakeLLMProvider(LLMProvider):
    """Scripted answers, no network. Records every call so tests can check
    what was sent, or that nothing was sent at all.

    answer: fixed text, or a function (system, user) -> text.
    failures: exceptions to raise at the start of the next calls, in order.

    Streaming controls:
    chunks: the pieces to stream (default: the answer, word by word).
    chunk_delay / chunk_sleep: wait before each piece (pass a fake clock's sleep).
    fail_after / fail_with: raise fail_with after that many pieces were sent.
    pause_after: after that many pieces, set `paused` and wait for `resume`,
        so a test can act mid-stream (disconnect, inspect the database...).
    """

    def __init__(
        self,
        answer: str | Callable[[str, str], str] = DEFAULT_ANSWER,
        failures: Sequence[Exception] = (),
        chunks: Sequence[str] | None = None,
        chunk_delay: float = 0.0,
        chunk_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        fail_after: int | None = None,
        fail_with: Exception | None = None,
        pause_after: int | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("model", "fake-llm")
        super().__init__(**kwargs)
        self.answer = answer
        self.failures = list(failures)
        self.chunks = chunks
        self.chunk_delay = chunk_delay
        self.chunk_sleep = chunk_sleep
        self.fail_after = fail_after
        self.fail_with = fail_with
        self.pause_after = pause_after
        self.paused = asyncio.Event()
        self.resume = asyncio.Event()
        self.calls: list[tuple[str, str, int]] = []
        self.streams_completed = 0
        self.streams_cancelled = 0

    def _text(self, system: str, user: str) -> str:
        return self.answer(system, user) if callable(self.answer) else self.answer

    async def _generate(self, system: str, user: str, max_tokens: int) -> LLMResult:
        self.calls.append((system, user, max_tokens))
        if self.failures:
            raise self.failures.pop(0)
        text = self._text(system, user)
        return LLMResult(text, count_tokens(system) + count_tokens(user), count_tokens(text))

    async def _stream(
        self, system: str, user: str, max_tokens: int
    ) -> AsyncIterator[str | LLMUsage]:
        self.calls.append((system, user, max_tokens))
        if self.failures:
            raise self.failures.pop(0)
        text = self._text(system, user)
        pieces = list(self.chunks) if self.chunks is not None else re.findall(r"\S+\s*", text)
        sent = 0
        try:
            for piece in pieces:
                if self.fail_after is not None and sent >= self.fail_after:
                    raise self.fail_with or RuntimeError("fake failure")
                if self.pause_after is not None and sent == self.pause_after:
                    self.paused.set()
                    await self.resume.wait()
                if self.chunk_delay:
                    await self.chunk_sleep(self.chunk_delay)
                yield piece
                sent += 1
            yield LLMUsage(count_tokens(system) + count_tokens(user), count_tokens("".join(pieces)))
            self.streams_completed += 1
        except (asyncio.CancelledError, GeneratorExit):
            # Closed early: by the consumer stopping, or its task being cancelled.
            self.streams_cancelled += 1
            raise
