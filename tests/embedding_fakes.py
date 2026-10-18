"""Embedding providers for tests: scripted failures, recorded calls, and
vectors with controllable meaning."""

import asyncio
from collections.abc import Callable, Sequence

from app.services.embeddings.base import BatchResult, EmbeddingProvider, InputType
from app.services.embeddings.fake import FakeEmbeddingProvider, fake_vector
from app.services.embeddings.rate_limit import SlidingWindowRateLimiter

DIMENSIONS = 1024


class FakeClock:
    """A clock that only moves when something sleeps on it, so minutes of
    waiting take no real time. Pass `clock` and `sleep` to the code under test."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        await asyncio.sleep(0)  # let other tasks run, as a real sleep would
        self.now += seconds


class RecordingSleep:
    """Stands in for asyncio.sleep: remembers the delays, doesn't wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class ScriptedProvider(EmbeddingProvider):
    """Raises the given exceptions on its first calls, then succeeds.
    Records every request it receives."""

    def __init__(
        self,
        failures: Sequence[Exception] = (),
        now: Callable[[], float] | None = None,
        reported_tokens: Callable[[list[str]], int] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("model", "test-model")
        kwargs.setdefault("dimensions", DIMENSIONS)
        kwargs.setdefault("sleep", RecordingSleep())
        super().__init__(**kwargs)
        self.failures = list(failures)
        self.calls: list[tuple[list[str], InputType]] = []
        self._now = now
        # How many tokens the "provider" says a request used (default: 1 per text).
        self._reported_tokens = reported_tokens or len
        self.times: list[float] = []  # when each request was sent (if a clock was given)

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        self.calls.append((list(texts), input_type))
        if self._now is not None:
            self.times.append(self._now())
        if self.failures:
            raise self.failures.pop(0)
        vectors = [fake_vector(t, self.dimensions) for t in texts]
        return BatchResult(vectors, self._reported_tokens(texts))


def limited_provider(
    clock: FakeClock,
    *,
    tokens: int = 8000,
    requests: int = 2,
    reserve_tokens: int = 0,
    reserve_requests: int = 0,
    provider_cls: type[ScriptedProvider] | None = None,
    **kwargs,
) -> ScriptedProvider:
    """A provider whose every request goes through a rate limiter running on `clock`."""
    limiter = SlidingWindowRateLimiter(
        token_budget=tokens,
        request_budget=requests,
        query_reserve_tokens=reserve_tokens,
        query_reserve_requests=reserve_requests,
        clock=clock.clock,
        sleep=clock.sleep,
    )
    cls = provider_cls or ScriptedProvider
    return cls(limiter=limiter, sleep=clock.sleep, now=clock.clock, **kwargs)


class KeywordProvider(FakeEmbeddingProvider):
    """Vectors with meaning we control: each keyword is its own direction, and
    a text's vector is the sum of the directions of the keywords it contains.
    So "cats" is close to a chunk about cats and far from one about databases.
    """

    def __init__(self, keywords: Sequence[str], **kwargs) -> None:
        kwargs.setdefault("model", "keyword-model")
        kwargs.setdefault("dimensions", DIMENSIONS)
        super().__init__(**kwargs)
        self.keywords = [k.lower() for k in keywords]

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        return BatchResult([self._vector(t) for t in texts], len(texts))

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = text.lower()
        for axis, keyword in enumerate(self.keywords):
            vector[axis] = float(words.count(keyword))
        # A tiny constant on the last axis keeps texts with no keywords from
        # being an all-zero vector (cosine distance is undefined for those).
        vector[-1] = 0.01
        return vector
