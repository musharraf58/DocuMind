"""Embedding providers for tests: scripted failures, recorded calls, and
vectors with controllable meaning."""

from collections.abc import Sequence

from app.services.embeddings.base import BatchResult, EmbeddingProvider, InputType
from app.services.embeddings.fake import FakeEmbeddingProvider, fake_vector

DIMENSIONS = 1024


class RecordingSleep:
    """Stands in for asyncio.sleep: remembers the delays, doesn't wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class ScriptedProvider(EmbeddingProvider):
    """Raises the given exceptions on its first calls, then succeeds.
    Records every request it receives."""

    def __init__(self, failures: Sequence[Exception] = (), **kwargs) -> None:
        kwargs.setdefault("model", "test-model")
        kwargs.setdefault("dimensions", DIMENSIONS)
        kwargs.setdefault("sleep", RecordingSleep())
        super().__init__(**kwargs)
        self.failures = list(failures)
        self.calls: list[tuple[list[str], InputType]] = []

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        self.calls.append((list(texts), input_type))
        if self.failures:
            raise self.failures.pop(0)
        return BatchResult([fake_vector(t, self.dimensions) for t in texts], len(texts))


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
