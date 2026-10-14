import hashlib
import math
import random

from app.services.embeddings.base import BatchResult, EmbeddingProvider, InputType, Vector


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic vectors derived from a hash of the text; no network.

    The same text always gives the same vector, different texts give
    unrelated vectors, so storage and search plumbing can be exercised
    end to end. The vectors carry no meaning: similar texts are NOT close.
    """

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        vectors = [fake_vector(text, self.dimensions) for text in texts]
        return BatchResult(vectors, total_tokens=sum(len(t.split()) for t in texts))


def fake_vector(text: str, dimensions: int) -> Vector:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.gauss(0, 1) for _ in range(dimensions)]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]
