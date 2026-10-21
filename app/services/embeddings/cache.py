import hashlib
import logging
from array import array
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import EmbeddingCacheEntry
from app.services.embeddings.base import (
    BatchResult,
    EmbeddingProvider,
    Heartbeat,
    InputType,
    Vector,
)

logger = logging.getLogger(__name__)

# Keys per lookup query; keeps the IN (...) list well within Postgres limits.
LOOKUP_BATCH = 1000


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def as_stored(vector: Vector) -> Vector:
    """The vector as it comes back from the database: pgvector keeps 32-bit
    floats. Returning this form for fresh vectors too means a first run and a
    cached re-run compute exactly the same distances."""
    return array("f", vector).tolist()


@dataclass
class CacheStats:
    hits: int = 0  # texts answered from the cache
    misses: int = 0  # distinct texts sent to the provider


class CachingEmbeddingProvider(EmbeddingProvider):
    """Wraps a provider and remembers every vector it returns, in Postgres.

    The key is (model, input_type, dimensions, sha256 of the text): all of
    them change the vector, so a cached vector is only reused when the
    provider would return the same one. Only texts not yet cached reach the
    wrapped provider, which still applies its own batching, retries and the
    shared rate limiter. New vectors are saved after every request, so an
    interrupted run keeps what it already paid for.

    Usable anywhere an EmbeddingProvider is (the processing pipeline, search).
    """

    def __init__(
        self, inner: EmbeddingProvider, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        super().__init__(model=inner.model, dimensions=inner.dimensions)
        self.inner = inner
        self.session_factory = session_factory
        self.stats = CacheStats()

    async def embed_documents(
        self,
        texts: Sequence[str],
        token_counts: Sequence[int] | None = None,
        heartbeat: Heartbeat | None = None,
    ) -> list[Vector]:
        return await self._cached(texts, "document", token_counts, heartbeat)

    async def embed_queries(
        self, texts: Sequence[str], heartbeat: Heartbeat | None = None
    ) -> list[Vector]:
        return await self._cached(texts, "query", None, heartbeat)

    async def embed_query(self, text: str) -> Vector:
        found = await self._lookup("query", [text_key(text)])
        if (vector := found.get(text_key(text))) is not None:
            self.stats.hits += 1
            return vector
        vector = as_stored(await self.inner.embed_query(text))  # short wait: someone is waiting
        self.stats.misses += 1
        await self._store("query", {text_key(text): vector})
        return vector

    async def _embed_batch(self, texts: list[str], input_type: InputType) -> BatchResult:
        # Never reached: every public method above is overridden and the
        # wrapped provider does the sending.
        raise NotImplementedError

    async def _cached(
        self,
        texts: Sequence[str],
        input_type: InputType,
        token_counts: Sequence[int] | None,
        heartbeat: Heartbeat | None,
    ) -> list[Vector]:
        keys = [text_key(text) for text in texts]
        found = await self._lookup(input_type, keys)

        # Distinct missing texts, each sent once even if it appears twice.
        missing: dict[str, tuple[str, int | None]] = {}
        for index, (key, text) in enumerate(zip(keys, texts, strict=True)):
            if key not in found and key not in missing:
                missing[key] = (text, token_counts[index] if token_counts else None)
        cached = sum(key in found for key in keys)
        self.stats.hits += cached
        self.stats.misses += len(missing)
        if missing:
            logger.info(
                "Embedding cache: %d of %d %s texts cached, %d to embed",
                cached,
                len(texts),
                input_type,
                len(missing),
            )
            found |= await self._embed_missing(input_type, missing, heartbeat)
        return [found[key] for key in keys]

    async def _embed_missing(
        self,
        input_type: InputType,
        missing: dict[str, tuple[str, int | None]],
        heartbeat: Heartbeat | None,
    ) -> dict[str, Vector]:
        """Embed the missing texts one provider request at a time, saving
        each request's vectors before sending the next."""
        texts = [text for text, _ in missing.values()]
        counts = [count for _, count in missing.values()]
        known_counts = counts if all(count is not None for count in counts) else None
        new: dict[str, Vector] = {}
        for batch, _tokens in self.inner._plan_batches(texts, known_counts):
            batch_keys = [text_key(text) for text in batch]
            if input_type == "query":
                vectors = await self.inner.embed_queries(batch, heartbeat=heartbeat)
            else:
                batch_counts = [missing[key][1] for key in batch_keys] if known_counts else None
                vectors = await self.inner.embed_documents(batch, batch_counts, heartbeat)
            fresh = dict(zip(batch_keys, map(as_stored, vectors), strict=True))
            await self._store(input_type, fresh)
            new |= fresh
        return new

    async def _lookup(self, input_type: InputType, keys: Sequence[str]) -> dict[str, Vector]:
        found: dict[str, Vector] = {}
        distinct = list(dict.fromkeys(keys))
        async with self.session_factory() as db:
            for start in range(0, len(distinct), LOOKUP_BATCH):
                rows = await db.execute(
                    select(EmbeddingCacheEntry.text_sha256, EmbeddingCacheEntry.embedding).where(
                        EmbeddingCacheEntry.model == self.model,
                        EmbeddingCacheEntry.input_type == input_type,
                        EmbeddingCacheEntry.dimensions == self.dimensions,
                        EmbeddingCacheEntry.text_sha256.in_(distinct[start : start + LOOKUP_BATCH]),
                    )
                )
                found.update((key, as_stored(vector)) for key, vector in rows)
        return found

    async def _store(self, input_type: InputType, vectors: dict[str, Vector]) -> None:
        if not vectors:
            return
        rows = [
            {
                "model": self.model,
                "input_type": input_type,
                "dimensions": self.dimensions,
                "text_sha256": key,
                "embedding": vector,
            }
            for key, vector in vectors.items()
        ]
        async with self.session_factory() as db:
            # Another run may have stored the same text meanwhile: same key,
            # same vector, so keeping the existing row is correct.
            await db.execute(insert(EmbeddingCacheEntry).values(rows).on_conflict_do_nothing())
            await db.commit()
