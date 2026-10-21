"""The embedding cache: a text is sent to the provider at most once per
(model, input type, dimensions)."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import EmbeddingCacheEntry
from app.services.embeddings.cache import CachingEmbeddingProvider, as_stored
from app.services.embeddings.errors import EmbeddingError
from app.services.embeddings.fake import fake_vector
from tests.embedding_fakes import DIMENSIONS, FakeClock, ScriptedProvider, limited_provider


@pytest.fixture
def session_factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


def cached(inner, session_factory):
    return CachingEmbeddingProvider(inner, session_factory)


def sent(provider: ScriptedProvider) -> list[str]:
    return [text for texts, _ in provider.calls for text in texts]


async def test_second_run_makes_no_provider_calls(session_factory):
    inner = ScriptedProvider()
    texts = ["alpha", "beta", "gamma"]

    first = await cached(inner, session_factory).embed_documents(texts)
    again = cached(inner, session_factory)  # a new process: only the table is shared
    second = await again.embed_documents(texts)

    assert sent(inner) == texts
    assert second == first == [as_stored(fake_vector(t, DIMENSIONS)) for t in texts]
    assert (again.stats.hits, again.stats.misses, again.stats.requests) == (3, 0, 0)


async def test_only_new_texts_are_embedded_and_order_is_kept(session_factory):
    inner = ScriptedProvider()
    provider = cached(inner, session_factory)
    await provider.embed_documents(["one", "two"])
    inner.calls.clear()

    vectors = await provider.embed_documents(["two", "three", "one", "four"])

    assert sent(inner) == ["three", "four"]
    expected = [as_stored(fake_vector(t, DIMENSIONS)) for t in ["two", "three", "one", "four"]]
    assert vectors == expected


async def test_query_and_document_vectors_are_cached_separately(session_factory):
    inner = ScriptedProvider()
    provider = cached(inner, session_factory)
    await provider.embed_documents(["refund policy"])
    await provider.embed_query("refund policy")
    await provider.embed_queries(["refund policy"])

    assert inner.calls == [(["refund policy"], "document"), (["refund policy"], "query")]


async def test_another_model_does_not_reuse_vectors(session_factory):
    await cached(ScriptedProvider(model="model-a"), session_factory).embed_documents(["x"])
    other_model = ScriptedProvider(model="model-b")
    await cached(other_model, session_factory).embed_documents(["x"])
    assert sent(other_model) == ["x"]


async def test_fresh_vectors_equal_what_a_cached_rerun_returns(session_factory):
    # pgvector stores 32-bit floats; a first run must not see slightly
    # different numbers (and so distances) than every later, cached run.
    fresh = await cached(ScriptedProvider(), session_factory).embed_documents(["x"])
    again = await cached(ScriptedProvider(), session_factory).embed_documents(["x"])
    assert fresh == again


async def test_repeated_text_in_one_call_is_sent_once(session_factory):
    inner = ScriptedProvider()
    vectors = await cached(inner, session_factory).embed_documents(["same", "same", "other"])
    assert sent(inner) == ["same", "other"]
    assert vectors[0] == vectors[1]


async def test_queries_are_batched_into_few_requests(session_factory):
    inner = ScriptedProvider(batch_size=10)
    questions = [f"question {n}?" for n in range(25)]
    vectors = await cached(inner, session_factory).embed_queries(questions)
    assert len(inner.calls) == 3 and {kind for _, kind in inner.calls} == {"query"}
    assert vectors == [as_stored(fake_vector(q, DIMENSIONS)) for q in questions]


async def test_vectors_paid_for_survive_a_failed_run(session_factory):
    inner = ScriptedProvider(batch_size=2)
    texts = ["a", "b", "c", "d"]

    class FailsOnSecondRequest(ScriptedProvider):
        async def _embed_batch(self, texts, input_type):
            if len(self.calls) == 1:
                self.calls.append((list(texts), input_type))
                raise EmbeddingError("provider down")
            return await super()._embed_batch(texts, input_type)

    with pytest.raises(EmbeddingError):
        await cached(FailsOnSecondRequest(batch_size=2), session_factory).embed_documents(texts)

    await cached(inner, session_factory).embed_documents(texts)
    assert sent(inner) == ["c", "d"]  # a and b were saved before the failure


async def test_misses_still_go_through_the_rate_limiter(session_factory):
    clock = FakeClock()
    inner = limited_provider(clock, requests=1, batch_size=1)
    provider = cached(inner, session_factory)

    await provider.embed_documents(["a", "b"])
    assert inner.times[1] - inner.times[0] >= 60  # one request per minute

    clock.sleeps.clear()
    await provider.embed_documents(["a", "b"])
    assert clock.sleeps == []  # nothing to send, nothing to wait for


async def test_live_query_is_cached(session_factory):
    inner = ScriptedProvider()
    provider = cached(inner, session_factory)
    assert await provider.embed_query("where is my parcel?") == await provider.embed_query(
        "where is my parcel?"
    )
    assert len(inner.calls) == 1


async def test_cache_stores_one_row_per_distinct_key(session_factory, db):
    provider = cached(ScriptedProvider(), session_factory)
    await provider.embed_documents(["a", "b", "a"])
    await provider.embed_documents(["a"])
    assert await db.scalar(select(func.count()).select_from(EmbeddingCacheEntry)) == 2
