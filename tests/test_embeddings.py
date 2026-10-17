import math
import random

import pytest

from app.services.embeddings import (
    EmbeddingError,
    FakeEmbeddingProvider,
    RequestTooLargeError,
    RetryableEmbeddingError,
)
from app.services.embeddings.base import BatchResult
from tests.embedding_fakes import DIMENSIONS, ScriptedProvider


def texts(n: int) -> list[str]:
    return [f"chunk number {i}" for i in range(n)]


# --- Fake provider ---


async def test_fake_vectors_are_deterministic_and_sized():
    provider = FakeEmbeddingProvider(model="fake", dimensions=DIMENSIONS)
    first = await provider.embed_documents(["alpha", "beta"])
    again = await provider.embed_documents(["alpha", "beta"])
    assert first == again
    assert first[0] != first[1]
    assert all(len(v) == DIMENSIONS for v in first)
    assert math.isclose(sum(x * x for x in first[0]), 1.0)


async def test_fake_respects_configured_dimension():
    provider = FakeEmbeddingProvider(model="fake", dimensions=8)
    assert len(await provider.embed_query("hello")) == 8


# --- Batching ---


async def test_ten_texts_with_batch_size_three_make_four_calls():
    provider = ScriptedProvider(batch_size=3)
    vectors = await provider.embed_documents(texts(10))

    assert [len(batch) for batch, _ in provider.calls] == [3, 3, 3, 1]
    assert [t for batch, _ in provider.calls for t in batch] == texts(10)  # order kept
    assert len(vectors) == 10
    assert all(input_type == "document" for _, input_type in provider.calls)


async def test_batch_size_is_capped_by_provider_limit():
    class SmallLimit(ScriptedProvider):
        max_batch_texts = 4

    provider = SmallLimit(batch_size=64)
    await provider.embed_documents(texts(10))
    assert [len(batch) for batch, _ in provider.calls] == [4, 4, 2]


async def test_batches_stay_within_token_limit():
    class TokenLimited(ScriptedProvider):
        max_batch_tokens = 40  # we use half as a safety margin: 20 tokens per request

    provider = TokenLimited(batch_size=64)
    long_texts = [" ".join(["word"] * 9)] * 6  # ~9 tokens each
    await provider.embed_documents(long_texts)
    assert [len(batch) for batch, _ in provider.calls] == [2, 2, 2]


# --- Token-aware batching ---


def sizes(provider: ScriptedProvider) -> list[int]:
    return [len(batch) for batch, _ in provider.calls]


async def test_batch_ends_when_next_chunk_would_exceed_token_budget():
    provider = ScriptedProvider(batch_size=64, token_budget=1000)
    await provider.embed_documents(texts(5), token_counts=[400] * 5)
    # 400 + 400 fits; a third would make 1200 > 1000.
    assert sizes(provider) == [2, 2, 1]


async def test_batch_may_fill_the_budget_exactly():
    provider = ScriptedProvider(token_budget=1000)
    await provider.embed_documents(texts(4), token_counts=[500, 500, 1000, 1])
    assert sizes(provider) == [2, 1, 1]  # 500+500 = budget; 1000 alone; then 1


async def test_batch_size_still_applies_when_tokens_are_small():
    provider = ScriptedProvider(batch_size=3, token_budget=1000)
    await provider.embed_documents(texts(7), token_counts=[10] * 7)
    assert sizes(provider) == [3, 3, 1]


async def test_given_token_counts_are_used_instead_of_recounting():
    # "tiny" is one token to tiktoken, but the caller says each is 900.
    provider = ScriptedProvider(token_budget=1000)
    await provider.embed_documents(["tiny", "tiny", "tiny"], token_counts=[900, 900, 900])
    assert sizes(provider) == [1, 1, 1]


async def test_tokens_are_counted_when_the_caller_gives_no_counts():
    provider = ScriptedProvider(token_budget=20)
    nine_tokens = " ".join(["word"] * 9)
    await provider.embed_documents([nine_tokens] * 6)
    assert sizes(provider) == [2, 2, 2]


async def test_chunk_larger_than_budget_is_refused_before_any_request():
    provider = ScriptedProvider(token_budget=1000)
    with pytest.raises(RequestTooLargeError, match=r"1500 tokens; at most 1000") as info:
        await provider.embed_documents(texts(3), token_counts=[300, 300, 1500])
    assert provider.calls == []  # not even the chunks before it were sent
    assert "CHUNK_TOKENS" in str(info.value)


async def test_without_a_token_budget_nothing_is_counted():
    provider = ScriptedProvider(batch_size=2)
    assert provider.token_budget is None
    await provider.embed_documents(texts(5), token_counts=[10**9] * 5)
    assert sizes(provider) == [2, 2, 1]


async def test_mismatched_token_counts_are_a_bug():
    provider = ScriptedProvider(token_budget=1000)
    with pytest.raises(ValueError, match="one entry per text"):
        await provider.embed_documents(texts(3), token_counts=[1, 2])


@pytest.mark.parametrize("seed", range(10))
async def test_no_batch_ever_exceeds_the_budget(seed):
    rng = random.Random(seed)
    budget = rng.randint(200, 2000)
    counts = [rng.randint(1, budget) for _ in range(rng.randint(1, 60))]
    provider = ScriptedProvider(batch_size=rng.randint(1, 20), token_budget=budget)
    await provider.embed_documents(texts(len(counts)), token_counts=counts)

    position = 0
    for batch, _ in provider.calls:
        batch_counts = counts[position : position + len(batch)]
        assert sum(batch_counts) <= budget
        assert len(batch) <= provider.batch_size
        position += len(batch)
    assert position == len(counts)  # everything sent exactly once, in order


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
async def test_blank_text_is_never_sent(bad):
    provider = ScriptedProvider()
    with pytest.raises(ValueError):
        await provider.embed_documents(["fine", bad])
    with pytest.raises(ValueError):
        await provider.embed_query(bad)
    assert provider.calls == []


async def test_query_uses_query_input_type():
    provider = ScriptedProvider()
    vector = await provider.embed_query("what is pgvector?")
    assert provider.calls == [(["what is pgvector?"], "query")]
    assert len(vector) == DIMENSIONS


# --- Retries ---


async def test_retry_succeeds_after_two_retryable_failures():
    provider = ScriptedProvider(
        failures=[RetryableEmbeddingError("rate limited"), RetryableEmbeddingError("503")]
    )
    vectors = await provider.embed_documents(texts(2))

    assert len(vectors) == 2
    assert len(provider.calls) == 3
    delays = provider._sleep.delays
    assert len(delays) == 2
    assert 0.5 <= delays[0] <= 1.0  # ~1s with jitter
    assert 1.0 <= delays[1] <= 2.0  # ~2s with jitter


async def test_gives_up_after_max_retries():
    failures = [RetryableEmbeddingError("Embedding service is unavailable")] * 10
    provider = ScriptedProvider(failures=failures, max_retries=3)
    with pytest.raises(RetryableEmbeddingError, match="unavailable"):
        await provider.embed_documents(texts(1))
    assert len(provider.calls) == 4  # first try + 3 retries
    assert len(provider._sleep.delays) == 3


async def test_non_retryable_error_fails_at_once():
    provider = ScriptedProvider(failures=[EmbeddingError("rejected the API key")])
    with pytest.raises(EmbeddingError, match="API key"):
        await provider.embed_documents(texts(5))
    assert len(provider.calls) == 1
    assert provider._sleep.delays == []


async def test_retry_after_hint_is_respected():
    provider = ScriptedProvider(failures=[RetryableEmbeddingError("429", retry_after=7)])
    await provider.embed_documents(texts(1))
    assert provider._sleep.delays == [7]


async def test_backoff_is_capped():
    delays = [ScriptedProvider._backoff(attempt, None) for attempt in range(20)]
    assert max(delays) <= 30


# --- Response checks ---


class FixedOutput(ScriptedProvider):
    """Returns whatever vectors it's given, regardless of its settings."""

    def __init__(self, vectors, **kwargs):
        super().__init__(**kwargs)
        self.vectors = vectors

    async def _embed_batch(self, texts, input_type):
        return BatchResult(self.vectors, 0)


async def test_wrong_dimension_from_provider_is_an_error():
    provider = FixedOutput([[0.1] * 1024], dimensions=512)
    with pytest.raises(EmbeddingError, match="wrong size"):
        await provider.embed_documents(texts(1))


async def test_wrong_vector_count_from_provider_is_an_error():
    provider = FixedOutput([[0.1] * DIMENSIONS], dimensions=DIMENSIONS)
    with pytest.raises(EmbeddingError, match="unexpected response"):
        await provider.embed_documents(texts(2))
