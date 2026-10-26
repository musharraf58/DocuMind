"""Daily quotas and the global LLM budget (fake Redis, fake clock, fake providers)."""

# The embedder/llm fixtures are imported from test_ask_api and then requested
# by name, which ruff reads as redefinitions.
# ruff: noqa: F811

from datetime import UTC, datetime, timedelta

import fakeredis
import pytest

from app.api.deps import llm_provider
from app.core import clock
from app.core.config import get_settings
from app.core.redis import get_redis
from app.main import app
from app.services.llm import FakeLLMProvider, GlobalLLMBudget, LLMBudgetExhausted
from app.services.quotas import Quotas, day_start, next_reset, quota_day
from tests.embedding_fakes import RecordingSleep
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload
from tests.test_ask_api import about, ask, embedder, llm, setup  # noqa: F401 (fixtures)
from tests.test_ask_stream import stream


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def fake_clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(clock, "utcnow", fake)
    return fake


def set_quota(monkeypatch, name: str, value: int) -> None:
    monkeypatch.setattr(get_settings(), name, value)


def fake_redis():
    return app.dependency_overrides[get_redis]()


# --- the quota day ---------------------------------------------------------------


def test_quota_day_starts_at_the_reset_hour():
    at = datetime(2026, 10, 22, 3, 30, tzinfo=UTC)
    assert quota_day(at, 0).isoformat() == "2026-10-22"
    assert quota_day(at, 4).isoformat() == "2026-10-21"  # before 04:00: still yesterday
    assert next_reset(at, 4) == datetime(2026, 10, 22, 4, tzinfo=UTC)
    assert next_reset(at, 0) == datetime(2026, 10, 23, 0, tzinfo=UTC)
    assert day_start(at, 4) == datetime(2026, 10, 21, 4, tzinfo=UTC)


# --- questions per day -------------------------------------------------------------


async def test_questions_per_day(client, embedder, llm, monkeypatch, fake_clock):
    set_quota(monkeypatch, "DAILY_QUESTIONS_PER_USER", 2)
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    for _ in range(2):
        assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200
    blocked = await ask(client, headers, collection_id, "What do cats do?")
    assert blocked.status_code == 429
    body = blocked.json()
    assert body["code"] == "daily_questions_quota_reached"
    assert body["limit"] == 2
    assert int(blocked.headers["Retry-After"]) == 12 * 3600  # until midnight UTC
    # The stream counts against the same quota.
    assert (await stream(headers, collection_id, "What do cats do?")).status == 429

    fake_clock.advance(hours=12)  # a new quota day
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200


async def test_quotas_are_per_user(client, embedder, llm, monkeypatch, fake_clock):
    set_quota(monkeypatch, "DAILY_QUESTIONS_PER_USER", 1)
    alice, alice_collection = await setup(client, {"cats.txt": about("cats")}, "a@example.com")
    bob, bob_collection = await setup(client, {"cats.txt": about("cats")}, "b@example.com")
    assert (await ask(client, alice, alice_collection, "What do cats do?")).status_code == 200
    assert (await ask(client, alice, alice_collection, "What do cats do?")).status_code == 429
    assert (await ask(client, bob, bob_collection, "What do cats do?")).status_code == 200


async def test_a_404_question_does_not_use_the_quota(client, embedder, llm, monkeypatch):
    set_quota(monkeypatch, "DAILY_QUESTIONS_PER_USER", 1)
    import uuid

    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    assert (await ask(client, headers, str(uuid.uuid4()), "What?")).status_code == 404
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200


async def test_reset_hour_is_configurable(client, embedder, llm, monkeypatch, fake_clock):
    set_quota(monkeypatch, "DAILY_QUESTIONS_PER_USER", 1)
    set_quota(monkeypatch, "QUOTA_RESET_HOUR_UTC", 14)
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200
    blocked = await ask(client, headers, collection_id, "What do cats do?")
    assert int(blocked.headers["Retry-After"]) == 2 * 3600  # 12:00 -> 14:00
    fake_clock.advance(hours=2)
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200


async def test_counter_keys_expire_by_themselves(client, embedder, llm, fake_clock):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    await ask(client, headers, collection_id, "What do cats do?")
    keys = [key async for key in fake_redis().scan_iter(match="documind:quota:*")]
    assert keys
    for key in keys:
        assert 0 < await fake_redis().ttl(key) <= 48 * 3600


# --- uploads per day -------------------------------------------------------------------


async def test_uploads_per_day(client, monkeypatch, fake_clock):
    set_quota(monkeypatch, "DAILY_UPLOADS_PER_USER", 2)
    headers = await auth_headers(client, "up@example.com")
    collection_id = await create_collection(client, headers)
    for i in range(2):
        response = await upload(client, headers, collection_id, f"{i}.txt", f"text {i}".encode())
        assert response.status_code == 202
    blocked = await upload(client, headers, collection_id, "3.txt", b"text 3")
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "daily_uploads_quota_reached"


async def test_refused_uploads_do_not_count(client, monkeypatch):
    set_quota(monkeypatch, "DAILY_UPLOADS_PER_USER", 1)
    headers = await auth_headers(client, "refused@example.com")
    collection_id = await create_collection(client, headers)
    assert (await upload(client, headers, collection_id, "x.exe", b"MZ...")).status_code == 415
    assert (await upload(client, headers, collection_id, "ok.txt", b"fine text")).status_code == 202


# --- embedding tokens per day ---------------------------------------------------------------


async def test_embedding_tokens_are_counted_from_provider_reports(client, embedder, llm):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    quotas = Quotas(fake_redis())
    me = (await client.get("/auth/me", headers=headers)).json()["id"]
    import uuid

    used = await quotas.used(uuid.UUID(me), "embed_tokens")
    assert used > 0
    await client.post(
        f"/collections/{collection_id}/search", json={"query": "cats"}, headers=headers
    )
    assert await quotas.used(uuid.UUID(me), "embed_tokens") > used


async def test_out_of_embedding_tokens_blocks_search_ask_and_upload(
    client, embedder, llm, monkeypatch
):
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    import uuid

    me = uuid.UUID((await client.get("/auth/me", headers=headers)).json()["id"])
    used = await Quotas(fake_redis()).used(me, "embed_tokens")
    set_quota(monkeypatch, "DAILY_EMBED_TOKENS_PER_USER", used)  # exactly used up

    search = await client.post(
        f"/collections/{collection_id}/search", json={"query": "cats"}, headers=headers
    )
    assert search.status_code == 429
    assert search.json()["code"] == "daily_embed_tokens_quota_reached"
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 429
    assert (
        await upload(client, headers, collection_id, "more.txt", b"more text")
    ).status_code == 429


async def test_worker_refuses_a_document_that_does_not_fit_today(client, db, monkeypatch):
    from app.models import Document, DocumentStatus

    set_quota(monkeypatch, "DAILY_EMBED_TOKENS_PER_USER", 50)
    headers = await auth_headers(client, "big@example.com")
    collection_id = await create_collection(client, headers)
    text = b"This sentence is long enough to pass the minimum text check. " * 50
    response = await upload(client, headers, collection_id, "big.txt", text)
    assert response.status_code == 202
    [result] = await run_queued_jobs()
    assert result.outcome == "failed"
    document = await db.get(Document, response.json()["id"], populate_existing=True)
    assert document.status == DocumentStatus.failed
    assert "Daily embedding limit reached" in document.error_message
    assert "Reprocess it after 00:00 UTC" in document.error_message


# --- the global LLM budget -------------------------------------------------------------------


async def test_global_budget_counts_all_users_and_search_keeps_working(
    client, embedder, monkeypatch, fake_clock
):
    """A budget of 2 LLM requests per day for everyone: the third question
    (from a different user) gets 503, while search still works."""
    set_quota(monkeypatch, "GLOBAL_DAILY_LLM_BUDGET", 2)
    budget = GlobalLLMBudget(fake_redis())
    fake = FakeLLMProvider(answer="Cats purr [1].", sleep=RecordingSleep(), budget=budget)
    app.dependency_overrides[llm_provider] = lambda: fake

    alice, alice_collection = await setup(client, {"cats.txt": about("cats")}, "a@example.com")
    bob, bob_collection = await setup(client, {"cats.txt": about("cats")}, "b@example.com")
    assert (await ask(client, alice, alice_collection, "What do cats do?")).status_code == 200
    assert (await ask(client, alice, alice_collection, "What do cats do?")).status_code == 200

    response = await ask(client, bob, bob_collection, "What do cats do?")
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "daily_capacity_reached"
    assert body["detail"] == "Daily capacity reached; please try again tomorrow"
    assert int(response.headers["Retry-After"]) == 12 * 3600
    assert len(fake.calls) == 2  # the provider was never asked a third time

    # The stream answers the same way, before streaming starts.
    assert (await stream(bob, bob_collection, "What do cats do?")).status == 503
    # Search needs no LLM: it keeps working.
    search = await client.post(
        f"/collections/{bob_collection}/search", json={"query": "cats"}, headers=bob
    )
    assert search.status_code == 200

    fake_clock.advance(hours=12)  # tomorrow
    assert (await ask(client, bob, bob_collection, "What do cats do?")).status_code == 200


async def test_a_capacity_503_does_not_use_the_users_question_quota(
    client, embedder, llm, monkeypatch
):
    set_quota(monkeypatch, "GLOBAL_DAILY_LLM_BUDGET", 0)
    set_quota(monkeypatch, "DAILY_QUESTIONS_PER_USER", 1)
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 503
    set_quota(monkeypatch, "GLOBAL_DAILY_LLM_BUDGET", 10)
    assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200


async def test_budget_runs_out_mid_question(client, embedder, monkeypatch, fake_clock):
    """The entry check passed, but the last unit went to someone else: the
    LLM client itself refuses, and the answer is a 503 too."""
    set_quota(monkeypatch, "GLOBAL_DAILY_LLM_BUDGET", 1)
    budget = GlobalLLMBudget(fake_redis())
    fake = FakeLLMProvider(answer="Cats purr [1].", sleep=RecordingSleep(), budget=budget)
    app.dependency_overrides[llm_provider] = lambda: fake
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    original = budget.reserve

    async def someone_else_first(keep=0):
        await Quotas(fake_redis()).take_global_llm()  # another request takes the last unit
        await original(keep)

    monkeypatch.setattr(budget, "reserve", someone_else_first)
    response = await ask(client, headers, collection_id, "What do cats do?")
    assert response.status_code == 503
    assert response.json()["code"] == "daily_capacity_reached"
    assert fake.calls == []


async def test_rewrite_never_takes_the_last_unit(client, db, embedder, monkeypatch, fake_clock):
    """A follow-up costs two requests (rewrite + answer). With only one left,
    the rewrite is skipped (it falls back to the original question) and the
    one request goes to the answer, instead of the rewrite using it up and
    the answer getting a 503."""
    from sqlalchemy import select

    from app.models import UsageEvent, UsageKind

    set_quota(monkeypatch, "GLOBAL_DAILY_LLM_BUDGET", 2)
    budget = GlobalLLMBudget(fake_redis())
    fake = FakeLLMProvider(answer="Cats purr [1].", sleep=RecordingSleep(), budget=budget)
    app.dependency_overrides[llm_provider] = lambda: fake
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    first = await ask(client, headers, collection_id, "What do cats do?")
    assert first.status_code == 200  # 1 of 2 used

    follow = await client.post(
        f"/collections/{collection_id}/ask",
        # Mentions cats, so the original question still finds the source.
        json={"question": "And why do cats?", "conversation_id": first.json()["conversation_id"]},
        headers=headers,
    )
    assert follow.status_code == 200
    assert len(fake.calls) == 2  # the two answers; the rewrite never reached the provider
    kinds = [e.kind for e in (await db.scalars(select(UsageEvent))).all()]
    assert kinds.count(UsageKind.llm_answer) == 2
    assert UsageKind.llm_rewrite not in kinds


async def test_rewrite_runs_when_two_units_are_left(client, embedder, monkeypatch, fake_clock):
    set_quota(monkeypatch, "GLOBAL_DAILY_LLM_BUDGET", 3)
    budget = GlobalLLMBudget(fake_redis())
    fake = FakeLLMProvider(answer="Cats purr [1].", sleep=RecordingSleep(), budget=budget)
    app.dependency_overrides[llm_provider] = lambda: fake
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    first = await ask(client, headers, collection_id, "What do cats do?")
    follow = await client.post(
        f"/collections/{collection_id}/ask",
        json={"question": "And why?", "conversation_id": first.json()["conversation_id"]},
        headers=headers,
    )
    assert follow.status_code == 200
    assert len(fake.calls) == 3  # answer, rewrite, answer


async def test_budget_reserve_unit(fake_clock):
    redis = fakeredis.aioredis.FakeRedis()
    settings = get_settings().model_copy(update={"GLOBAL_DAILY_LLM_BUDGET": 2})
    budget = GlobalLLMBudget(redis, settings)
    await budget.reserve()
    await budget.reserve()
    with pytest.raises(LLMBudgetExhausted) as caught:
        await budget.reserve()
    assert caught.value.retry_after == 12 * 3600
    assert await Quotas(redis, settings).global_llm_used() == 2  # the refusal took nothing


async def test_retries_count_against_the_budget(fake_clock):
    from app.services.llm import RetryableLLMError

    redis = fakeredis.aioredis.FakeRedis()
    settings = get_settings().model_copy(update={"GLOBAL_DAILY_LLM_BUDGET": 10})
    fake = FakeLLMProvider(
        failures=[RetryableLLMError("timed out")],
        sleep=RecordingSleep(),
        budget=GlobalLLMBudget(redis, settings),
    )
    await fake.generate("s", "u", 10)
    assert await Quotas(redis, settings).global_llm_used() == 2  # the failed try + the retry


async def test_redis_down_quotas_fail_open(client, embedder, llm, monkeypatch, caplog):
    from tests.test_rate_limit_api import DownRedis

    set_quota(monkeypatch, "DAILY_QUESTIONS_PER_USER", 1)
    headers, collection_id = await setup(client, {"cats.txt": about("cats")})
    app.dependency_overrides[get_redis] = lambda: DownRedis()
    for _ in range(3):
        assert (await ask(client, headers, collection_id, "What do cats do?")).status_code == 200
