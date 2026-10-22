"""GET /usage: per-kind totals for today and this month, and the live quota
counters. Owner-scoped: user B never sees user A's numbers."""

import uuid
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.core.redis import get_redis
from app.main import app
from app.models import UsageEvent, UsageKind
from tests.helpers import auth_headers


async def me(client, headers) -> uuid.UUID:
    return uuid.UUID((await client.get("/auth/me", headers=headers)).json()["id"])


def event(user_id, kind, tokens_in, tokens_out=0, created_at=None) -> UsageEvent:
    row = UsageEvent(
        user_id=user_id, kind=kind, tokens_in=tokens_in, tokens_out=tokens_out, model="m"
    )
    if created_at is not None:
        row.created_at = created_at
    return row


async def test_totals_for_today_and_this_month(client, db):
    headers = await auth_headers(client, "alice@example.com")
    uid = await me(client, headers)
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    db.add_all(
        [
            event(uid, UsageKind.embed_document, 1000),
            event(uid, UsageKind.embed_query, 7),
            event(uid, UsageKind.llm_answer, 3000, 200),
            event(uid, UsageKind.llm_answer, 2500, 150),
            # Earlier this month (if today isn't the 1st), and last month.
            event(uid, UsageKind.llm_rewrite, 400, 30, created_at=month_start + timedelta(hours=1)),
            event(uid, UsageKind.llm_answer, 9999, 999, created_at=month_start - timedelta(days=1)),
        ]
    )
    await db.commit()

    response = await client.get("/usage", headers=headers)
    assert response.status_code == 200
    body = response.json()
    today = body["today"]["by_kind"]
    assert today["llm_answer"] == {"requests": 2, "tokens_in": 5500, "tokens_out": 350}
    assert today["embed_document"] == {"requests": 1, "tokens_in": 1000, "tokens_out": 0}
    assert today["embed_query"]["requests"] == 1
    month = body["month"]
    assert month["by_kind"]["llm_rewrite"] == {"requests": 1, "tokens_in": 400, "tokens_out": 30}
    assert month["by_kind"]["llm_answer"]["requests"] == 2  # last month's isn't counted
    assert month["total"]["requests"] == 5
    assert month["total"]["tokens_in"] == 1000 + 7 + 5500 + 400


async def test_every_kind_is_listed_even_when_unused(client):
    headers = await auth_headers(client, "new@example.com")
    body = (await client.get("/usage", headers=headers)).json()
    assert set(body["today"]["by_kind"]) == {k.value for k in UsageKind}
    assert body["today"]["total"] == {"requests": 0, "tokens_in": 0, "tokens_out": 0}


async def test_user_b_cannot_see_user_a(client, db):
    alice = await auth_headers(client, "alice2@example.com")
    bob = await auth_headers(client, "bob2@example.com")
    db.add(event(await me(client, alice), UsageKind.llm_answer, 5000, 500))
    await db.commit()

    bob_view = (await client.get("/usage", headers=bob)).json()
    assert bob_view["today"]["total"]["requests"] == 0
    assert bob_view["month"]["total"]["requests"] == 0
    alice_view = (await client.get("/usage", headers=alice)).json()
    assert alice_view["today"]["total"]["requests"] == 1
    # There is no way to ask for someone else's: no user parameter is accepted.
    sneaky = await client.get(f"/usage?user_id={await me(client, alice)}", headers=bob)
    assert sneaky.json()["today"]["total"]["requests"] == 0


async def test_quotas_show_live_counters(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "DAILY_QUESTIONS_PER_USER", 50)
    headers = await auth_headers(client, "quota@example.com")
    collection_id = (await client.post("/collections", json={"name": "c"}, headers=headers)).json()[
        "id"
    ]
    await client.post(
        f"/collections/{collection_id}/ask", json={"question": "Anything?"}, headers=headers
    )
    quotas = (await client.get("/usage", headers=headers)).json()["quotas"]
    assert quotas["questions"] == {"used": 1, "limit": 50}
    assert quotas["uploads"]["used"] == 0
    assert quotas["resets_at"]


async def test_quotas_null_when_redis_is_down(client):
    from tests.test_rate_limit_api import DownRedis

    headers = await auth_headers(client, "down@example.com")
    app.dependency_overrides[get_redis] = lambda: DownRedis()
    response = await client.get("/usage", headers=headers)
    assert response.status_code == 200
    assert response.json()["quotas"] is None


async def test_requires_authentication(client):
    assert (await client.get("/usage")).status_code == 401
