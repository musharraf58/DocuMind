"""The demo account: seeded by scripts/seed_demo.py, read-only in the API."""

import shutil
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.models import Collection, Conversation, Document, DocumentStatus, User
from app.services.embeddings import FakeEmbeddingProvider
from app.services.embeddings.cache import CachingEmbeddingProvider
from app.services.storage import get_storage
from scripts.seed_demo import CORPUS_DIR, SeedError, seed
from tests.helpers import files_on_disk, login

EMAIL = "demo@example.com"
PASSWORD = "a-demo-passphrase-for-tests"
FAKE_MODEL = "fake-sha256"  # what the API embeds queries with in tests


class CountingProvider(FakeEmbeddingProvider):
    """The fake provider, counting the texts that really reach it."""

    def __init__(self) -> None:
        super().__init__(model=FAKE_MODEL, dimensions=get_settings().EMBEDDING_DIMENSIONS)
        self.texts = 0

    async def _embed_batch(self, texts, input_type):
        self.texts += len(texts)
        return await super()._embed_batch(texts, input_type)


@pytest.fixture
def sessions(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


async def run_seed(sessions, provider=None, corpus=CORPUS_DIR, password=PASSWORD):
    provider = provider or CountingProvider()
    embedder = CachingEmbeddingProvider(provider, sessions)
    return await seed(sessions, get_storage(), embedder, EMAIL, password, corpus), provider


@pytest.fixture
async def demo(client, sessions) -> dict:
    """The seeded demo account, logged in."""
    await run_seed(sessions)
    token = (await login(client, EMAIL, PASSWORD)).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    collection = (await client.get("/collections", headers=headers)).json()[0]
    documents = (
        await client.get(f"/collections/{collection['id']}/documents", headers=headers)
    ).json()
    ask = await client.post(
        f"/collections/{collection['id']}/ask",
        json={"question": "What grind size for a French press?"},
        headers=headers,
    )
    assert ask.status_code == 200
    return {
        "headers": headers,
        "collection_id": collection["id"],
        "document_id": documents[0]["id"],
        "conversation_id": ask.json()["conversation_id"],
    }


# --- seeding --------------------------------------------------------------------


async def test_seed_creates_the_demo_account_and_its_collection(db, sessions, storage_dir):
    report, provider = await run_seed(sessions)
    assert report.created_user
    corpus = sorted(path.name for path in CORPUS_DIR.glob("*.md"))
    assert sorted(report.added) == corpus and not report.failed
    user = await db.scalar(select(User).where(User.email == EMAIL))
    assert user.is_demo
    documents = (await db.scalars(select(Document).where(Document.owner_id == user.id))).all()
    assert {d.original_filename for d in documents} == set(corpus)
    assert all(d.status == DocumentStatus.ready for d in documents)
    assert len(files_on_disk(storage_dir)) == len(corpus)
    assert provider.texts > 0


async def test_seeding_again_changes_nothing_and_calls_no_provider(db, sessions):
    await run_seed(sessions)
    report, provider = await run_seed(sessions)
    assert not report.created_user and not report.password_changed
    assert report.added == report.replaced == report.removed == []
    assert len(report.unchanged) == len(list(CORPUS_DIR.glob("*.md")))
    assert provider.texts == 0


async def test_reseeding_from_scratch_reuses_the_embedding_cache(db, sessions):
    """The account and its documents are gone, but every chunk's vector is
    still in the cache: building it all again costs no provider call."""
    _, first = await run_seed(sessions)
    assert first.texts > 0
    user = await db.scalar(select(User).where(User.email == EMAIL))
    await db.delete(user)
    await db.commit()

    report, second = await run_seed(sessions)
    assert report.created_user and not report.failed
    assert second.texts == 0


async def test_the_collection_follows_the_corpus_folder(db, sessions, tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS_DIR, corpus)
    await run_seed(sessions, corpus=corpus)

    (corpus / "bicycle-maintenance.md").unlink()
    (corpus / "coffee-brewing.md").write_text("# Coffee\n\nNow about tea, really. " * 20)
    (corpus / "new-notes.txt").write_text("Brand new notes about kettles. " * 20)
    report, _ = await run_seed(sessions, corpus=corpus)

    assert report.removed == ["bicycle-maintenance.md"]
    assert report.replaced == ["coffee-brewing.md"]
    assert report.added == ["new-notes.txt"]
    names = set(await db.scalars(select(Document.original_filename)))
    assert names == {path.name for path in corpus.iterdir()}


async def test_a_new_password_replaces_the_old_and_ends_old_sessions(client, sessions):
    await run_seed(sessions)
    old_token = (await login(client, EMAIL, PASSWORD)).json()["access_token"]
    report, _ = await run_seed(sessions, password="another-demo-passphrase")
    assert report.password_changed
    assert (await login(client, EMAIL, PASSWORD)).status_code == 401
    assert (await login(client, EMAIL, "another-demo-passphrase")).status_code == 200
    old = await client.get("/auth/me", headers={"Authorization": f"Bearer {old_token}"})
    assert old.status_code == 401


async def test_an_ordinary_account_is_never_turned_into_the_demo(client, sessions, db):
    await client.post("/auth/register", json={"email": EMAIL, "password": "someone-elses-pw-1"})
    with pytest.raises(SeedError, match="ordinary account"):
        await run_seed(sessions)
    user = await db.scalar(select(User).where(User.email == EMAIL))
    assert not user.is_demo


async def test_a_weak_password_is_refused_without_being_echoed(sessions):
    with pytest.raises(SeedError) as caught:
        await run_seed(sessions, password="short")
    assert "short" not in str(caught.value).replace("too short", "")


# --- read-only in the API ---------------------------------------------------------

FORBIDDEN = [
    ("upload", "POST", "/collections/{collection_id}/documents"),
    ("delete document", "DELETE", "/documents/{document_id}"),
    ("reprocess", "POST", "/documents/{document_id}/reprocess"),
    ("create collection", "POST", "/collections"),
    ("rename collection", "PATCH", "/collections/{collection_id}"),
    ("delete collection", "DELETE", "/collections/{collection_id}"),
    ("rename conversation", "PATCH", "/conversations/{conversation_id}"),
    ("delete conversation", "DELETE", "/conversations/{conversation_id}"),
    ("change password", "POST", "/auth/change-password"),
    ("delete account", "DELETE", "/auth/me"),
]
BODIES = {
    "upload": {"files": {"file": ("x.txt", b"vandalism " * 20)}},
    "create collection": {"json": {"name": "Mine now"}},
    "rename collection": {"json": {"name": "Renamed"}},
    "rename conversation": {"json": {"title": "Renamed"}},
    "change password": {
        "json": {"current_password": PASSWORD, "new_password": "a-brand-new-passphrase"}
    },
    "delete account": {"json": {"password": PASSWORD}},
}


async def snapshot(db) -> tuple:
    counts = []
    for model in (User, Collection, Document, Conversation):
        counts.append(await db.scalar(select(func.count()).select_from(model)))
    names = tuple(sorted(await db.scalars(select(Collection.name))))
    user = await db.scalar(select(User).where(User.email == EMAIL))
    return (*counts, names, user.hashed_password, user.token_version)


@pytest.mark.parametrize(("action", "method", "path"), FORBIDDEN, ids=[f[0] for f in FORBIDDEN])
async def test_every_change_is_refused(client, demo, db, storage_dir, action, method, path):
    before = await snapshot(db)
    files_before = files_on_disk(storage_dir)
    url = path.format(**demo)
    response = await client.request(method, url, headers=demo["headers"], **BODIES.get(action, {}))
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "demo_read_only"
    assert "read-only" in response.json()["detail"]
    db.expire_all()
    assert await snapshot(db) == before
    assert files_on_disk(storage_dir) == files_before


async def test_reading_searching_and_asking_work(client, demo):
    h, c, d = demo["headers"], demo["collection_id"], demo["document_id"]
    me = (await client.get("/auth/me", headers=h)).json()
    assert me["is_demo"] is True
    for url in (
        "/collections",
        f"/collections/{c}",
        f"/collections/{c}/documents",
        f"/documents/{d}",
        f"/documents/{d}/download",
        f"/documents/{d}/chunks",
        "/conversations",
        f"/conversations/{demo['conversation_id']}",
        "/usage",
    ):
        assert (await client.get(url, headers=h)).status_code == 200, url
    search = await client.post(f"/collections/{c}/search", json={"query": "chain"}, headers=h)
    assert search.status_code == 200 and search.json()
    question = {"question": "How often should I oil a bike chain?"}
    assert (await client.post(f"/collections/{c}/ask", json=question, headers=h)).status_code == 200
    async with client.stream("POST", f"/collections/{c}/ask/stream", json=question, headers=h) as r:
        assert r.status_code == 200


async def test_the_demo_accounts_quotas_still_apply(client, demo, monkeypatch):
    monkeypatch.setattr(get_settings(), "DAILY_QUESTIONS_PER_USER", 2)  # one used by the fixture
    url = f"/collections/{demo['collection_id']}/ask"
    question = {"question": "Is light important for houseplants?"}
    assert (await client.post(url, json=question, headers=demo["headers"])).status_code == 200
    refused = await client.post(url, json=question, headers=demo["headers"])
    assert refused.status_code == 429
    assert refused.json()["code"] == "daily_questions_quota_reached"


async def test_ordinary_accounts_are_not_affected(client):
    from tests.helpers import auth_headers

    headers = await auth_headers(client, f"{uuid.uuid4().hex[:8]}@example.com")
    response = await client.post("/collections", json={"name": "Mine"}, headers=headers)
    assert response.status_code == 201
    assert (await client.get("/auth/me", headers=headers)).json()["is_demo"] is False
