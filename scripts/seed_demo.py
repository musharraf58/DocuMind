"""Create or update the shared, read-only demo account and its sample
collection, built from the documents in demo/corpus/.

    DEMO_EMAIL=demo@example.com DEMO_PASSWORD='...' python -m scripts.seed_demo

Uses the same configuration as the app (DATABASE_URL, STORAGE_BACKEND and
S3_*, EMBEDDING_*), so run it with the deployment's environment to seed the
deployment. The email and password come from the environment only: never
commit them (they are in .env.lite.example without values).

Safe to run again at any time: the account's password is updated if it
changed, the collection is kept in sync with demo/corpus/ (new files added,
changed files replaced, removed files deleted), and documents already
processed are left alone. Embeddings go through the Postgres embedding
cache, so text embedded before costs no provider call.

Refuses to turn an existing ordinary account into the demo account.
"""

import asyncio
import hashlib
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import clock
from app.core.config import Settings, get_settings
from app.core.passwords import password_problem
from app.core.security import hash_password, verify_password
from app.models import Collection, Document, DocumentStatus, User
from app.services import documents as document_service
from app.services import jobs
from app.services.documents import ALLOWED_TYPES
from app.services.embeddings import EmbeddingProvider
from app.services.storage import StorageBackend
from app.services.users import normalize_email

logger = logging.getLogger("scripts.seed_demo")

CORPUS_DIR = Path(__file__).resolve().parent.parent / "demo" / "corpus"
COLLECTION_NAME = "Sample documents"
COLLECTION_DESCRIPTION = "Short guides written for the DocuMind demo"
SEEDED_TYPES = {".md", ".txt"}


class SeedError(Exception):
    """Something the operator must fix; the message says what."""


@dataclass
class SeedReport:
    created_user: bool = False
    password_changed: bool = False
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    retried: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def read_corpus(corpus_dir: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in SEEDED_TYPES or not path.is_file():
            continue
        content = path.read_bytes()
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SeedError(f"{path.name} is not valid UTF-8") from exc
        files[path.name] = content
    if not files:
        raise SeedError(f"No .md or .txt files in {corpus_dir}")
    return files


async def seed(
    sessions: async_sessionmaker[AsyncSession],
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    email: str,
    password: str,
    corpus_dir: Path = CORPUS_DIR,
) -> SeedReport:
    email = normalize_email(email)
    if problem := password_problem(password, email):
        raise SeedError(f"DEMO_PASSWORD: {problem}")
    corpus = read_corpus(corpus_dir)
    report = SeedReport()

    async with sessions() as db:
        user = await _demo_user(db, email, password, report)
        collection = await _collection(db, user.id)
        to_process = await _sync_documents(db, storage, collection, corpus, report)

    for document_id, name in to_process:
        async with sessions() as db:
            result = await jobs.run_job(db, document_id, storage, embedder, sessions=sessions)
        if result.outcome != jobs.Outcome.ready:
            report.failed[name] = result.message or result.outcome
    return report


async def _demo_user(db: AsyncSession, email: str, password: str, report: SeedReport) -> User:
    user = await db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, hashed_password=hash_password(password), is_demo=True)
        db.add(user)
        await db.commit()
        report.created_user = True
        return user
    if not user.is_demo:
        raise SeedError(
            "An ordinary account already uses DEMO_EMAIL; refusing to turn it into the "
            "shared demo account. Choose another address."
        )
    if not verify_password(password, user.hashed_password):
        # A new password also ends every session made with the old one.
        await db.execute(
            update(User)
            .where(User.id == user.id)
            .values(hashed_password=hash_password(password), token_version=User.token_version + 1)
        )
        await db.commit()
        report.password_changed = True
    return user


async def _collection(db: AsyncSession, owner_id: uuid.UUID) -> Collection:
    collection = await db.scalar(
        select(Collection).where(
            Collection.owner_id == owner_id, Collection.name == COLLECTION_NAME
        )
    )
    if collection is None:
        collection = Collection(
            owner_id=owner_id, name=COLLECTION_NAME, description=COLLECTION_DESCRIPTION
        )
        db.add(collection)
        await db.commit()
    return collection


async def _sync_documents(
    db: AsyncSession,
    storage: StorageBackend,
    collection: Collection,
    corpus: dict[str, bytes],
    report: SeedReport,
) -> list[tuple[uuid.UUID, str]]:
    """Make the collection match the corpus. Returns the documents to process."""
    existing = {
        document.original_filename: document
        for document in await db.scalars(
            select(Document).where(Document.collection_id == collection.id)
        )
    }
    to_process = []
    for name, document in existing.items():
        if name not in corpus:
            await document_service.delete_document(db, storage, document)
            report.removed.append(name)
    for name, content in corpus.items():
        digest = hashlib.sha256(content).hexdigest()
        document = existing.get(name)
        if document is not None and document.content_hash == digest:
            if document.status == DocumentStatus.ready:
                report.unchanged.append(name)
                continue
            # Failed or interrupted earlier: try again.
            await document_service.queue_for_processing(db, document)
            to_process.append((document.id, name))
            report.retried.append(name)
            continue
        if document is not None:  # the file changed
            await document_service.delete_document(db, storage, document)
            report.replaced.append(name)
        else:
            report.added.append(name)
        to_process.append((await _store(db, storage, collection, name, content, digest), name))
    return to_process


async def _store(
    db: AsyncSession,
    storage: StorageBackend,
    collection: Collection,
    name: str,
    content: bytes,
    digest: str,
) -> uuid.UUID:
    document_id = uuid.uuid4()
    key = f"{collection.owner_id}/{document_id}"  # generated, like an upload's

    async def one_chunk() -> AsyncIterator[bytes]:
        yield content

    await storage.save(key, one_chunk())
    db.add(
        Document(
            id=document_id,
            collection_id=collection.id,
            owner_id=collection.owner_id,
            original_filename=name,
            content_type=ALLOWED_TYPES[Path(name).suffix.lower()],
            size_bytes=len(content),
            content_hash=digest,
            storage_key=key,
            status=DocumentStatus.pending,
            queued_at=clock.utcnow(),
        )
    )
    await db.commit()
    return document_id


async def main(settings: Settings) -> SeedReport:
    from app.core.database import make_engine
    from app.core.redis import make_redis
    from app.services.embeddings import get_embedding_provider
    from app.services.embeddings.cache import CachingEmbeddingProvider
    from app.services.storage import get_storage

    email = settings.DEMO_EMAIL
    password = settings.DEMO_PASSWORD.get_secret_value()
    if not email or not password:
        raise SeedError("Set DEMO_EMAIL and DEMO_PASSWORD in the environment")
    engine = make_engine(statement_timeout_ms=settings.DB_WORKER_STATEMENT_TIMEOUT_MS)
    redis = make_redis()
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        embedder = CachingEmbeddingProvider(get_embedding_provider(redis), sessions)
        report = await seed(sessions, get_storage(), embedder, email, password)
        hits, misses = embedder.stats.hits, embedder.stats.misses
    finally:
        await redis.aclose()
        await engine.dispose()
    print(f"Demo account: {'created' if report.created_user else 'up to date'}", end="")
    print(" (password changed)" if report.password_changed else "")
    for label, names in (
        ("added", report.added),
        ("replaced", report.replaced),
        ("removed", report.removed),
        ("retried", report.retried),
        ("unchanged", report.unchanged),
    ):
        if names:
            print(f"  {label}: {', '.join(names)}")
    print(f"  embeddings: {hits} from the cache, {misses} from the provider")
    for name, reason in report.failed.items():
        print(f"  FAILED {name}: {reason}")
    return report


if __name__ == "__main__":
    from app.core.logging import configure_logging

    configure_logging()
    try:
        result = asyncio.run(main(get_settings()))
    except SeedError as exc:
        print(f"seed_demo: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(1 if result.failed else 0)
