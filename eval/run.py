"""Measure retrieval quality: python -m eval.run [options]

Ingests eval/corpus/ through the real processing pipeline into a collection
owned by a dedicated eval user, asks every question in eval/questions.yaml
through the real search code, and reports how often the right chunk came
back and how the relevance gate behaved. It NEVER calls the LLM: it judges
what retrieval found, not what an answer would say.

Embeddings go through the cache (app/services/embeddings/cache.py), so a
repeat run with unchanged documents and questions makes no provider calls.
See eval/README.md for how to read the report.
"""

import argparse
import asyncio
import hashlib
import io
import json
import logging
import secrets
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.models import Chunk, Collection, Document, DocumentStatus, User
from app.services import jobs, relevance
from app.services.documents import delete_document, upload_document
from app.services.embeddings.base import EmbeddingProvider
from app.services.retrieval import (
    KeywordMatch,
    RetrievedChunk,
    SearchMode,
    SearchOptions,
    search_chunks,
)
from app.services.storage import StorageBackend
from app.services.users import register_user
from eval.dataset import (
    EVAL_DIR,
    QUESTION_TYPES,
    Question,
    corpus_files,
    is_expected_chunk,
    load_questions,
)
from eval.metrics import Outcome, false_rejection_rate, hit_rate_at_k, mrr, rejection_rate

logger = logging.getLogger("eval")

EVAL_EMAIL = "eval@documind.local"
REPORTS_DIR = EVAL_DIR / "reports"
KS = (1, 3, 5)
# How deep each search goes, so the report can say "found at rank 14" rather
# than just "missed". The gate and hit@k only ever look at the first top_k:
# results are fully ordered, so the first top_k of a deeper search are the
# same chunks /ask would get.
DEPTH = 20
SHOWN_RESULTS = 3  # results listed per failed question


class EvalError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvalConfig:
    """Everything that can change the numbers. Saved in every report."""

    search_mode: SearchMode
    chunk_tokens: int
    chunk_overlap_tokens: int
    embedding_model: str
    embedding_dimensions: int
    max_distance: float
    keyword_slack: float
    keyword_match: KeywordMatch
    text_search_config: str
    top_k: int
    hybrid_candidates: int
    rrf_k: int

    @classmethod
    def from_settings(cls, settings: Settings, model: str, **overrides: Any) -> "EvalConfig":
        values = {
            "search_mode": settings.RAG_SEARCH_MODE,
            "chunk_tokens": settings.CHUNK_TOKENS,
            "chunk_overlap_tokens": settings.CHUNK_OVERLAP_TOKENS,
            "embedding_model": model,
            "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
            "max_distance": settings.RAG_MAX_DISTANCE,
            "keyword_slack": settings.RAG_KEYWORD_GATE_SLACK,
            "keyword_match": settings.RAG_KEYWORD_MATCH,
            "text_search_config": settings.TEXT_SEARCH_CONFIG,
            "top_k": settings.RAG_TOP_K,
            "hybrid_candidates": settings.HYBRID_CANDIDATES,
            "rrf_k": settings.RRF_K,
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    @property
    def collection_name(self) -> str:
        # One collection per chunking: other options only change the search.
        return f"eval: {self.chunk_tokens} tokens, {self.chunk_overlap_tokens} overlap"

    def processing_settings(self, settings: Settings) -> Settings:
        return settings.model_copy(
            update={
                "CHUNK_TOKENS": self.chunk_tokens,
                "CHUNK_OVERLAP_TOKENS": self.chunk_overlap_tokens,
            }
        )

    def search_options(self) -> SearchOptions:
        depth = max(DEPTH, self.top_k)
        return SearchOptions(
            mode=self.search_mode,
            top_k=depth,
            candidates=max(self.hybrid_candidates, depth),
            rrf_k=self.rrf_k,
            text_search_config=self.text_search_config,
            keyword_match=self.keyword_match,
        )


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    retrieved: list[RetrievedChunk]  # best first, DEPTH deep
    correct: list[bool]  # per retrieved chunk: is it the expected passage?

    @property
    def rank(self) -> int | None:
        """1-based rank of the first correct chunk; None if not retrieved."""
        return next((i for i, ok in enumerate(self.correct, start=1) if ok), None)

    def outcome(self, config: EvalConfig, max_distance: float | None = None) -> Outcome:
        """What /ask would do: gate the first top_k chunks exactly as
        answering.find_sources does. max_distance overrides the config's
        (the sweep tries many)."""
        kept = relevance.relevant(
            self.retrieved[: config.top_k],
            config.max_distance if max_distance is None else max_distance,
            config.keyword_slack,
            hybrid=config.search_mode == "hybrid",
        )
        return Outcome(self.question.answerable, self.rank, rejected=not kept)


@dataclass
class Prepared:
    """The eval collection, ingested and searched: everything scoring needs."""

    config: EvalConfig
    results: list[QuestionResult]
    documents: int
    chunks: int
    embedding: dict[str, int] = field(default_factory=dict)


# --- Ingestion ---


async def eval_collection(
    session_factory: async_sessionmaker[AsyncSession], config: EvalConfig
) -> Collection:
    """The eval user's collection for this chunking, created on first use.
    The eval user has a random password nobody knows: it cannot log in."""
    async with session_factory() as db:
        user = await db.scalar(select(User).where(User.email == EVAL_EMAIL))
        if user is None:
            user = await register_user(db, EVAL_EMAIL, secrets.token_urlsafe(32))
        collection = await db.scalar(
            select(Collection).where(
                Collection.owner_id == user.id, Collection.name == config.collection_name
            )
        )
        if collection is None:
            collection = Collection(
                owner_id=user.id,
                name=config.collection_name,
                description="Created by python -m eval.run; safe to delete.",
            )
            db.add(collection)
            await db.commit()
        return collection


async def sync_corpus(
    session_factory: async_sessionmaker[AsyncSession],
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    collection: Collection,
    settings: Settings,
    files: Sequence[Path],
) -> None:
    """Make the collection hold exactly the corpus, processed with these
    settings and this embedding model.

    An unchanged document already processed is left alone. Anything else
    (new, edited, failed, or embedded with another model) is uploaded and
    processed right here through the real pipeline (jobs.run_job), never
    left pending for a worker to pick up. Documents no longer in the corpus
    are deleted.
    """
    async with session_factory() as db:
        existing = {
            document.original_filename: document
            for document in await db.scalars(
                select(Document).where(Document.collection_id == collection.id)
            )
        }
        for file in files:
            data = file.read_bytes()
            document = existing.pop(file.name, None)
            if document is not None and await _up_to_date(db, document, data, embedder.model):
                continue
            if document is not None:
                await delete_document(db, storage, document)
            document = await upload_document(
                db, storage, collection, UploadFile(io.BytesIO(data), filename=file.name)
            )
            logger.info("Processing %s", file.name)
            document_id = document.id  # the job may roll back, expiring `document`
            result = await jobs.run_job(db, document_id, storage, embedder, settings)
            if result.outcome != jobs.Outcome.ready:
                document = await db.get(Document, document_id, populate_existing=True)
                if document is not None:  # never leave it pending for the worker
                    await delete_document(db, storage, document)
                raise EvalError(f"{file.name} could not be processed: {result.message}")
        for stale in existing.values():
            await delete_document(db, storage, stale)


async def _up_to_date(db: AsyncSession, document: Document, data: bytes, model: str) -> bool:
    if document.status != DocumentStatus.ready:
        return False
    if document.content_hash != hashlib.sha256(data).hexdigest():
        return False
    other_model = await db.scalar(
        select(exists().where(Chunk.document_id == document.id, Chunk.embedding_model != model))
    )
    return not other_model


async def _corpus_size(db: AsyncSession, collection: Collection) -> tuple[int, int]:
    documents = await db.scalars(select(Document.id).where(Document.collection_id == collection.id))
    chunks = await db.scalars(select(Chunk.id).where(Chunk.collection_id == collection.id))
    return len(documents.all()), len(chunks.all())


# --- Retrieval ---


async def prepare(
    session_factory: async_sessionmaker[AsyncSession],
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    settings: Settings,
    config: EvalConfig,
    questions: Sequence[Question],
    files: Sequence[Path],
) -> Prepared:
    """Ingest the corpus and run every question through search, once."""
    if config.embedding_model != embedder.model:
        raise EvalError("config.embedding_model must be the embedder's model")
    collection = await eval_collection(session_factory, config)
    await sync_corpus(
        session_factory,
        storage,
        embedder,
        collection,
        config.processing_settings(settings),
        files,
    )
    # All questions in as few provider requests as possible (none when cached).
    vectors = await embedder.embed_queries([q.question for q in questions])
    options = config.search_options()
    results = []
    async with session_factory() as db:
        for question, vector in zip(questions, vectors, strict=True):
            retrieved = await search_chunks(
                db,
                collection.owner_id,
                collection.id,
                embedder.model,
                vector,
                question.question,
                options,
            )
            correct = [
                question.expected is not None
                and is_expected_chunk(
                    question.expected, chunk.document_name, chunk.content, chunk.page_number
                )
                for chunk in retrieved
            ]
            results.append(QuestionResult(question, retrieved, correct))
        documents, chunks = await _corpus_size(db, collection)
    stats = getattr(embedder, "stats", None)
    embedding = asdict(stats) if stats is not None else {}
    return Prepared(config, results, documents, chunks, embedding)


# --- Scoring ---


def score(
    results: Sequence[QuestionResult], config: EvalConfig, max_distance: float | None = None
) -> dict[str, Any]:
    """The headline numbers, overall and per question type."""
    outcomes = [r.outcome(config, max_distance) for r in results]
    answerable = [r for r in results if r.question.answerable]
    ranks = [r.rank for r in answerable]
    metrics: dict[str, Any] = {}
    if ranks:
        metrics |= {f"hit@{k}": hit_rate_at_k(ranks, k) for k in KS}
        metrics["mrr"] = mrr(ranks)
        metrics["false_rejection_rate"] = false_rejection_rate(outcomes)
    if any(not o.answerable for o in outcomes):
        metrics["rejection_rate"] = rejection_rate(outcomes)
    by_type = {}
    for kind in QUESTION_TYPES:
        group = [(r, o) for r, o in zip(results, outcomes, strict=True) if r.question.type == kind]
        if not group:
            continue
        entry: dict[str, Any] = {"count": len(group)}
        if kind == "unanswerable":
            entry["rejection_rate"] = rejection_rate([o for _, o in group])
        else:
            entry |= {f"hit@{k}": hit_rate_at_k([r.rank for r, _ in group], k) for k in KS}
            entry["false_rejection_rate"] = false_rejection_rate([o for _, o in group])
        by_type[kind] = entry
    return {"metrics": metrics, "by_type": by_type}


def question_record(result: QuestionResult, config: EvalConfig) -> dict[str, Any]:
    outcome = result.outcome(config)
    return {
        "id": result.question.id,
        "type": result.question.type,
        "question": result.question.question,
        "rank": result.rank,
        "rejected": outcome.rejected,
        "failed": _failed(result, outcome),
        "top": [
            {
                "document": chunk.document_name,
                "page": chunk.page_number,
                "distance": round(chunk.distance, 4),
                "vector_rank": chunk.vector_rank,
                "keyword_rank": chunk.keyword_rank,
                "correct": correct,
            }
            for chunk, correct in zip(
                result.retrieved[:SHOWN_RESULTS], result.correct, strict=False
            )
        ],
    }


def _failed(result: QuestionResult, outcome: Outcome) -> bool:
    """Answerable: the right chunk wasn't first, or the gate turned the
    question away. Unanswerable: the gate let something through."""
    if result.question.answerable:
        return result.rank != 1 or outcome.rejected
    return not outcome.rejected


def build_report(prepared: Prepared) -> dict[str, Any]:
    config = prepared.config
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": asdict(config),
        "corpus": {"documents": prepared.documents, "chunks": prepared.chunks},
        "questions": len(prepared.results),
        **score(prepared.results, config),
        "embedding_calls": prepared.embedding,
        "results": [question_record(r, config) for r in prepared.results],
    }


# --- Output ---


def _percent(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.0%}"


def format_report(report: dict[str, Any]) -> str:
    config, metrics = report["config"], report["metrics"]
    lines = [
        f"Mode {config['search_mode']} | chunks {config['chunk_tokens']}/"
        f"{config['chunk_overlap_tokens']} | {config['embedding_model']} | "
        f"max distance {config['max_distance']} | slack {config['keyword_slack']} | "
        f"keywords {config['keyword_match']}/{config['text_search_config']}",
        f"{report['corpus']['documents']} documents, {report['corpus']['chunks']} chunks, "
        f"{report['questions']} questions",
        "",
        f"{'type':<13} {'n':>3} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'falseRej':>9} "
        f"{'rejected':>9}",
    ]
    rows = [("all", report["questions"], metrics)] + [
        (kind, entry["count"], entry) for kind, entry in report["by_type"].items()
    ]
    for name, count, values in rows:
        lines.append(
            f"{name:<13} {count:>3} "
            + " ".join(f"{_percent(values.get(f'hit@{k}')):>6}" for k in KS)
            + f" {_percent(values.get('false_rejection_rate')):>9}"
            + f" {_percent(values.get('rejection_rate')):>9}"
        )
    lines.append("")
    lines.append(f"MRR: {metrics['mrr']:.3f}")

    failures = [r for r in report["results"] if r["failed"]]
    lines += ["", f"Failed questions: {len(failures)}"]
    for record in failures:
        where = "not in top " + str(DEPTH) if record["rank"] is None else f"rank {record['rank']}"
        if record["type"] == "unanswerable":
            where = "let through by the gate"
        elif record["rejected"]:
            where += ", rejected by the gate"
        lines.append(f"  {record['id']:<24} {record['type']:<12} {where}")
        for hit in record["top"]:
            mark = "*" if hit["correct"] else " "
            lines.append(
                f"      {mark} {hit['document']:<26} d={hit['distance']:.3f} "
                f"v#{hit['vector_rank'] or '-'} k#{hit['keyword_rank'] or '-'}"
            )
    calls = report.get("embedding_calls") or {}
    if calls:
        lines += [
            "",
            f"Embeddings: {calls.get('hits', 0)} from cache, {calls.get('misses', 0)} new, "
            f"{calls.get('requests', 0)} provider requests",
        ]
    return "\n".join(lines)


def save_report(report: dict[str, Any], path: Path | None = None) -> Path:
    if path is None:
        config = report["config"]
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{config['search_mode']}-{config['chunk_tokens']}.json"
        path = REPORTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


# --- Command line ---


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by run, sweep and experiments. Unset = from settings."""
    parser.add_argument("--mode", dest="search_mode", choices=["vector", "keyword", "hybrid"])
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--overlap", dest="chunk_overlap_tokens", type=int)
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--keyword-slack", type=float)
    parser.add_argument("--keyword-match", choices=["strict", "loose"])
    parser.add_argument("--text-search-config")
    parser.add_argument("--top-k", type=int)


CONFIG_ARGUMENTS = (
    "search_mode",
    "chunk_tokens",
    "chunk_overlap_tokens",
    "max_distance",
    "keyword_slack",
    "keyword_match",
    "text_search_config",
    "top_k",
)


@dataclass
class Environment:
    """The real database, storage and (cached, rate-limited) embedder."""

    session_factory: async_sessionmaker[AsyncSession]
    storage: StorageBackend
    embedder: EmbeddingProvider
    settings: Settings


def real_environment() -> Environment:
    # Imported here so that importing this module (tests) doesn't connect.
    from app.core.database import SessionLocal
    from app.core.redis import make_redis
    from app.services.embeddings import get_embedding_provider
    from app.services.embeddings.cache import CachingEmbeddingProvider
    from app.services.storage import get_storage

    embedder = CachingEmbeddingProvider(get_embedding_provider(make_redis()), SessionLocal)
    return Environment(SessionLocal, get_storage(), embedder, get_settings())


def config_from_args(args: argparse.Namespace, env: Environment) -> EvalConfig:
    overrides = {name: getattr(args, name) for name in CONFIG_ARGUMENTS}
    return EvalConfig.from_settings(env.settings, env.embedder.model, **overrides)


async def run(
    env: Environment, config: EvalConfig, questions_file: Path | None = None
) -> dict[str, Any]:
    questions = load_questions(questions_file) if questions_file else load_questions()
    prepared = await prepare(
        env.session_factory,
        env.storage,
        env.embedder,
        env.settings,
        config,
        questions,
        corpus_files(),
    )
    return build_report(prepared)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_arguments(parser)
    parser.add_argument("--out", type=Path, help="report path (default: eval/reports/...)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    env = real_environment()
    report = asyncio.run(run(env, config_from_args(args, env)))
    print(format_report(report))
    print(f"\nReport saved to {save_report(report, args.out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
