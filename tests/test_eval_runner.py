"""The evaluation runner and sweep, end to end against the test database with
a fake embedding provider: deterministic, cached, isolated, and LLM-free."""

import subprocess
import sys
from dataclasses import replace

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.models import Chunk, Collection, Document, DocumentStatus, User
from app.services.embeddings.cache import CachingEmbeddingProvider
from app.services.storage import get_storage
from eval import run as runner
from eval.dataset import corpus_files, load_questions
from eval.run import EVAL_EMAIL, Environment, EvalConfig
from eval.sweep import safe_range, sweep, thresholds
from tests.embedding_fakes import KeywordProvider
from tests.helpers import auth_headers, create_collection, run_queued_jobs, upload

KEYWORDS = ["refund", "warranty", "shipping", "battery", "points", "scanner", "lamp", "err-4471"]


class CountingProvider(KeywordProvider):
    def __init__(self) -> None:
        super().__init__(KEYWORDS)
        self.calls: list[list[str]] = []

    async def _embed_batch(self, texts, input_type):
        self.calls.append(list(texts))
        return await super()._embed_batch(texts, input_type)


@pytest.fixture
def session_factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture
def inner():
    return CountingProvider()


def environment(session_factory, inner) -> Environment:
    """A fresh cache wrapper each time, as a new `python -m eval.run` would get."""
    embedder = CachingEmbeddingProvider(inner, session_factory)
    return Environment(session_factory, get_storage(), embedder, get_settings())


def config(env: Environment, **changes) -> EvalConfig:
    return EvalConfig.from_settings(env.settings, env.embedder.model, **changes)


def comparable(report):
    """A report minus what legitimately differs between runs."""
    return {k: v for k, v in report.items() if k not in {"created_at", "embedding_calls"}}


async def test_repeat_run_is_identical_and_makes_no_provider_calls(session_factory, inner):
    first_env = environment(session_factory, inner)
    first = await runner.run(first_env, config(first_env))
    assert inner.calls, "the first run has to embed the corpus and questions"

    inner.calls.clear()
    second_env = environment(session_factory, inner)
    second = await runner.run(second_env, config(second_env))

    assert inner.calls == []
    assert second["embedding_calls"]["requests"] == 0
    assert comparable(second) == comparable(first)


async def test_report_has_config_metrics_and_every_question(session_factory, inner):
    env = environment(session_factory, inner)
    report = await runner.run(env, config(env, search_mode="hybrid", max_distance=0.5))

    assert report["config"]["search_mode"] == "hybrid"
    assert report["config"]["max_distance"] == 0.5
    assert {"chunk_tokens", "chunk_overlap_tokens", "embedding_model", "keyword_slack"} <= set(
        report["config"]
    )
    assert set(report["metrics"]) == {
        "hit@1",
        "hit@3",
        "hit@5",
        "mrr",
        "false_rejection_rate",
        "rejection_rate",
    }
    assert report["metrics"]["mrr"] is None  # until mrr() is implemented
    assert [r["id"] for r in report["results"]] == [q.id for q in load_questions()]
    assert report["corpus"]["documents"] == len(corpus_files())
    assert "MRR: not implemented yet" in runner.format_report(report)


async def test_eval_data_lives_with_the_eval_user_and_nothing_is_left_pending(
    session_factory, inner, client, db
):
    # Someone else's matching document must never show up in the results.
    headers = await auth_headers(client, "alice@example.com")
    collection_id = await create_collection(client, headers)
    await upload(client, headers, collection_id, "alice.md", b"ERR-4471 in Alice's own notes.")
    await run_queued_jobs()

    env = environment(session_factory, inner)
    report = await runner.run(env, config(env, search_mode="keyword"))

    shown = {hit["document"] for r in report["results"] for hit in r["top"]}
    assert "alice.md" not in shown
    eval_user = await db.scalar(select(User).where(User.email == EVAL_EMAIL))
    owners = await db.scalars(
        select(Document.owner_id).where(
            Document.original_filename.in_(f.name for f in corpus_files())
        )
    )
    assert set(owners) == {eval_user.id}
    statuses = await db.scalars(select(Document.status).where(Document.owner_id == eval_user.id))
    assert set(statuses) == {DocumentStatus.ready}  # nothing for the worker to pick up


async def test_each_chunk_size_gets_its_own_collection(session_factory, inner, db):
    env = environment(session_factory, inner)
    small = await runner.run(env, config(env, chunk_tokens=300))
    large = await runner.run(env, config(env, chunk_tokens=800))

    names = set(await db.scalars(select(Collection.name)))
    assert names == {"eval: 300 tokens, 60 overlap", "eval: 800 tokens, 60 overlap"}
    assert small["corpus"]["chunks"] > large["corpus"]["chunks"]


async def test_changed_document_only_re_embeds_its_own_chunks(session_factory, inner, tmp_path, db):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for file in corpus_files():
        (corpus / file.name).write_bytes(file.read_bytes())
    questions = load_questions()
    env = environment(session_factory, inner)
    cfg = config(env)

    async def prepare():
        return await runner.prepare(
            session_factory,
            env.storage,
            env.embedder,
            env.settings,
            cfg,
            questions,
            sorted(corpus.iterdir()),
        )

    await prepare()
    inner.calls.clear()
    edited = corpus / "loyalty-programme.md"
    edited.write_text(edited.read_text() + "\nMembers also get a free tote bag.\n")
    (corpus / "product-care.md").unlink()

    prepared = await prepare()

    sent = [text for call in inner.calls for text in call]
    assert sent and all("tote bag" in text for text in sent)  # only the changed chunk
    names = set(
        await db.scalars(select(Document.original_filename).where(Document.status == "ready"))
    )
    assert "product-care.md" not in names and "loyalty-programme.md" in names
    chunk_count = await db.scalar(select(func.count()).select_from(Chunk))
    assert prepared.chunks == chunk_count


async def test_failed_processing_is_cleaned_up_and_reported(
    session_factory, inner, tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "empty.md").write_text("   \n")
    env = environment(session_factory, inner)

    with pytest.raises(runner.EvalError, match="empty.md could not be processed"):
        await runner.prepare(
            session_factory,
            env.storage,
            env.embedder,
            env.settings,
            config(env),
            [],
            [corpus / "empty.md"],
        )
    async with session_factory() as db:
        assert await db.scalar(select(func.count()).select_from(Document)) == 0


def test_evaluation_code_never_imports_the_llm():
    code = (
        "import sys, eval.run, eval.sweep, eval.experiments, eval.metrics, eval.dataset\n"
        "loaded = [m for m in sys.modules if m.startswith(('app.services.llm', "
        "'app.services.answering', 'app.services.ask', 'google'))]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


async def test_runner_works_with_the_llm_disabled(session_factory, inner, monkeypatch):
    def no_llm(*args, **kwargs):
        raise AssertionError("the evaluation must never call the LLM")

    import app.services.llm as llm

    monkeypatch.setattr(llm, "get_llm_provider", no_llm)
    env = environment(session_factory, inner)
    await runner.run(env, config(env, search_mode="hybrid"))


# --- Sweep ---


def test_thresholds_cover_the_range_inclusively():
    assert thresholds(0.5, 0.6, 0.05) == [0.5, 0.55, 0.6]
    with pytest.raises(ValueError):
        thresholds(0.6, 0.5, 0.05)


async def test_sweep_rates_move_in_opposite_directions(session_factory, inner):
    env = environment(session_factory, inner)
    cfg = config(env)
    prepared = await runner.prepare(
        session_factory,
        env.storage,
        env.embedder,
        env.settings,
        cfg,
        load_questions(),
        corpus_files(),
    )
    rows = sweep(prepared.results, cfg, thresholds(0.0, 2.0, 0.1))

    false_rej = [row["false_rejection_rate"] for row in rows]
    rejected = [row["rejection_rate"] for row in rows]
    assert false_rej == sorted(false_rej, reverse=True)  # looser gate: fewer false rejections
    assert rejected == sorted(rejected, reverse=True)  # ...and fewer correct rejections
    assert (false_rej[-1], rejected[-1]) == (0.0, 0.0)  # 2 is the largest distance: all pass
    # The sweep is the same as running at each threshold.
    report = runner.build_report(replace(prepared, config=replace(cfg, max_distance=0.5)))
    assert rows[5]["false_rejection_rate"] == report["metrics"]["false_rejection_rate"]


def test_safe_range():
    rows = [
        {"max_distance": 0.5, "false_rejection_rate": 0.4, "rejection_rate": 1.0},
        {"max_distance": 0.6, "false_rejection_rate": 0.0, "rejection_rate": 1.0},
        {"max_distance": 0.7, "false_rejection_rate": 0.0, "rejection_rate": 1.0},
        {"max_distance": 0.8, "false_rejection_rate": 0.0, "rejection_rate": 0.5},
    ]
    assert safe_range(rows) == (0.6, 0.7)
