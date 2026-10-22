"""Choose RAG_MAX_DISTANCE from measurements: python -m eval.sweep [options]

Retrieves once per question (same as eval.run, same cache), then re-applies
the relevance gate offline at every threshold in the range. For each it
reports the false rejection rate (answerable questions turned away) and
the rejection rate (unanswerable questions correctly turned away). A good
threshold keeps the first near 0 and the second near 100%.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dataset import corpus_files, load_questions
from eval.run import (
    REPORTS_DIR,
    EvalConfig,
    QuestionResult,
    add_config_arguments,
    config_from_args,
    prepare,
    real_environment,
    score,
)


def thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0 or stop < start:
        raise ValueError("need step > 0 and stop >= start")
    count = round((stop - start) / step)
    return [round(start + i * step, 4) for i in range(count + 1)]


def sweep(
    results: Sequence[QuestionResult], config: EvalConfig, values: Sequence[float]
) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        metrics = score(results, config, max_distance=value)["metrics"]
        rows.append(
            {
                "max_distance": value,
                "false_rejection_rate": metrics.get("false_rejection_rate"),
                "rejection_rate": metrics.get("rejection_rate"),
            }
        )
    return rows


def best_distances(results: Sequence[QuestionResult], config: EvalConfig) -> list[dict[str, Any]]:
    """Per question, the smallest distance among the chunks the gate looks
    at, and for answerable ones the distance of the right chunk. The
    threshold has to fall between the answerable and unanswerable groups."""
    rows = []
    for result in results:
        considered = result.retrieved[: config.top_k]
        correct = [
            chunk.distance for chunk, ok in zip(considered, result.correct, strict=False) if ok
        ]
        rows.append(
            {
                "id": result.question.id,
                "answerable": result.question.answerable,
                "closest": round(min((c.distance for c in considered), default=1.0), 4),
                "correct": round(correct[0], 4) if correct else None,
            }
        )
    return sorted(rows, key=lambda row: row["closest"])


def safe_range(rows: Sequence[dict[str, Any]]) -> tuple[float | None, float | None]:
    """(lowest threshold with no false rejections, highest that still rejects
    every unanswerable question). If the first is <= the second, every value
    between them is perfect on this set; otherwise they overlap and the
    choice is a trade-off."""
    no_false = [r["max_distance"] for r in rows if r["false_rejection_rate"] == 0]
    all_rejected = [r["max_distance"] for r in rows if r["rejection_rate"] == 1]
    return (min(no_false, default=None), max(all_rejected, default=None))


def format_sweep(
    rows: Sequence[dict[str, Any]], distances: Sequence[dict[str, Any]], config: EvalConfig
) -> str:
    lines = [
        f"Mode {config.search_mode} | chunks {config.chunk_tokens} | slack "
        f"{config.keyword_slack} | gate over the first {config.top_k} results",
        "",
        "Closest chunk per question (A = answerable, U = unanswerable):",
    ]
    for row in distances:
        kind = "A" if row["answerable"] else "U"
        right = "" if row["correct"] is None else f"  right chunk {row['correct']:.3f}"
        if row["answerable"] and row["correct"] is None:
            right = "  right chunk not in top results"
        lines.append(f"  {kind} {row['closest']:.3f}  {row['id']}{right}")
    lines += ["", f"{'max_distance':>12} {'falseRej':>9} {'rejected':>9}"]
    for row in rows:
        lines.append(
            f"{row['max_distance']:>12.2f} {_percent(row['false_rejection_rate']):>9} "
            f"{_percent(row['rejection_rate']):>9}"
        )
    low, high = safe_range(rows)
    lines.append("")
    if low is None or high is None:
        lines.append("No threshold in this range gets either rate perfect.")
    elif low <= high:
        lines.append(f"Every threshold from {low:.2f} to {high:.2f} is perfect on this set.")
    else:
        lines.append(
            f"No perfect threshold: no false rejections needs >= {low:.2f}, rejecting every "
            f"unanswerable question needs <= {high:.2f}. Pick by which mistake costs more."
        )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


async def run_sweep(env, config: EvalConfig, values: Sequence[float]) -> dict[str, Any]:
    prepared = await prepare(
        env.session_factory,
        env.storage,
        env.embedder,
        env.settings,
        config,
        load_questions(),
        corpus_files(),
    )
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": asdict(config),
        "rows": sweep(prepared.results, config, values),
        "distances": best_distances(prepared.results, config),
        "embedding_calls": prepared.embedding,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_arguments(parser)
    parser.add_argument("--start", type=float, default=0.40)
    parser.add_argument("--stop", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    env = real_environment()
    config = config_from_args(args, env)
    report = asyncio.run(run_sweep(env, config, thresholds(args.start, args.stop, args.step)))
    print(format_sweep(report["rows"], report["distances"], config))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = args.out or REPORTS_DIR / f"{stamp}-sweep-{config.search_mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
