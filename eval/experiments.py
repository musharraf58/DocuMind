"""Run the comparison matrix: python -m eval.experiments

Vector vs keyword vs hybrid, strict vs loose keyword matching, english vs
simple text search, and chunk sizes 300 / 500 / 800, each chunk size in its
own eval collection. Every run uses the embedding cache, so only chunks
never embedded before cost provider calls: on a fresh database the three
chunk sizes are the only real cost; repeat runs are free.

Saves every report under eval/reports/experiments-<time>/ and prints one
summary table (Markdown, ready for docs/EVALUATION.md).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from eval.run import (
    REPORTS_DIR,
    Environment,
    EvalConfig,
    add_config_arguments,
    config_from_args,
    real_environment,
    run,
    save_report,
)

# (name, changes to the base configuration)
EXPERIMENTS: list[tuple[str, dict[str, Any]]] = [
    ("vector", {"search_mode": "vector"}),
    ("keyword strict", {"search_mode": "keyword", "keyword_match": "strict"}),
    ("keyword loose", {"search_mode": "keyword", "keyword_match": "loose"}),
    ("hybrid strict", {"search_mode": "hybrid", "keyword_match": "strict"}),
    ("hybrid loose", {"search_mode": "hybrid", "keyword_match": "loose"}),
    (
        "hybrid loose, simple",
        {"search_mode": "hybrid", "keyword_match": "loose", "text_search_config": "simple"},
    ),
    ("vector, 300 tokens", {"search_mode": "vector", "chunk_tokens": 300}),
    ("hybrid loose, 300", {"search_mode": "hybrid", "keyword_match": "loose", "chunk_tokens": 300}),
    ("vector, 800 tokens", {"search_mode": "vector", "chunk_tokens": 800}),
    ("hybrid loose, 800", {"search_mode": "hybrid", "keyword_match": "loose", "chunk_tokens": 800}),
]


def summary_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    def cell(value: float | None, digits: str = ".0%") -> str:
        return "n/a" if value is None else format(value, digits)

    lines = [
        "| Experiment | Chunks | hit@1 | hit@3 | hit@5 | MRR | False rej. | Rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in rows:
        m = report["metrics"]
        lines.append(
            f"| {name} | {report['corpus']['chunks']} | {cell(m.get('hit@1'))} | "
            f"{cell(m.get('hit@3'))} | {cell(m.get('hit@5'))} | {cell(m.get('mrr'), '.3f')} | "
            f"{cell(m.get('false_rejection_rate'))} | {cell(m.get('rejection_rate'))} |"
        )
    return "\n".join(lines)


async def run_all(
    env: Environment, base: EvalConfig, only: Sequence[str] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for name, changes in EXPERIMENTS:
        if only and name not in only:
            continue
        logging.getLogger("eval").info("Experiment: %s", name)
        rows.append((name, await run(env, replace(base, **changes))))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_arguments(parser)
    parser.add_argument("--only", action="append", help="run just this experiment (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    env = real_environment()
    rows = asyncio.run(run_all(env, config_from_args(args, env), args.only))
    folder = REPORTS_DIR / f"experiments-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    for name, report in rows:
        save_report(report, folder / f"{name.replace(', ', '-').replace(' ', '-')}.json")
    print(summary_table(rows))
    calls = env.embedder.stats if hasattr(env.embedder, "stats") else None
    if calls is not None:
        print(
            f"\nEmbeddings: {calls.hits} from cache, {calls.misses} new, "
            f"{calls.requests} provider requests",
            file=sys.stderr,
        )
    print(f"Reports saved to {folder}", file=sys.stderr)


if __name__ == "__main__":
    main()
