"""Retrieval metrics. Pure functions over plain numbers: no database, no
provider, so each one can be checked by hand on a tiny example.

Conventions:
- a *rank* is the 1-based position of the first correct chunk in what
  retrieval returned, before the relevance gate; None when no correct
  chunk was returned at all.
- an *outcome* also says whether the question was answerable and whether
  the relevance gate rejected it (kept no chunk, so /ask would say
  "I couldn't find that").
"""

from collections.abc import Sequence
from dataclasses import dataclass

Rank = int | None


@dataclass(frozen=True)
class Outcome:
    answerable: bool
    rank: Rank  # always None for unanswerable questions
    rejected: bool


def hit_rate_at_k(ranks: Sequence[Rank], k: int) -> float:
    """The share of questions whose correct chunk is among the first k
    results. 1.0 = always found within k; says nothing about *where* in the
    top k."""
    _check_ranks(ranks)
    if k < 1:
        raise ValueError("k must be at least 1")
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: Sequence[Rank]) -> float:
    """Mean Reciprocal Rank: the average, over all questions, of 1 / rank of
    the first correct chunk, counting 0 for a question whose correct chunk
    was not returned (rank None).

    Examples:
        [1, 1, 1]          -> 1.0     (always first)
        [1, 2, None, 4]    -> (1 + 1/2 + 0 + 1/4) / 4 = 0.4375
        [None, None]       -> 0.0

    Unlike hit@k, MRR rewards putting the right chunk *higher*: rank 1 is
    worth 1, rank 2 only 0.5, rank 10 only 0.1.

    Raise ValueError for an empty list or a rank below 1
    (_check_ranks does exactly that).

    TODO(you): implement this. tests/test_eval_metrics.py has the cases;
    they fail until you do. Until then the report prints
    "MRR: not implemented yet".
    """
    raise NotImplementedError("mrr() is left for you to implement; see its docstring")


def rejection_rate(outcomes: Sequence[Outcome]) -> float:
    """Of the UNANSWERABLE questions, the share the gate rejected. Higher is
    better: junk never reaches the LLM."""
    return _share_rejected([o for o in outcomes if not o.answerable], "unanswerable")


def false_rejection_rate(outcomes: Sequence[Outcome]) -> float:
    """Of the ANSWERABLE questions, the share the gate rejected even though
    the corpus has the answer. Lower is better: each one is an
    "I couldn't find that" the user should not have seen."""
    return _share_rejected([o for o in outcomes if o.answerable], "answerable")


def _share_rejected(outcomes: Sequence[Outcome], kind: str) -> float:
    if not outcomes:
        raise ValueError(f"no {kind} questions to score")
    return sum(1 for o in outcomes if o.rejected) / len(outcomes)


def _check_ranks(ranks: Sequence[Rank]) -> None:
    if not ranks:
        raise ValueError("no questions to score")
    if any(rank is not None and rank < 1 for rank in ranks):
        raise ValueError("ranks start at 1 (None = not found)")
