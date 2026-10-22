"""The evaluation metrics, on examples small enough to check by hand.

The three MRR tests fail until mrr() in eval/metrics.py is implemented:
that function is deliberately left as an exercise.
"""

import pytest

from eval.metrics import (
    Outcome,
    false_rejection_rate,
    hit_rate_at_k,
    mrr,
    rejection_rate,
)

RANKS = [1, 2, None, 4, 3]


@pytest.mark.parametrize(("k", "expected"), [(1, 1 / 5), (2, 2 / 5), (3, 3 / 5), (5, 4 / 5)])
def test_hit_rate_counts_correct_chunks_within_k(k, expected):
    assert hit_rate_at_k(RANKS, k) == pytest.approx(expected)


def test_hit_rate_extremes():
    assert hit_rate_at_k([1, 1], 1) == 1.0
    assert hit_rate_at_k([None, None], 5) == 0.0
    assert hit_rate_at_k([9], 5) == 0.0


def test_hit_rate_ignores_position_within_k():
    # Same hit@5 whether the right chunk is first or fifth: that's MRR's job.
    assert hit_rate_at_k([1, 1], 5) == hit_rate_at_k([5, 5], 5)


@pytest.mark.parametrize(("ranks", "k"), [([], 3), ([0], 3), ([1], 0)])
def test_hit_rate_rejects_bad_input(ranks, k):
    with pytest.raises(ValueError):
        hit_rate_at_k(ranks, k)


def test_mrr_on_hand_built_examples():
    assert mrr([1, 1, 1]) == 1.0
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 1 / 2 + 0 + 1 / 4) / 4)
    assert mrr([None, None]) == 0.0
    assert mrr([2]) == 0.5


def test_mrr_rewards_higher_ranks_where_hit_rate_cannot():
    early, late = [1, 1], [5, 5]
    assert hit_rate_at_k(early, 5) == hit_rate_at_k(late, 5)
    assert mrr(early) > mrr(late)
    assert mrr(late) == pytest.approx(0.2)


@pytest.mark.parametrize("ranks", [[], [0, 1], [-1]])
def test_mrr_rejects_bad_input(ranks):
    with pytest.raises(ValueError):
        mrr(ranks)


OUTCOMES = [
    Outcome(answerable=True, rank=1, rejected=False),
    Outcome(answerable=True, rank=None, rejected=True),  # a false rejection
    Outcome(answerable=True, rank=3, rejected=False),
    Outcome(answerable=True, rank=2, rejected=True),  # found, but gated away
    Outcome(answerable=False, rank=None, rejected=True),  # correctly rejected
    Outcome(answerable=False, rank=None, rejected=False),  # junk let through
    Outcome(answerable=False, rank=None, rejected=True),
]


def test_rejection_rate_is_over_unanswerable_questions_only():
    assert rejection_rate(OUTCOMES) == pytest.approx(2 / 3)


def test_false_rejection_rate_is_over_answerable_questions_only():
    assert false_rejection_rate(OUTCOMES) == pytest.approx(2 / 4)


def test_rejection_rates_need_questions_of_their_kind():
    only_answerable = [o for o in OUTCOMES if o.answerable]
    with pytest.raises(ValueError, match="unanswerable"):
        rejection_rate(only_answerable)
    with pytest.raises(ValueError, match="answerable"):
        false_rejection_rate([o for o in OUTCOMES if not o.answerable])
