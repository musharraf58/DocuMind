"""Reciprocal Rank Fusion (RRF): merging ranked lists by position.

Each list votes for its items with 1 / (k + rank): the first place gets
1/(k+1), the second 1/(k+2), and so on. An item's score is the sum of its
votes across lists, so something ranked well by BOTH searches beats
something ranked well by only one. Only positions are used, never the raw
scores: cosine distances and keyword ranks are on unrelated scales and
can't be added together meaningfully.
"""

from collections.abc import Hashable, Sequence

DEFAULT_K = 60  # the usual value; larger k = flatter votes between ranks


def reciprocal_rank_fusion[Key: Hashable](
    ranked_lists: Sequence[Sequence[Key]], k: int = DEFAULT_K
) -> list[tuple[Key, float]]:
    """Fuse ranked lists (best first) into one: [(item, score)], best first.

    An item appearing twice in one list only counts at its first (best)
    position. Ties are broken by the item's best rank in any list, then by
    the order of first appearance, so the result is always the same for the
    same input.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: dict[Key, float] = {}
    best_rank: dict[Key, int] = {}
    first_seen: dict[Key, int] = {}
    for items in ranked_lists:
        counted: set[Key] = set()
        for rank, item in enumerate(items, start=1):
            if item in counted:
                continue
            counted.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            first_seen.setdefault(item, len(first_seen))
    ordered = sorted(scores, key=lambda item: (-scores[item], best_rank[item], first_seen[item]))
    return [(item, scores[item]) for item in ordered]
