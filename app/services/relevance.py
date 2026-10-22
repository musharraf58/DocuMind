"""The relevance gate: which retrieved chunks are close enough to the
question to be worth answering from. If none pass, /ask answers "I could not
find this in your documents" without calling the LLM.

Rule (justified by the evaluation, docs/EVALUATION.md):
1. A chunk passes if its vector distance is at most RAG_MAX_DISTANCE. This
   holds in every search mode, because meaning is what tells an answerable
   question from an unrelated one: two texts can share a word and nothing else.
2. In hybrid mode, optionally, a STRONG keyword match also passes with a
   slightly looser distance: the chunk contains every term of the question,
   is among the top KEYWORD_TOP keyword results, and is within
   RAG_MAX_DISTANCE + RAG_KEYWORD_GATE_SLACK. Rare exact terms (error codes,
   part numbers) are what embeddings handle worst, and an all-terms match is
   strong evidence. The distance cap still applies, so a question that only
   shares a word with a document doesn't get through. Slack 0 turns this off.
"""

from collections.abc import Sequence

from app.services.retrieval import RetrievedChunk

KEYWORD_TOP = 3


def passes(chunk: RetrievedChunk, max_distance: float, keyword_slack: float, hybrid: bool) -> bool:
    if chunk.distance <= max_distance:
        return True
    return bool(
        hybrid
        and keyword_slack > 0
        and chunk.keyword_all_terms
        and chunk.keyword_rank is not None
        and chunk.keyword_rank <= KEYWORD_TOP
        and chunk.distance <= max_distance + keyword_slack
    )


def relevant(
    chunks: Sequence[RetrievedChunk], max_distance: float, keyword_slack: float, hybrid: bool
) -> list[RetrievedChunk]:
    """The chunks that pass, in their original (best-first) order."""
    return [c for c in chunks if passes(c, max_distance, keyword_slack, hybrid)]
