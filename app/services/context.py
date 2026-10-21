"""Choosing which retrieved chunks go into the prompt."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.services.chunking import count_tokens
from app.services.prompts import render_source
from app.services.retrieval import RetrievedChunk


@dataclass(frozen=True)
class Source:
    """A chunk as it appears in the prompt, under its citation number."""

    number: int
    chunk: RetrievedChunk
    text: str  # the chunk's text as included (shortened only if it alone exceeded the budget)

    def render(self) -> str:
        return render_source(
            self.number,
            self.chunk.document_name,
            self.chunk.page_number,
            self.chunk.section_title,
            self.text,
        )


def assemble_context(
    chunks: Sequence[RetrievedChunk],
    token_budget: int,
    count: Callable[[str], int] = count_tokens,
    keep_order: bool = False,
) -> list[Source]:
    """Number the chunks [1], [2], ... best (smallest distance) first, adding
    them until the next one would exceed `token_budget` (counted on each
    rendered block, header included). So when space runs out it is the
    lowest-ranked chunks that are left out. The best chunk is always
    included, shortened if it alone is over the budget. keep_order: the
    chunks are already ranked (e.g. fused hybrid results); don't re-sort."""
    sources: list[Source] = []
    used = 0
    ordered = list(chunks) if keep_order else sorted(chunks, key=lambda c: c.distance)
    for chunk in ordered:
        source = Source(len(sources) + 1, chunk, chunk.content)
        cost = count(source.render())
        if used + cost > token_budget:
            if not sources:
                sources.append(_shortened(source, token_budget, count))
            break
        sources.append(source)
        used += cost
    return sources


def _shortened(source: Source, token_budget: int, count: Callable[[str], int]) -> Source:
    """Cut the text (at a word boundary) until the rendered block fits."""
    words = source.text.split()
    low, high = 0, len(words)
    while low < high:  # longest prefix of words that fits
        middle = (low + high + 1) // 2
        candidate = Source(source.number, source.chunk, " ".join(words[:middle]) + " ...")
        if count(candidate.render()) <= token_budget:
            low = middle
        else:
            high = middle - 1
    return Source(source.number, source.chunk, " ".join(words[:low]) + " ...")
