"""Split extracted segments into overlapping chunks of at most N tokens.

Pure functions only (no I/O, no database), so they are easy to test.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache

import tiktoken

from app.services.extraction import Segment

TokenCounter = Callable[[str], int]

# Tried in order: paragraphs, lines, sentences, words. Each pattern captures
# its separator, so the pieces can be glued back together exactly. Anything
# still too long after words (say, a 5,000-character URL) is cut by characters.
_SEPARATORS = [
    re.compile(r"(\n[ \t]*\n\s*)"),
    re.compile(r"(\n)"),
    re.compile(r"(?<=[.!?])([ \t]+)"),
    re.compile(r"(\s+)"),
]
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class ChunkData:
    chunk_index: int
    content: str
    page_number: int | None
    section_title: str | None
    token_count: int


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count (cl100k_base). Text like "<|endoftext|>" in a
    document is counted as ordinary text instead of raising an error."""
    return len(_encoding().encode(text, disallowed_special=()))


def chunk_segments(
    segments: Iterable[Segment],
    chunk_tokens: int,
    overlap_tokens: int,
    count: TokenCounter = count_tokens,
) -> list[ChunkData]:
    """Chunk each segment separately and number the chunks from 0.

    Guarantees: no chunk is empty or over chunk_tokens; each chunk keeps its
    segment's page number and section title; consecutive chunks from the same
    segment share up to overlap_tokens of text (whole words).
    """
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap_tokens must be at least 0 and less than chunk_tokens")

    chunks: list[ChunkData] = []
    for segment in segments:
        for content in _chunk_text(segment.text, chunk_tokens, overlap_tokens, count):
            chunks.append(
                ChunkData(
                    chunk_index=len(chunks),
                    content=content,
                    page_number=segment.page_number,
                    section_title=segment.section_title,
                    token_count=count(content),
                )
            )
    return chunks


def _chunk_text(
    text: str, chunk_tokens: int, overlap_tokens: int, count: TokenCounter
) -> list[str]:
    # Pieces leave room for the overlap that will be put in front of them.
    pieces = _split(text, chunk_tokens - overlap_tokens, count)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if not current or count(candidate.strip()) <= chunk_tokens:
            current = candidate
            continue
        # `piece` doesn't fit: close the current chunk and start the next one
        # with the end of this one, joined by the original separator.
        content = current.strip()
        chunks.append(content)
        separator = current[len(current.rstrip()) :]
        tail = _tail(content, overlap_tokens, count)
        while tail and count((tail + separator + piece).strip()) > chunk_tokens:
            tail = _drop_first_word(tail)
        current = tail + separator + piece if tail else piece

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split(
    text: str, limit: int, count: TokenCounter, level: int = 0, too_long: bool = False
) -> list[str]:
    """Cut text into pieces of at most `limit` tokens that concatenate back to
    exactly `text`, preferring the coarsest separator that works.

    too_long=True means the caller already counted `text` and it didn't fit.
    """
    if not too_long and count(text) <= limit:
        return [text]
    if level == len(_SEPARATORS):
        return _split_by_characters(text, limit, count)

    parts = _SEPARATORS[level].split(text)  # [text, sep, text, sep, ..., text]
    pieces = ["".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
    if len(pieces) == 1:  # this separator doesn't occur; try the next one
        return _split(text, limit, count, level + 1, too_long=True)
    # Only the pieces that are still too long get split further, so a long
    # paragraph doesn't cause its short neighbours to be broken into sentences.
    return [small for piece in pieces for small in _split(piece, limit, count, level + 1)]


def _split_by_characters(text: str, limit: int, count: TokenCounter) -> list[str]:
    pieces = []
    while text:
        # Find a prefix length that no longer fits by doubling, then binary
        # search below it for the longest prefix that does (at least 1 char).
        # Searching over the whole remaining text instead would re-count up to
        # megabytes for every piece, which is quadratic on huge inputs.
        low, high = 1, min(len(text), limit)
        while high < len(text) and count(text[:high]) <= limit:
            low, high = high, min(len(text), high * 2)
        while low < high:
            middle = (low + high + 1) // 2
            if count(text[:middle]) <= limit:
                low = middle
            else:
                high = middle - 1
        pieces.append(text[:low])
        text = text[low:]
    return pieces


def _tail(text: str, max_tokens: int, count: TokenCounter) -> str:
    """The longest run of whole words at the end of `text` within max_tokens.

    Never the entire text: repeating a whole chunk adds nothing.
    """
    starts = [match.start() for match in _WORD.finditer(text)]
    if max_tokens <= 0 or len(starts) < 2:
        return ""
    # Binary search for the earliest word start whose suffix still fits.
    low, high = 1, len(starts)
    while low < high:
        middle = (low + high) // 2
        if count(text[starts[middle] :]) <= max_tokens:
            high = middle
        else:
            low = middle + 1
    return text[starts[low] :] if low < len(starts) else ""


def _drop_first_word(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1] if len(parts) == 2 else ""
