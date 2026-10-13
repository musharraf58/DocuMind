import random

import pytest

from app.services.chunking import chunk_segments, count_tokens
from app.services.extraction import Segment

CHUNK = 100
OVERLAP = 15

WORDS = (
    "the quick brown fox jumps over lazy dog while researchers measure retrieval "
    "quality across many documents and careful engineers write small tests first"
).split()


def sentence(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18))).capitalize() + "."


def paragraph(rng: random.Random, sentences: int) -> str:
    return " ".join(sentence(rng) for _ in range(sentences))


def long_text(seed: int = 1, paragraphs: int = 30) -> str:
    rng = random.Random(seed)
    return "\n\n".join(paragraph(rng, rng.randint(1, 8)) for _ in range(paragraphs))


def overlap_between(previous: str, current: str) -> str:
    """The longest suffix of `previous` that `current` starts with."""
    for start in range(len(previous)):
        if current.startswith(previous[start:]):
            return previous[start:]
    return ""


def chunk(text: str, **kwargs) -> list:
    return chunk_segments([Segment(text)], kwargs.pop("size", CHUNK), OVERLAP, **kwargs)


def test_short_text_is_one_chunk_with_segment_metadata():
    [only] = chunk_segments([Segment("Hello world.", 3, "Intro")], CHUNK, OVERLAP)
    assert (only.chunk_index, only.content, only.page_number, only.section_title) == (
        0,
        "Hello world.",
        3,
        "Intro",
    )
    assert only.token_count == count_tokens("Hello world.")


def test_no_segments_means_no_chunks():
    assert chunk_segments([], CHUNK, OVERLAP) == []


@pytest.mark.parametrize("seed", range(5))
def test_no_chunk_is_empty_or_over_the_limit(seed):
    chunks = chunk(long_text(seed))
    assert len(chunks) > 5
    for c in chunks:
        assert c.content.strip() == c.content != ""
        assert c.token_count == count_tokens(c.content) <= CHUNK


@pytest.mark.parametrize("seed", range(5))
def test_consecutive_chunks_overlap(seed):
    chunks = chunk(long_text(seed))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = overlap_between(previous.content, current.content)
        assert shared, f"chunk {current.chunk_index} doesn't overlap the previous one"
        assert count_tokens(shared) <= OVERLAP


@pytest.mark.parametrize("seed", range(5))
def test_chunking_is_lossless(seed):
    """Dropping each chunk's overlap and gluing them together gives back the text."""
    text = long_text(seed)
    chunks = chunk(text)
    rebuilt = chunks[0].content
    for previous, current in zip(chunks, chunks[1:], strict=False):
        rebuilt += " " + current.content[len(overlap_between(previous.content, current.content)) :]
    assert rebuilt.split() == text.split()


def test_prefers_paragraph_boundaries():
    # Paragraphs of ~40 tokens: two fit in a chunk, three don't. Every chunk
    # should therefore end exactly at the end of a paragraph.
    rng = random.Random(7)
    paragraphs = []
    while len(paragraphs) < 12:
        candidate = paragraph(rng, 3)
        if 35 <= count_tokens(candidate) <= 42:
            paragraphs.append(candidate)
    chunks = chunk("\n\n".join(paragraphs))
    assert len(chunks) >= 4
    for c in chunks:
        assert any(c.content.endswith(p) for p in paragraphs), c.content[-60:]


def test_long_paragraph_splits_at_sentence_ends():
    text = paragraph(random.Random(3), 40)
    chunks = chunk(text)
    assert len(chunks) > 2
    assert all(c.content.endswith(".") for c in chunks)


def test_single_huge_word_is_cut_to_fit():
    chunks = chunk("x" * 5000)
    assert len(chunks) > 1
    assert all(0 < c.token_count <= CHUNK for c in chunks)
    assert "".join(c.content for c in chunks) == "x" * 5000


def test_special_token_text_is_treated_as_plain_text():
    [only] = chunk("Docs sometimes contain <|endoftext|> literally.")
    assert "<|endoftext|>" in only.content


def test_chunks_never_span_segments_and_are_numbered_globally():
    segments = [
        Segment(long_text(1, 8), page_number=1),
        Segment("Page two is short.", page_number=2),
        Segment(long_text(2, 8), page_number=3, section_title="Later"),
    ]
    chunks = chunk_segments(segments, CHUNK, OVERLAP)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.page_number for c in chunks] == sorted(c.page_number for c in chunks)
    [page_two] = [c for c in chunks if c.page_number == 2]
    assert page_two.content == "Page two is short."
    assert {c.section_title for c in chunks if c.page_number == 3} == {"Later"}


def test_zero_overlap_is_allowed():
    chunks = chunk_segments([Segment(long_text())], CHUNK, 0)
    rebuilt = " ".join(c.content for c in chunks)
    assert rebuilt.split() == long_text().split()


def test_rejects_overlap_not_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_segments([Segment("text")], 50, 50)


def test_uses_given_token_counter():
    def words(text: str) -> int:
        return len(text.split())

    chunks = chunk_segments([Segment(" ".join(["word"] * 100))], 30, 5, count=words)
    assert all(c.token_count <= 30 for c in chunks)
    assert all(c.token_count == words(c.content) for c in chunks)


@pytest.mark.parametrize("text", ["x" * 200_000, long_text(1, 700)], ids=["no-spaces", "prose"])
def test_work_grows_linearly_with_input(text):
    """Guards against quadratic behaviour (e.g. re-counting the whole remaining
    text for every piece) by measuring how much text the counter is given."""
    counted = 0

    def counting(s: str) -> int:
        nonlocal counted
        counted += len(s)
        return count_tokens(s)

    chunk_segments([Segment(text)], CHUNK, OVERLAP, count=counting)
    assert counted <= 40 * len(text)
