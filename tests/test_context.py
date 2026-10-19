import uuid

from app.services.context import assemble_context
from app.services.retrieval import RetrievedChunk


def chunk(name: str, distance: float, words: int = 50, page=None, section=None):
    text = f"{name} " + " ".join(["word"] * words)
    return RetrievedChunk(uuid.uuid4(), uuid.uuid4(), f"{name}.pdf", text, page, section, distance)


def test_sources_are_numbered_best_first_with_their_location():
    chunks = [
        chunk("third", 0.5),
        chunk("first", 0.1, page=3),
        chunk("second", 0.3, section="Intro"),
    ]
    sources = assemble_context(chunks, token_budget=10_000)

    assert [(s.number, s.chunk.document_name) for s in sources] == [
        (1, "first.pdf"),
        (2, "second.pdf"),
        (3, "third.pdf"),
    ]
    assert sources[0].render().startswith('<source id="1" document="first.pdf" location="page 3">')
    assert 'location="section Intro"' in sources[1].render()


def test_budget_leaves_out_the_lowest_ranked_chunks_first():
    chunks = [chunk(f"c{rank}", distance=rank / 10) for rank in range(1, 6)]
    one_block = len(assemble_context(chunks[:1], 10_000)[0].render().split())
    # Room for about two and a half blocks (counting words as tokens here).
    sources = assemble_context(
        chunks, token_budget=int(one_block * 2.5), count=lambda t: len(t.split())
    )

    assert [s.chunk.document_name for s in sources] == ["c1.pdf", "c2.pdf"]  # c3-c5 left out


def test_stops_at_the_first_chunk_that_does_not_fit():
    # A small, low-ranked chunk is not squeezed in after a big one was left out:
    # sources are always the best ones, in order.
    chunks = [chunk("big1", 0.1, 100), chunk("big2", 0.2, 100), chunk("small", 0.3, 1)]
    words = lambda t: len(t.split())  # noqa: E731
    first_cost = words(assemble_context(chunks[:1], 10_000, count=words)[0].render())
    sources = assemble_context(chunks, token_budget=first_cost + 50, count=words)
    assert [s.chunk.document_name for s in sources] == ["big1.pdf"]


def test_best_chunk_alone_over_budget_is_shortened_not_dropped():
    sources = assemble_context([chunk("huge", 0.1, 2000)], token_budget=200)
    [only] = sources
    assert only.number == 1
    assert only.text.endswith(" ...")
    assert len(only.text) < len(only.chunk.content)
    from app.services.chunking import count_tokens

    assert count_tokens(only.render()) <= 200


def test_no_chunks_means_no_sources():
    assert assemble_context([], 3000) == []
