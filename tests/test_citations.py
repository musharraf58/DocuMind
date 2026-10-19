import uuid

from app.services.citations import check_citations
from app.services.context import Source
from app.services.retrieval import RetrievedChunk


def sources(count: int) -> list[Source]:
    return [
        Source(
            n,
            RetrievedChunk(uuid.uuid4(), uuid.uuid4(), f"doc{n}.pdf", f"text {n}", n, None, n / 10),
            f"text {n}",
        )
        for n in range(1, count + 1)
    ]


def test_citations_map_to_the_sources_with_those_numbers():
    supplied = sources(3)
    checked = check_citations("Refunds take 30 days [2]. Shipping is free [3].", supplied)
    assert [s.chunk.chunk_id for s in checked.cited] == [
        supplied[1].chunk.chunk_id,
        supplied[2].chunk.chunk_id,
    ]
    assert checked.grounded
    assert checked.text == "Refunds take 30 days [2]. Shipping is free [3]."


def test_grouped_repeated_and_adjacent_citations():
    checked = check_citations("A [1, 3]. B [3][1]. C [1,2].", sources(3))
    assert [s.number for s in checked.cited] == [1, 3, 2]  # first mention order, no repeats


def test_unknown_numbers_are_dropped_from_citations_and_text():
    checked = check_citations("Paris [1] [7]. Also [2, 9]. Never [0].", sources(2))
    assert [s.number for s in checked.cited] == [1, 2]
    assert checked.text == "Paris [1]. Also [2]. Never."


def test_only_invalid_citations_means_not_grounded():
    checked = check_citations("It is 42 [5].", sources(2))
    assert checked.cited == []
    assert not checked.grounded
    assert checked.text == "It is 42."


def test_no_citations_at_all_means_not_grounded():
    checked = check_citations("I think it is probably 42.", sources(3))
    assert (checked.cited, checked.grounded) == ([], False)


def test_bracketed_years_are_not_citations():
    checked = check_citations("Published in [2024], see [1].", sources(1))
    assert checked.text == "Published in [2024], see [1]."
    assert [s.number for s in checked.cited] == [1]
