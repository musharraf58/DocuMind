import pytest

from app.services.fusion import reciprocal_rank_fusion as rrf


def items(result):
    return [item for item, _ in result]


def test_single_list_keeps_its_order_with_reciprocal_scores():
    result = rrf([["a", "b", "c"]], k=60)
    assert items(result) == ["a", "b", "c"]
    assert [score for _, score in result] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_item_in_both_lists_beats_items_in_one():
    # "c" is only 3rd in each list, but found by both searches.
    result = rrf([["a", "b", "c"], ["d", "e", "c"]], k=60)
    assert items(result)[0] == "c"
    assert dict(result)["c"] == pytest.approx(2 / 63)


def test_known_ordering_from_two_lists():
    vector = ["v1", "both", "v3"]
    keyword = ["both", "k2"]
    result = rrf([vector, keyword], k=60)
    # both: 1/62 + 1/61; v1: 1/61; k2: 1/62; v3: 1/63
    assert items(result) == ["both", "v1", "k2", "v3"]


def test_ties_are_broken_by_best_rank_then_first_appearance():
    # x is 1st in one list, y is 1st in the other: equal scores, equal best rank.
    assert items(rrf([["x"], ["y"]], k=60)) == ["x", "y"]
    assert items(rrf([["y"], ["x"]], k=60)) == ["y", "x"]
    # p = 1/61 + 1/63 vs q = 1/62 + 1/62: not equal, so ties don't decide here.
    result = rrf([["p", "q"], ["q", "z", "p"]], k=60)
    assert dict(result)["p"] != dict(result)["q"]


def test_no_duplicates_even_if_a_list_repeats_an_item():
    result = rrf([["a", "a", "b"], ["b", "a"]], k=60)
    assert sorted(items(result)) == ["a", "b"]
    assert dict(result)["a"] == pytest.approx(1 / 61 + 1 / 62)  # first position only


def test_smaller_k_rewards_top_positions_more():
    lists = [["a", "b"], ["b", "c"]]
    assert dict(rrf(lists, k=1))["a"] / dict(rrf(lists, k=1))["c"] > 1.4
    assert dict(rrf(lists, k=60))["a"] / dict(rrf(lists, k=60))["c"] < 1.02


def test_empty_input():
    assert rrf([]) == [] and rrf([[], []]) == []


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)


def test_same_input_same_output():
    lists = [["a", "b", "c", "d"], ["d", "c", "e"]]
    assert rrf(lists) == rrf(lists)
