import math

from ez_tool.data.text import normalize
from ez_tool.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_counts_only_within_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "d"}, 2) == 0.5
    assert recall_at_k(ranked, {"a", "d"}, 4) == 1.0


def test_precision_divides_by_k_not_by_hits():
    assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5


def test_ndcg_is_one_for_a_perfect_prefix():
    # Both relevant docs ranked first -> DCG equals IDCG.
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_ndcg_penalises_a_later_hit():
    good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    worse = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert good == 1.0 and worse < good


def test_mrr_uses_the_first_hit_only():
    assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_unlabelled_queries_are_nan_not_zero():
    # A query with no ground truth must not be averaged in as a failure.
    assert math.isnan(recall_at_k(["a"], set(), 5))
    assert math.isnan(ndcg_at_k(["a"], set(), 5))


def test_normalize_splits_code_shaped_identifiers():
    assert normalize("getUserById") == "get user by id"
    assert normalize("Get_All-Climate/News") == "get all climate news"
    assert normalize("v1/stock-quote.json") == "v1 stock quote json"
