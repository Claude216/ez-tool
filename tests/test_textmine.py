"""Unit tests for the section 7 replication.

Deliberately free of the corpora: fidelity to vendored ezr is checked by
`ez_tool.textmine.validate` against a reference dump, which needs beegfs and
a Python 3.12 interpreter. What is checked here is the logic that no reference
dump covers -- ordering rules, the budget contract, and the metric definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ez_tool.textmine import active, cnb, features, metrics


def test_tfidf_keeps_top_features_by_score():
    # "alpha" is in every document, so log(N/df) = 0 and its score is 0.
    docs = [["alpha", "beta", "beta"], ["alpha", "gamma"], ["alpha", "beta"]]
    f = features.tfidf(docs, top=2)
    assert f.vocab[0] == "beta"
    assert "alpha" not in f.vocab or f.vocab.index("alpha") == 1
    assert f.X.shape == (3, 2)
    assert f.X[0, 0] == 2  # beta appears twice in the first document


def test_tfidf_ties_keep_first_appearance_order():
    """Equal scores must fall back to document-frequency insertion order.

    "ccc" wins outright on rarity; "bbb" and "aaa" score identically, so their
    order is decided by which was seen first while scanning documents.
    """
    docs = [["bbb", "aaa"], ["bbb", "aaa"], ["ccc"]]
    f = features.tfidf(docs, top=3)
    assert f.vocab == ["ccc", "bbb", "aaa"]


def test_stem_strips_one_suffix_only():
    """ezr recurses with n-1 from a default of 1, so exactly one strip."""
    cache: dict[str, str] = {}
    assert features._stem1("willingness", ["ness", "ing"], cache) == "willing"


def test_stem_keeps_words_it_would_halve():
    """ezr refuses a strip leaving a stem under half the original length."""
    cache: dict[str, str] = {}
    assert features._stem1("testing", ["ing"], cache) == "test"
    assert features._stem1("organization", ["ization"], cache) == "organization"


def test_cnb_weights_are_complementary():
    """A class's weights come from the documents that are not in it."""
    X = np.array([[4, 0], [0, 4], [3, 1]], dtype=np.int32)
    y = np.array([True, False, False])
    labeled = np.ones(3, dtype=bool)
    w = cnb.fit(X, y, labeled)
    assert set(w) == {"yes", "no"}
    # Feature 0 is common outside "no", so it weighs against "no" more than
    # feature 1 does.
    assert w["no"][0] < w["no"][1]


def test_cnb_single_class_predicts_that_class():
    X = np.array([[1, 0], [0, 1]], dtype=np.int32)
    y = np.array([False, False])
    w = cnb.fit(X, y, np.ones(2, dtype=bool))
    assert set(w) == {"no"}
    assert not cnb.predict_yes(X, w).any()


def test_predict_tie_rule():
    """Documents with no features score 0.0 either way; ezr calls them yes."""
    s = np.zeros(3)
    assert cnb.predict_from_scores(s, s, n=3, tie="yes").all()
    assert not cnb.predict_from_scores(s, s, n=3, tie="no").any()


def test_metrics_definitions():
    y = np.array([True, True, False, False])
    pred = np.array([True, False, True, False])
    assert metrics.classifier_recall(pred, y) == 0.5
    assert metrics.false_alarm(pred, y) == 0.5
    assert metrics.found_recall(np.array([True, False, False, False]), y) == 0.5
    assert metrics.ezr_recall_int(pred, y) == 50


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(200, 8)).astype(np.int32)
    y = np.zeros(200, dtype=bool)
    y[:20] = True
    X[:20] += 3  # give the positives a separable signal
    return X, y


def test_budget_counts_warm_start(toy):
    X, y = toy
    warm = active.WarmStart(mode="oracle", yes_=5, no_=5)
    t = active.run_trial(X, y, budget=30, warm=warm, seed=0)
    assert t.n_labeled[0] == warm.size == 10
    assert t.n_labeled[-1] == 30
    assert len(t.n_labeled) == 30 - warm.size + 1


def test_found_recall_is_monotone(toy):
    X, y = toy
    t = active.run_trial(X, y, budget=60, warm=active.WarmStart(n=10), seed=1)
    fr = t.found_recall
    assert all(b >= a for a, b in zip(fr, fr[1:]))
    assert fr[-1] <= 1.0


def test_random_warm_start_falls_back_until_first_positive():
    """FASTREAD step 1: with no positive seen, picks are uniform, on budget."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(400, 6)).astype(np.int32)
    y = np.zeros(400, dtype=bool)
    y[:2] = True  # 2 positives in 400 -- a random 4 will almost never hit one
    t = active.run_trial(X, y, budget=20, warm=active.WarmStart(n=4), seed=3)
    assert t.n_random_picks >= 1
    assert t.n_labeled[-1] == 20


def test_warm_start_modes_produce_requested_sizes(toy):
    _, y = toy
    rng = np.random.default_rng(0)
    assert active.draw_warm(y, active.WarmStart(n=17), rng).sum() == 17
    oracle = active.WarmStart(mode="oracle", yes_=6, no_=9)
    m = active.draw_warm(y, oracle, rng)
    assert m.sum() == 15
    assert (m & y).sum() >= 6  # negatives are drawn from the rest, may add more
