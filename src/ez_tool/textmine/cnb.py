"""Complementary Naive Bayes, vectorised. Ports ezr `cnb`/`cnbLikes`.

Rennie et al.'s inversion: a class's weights are built from the documents that
do *not* belong to it, and classification reports the class a row is least
unlikely to not belong to. That behaves better than plain Naive Bayes when the
positive class is a one-in-200 rarity, which is exactly the SLR setting.

The arithmetic is ezr's, unchanged. Only the data layout differs: ezr walks
lists of rows through nested dicts, this holds the corpus as one (n_docs,
n_features) integer array, so training a class is a masked column sum and
scoring every document at once is a single matrix-vector product.

One inherited subtlety, see `predict_yes`: a document containing none of the
100 selected features scores exactly 0.0 for every class. ezr resolves that
tie with `max()` over a dict whose key order comes from set iteration, so the
outcome is hash-order dependent. Measured against the reference dump it always
resolves to "yes", which is the default here -- stated as an explicit rule so
this port does not inherit the hash dependence.
"""

from __future__ import annotations

import numpy as np

YES, NO = "yes", "no"


def fit(
    X: np.ndarray,
    y: np.ndarray,
    labeled: np.ndarray,
    alpha: float = 1.0,
    norm: bool = False,
) -> dict[str, np.ndarray]:
    """Weights per class present in `labeled`. Ports ezr `cnb`.

    `norm` is Rennie et al.'s recommendation to divide a class's weights by
    their absolute sum; ezr defaults it off (`--textmine.norm=0`) and the EZR
    paper plots both settings.
    """
    n_features = X.shape[1]
    # `np.asarray(...).ravel()` so a scipy sparse matrix works unchanged: its
    # .sum(axis=0) returns a 2-D matrix where a dense array returns 1-D.
    total = np.asarray(X[labeled].sum(axis=0), dtype=np.float64).ravel()
    T = total.sum()

    weights: dict[str, np.ndarray] = {}
    for klass, member in ((YES, y), (NO, ~y)):
        rows = labeled & member
        if not rows.any():
            continue  # ezr only builds weights for classes it has seen
        freq = np.asarray(X[rows].sum(axis=0), dtype=np.float64).ravel()
        den = T + n_features * alpha - freq.sum() + 1e-32
        w = -np.log((total + alpha - freq + 1e-32) / den)
        if norm:
            w = w / (np.abs(w).sum() or 1e-32)
        weights[klass] = w
    return weights


def scores(X: np.ndarray, weights: dict[str, np.ndarray], klass: str) -> np.ndarray:
    """Per-document score for one class. Ports ezr `cnbLikes`, all rows at once."""
    return X @ weights[klass]


def predict_yes(
    X: np.ndarray, weights: dict[str, np.ndarray], tie: str = YES
) -> np.ndarray:
    """Boolean mask of documents classified relevant. Ports ezr `_tm_best`.

    `tie` decides documents scoring identically under both classes -- in
    practice those holding none of the selected features, which score 0.0
    either way. `validate` shows ezr resolves these to "yes", so that is the
    default; every such document is a negative in all four corpora, so the
    rule puts a small floor under false alarm (6 of 1572 negatives on
    Kitchenham, 1 of 8807 on Hall, none elsewhere).
    """
    if YES not in weights:
        return np.zeros(X.shape[0], dtype=bool)
    if NO not in weights:
        return np.ones(X.shape[0], dtype=bool)
    return predict_from_scores(
        scores(X, weights, YES), scores(X, weights, NO), n=X.shape[0], tie=tie
    )


def predict_from_scores(
    s_yes: np.ndarray | None,
    s_no: np.ndarray | None,
    n: int,
    tie: str = YES,
) -> np.ndarray:
    """`predict_yes` for callers that already hold the score vectors."""
    if s_yes is None:
        return np.zeros(n, dtype=bool)
    if s_no is None:
        return np.ones(n, dtype=bool)
    return s_yes >= s_no if tie == YES else s_yes > s_no
