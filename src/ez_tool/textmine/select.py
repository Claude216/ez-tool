"""CNB as a feedback re-ranker over a BM25 shortlist.

The setting: an agent has a query and a shortlist of candidate APIs. It tries
one, and the attempt tells it whether that API was the right one -- a label.
Complementary Bayes trains on what has been tried so far and reorders what is
left, so the next attempt is better informed than the last.

This is the shape section 7's active learner actually fits on this benchmark.
It is not a retriever: `retrieval.base.Retriever` promises a ranking with no
labels at all, and CNB cannot honour that. What it can do is spend an agent's
trials better than a fixed ranking does.

The comparison is therefore against BM25's own order over the same shortlist,
which is the no-learning control: identical candidates, identical budget, the
only difference being whether feedback is used. Recall after k attempts is
recall@k, so the numbers line up with the BM25 baseline in the README.

**This gives CNB information BM25 never sees.** Every attempt returns ground
truth. The question it answers is "does trial feedback help inside a
shortlist", not "is CNB a better retriever than BM25" -- the latter would be
an unfair comparison and is not claimed anywhere here.

Warm start: none. There is no pool of known positives to seed from -- about
two APIs in 22,000 are relevant. The agent simply works down the BM25 ranking
until the first success, which is FASTREAD's step 1, and only then does CNB
have a positive class to model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import cnb


@dataclass
class Attempt:
    """One query's trial sequence, under one policy."""

    order: list[int] = field(default_factory=list)  # pool positions, in order tried
    hit: list[bool] = field(default_factory=list)  # was each attempt relevant
    n_switched: int = 0  # attempts chosen by CNB rather than by BM25

    def recall_at(self, ks: tuple[int, ...], n_relevant: int) -> dict[int, float]:
        found = np.cumsum(np.asarray(self.hit, dtype=int))
        out = {}
        for k in ks:
            if not n_relevant:
                out[k] = float("nan")
            elif k <= len(found):
                out[k] = float(found[k - 1]) / n_relevant
            else:
                out[k] = float(found[-1]) / n_relevant if len(found) else 0.0
        return out


def bm25_order(pool: np.ndarray, y_pool: np.ndarray, budget: int) -> Attempt:
    """The control: work down BM25's ranking, no learning."""
    a = Attempt()
    for j in range(min(budget, len(pool))):
        a.order.append(j)
        a.hit.append(bool(y_pool[j]))
    return a


def cnb_order(
    X_pool,
    y_pool: np.ndarray,
    budget: int,
    norm: bool = False,
    alpha: float = 1.0,
    min_positives: int = 1,
    score: str = "margin",
) -> Attempt:
    """Try BM25's next candidate until a hit, then let CNB choose.

    `X_pool` is the shortlist's rows of the corpus matrix (sparse is fine).
    `y_pool[j]` is revealed only once position j has been tried.

    `score="yes"` ranks on `X @ w_yes`, which is what ezr's `tmActive` does.
    That is the faithful transplant and it is also the weaker one: because a
    class's weights are built from the documents outside it, `w_yes` is a
    function of the negatives alone. `score="margin"` uses
    `X @ w_yes - X @ w_no`, the comparison `_tm_best` makes when classifying,
    which is the only form that lets the positives influence the ranking.
    """
    n = len(y_pool)
    a = Attempt()
    tried = np.zeros(n, dtype=bool)

    for _ in range(min(budget, n)):
        seen_pos = int((tried & y_pool).sum())
        pick = None
        if seen_pos >= min_positives:
            w = cnb.fit(X_pool, y_pool, tried, alpha=alpha, norm=norm)
            if cnb.YES in w:
                s = np.asarray(cnb.scores(X_pool, w, cnb.YES)).ravel()
                if score == "margin" and cnb.NO in w:
                    s = s - np.asarray(cnb.scores(X_pool, w, cnb.NO)).ravel()
                s[tried] = -np.inf
                cand = int(np.argmax(s))
                if not tried[cand]:
                    pick = cand
                    a.n_switched += 1
        if pick is None:
            # Fall back to the best untried BM25 position: FASTREAD step 1.
            remaining = np.flatnonzero(~tried)
            if not remaining.size:
                break
            pick = int(remaining[0])
        tried[pick] = True
        a.order.append(pick)
        a.hit.append(bool(y_pool[pick]))
    return a


def fused_order(
    X_pool,
    y_pool: np.ndarray,
    budget: int,
    norm: bool = False,
    alpha: float = 1.0,
    k_rrf: float = 60.0,
) -> Attempt:
    """Blend CNB's opinion with BM25's instead of replacing it.

    `cnb_order` abandons the BM25 ranking the moment it has one positive to
    train on. With roughly two relevant APIs per query that positive is a
    single example over thousands of features, which is thin evidence to
    overturn a ranking built from the whole corpus. Reciprocal rank fusion
    keeps both: a candidate has to look good to BM25 *and* to CNB to be
    promoted.

    Pool position doubles as BM25 rank, since the pool arrives in BM25 order.
    """
    n = len(y_pool)
    a = Attempt()
    tried = np.zeros(n, dtype=bool)
    bm25_rr = 1.0 / (k_rrf + np.arange(n))

    for _ in range(min(budget, n)):
        remaining = np.flatnonzero(~tried)
        if not remaining.size:
            break
        pick = int(remaining[0])
        if (tried & y_pool).any():
            w = cnb.fit(X_pool, y_pool, tried, alpha=alpha, norm=norm)
            if cnb.YES in w:
                s = np.asarray(cnb.scores(X_pool, w, cnb.YES)).ravel()
                order = np.argsort(-s)
                cnb_rank = np.empty(n, dtype=float)
                cnb_rank[order] = np.arange(n)
                fused = bm25_rr + 1.0 / (k_rrf + cnb_rank)
                fused[tried] = -np.inf
                cand = int(np.argmax(fused))
                if not tried[cand]:
                    if cand != pick:
                        a.n_switched += 1
                    pick = cand
        tried[pick] = True
        a.order.append(pick)
        a.hit.append(bool(y_pool[pick]))
    return a
