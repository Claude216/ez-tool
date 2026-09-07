"""The three curves section 7 needs, and why there are three.

`classifier_recall` is what ezr's `_tm_recall` computes and what figure 4
plots: train on the labels acquired so far, predict every document in the
corpus, and report the fraction of truly relevant papers the model calls
relevant. It answers "with a budget of N labels, what share of the relevant
papers can the model point at".

`found_recall` is Yu et al.'s |LR|/|R|: the fraction of relevant papers a
human has physically read and confirmed. It is what FASTREAD's X95 counts,
and it is bounded by the budget -- with 50 labels on Hall it cannot exceed
50/104 = 48%, whatever the model does.

Both are honest, but they are not the same number, and figure 4 draws
FASTREAD's X95 as a vertical rule on the classifier-recall axis. Only
`found_recall` is comparable to that rule, so every run records both.

`false_alarm` is fp/(fp+tn) over the whole corpus, matching the denominator
`classifier_recall` uses. The vendored commit implements neither it nor
`found_recall`; both are added here.
"""

from __future__ import annotations

import numpy as np


def classifier_recall(pred_yes: np.ndarray, y: np.ndarray) -> float:
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    return float((pred_yes & y).sum()) / n_pos


def false_alarm(pred_yes: np.ndarray, y: np.ndarray) -> float:
    n_neg = int((~y).sum())
    if not n_neg:
        return float("nan")
    return float((pred_yes & ~y).sum()) / n_neg


def found_recall(labeled: np.ndarray, y: np.ndarray) -> float:
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    return float((labeled & y).sum()) / n_pos


def ezr_recall_int(pred_yes: np.ndarray, y: np.ndarray) -> int:
    """`_tm_recall` verbatim, truncated to a whole percent, for validation."""
    n_pos = int(y.sum())
    if not n_pos:
        return 0
    return int(100 * (pred_yes & y).sum() / n_pos)
