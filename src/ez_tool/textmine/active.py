"""The active-learning loop of section 7. Ports ezr `tmActive`/`_tm_warm`.

Warm start then greedy certainty sampling: retrain CNB on everything labelled
so far, label the unlabelled paper it ranks most strongly "yes", repeat. The
budget counts *total* labels, warm start included -- a budget of 200 with a
warm start of 24 buys 176 acquisitions, which is ezr's
`len(lab) >= the.learn.budget` semantics.

Warm start is the one place the paper and the code genuinely disagree, so it
is a switch:

* `random`  -- section 7.4's literal words, "24 randomly drawn papers".
* `oracle`  -- ezr `_tm_warm`: `yes` papers drawn from the known relevant set
               plus `no` drawn at random, which is Yu et al.'s expert seeding
               plus presumptive sampling. The vendored default is 20 + 20.

A uniform draw of 24 from Hall (104 relevant in 8911) yields no relevant paper
about three quarters of the time, leaving CNB with a single class -- vendored
ezr raises KeyError there. Rather than special-case it, `step` falls back to a
uniform draw whenever the model has never seen a positive. That is FASTREAD's
own step 1, "randomly sample from unlabeled candidate studies until 1
'relevant' example retrieved", and it spends budget like any other label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import cnb, metrics


@dataclass(frozen=True)
class WarmStart:
    mode: str = "random"  # "random" | "oracle"
    n: int = 24  # used by "random"
    yes_: int = 12  # used by "oracle"; trailing _ keeps YAML 1.1 from
    no_: int = 12   # parsing the config keys `yes`/`no` as booleans

    @property
    def size(self) -> int:
        return self.n if self.mode == "random" else self.yes_ + self.no_


@dataclass
class Trace:
    """One trial: metrics recorded after every label, warm start included."""

    n_labeled: list[int] = field(default_factory=list)
    classifier_recall: list[float] = field(default_factory=list)
    false_alarm: list[float] = field(default_factory=list)
    found_recall: list[float] = field(default_factory=list)
    n_random_picks: int = 0  # acquisitions made before the first positive


def draw_warm(
    y: np.ndarray, warm: WarmStart, rng: np.random.Generator
) -> np.ndarray:
    """Initial labelled mask."""
    n = len(y)
    labeled = np.zeros(n, dtype=bool)
    if warm.mode == "random":
        labeled[rng.choice(n, size=min(warm.n, n), replace=False)] = True
        return labeled
    if warm.mode == "oracle":
        # Ports `_tm_warm`: negatives are drawn from everything except the
        # chosen positives, so they may include other positives.
        pos = np.flatnonzero(y)
        chosen = rng.choice(pos, size=min(warm.yes_, len(pos)), replace=False)
        labeled[chosen] = True
        rest = np.flatnonzero(~labeled)
        labeled[rng.choice(rest, size=min(warm.no_, len(rest)), replace=False)] = True
        return labeled
    raise ValueError(f"unknown warm start mode: {warm.mode!r}")


def run_trial(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    warm: WarmStart,
    seed: int,
    norm: bool = False,
    alpha: float = 1.0,
    tie: str = cnb.YES,
) -> Trace:
    rng = np.random.default_rng(seed)
    labeled = draw_warm(y, warm, rng)
    trace = Trace()

    n_labeled = int(labeled.sum())
    while True:
        weights = cnb.fit(X, y, labeled, alpha=alpha, norm=norm)
        # One pass of scores serves both the metrics and the next acquisition,
        # and masking beats slicing: `X[~labeled]` would copy most of the
        # corpus on every step.
        s_yes = cnb.scores(X, weights, cnb.YES) if cnb.YES in weights else None
        s_no = cnb.scores(X, weights, cnb.NO) if cnb.NO in weights else None
        pred = cnb.predict_from_scores(s_yes, s_no, n=len(y), tie=tie)

        trace.n_labeled.append(n_labeled)
        trace.classifier_recall.append(metrics.classifier_recall(pred, y))
        trace.false_alarm.append(metrics.false_alarm(pred, y))
        trace.found_recall.append(metrics.found_recall(labeled, y))

        if n_labeled >= budget or n_labeled >= len(y):
            break

        if s_yes is not None:
            # np.argmax breaks ties toward the lowest index; ezr's max() over a
            # set breaks them by iteration order. Exact ties are needed for the
            # two to diverge, and scores here are sums of distinct floats.
            masked = np.where(labeled, -np.inf, s_yes)
            pick = int(np.argmax(masked))
        else:
            pick = int(rng.choice(np.flatnonzero(~labeled)))  # FASTREAD step 1
            trace.n_random_picks += 1
        labeled[pick] = True
        n_labeled += 1

    return trace
