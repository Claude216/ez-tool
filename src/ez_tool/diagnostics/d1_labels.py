"""Diagnostic 1 -- label distribution and effective class count.

CNB's weight for class ``k`` is ``-log((total[a] + alpha - freq[k][a]) / den)``.
``freq[k][a]`` is estimated from that class's own rows, so when a class holds a
handful of rows its weight vector is dominated by the corpus-wide ``total[a]``
and converges on every other class's. The question this diagnostic answers is
how many classes are actually in that regime.

Levels differ in label cardinality: at tool and category level a G1 instance
carries one class, at API level it carries a set. Counts here are over
*mentions* -- a class's support is the number of instances listing it -- which
is the sample a per-class CNB estimator would actually see.
"""

from __future__ import annotations

import math
from collections import Counter

from ez_tool.diagnostics.g1 import G1Data, levels

LEVELS = ("category", "tool", "api")


def support(data: G1Data, level: str) -> Counter:
    """Instances per class at one level of the hierarchy."""
    c: Counter = Counter()
    for inst in data.instances:
        for k in levels(data, inst)[level]:
            c[k] += 1
    return c


def _quantile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolated quantile; avoids a numpy dependency in a stats path."""
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def perplexity(counts: Counter) -> float:
    """exp(H) of the class distribution: the effective number of classes.

    A uniform distribution over K classes gives exactly K. Anything skewed
    gives less, and the gap is the share of the nominal class count that
    carries no real mass.
    """
    n = sum(counts.values())
    if not n:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = v / n
        h -= p * math.log(p)
    return math.exp(h)


def summarise(data: G1Data, level: str) -> dict:
    counts = support(data, level)
    vals = sorted(counts.values())
    total = sum(vals)
    top10 = sum(v for _, v in counts.most_common(10))

    return {
        "level": level,
        "classes": len(counts),
        "label_mentions": total,
        "instances": len(data.instances),
        "min": vals[0],
        "p25": _quantile(vals, 0.25),
        "median": _quantile(vals, 0.50),
        "p75": _quantile(vals, 0.75),
        "max": vals[-1],
        "mean": total / len(vals),
        "n_eq_1": sum(1 for v in vals if v == 1),
        "n_le_3": sum(1 for v in vals if v <= 3),
        "n_le_5": sum(1 for v in vals if v <= 5),
        "n_le_10": sum(1 for v in vals if v <= 10),
        "top10_share": top10 / total,
        "perplexity": perplexity(counts),
        "top10_classes": counts.most_common(10),
        # The counts above say how much of the *label space* is starved; these
        # say how much of the *data* lands in a starved class, which is what a
        # per-query error rate would actually see.
        "share_classes_le_3": sum(1 for v in vals if v <= 3) / len(vals),
        "share_classes_le_5": sum(1 for v in vals if v <= 5) / len(vals),
        "mentions_in_classes_le_3": sum(v for v in vals if v <= 3) / total,
        "mentions_in_classes_le_5": sum(v for v in vals if v <= 5) / total,
        "mentions_in_classes_le_10": sum(v for v in vals if v <= 10) / total,
        "histogram": dict(Counter(vals)),
    }


def verdict(row: dict) -> str:
    """The decision rule, stated rather than left to the reader."""
    if row["median"] < 5:
        return (
            f"NOT VIABLE at {row['level']} level: median support "
            f"{row['median']:.0f} < 5 instances per class."
        )
    return (
        f"VIABLE at {row['level']} level on support grounds: median "
        f"{row['median']:.0f} instances per class."
    )


def run(data: G1Data) -> dict:
    out = {}
    for level in LEVELS:
        row = summarise(data, level)
        row["verdict"] = verdict(row)
        out[level] = row
    return out
