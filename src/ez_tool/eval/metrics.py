"""Binary-relevance ranking metrics.

All labels are binary (an API is relevant or it is not), so NDCG uses gain 1
per relevant document and the ideal ranking places every relevant document
first -- IDCG is the sum of the first ``min(|relevant|, k)`` discounts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_KS = (1, 3, 5, 10, 20, 50, 100)


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / k


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, did in enumerate(ranked[:k])
        if did in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def ndcg_at_k_toolgen(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    """NDCG as ToolGen computes it, for like-for-like comparison.

    Their ``compute_ndcg`` (evaluation/retrieval/eval_bm25.py) marks
    ``true_relevance`` only for documents that appear in the retrieved list,
    so relevant documents the retriever *missed* never enter the ideal
    ranking. IDCG is therefore taken over the number of relevant documents
    actually retrieved rather than over all of them, which forgives misses and
    scores strictly higher than the standard definition whenever recall is
    below 1. Reproduced here so their published numbers can be read on the
    same footing as ours, not because it is the better metric.
    """
    if not relevant:
        return float("nan")
    n_hits = sum(1 for did in ranked if did in relevant)
    if n_hits == 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, did in enumerate(ranked[:k])
        if did in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_hits, k)))
    return dcg / idcg if idcg else 0.0


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, did in enumerate(ranked):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_one(
    ranked: Sequence[str], relevant: set[str], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    scores = {"mrr": mrr(ranked, relevant)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        scores[f"ndcg_toolgen@{k}"] = ndcg_at_k_toolgen(ranked, relevant, k)
    return scores


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average over queries, skipping NaN (unlabelled) entries."""
    if not per_query:
        return {}
    out: dict[str, float] = {}
    for metric in per_query[0]:
        vals = [q[metric] for q in per_query if not math.isnan(q[metric])]
        out[metric] = sum(vals) / len(vals) if vals else float("nan")
    return out
