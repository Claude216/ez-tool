"""Follow-up A -- decompose the leakage number under harder candidate pools.

Round one ranked ground truth against all 10,355 G1 APIs and found top-5 =
53.7%. A uniformly drawn competitor almost always comes from an unrelated
category, so that number mixes two effects: topical relatedness between query
and correct API, which is signal any retriever should use, and vocabulary the
generator copied verbatim out of the documentation, which is the artifact. Only
the second is leakage.

Restricting the pool separates them. Everything except the pool is held fixed --
same tokenizer (EZR pipeline, identifier splitting on), same documentation
fields, same length-normalised Jaccard ranker.

    P0  all APIs                     baseline, reproduces round one
    P1  same category as ground truth  removes cross-domain separation
    P2  same tool as ground truth      hardest; near-identical documentation
    P3  random non-relevant, |P3| = |P1|   controls for pool size alone

P3 is the control and carries the argument: without it, any P0 -> P1 improvement
is confounded with the pool merely being smaller.
"""

from __future__ import annotations

import numpy as np

from ez_tool.diagnostics import textpipe
from ez_tool.diagnostics.d2_leakage import _binary_matrix, api_doc_text
from ez_tool.diagnostics.g1 import G1Data, _norm

SEED = 20260902
QUERY_CHUNK = 2000
POOLS = ("P0", "P1", "P2", "P3")

# A pool this small cannot express a rank; the instance is excluded and counted.
MIN_P1 = 5
MIN_P2 = 2


def _chance_topk(n: int, g: int, k: int) -> float:
    """P(a random ranking puts some ground truth in the top k).

    1 - C(n-g, k)/C(n, k). Without this, small pools look like triumphs: P2's
    median pool holds 4 APIs of which ~2 are ground truth, so "top-5" is every
    candidate and "top-1" is a coin flip.
    """
    if k >= n:
        return 1.0
    p_miss = 1.0
    for j in range(k):
        p_miss *= (n - g - j) / (n - j)
        if p_miss <= 0:
            return 1.0
    return 1.0 - p_miss


def _rank(scores: np.ndarray, gt_local: np.ndarray) -> tuple[float, float]:
    """(optimistic, random-tie-break) rank of the best ground-truth entry."""
    best = scores[gt_local].max()
    greater = int((scores > best).sum())
    ties = int((scores == best).sum())
    return greater + 1.0, greater + (ties + 1) / 2.0


def run(data: G1Data) -> dict:
    rng = np.random.default_rng(SEED)
    cache: dict = {}

    keys = list(data.docs)
    key_index = {k: i for i, k in enumerate(keys)}
    n_doc = len(keys)

    doc_sets = [
        set(textpipe.prepare(api_doc_text(data.docs[k]), cache, split_identifiers=True))
        for k in keys
    ]
    query_sets = [
        set(textpipe.prepare(i.query, cache, split_identifiers=True))
        for i in data.instances
    ]
    doc_len = np.array([len(s) for s in doc_sets], dtype=np.float32)
    q_len = np.array([len(s) for s in query_sets], dtype=np.float32)

    vocab_index = {t: i for i, t in enumerate(sorted(set().union(*query_sets)))}
    Q = _binary_matrix(query_sets, vocab_index)
    Dt = _binary_matrix(doc_sets, vocab_index).T.tocsc()

    # Pool membership, precomputed once. Category comes from each API's own
    # category_name; the 41 tool names that span categories therefore
    # contribute to each category they appear under.
    by_cat: dict[str, list[int]] = {}
    by_tool: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_cat.setdefault(_norm(data.docs[k].category), []).append(i)
        by_tool.setdefault(k[0], []).append(i)
    by_cat = {k: np.asarray(v) for k, v in by_cat.items()}
    by_tool = {k: np.asarray(v) for k, v in by_tool.items()}

    n_q = len(data.instances)
    res = {
        p: {
            "opt": np.full(n_q, np.nan),
            "exp": np.full(n_q, np.nan),
            "size": np.full(n_q, np.nan),
            "chance1": np.full(n_q, np.nan),
            "chance5": np.full(n_q, np.nan),
        }
        for p in POOLS
    }
    excluded = {p: 0 for p in POOLS}

    # Round one's headline was a 12.4x Jaccard ratio against uniformly drawn
    # negatives. The same ratio against same-category and same-tool negatives
    # says how much of it was cross-domain separation.
    ov_true = np.full(n_q, np.nan)
    ov_rand = np.full(n_q, np.nan)
    ov_cat = np.full(n_q, np.nan)
    ov_tool = np.full(n_q, np.nan)

    all_idx = np.arange(n_doc)

    for start in range(0, n_q, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, n_q)
        inter = np.asarray((Q[start:stop] @ Dt).todense(), dtype=np.float32)
        union = q_len[start:stop, None] + doc_len[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        for r in range(stop - start):
            i = start + r
            inst = data.instances[i]
            row = jac[r]
            gt = np.asarray([key_index[k] for k in inst.apis])

            gt_cats = {_norm(data.docs[k].category) for k in inst.apis}
            p1 = np.unique(np.concatenate([by_cat[c] for c in gt_cats]))
            p2 = by_tool[inst.apis[0][0]]

            # P3: ground truth plus uniform non-relevant filler, sized to P1.
            n_fill = max(len(p1) - len(gt), 0)
            mask = np.ones(n_doc, dtype=bool)
            mask[gt] = False
            others = all_idx[mask]
            fill = rng.choice(others, size=min(n_fill, len(others)), replace=False)
            p3 = np.concatenate([gt, fill])

            gt_set = set(gt.tolist())
            ov_true[i] = row[gt].max()
            ov_rand[i] = row[rng.choice(others, size=min(10, len(others)),
                                        replace=False)].mean()
            cat_neg = np.asarray([j for j in p1.tolist() if j not in gt_set])
            if len(cat_neg):
                ov_cat[i] = row[rng.choice(cat_neg,
                                           size=min(10, len(cat_neg)),
                                           replace=False)].mean()
            tool_neg = np.asarray([j for j in p2.tolist() if j not in gt_set])
            if len(tool_neg):
                ov_tool[i] = row[tool_neg].mean()

            for name, pool, floor in (
                ("P0", all_idx, 0),
                ("P1", p1, MIN_P1),
                ("P2", p2, MIN_P2),
                ("P3", p3, MIN_P1),
            ):
                if len(pool) < floor:
                    excluded[name] += 1
                    continue
                local = np.flatnonzero(np.isin(pool, gt))
                opt, exp = _rank(row[pool], local)
                res[name]["opt"][i] = opt
                res[name]["exp"][i] = exp
                res[name]["size"][i] = len(pool)
                res[name]["chance1"][i] = _chance_topk(len(pool), len(local), 1)
                res[name]["chance5"][i] = _chance_topk(len(pool), len(local), 5)

    out: dict = {
        "n_queries": n_q,
        "n_apis": n_doc,
        "ranker": "length-normalised Jaccard, unchanged from round one",
        "excluded": excluded,
        "pools": {},
    }
    for p in POOLS:
        ok = ~np.isnan(res[p]["exp"])
        exp = res[p]["exp"][ok]
        opt = res[p]["opt"][ok]
        size = res[p]["size"][ok]
        out["pools"][p] = {
            "instances": int(ok.sum()),
            "pool_size": {
                "min": float(size.min()),
                "median": float(np.median(size)),
                "max": float(size.max()),
                "mean": float(size.mean()),
            },
            "median_rank_expected": float(np.median(exp)),
            "median_rank_optimistic": float(np.median(opt)),
            "top1": float((exp <= 1).mean()),
            "top5": float((exp <= 5).mean()),
            "top10": float((exp <= 10).mean()),
            "top1_optimistic": float((opt <= 1).mean()),
            "top5_optimistic": float((opt <= 5).mean()),
            "top10_optimistic": float((opt <= 10).mean()),
            # Rank as a fraction of the pool: the only cross-pool comparison
            # that is not dominated by how many candidates each pool holds.
            "normalised_rank_median": float(np.median(exp / size)),
            "normalised_rank_mean": float((exp / size).mean()),
            # What a ranker that ignored the query entirely would score on
            # these same pools, accounting for how many entries are ground truth.
            "chance_top1": float(res[p]["chance1"][ok].mean()),
            "chance_top5": float(res[p]["chance5"][ok].mean()),
            "lift_top1": float((exp <= 1).mean() / max(res[p]["chance1"][ok].mean(), 1e-12)),
            "lift_top5": float((exp <= 5).mean() / max(res[p]["chance5"][ok].mean(), 1e-12)),
        }

    def _ov(a):
        m = ~np.isnan(a)
        return {"n": int(m.sum()), "mean": float(a[m].mean()),
                "median": float(np.median(a[m]))}

    out["overlap"] = {
        "ground_truth": _ov(ov_true),
        "random_any_category": _ov(ov_rand),
        "random_same_category": _ov(ov_cat),
        "same_tool_non_relevant": _ov(ov_tool),
    }
    base = out["overlap"]["ground_truth"]["mean"]
    out["overlap_ratios"] = {
        "vs_random_any_category": base / out["overlap"]["random_any_category"]["mean"],
        "vs_random_same_category": base / out["overlap"]["random_same_category"]["mean"],
        "vs_same_tool": base / out["overlap"]["same_tool_non_relevant"]["mean"],
    }

    out["_arrays"] = {p: res[p] for p in POOLS}
    out["_arrays"]["overlap"] = {
        "true": ov_true, "rand": ov_rand, "cat": ov_cat, "tool": ov_tool,
    }
    p1, p3 = out["pools"]["P1"], out["pools"]["P3"]
    gap = p1["top5"] - p3["top5"]
    if gap >= 0.10:
        out["verdict"] = (
            f"LEAKAGE SUBSTANTIAL: at equal pool size, same-category negatives "
            f"leave top-5 at {p1['top5']:.1%} against the control's "
            f"{p3['top5']:.1%} (+{gap:.1%}); round one's verdict stands."
        )
    else:
        direction = "below" if gap < 0 else "within reach of"
        out["verdict"] = (
            f"DOWNGRADE round one: at equal pool size, same-category negatives "
            f"drop top-5 to {p1['top5']:.1%}, {direction} the size-matched "
            f"control's {p3['top5']:.1%} ({gap:+.1%}). The 53.7% figure measured "
            f"pool size and cross-domain topicality, not copied wording."
        )
    return out
