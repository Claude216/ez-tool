"""Diagnostic 2 -- lexical leakage from reverse generation.

ToolBench instructions were written by prompting ChatGPT with sampled API
documentation, so documentation vocabulary flows into the queries. Any lexical
method scored on this data is partly measuring that flow. Two measurements:

* **Jaccard** between a query's term set and its ground-truth API's
  documentation, against the same statistic for randomly drawn non-relevant
  APIs. The ratio is the leakage signal, isolated from how wordy either side is.
* **A zero-parameter overlap ranker.** Rank the whole G1 API pool by raw
  ``|Q & D|`` and ask where ground truth lands. No weighting, no tuning, no
  training -- if this already succeeds, lexical matching is not the hard part
  of the task and the benchmark cannot support a usability claim.

Both sides go through the same tokenizer, EZR's pipeline *with* identifier
splitting enabled. Splitting is on because leakage is a question about
vocabulary, not about EZR's tokenizer; what EZR's own tokenizer destroys is
measured separately in :mod:`d3_sparsity`.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ez_tool.diagnostics import textpipe
from ez_tool.diagnostics.g1 import G1Data

N_RANDOM = 10
SEED = 20260902
QUERY_CHUNK = 2000


def api_doc_text(doc) -> str:
    """The documentation fields concatenated for this diagnostic.

    ``api_name`` + ``api_description`` + every required and optional parameter's
    name and description. Deliberately excludes ``tool_name`` and
    ``category_name`` -- both are label components at the coarser levels, and
    folding them in would let a label leak into its own feature vector.
    Parameter *defaults* are excluded everywhere in this repo (base64 payloads).
    """
    parts = [doc.api, doc.description]
    for name, desc in doc.required + doc.optional:
        parts.append(name)
        parts.append(desc)
    return " ".join(parts)


def _binary_matrix(term_sets, vocab_index) -> sparse.csr_matrix:
    indptr, indices = [0], []
    for ts in term_sets:
        indices.extend(vocab_index[t] for t in ts if t in vocab_index)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(indptr) - 1, len(vocab_index)),
    )


def run(data: G1Data) -> dict:
    rng = np.random.default_rng(SEED)
    cache: dict = {}

    keys = list(data.docs)
    key_index = {k: i for i, k in enumerate(keys)}
    doc_sets = [
        set(textpipe.prepare(api_doc_text(data.docs[k]), cache, split_identifiers=True))
        for k in keys
    ]
    query_sets = [
        set(textpipe.prepare(i.query, cache, split_identifiers=True))
        for i in data.instances
    ]

    # |Q & D| only involves terms a query actually has, so restricting the
    # matrices to query vocabulary is exact, not an approximation. Union sizes
    # use the untruncated |D| computed here.
    doc_len = np.array([len(s) for s in doc_sets], dtype=np.float32)
    q_len = np.array([len(s) for s in query_sets], dtype=np.float32)

    vocab = sorted(set().union(*query_sets)) if query_sets else []
    vocab_index = {t: i for i, t in enumerate(vocab)}
    Q = _binary_matrix(query_sets, vocab_index)
    D = _binary_matrix(doc_sets, vocab_index)
    Dt = D.T.tocsc()

    n_q, n_doc = len(query_sets), len(doc_sets)
    gt_rows = [[key_index[k] for k in inst.apis] for inst in data.instances]

    overlap_true = np.zeros(n_q, dtype=np.float32)
    overlap_rand = np.zeros(n_q, dtype=np.float32)
    rank_expected = np.zeros(n_q, dtype=np.float64)
    rank_optimistic = np.zeros(n_q, dtype=np.float64)
    rank_jaccard = np.zeros(n_q, dtype=np.float64)

    for start in range(0, n_q, QUERY_CHUNK):
        stop = min(start + QUERY_CHUNK, n_q)
        inter = np.asarray((Q[start:stop] @ Dt).todense(), dtype=np.float32)

        # Jaccard against every API at once; |Q u D| = |Q| + |D| - |Q & D|.
        union = q_len[start:stop, None] + doc_len[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        for r in range(stop - start):
            i = start + r
            gt = gt_rows[i]
            # Best-matching ground-truth API: a query with several relevant
            # APIs is served by finding any one of them.
            overlap_true[i] = jac[r, gt].max()

            mask = np.ones(n_doc, dtype=bool)
            mask[gt] = False
            pool = np.flatnonzero(mask)
            draws = rng.choice(pool, size=N_RANDOM, replace=False)
            overlap_rand[i] = jac[r, draws].mean()

            # Rank by raw overlap. Integer scores tie heavily, so report both
            # the optimistic bound (all ties resolved in ground truth's favour)
            # and the expectation under random tie-breaking.
            best = inter[r, gt].max()
            greater = int((inter[r] > best).sum())
            ties = int((inter[r] == best).sum())
            rank_optimistic[i] = greater + 1
            rank_expected[i] = greater + (ties + 1) / 2.0

            # Raw overlap rewards long documents; Jaccard is the same ranker
            # with the one correction every real lexical retriever makes.
            jbest = jac[r, gt].max()
            jgreater = int((jac[r] > jbest).sum())
            jties = int((jac[r] == jbest).sum())
            rank_jaccard[i] = jgreater + (jties + 1) / 2.0

    def pct(a, qs=(5, 25, 50, 75, 95)):
        return {f"p{q}": float(np.percentile(a, q)) for q in qs}

    ratios = overlap_true / np.maximum(overlap_rand, 1e-12)

    by_cat: dict[str, dict] = {}
    cats = np.array(
        [data.docs[i.apis[0]].category or "(none)" for i in data.instances]
    )
    for c in sorted(set(cats.tolist())):
        m = cats == c
        by_cat[c] = {
            "instances": int(m.sum()),
            "overlap_true_mean": float(overlap_true[m].mean()),
            "overlap_rand_mean": float(overlap_rand[m].mean()),
            "ratio_of_means": float(
                overlap_true[m].mean() / max(overlap_rand[m].mean(), 1e-12)
            ),
            "top5_rate_expected": float((rank_expected[m] <= 5).mean()),
            "median_rank_expected": float(np.median(rank_expected[m])),
            "top5_rate_jaccard": float((rank_jaccard[m] <= 5).mean()),
            "median_rank_jaccard": float(np.median(rank_jaccard[m])),
        }

    out = {
        "n_queries": n_q,
        "n_apis": n_doc,
        "tokenizer": "ezr pipeline, split_identifiers=True, both sides",
        "doc_fields": "api_name + api_description + required/optional param names and descriptions",
        "overlap_true": {"mean": float(overlap_true.mean()), **pct(overlap_true)},
        "overlap_rand": {"mean": float(overlap_rand.mean()), **pct(overlap_rand)},
        "ratio_of_means": float(overlap_true.mean() / max(overlap_rand.mean(), 1e-12)),
        # Per-instance ratios divide by an overlap_rand that is exactly zero
        # for some instances, so only order statistics are meaningful here.
        "ratio_per_instance": pct(ratios),
        "ratio_per_instance_undefined": int((overlap_rand == 0).sum()),
        "rank_expected": {
            "median": float(np.median(rank_expected)),
            "mean": float(rank_expected.mean()),
            **pct(rank_expected),
            "top1": float((rank_expected <= 1).mean()),
            "top5": float((rank_expected <= 5).mean()),
            "top10": float((rank_expected <= 10).mean()),
        },
        "rank_optimistic": {
            "median": float(np.median(rank_optimistic)),
            "top1": float((rank_optimistic <= 1).mean()),
            "top5": float((rank_optimistic <= 5).mean()),
            "top10": float((rank_optimistic <= 10).mean()),
        },
        "rank_jaccard": {
            "median": float(np.median(rank_jaccard)),
            "mean": float(rank_jaccard.mean()),
            **pct(rank_jaccard),
            "top1": float((rank_jaccard <= 1).mean()),
            "top5": float((rank_jaccard <= 5).mean()),
            "top10": float((rank_jaccard <= 10).mean()),
        },
        "by_category": by_cat,
        "_arrays": {
            "overlap_true": overlap_true,
            "overlap_rand": overlap_rand,
            "rank_expected": rank_expected,
            "rank_jaccard": rank_jaccard,
        },
    }
    # The decision rule keys on the best parameter-free ranker available, which
    # is the length-normalised one; raw overlap is reported beside it to show
    # how much of the gap is length normalisation rather than leakage.
    best_top5 = max(out["rank_expected"]["top5"], out["rank_jaccard"]["top5"])
    out["verdict"] = (
        f"CONFOUNDED: a parameter-free overlap ranker puts ground truth in the "
        f"top 5 for {best_top5:.1%} of instances."
        if best_top5 >= 0.5
        else (
            f"NOT confounded by this test: the best parameter-free overlap "
            f"ranker reaches top-5 on only {best_top5:.1%} of instances, well "
            f"short of a large majority -- but see the {out['ratio_of_means']:.1f}x "
            f"Jaccard ratio, which shows the leakage is real."
        )
    )
    return out
