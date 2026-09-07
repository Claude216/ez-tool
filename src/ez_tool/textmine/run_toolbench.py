"""Does trial feedback beat a fixed ranking, inside a BM25 shortlist?

BM25 supplies the top-K candidates for each query. Two policies then spend the
same budget of attempts over those same candidates:

  bm25  -- work down the ranking (the no-learning control)
  cnb   -- work down the ranking until the first hit, then let complementary
           Bayes, trained on what has been tried, choose the next attempt

Recall after k attempts is recall@k, so these sit directly beside the BM25
baseline in the README. CNB sees ground truth on every attempt and BM25 sees
none, so this measures the value of feedback, not retrieval quality.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..data.corpus import load_corpus
from ..paths import RESULTS_DIR, require_beegfs
from ..retrieval.bm25 import BM25Retriever
from . import select, toolbench

KS = (1, 3, 5, 10, 20, 50, 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-features", type=int, default=5000)
    ap.add_argument("--pool", type=int, default=100, help="BM25 shortlist size")
    ap.add_argument("--budget", type=int, default=100, help="attempts per query")
    ap.add_argument("--norm", action="store_true", help="Rennie normalisation")
    ap.add_argument("--run-name", default="toolbench_cnb")
    args = ap.parse_args()
    require_beegfs()

    corpus = toolbench.build(top_features=args.top_features)
    queries, stats = toolbench.labelled_queries(corpus)
    print(f"corpus {corpus.n_docs} APIs x {len(corpus.vocab)} features; "
          f"{len(queries)} queries ({stats['policy']})")

    docs, _ = load_corpus()
    texts = [d.text(backfill_description=False, include_parameters=True)
             for d in docs]
    bm25 = BM25Retriever()
    bm25.index([d.doc_id for d in docs], texts)
    t0 = time.time()
    ranked = bm25.retrieve([q.normalized for q in queries], top_k=args.pool)
    print(f"bm25 shortlists in {time.time() - t0:.1f}s")

    per_policy: dict[str, list[dict[int, float]]] = {"bm25": [], "cnb": [], "fused": []}
    switched, ceiling = [], []
    t0 = time.time()
    for q, hits in zip(queries, ranked):
        pool = np.asarray([corpus.at[d] for d, _ in hits if d in corpus.at])
        rel = set(toolbench.relevant_rows(corpus, q).tolist())
        y_pool = np.asarray([r in rel for r in pool], dtype=bool)
        n_rel = len(rel)
        # What any re-ranking of this shortlist could reach at best.
        ceiling.append(int(y_pool.sum()) / n_rel if n_rel else float("nan"))

        X_pool = corpus.X[pool]
        a_b = select.bm25_order(pool, y_pool, args.budget)
        a_c = select.cnb_order(X_pool, y_pool, args.budget, norm=args.norm)
        a_f = select.fused_order(X_pool, y_pool, args.budget, norm=args.norm)
        per_policy["bm25"].append(a_b.recall_at(KS, n_rel))
        per_policy["cnb"].append(a_c.recall_at(KS, n_rel))
        per_policy["fused"].append(a_f.recall_at(KS, n_rel))
        switched.append(a_c.n_switched)
    print(f"{len(queries)} queries in {time.time() - t0:.1f}s")

    report = {
        "run_name": args.run_name,
        "config": vars(args),
        "n_queries": len(queries),
        "pool_recall_ceiling": round(float(np.nanmean(ceiling)), 4),
        "cnb_attempts_reordered_mean": round(float(np.mean(switched)), 1),
        "recall": {},
    }
    print(f"\nshortlist ceiling (recall@{args.pool}): "
          f"{report['pool_recall_ceiling']:.4f}")
    print(f"{'policy':<8}" + "".join(f"{'R@' + str(k):>9}" for k in KS))
    for policy, rows in per_policy.items():
        vals = {k: float(np.nanmean([r[k] for r in rows])) for k in KS}
        report["recall"][policy] = {str(k): round(v, 4) for k, v in vals.items()}
        print(f"{policy:<8}" + "".join(f"{vals[k]:>9.4f}" for k in KS))
    report["delta"] = {}
    for policy in ("cnb", "fused"):
        d = {k: report["recall"][policy][str(k)] - report["recall"]["bm25"][str(k)]
             for k in KS}
        report["delta"][policy] = {str(k): round(v, 4) for k, v in d.items()}
        print(f"{'d ' + policy:<8}" + "".join(f"{d[k]:>+9.4f}" for k in KS))

    out = RESULTS_DIR / "textmine" / f"{args.run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
