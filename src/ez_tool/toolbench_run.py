"""Reproduce the BM25 row of ToolGen Table 1 on ToolBench's retrieval split."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from .data.text import normalize
from .data.toolbench import SUBSET_OF_STAGE, load_corpus, load_queries
from .eval.metrics import aggregate, evaluate_one
from .paths import RESULTS_DIR, ensure_dirs, require_beegfs
from .retrieval.bm25 import BM25Retriever

# ToolGen Table 1, BM25 row. NDCG@1 / @3 / @5, x100.
TOOLGEN_PUBLISHED = {
    "multi_domain": {
        "I1": (22.77, 22.64, 25.61),
        "I2": (18.29, 20.74, 22.18),
        "I3": (10.00, 10.08, 12.33),
    },
    "in_domain": {
        "I1": (29.46, 31.12, 33.27),
        "I2": (24.13, 25.29, 27.65),
        "I3": (32.00, 25.88, 29.78),
    },
}


def _build_retriever(spec: dict):
    cfg = dict(spec)
    cfg.pop("name", None)
    cfg.pop("normalize_text", None)  # applied to text, not a retriever arg
    rtype = cfg.pop("type")
    if rtype == "bm25":
        return BM25Retriever(**cfg)
    if rtype == "cnb_prf":
        from .retrieval.cnb_prf import CnbPrfRetriever

        return CnbPrfRetriever(**cfg)
    if rtype == "bm25_okapi":
        # Imported lazily: rank_bm25 and nltk are only needed for replication.
        from .retrieval.bm25_okapi import BM25OkapiRetriever

        return BM25OkapiRetriever(**cfg)
    raise SystemExit(f"unknown retriever type: {rtype!r}")


def run(config_path: str) -> int:
    require_beegfs()
    ensure_dirs()
    cfg = yaml.safe_load(Path(config_path).read_text())

    root = Path(cfg["data_root"])
    if not root.exists():
        raise SystemExit(f"{root} not found -- ToolBench retrieval split missing")

    top_k = int(cfg.get("top_k", 100))
    ks = tuple(cfg.get("ks", [1, 3, 5]))
    splits = [tuple(s) for s in cfg["splits"]]

    record: dict = {"run_name": cfg["run_name"], "config": cfg, "results": {}}

    for setting in cfg["settings"]:
        record["results"][setting] = {}
        for rspec in cfg["retrievers"]:
            rname = rspec["name"]
            print(f"\n{'=' * 78}")
            print(f"  setting={setting}   retriever={rname}")
            print("=" * 78)

            per_group: dict[str, dict] = {}
            # In multi_domain the corpus is shared across stages, so index it
            # once and reuse; in_domain needs a fresh index per stage.
            cached_stage = None
            retriever = None

            for stage, split in splits:
                corpus_key = "G123" if setting == "multi_domain" else stage
                use_norm = bool(rspec.get("normalize_text", False))
                if corpus_key != cached_stage:
                    doc_ids, texts, cstats = load_corpus(root, stage, setting)
                    if use_norm:
                        texts = [normalize(t) for t in texts]
                    print(
                        f"  corpus[{corpus_key}]: {cstats['documents']:,} docs "
                        f"({cstats['rows_in_file']:,} rows, "
                        f"{cstats.get('duplicate_rows', 0):,} duplicates dropped)"
                    )
                    retriever = _build_retriever(rspec)
                    t0 = time.perf_counter()
                    retriever.index(doc_ids, texts)
                    print(f"  indexed in {time.perf_counter() - t0:.1f}s")
                    cached_stage = corpus_key

                queries, qstats = load_queries(root, stage, split, setting)
                qtexts = [normalize(q.text) if use_norm else q.text for q in queries]
                t0 = time.perf_counter()
                ranked = retriever.retrieve(qtexts, top_k=top_k)
                dt = time.perf_counter() - t0

                scores = [
                    evaluate_one([d for d, _ in hits], q.relevant, ks=ks)
                    for q, hits in zip(queries, ranked)
                ]
                agg = aggregate(scores)
                per_group[f"{stage}_{split}"] = agg

                print(
                    f"    {stage}_{split:<12} {qstats['queries']:>4} queries "
                    f"({qstats['query_rows']:>4} rows) | "
                    f"NDCG@1/3/5 {100 * agg['ndcg_toolgen@1']:5.2f} "
                    f"{100 * agg['ndcg_toolgen@3']:5.2f} "
                    f"{100 * agg['ndcg_toolgen@5']:5.2f} | {dt:5.1f}s"
                )

            record["results"][setting][rname] = per_group
            _print_table(per_group, setting, rname)

    out = RESULTS_DIR / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out / 'metrics.json'}")
    return 0


def _fold(per_group: dict, subset: str, metric: str, ks=(1, 3, 5)):
    """Average the (stage, split) groups belonging to one I1/I2/I3 column."""
    members = [
        g for g in per_group if SUBSET_OF_STAGE.get(g.split("_")[0]) == subset
    ]
    if not members:
        return None
    return [
        100 * sum(per_group[g][f"{metric}@{k}"] for g in members) / len(members)
        for k in ks
    ]


def _print_table(per_group: dict, setting: str, rname: str) -> None:
    pub = TOOLGEN_PUBLISHED[setting]
    print(f"\n  --- ToolGen Table 1 layout, {setting} / {rname} ---")
    print(
        "  %-30s %-20s %-20s" % ("row", "NDCG@1  @3     @5", "vs published")
    )
    for subset in ("I1", "I2", "I3"):
        ours_tg = _fold(per_group, subset, "ndcg_toolgen")
        ours_std = _fold(per_group, subset, "ndcg")
        if ours_tg is None:
            continue
        p = pub[subset]
        delta = [o - q for o, q in zip(ours_tg, p)]
        print(
            "  %-6s ours(their metric) %5.2f %5.2f %5.2f   published %5.2f %5.2f %5.2f"
            "   delta %+5.2f %+5.2f %+5.2f"
            % (subset, *ours_tg, *p, *delta)
        )
        print("  %-6s ours(standard NDCG) %4.2f %5.2f %5.2f" % ("", *ours_std))
