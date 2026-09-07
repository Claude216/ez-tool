"""Command line entry point.

    ez-tool inspect                      # corpus / query / label-coverage stats
    ez-tool run --config configs/bm25_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import orjson
import yaml

from .data.corpus import load_corpus
from .data.queries import apply_label_policy, check_label_coverage, load_queries
from .eval.metrics import aggregate, evaluate_one
from .paths import GROUPS, RESULTS_DIR, RUNS_DIR, ensure_dirs, require_beegfs
from .retrieval.bm25 import BM25Retriever

from .retrieval.cnb_prf import CnbPrfRetriever

RETRIEVERS = {"bm25": BM25Retriever, "cnb_prf": CnbPrfRetriever}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _fmt_row(label: str, scores: dict, ks=(1, 5, 10, 20, 100)) -> str:
    cells = "".join(f"{scores.get(f'recall@{k}', float('nan')):>9.3f}" for k in ks)
    return f"  {label:<22}{cells}{scores.get('ndcg@10', float('nan')):>10.3f}{scores.get('mrr', float('nan')):>8.3f}"


def _header(ks=(1, 5, 10, 20, 100)) -> str:
    cells = "".join(f"{'R@' + str(k):>9}" for k in ks)
    return f"  {'group':<22}{cells}{'NDCG@10':>10}{'MRR':>8}"


def cmd_inspect(_: argparse.Namespace) -> int:
    require_beegfs()
    docs, corpus_stats = load_corpus()
    queries = load_queries()
    coverage = check_label_coverage(queries, {d.doc_id for d in docs})

    print("=== corpus ===")
    for k, v in corpus_stats.items():
        print(f"  {k:<28} {v:>8,}")

    empty = corpus_stats.get("apis_empty_description", 0)
    print(f"  {'empty-description share':<28} {empty / max(corpus_stats['docs'],1):>8.1%}")

    print("\n=== queries ===")
    per_group: dict[str, int] = {}
    for q in queries:
        per_group[q.group] = per_group.get(q.group, 0) + 1
    for g in GROUPS:
        print(f"  {g:<28} {per_group.get(g, 0):>8,}")
    print(f"  {'TOTAL':<28} {len(queries):>8,}")

    print("\n=== label coverage ===")
    for k, v in coverage.items():
        if k == "unresolved_examples":
            continue
        print(f"  {k:<28} {v:>8,.4f}" if isinstance(v, float) else f"  {k:<28} {v:>8,}")
    if coverage["unresolved_examples"]:
        print("  unresolved examples:")
        for e in coverage["unresolved_examples"]:
            print(f"    - {e}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    require_beegfs()
    ensure_dirs()
    cfg = yaml.safe_load(Path(args.config).read_text())

    run_name = cfg.get("run_name", "run")
    top_k = int(cfg.get("top_k", 100))
    ks = tuple(cfg.get("ks", [1, 3, 5, 10, 20, 50, 100]))
    groups = tuple(cfg.get("groups", GROUPS))

    docs, corpus_stats = load_corpus()
    corpus_ids = {d.doc_id for d in docs}
    queries = load_queries(groups=groups)
    coverage = check_label_coverage(queries, corpus_ids)
    print(
        f"corpus: {corpus_stats['docs']:,} APIs | queries: {len(queries):,} | "
        f"label resolution: {coverage['resolution_rate']:.1%}"
    )
    if coverage["resolution_rate"] < 0.99:
        print(
            f"  NOTE: {coverage['labels_unresolved']:,} labels reference APIs absent "
            "from the 2024 filtered corpus and are unreachable by ANY retriever.",
            file=sys.stderr,
        )

    policy = cfg.get("label_policy", "drop_dead")
    queries, policy_stats = apply_label_policy(queries, corpus_ids, policy)
    ceiling = 1.0 if policy == "drop_dead" else coverage["resolution_rate"]
    print(
        f"label_policy={policy}: {policy_stats['queries_out']:,} of "
        f"{policy_stats['queries_in']:,} queries retained | "
        f"recall ceiling {ceiling:.3f}"
    )

    doc_ids = [d.doc_id for d in docs]
    rcfg = dict(cfg["retriever"])
    rtype = rcfg.pop("type")
    if rtype not in RETRIEVERS:
        raise SystemExit(f"unknown retriever '{rtype}'; have {sorted(RETRIEVERS)}")

    out_root = RESULTS_DIR / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for variant in cfg.get("variants", [{"name": "default"}]):
        vname = variant["name"]
        backfill = bool(variant.get("backfill_description", False))
        with_params = bool(variant.get("include_parameters", False))
        tg_style = bool(variant.get("toolgen_style", False))
        print(
            f"\n--- variant: {vname} (backfill_description={backfill}, "
            f"include_parameters={with_params}, toolgen_style={tg_style}) ---"
        )

        texts = [
            d.text(
                backfill_description=backfill,
                include_parameters=with_params,
                toolgen_style=tg_style,
            )
            for d in docs
        ]
        empty_docs = sum(1 for t in texts if not t.strip())
        avg_terms = sum(t.count(" ") + 1 for t in texts) / max(len(texts), 1)

        retriever = RETRIEVERS[rtype](**rcfg)
        t0 = time.perf_counter()
        retriever.index(doc_ids, texts)
        t_index = time.perf_counter() - t0

        t0 = time.perf_counter()
        ranked = retriever.retrieve([q.normalized for q in queries], top_k=top_k)
        t_retrieve = time.perf_counter() - t0

        per_query, by_group = [], {}
        for q, hits in zip(queries, ranked):
            scores = evaluate_one([d for d, _ in hits], q.relevant, ks=ks)
            per_query.append(scores)
            by_group.setdefault(q.group, []).append(scores)

        overall = aggregate(per_query)
        grouped = {g: aggregate(v) for g, v in by_group.items()}

        print(_header())
        for g in groups:
            if g in grouped:
                print(_fmt_row(g, grouped[g]))
        print(_fmt_row("OVERALL", overall))
        print(
            f"  index {t_index:.1f}s | retrieve {t_retrieve:.1f}s "
            f"({1000 * t_retrieve / max(len(queries),1):.1f} ms/query) | "
            f"empty docs {empty_docs:,} | mean doc length {avg_terms:.1f} terms"
        )

        summary[vname] = {
            "backfill_description": backfill,
            "include_parameters": with_params,
            "toolgen_style": tg_style,
            "overall": overall,
            "by_group": grouped,
            "empty_documents": empty_docs,
            "mean_doc_terms": avg_terms,
            "timing_seconds": {"index": t_index, "retrieve": t_retrieve},
        }

        (RUNS_DIR / run_name).mkdir(parents=True, exist_ok=True)
        with open(RUNS_DIR / run_name / f"{vname}.rankings.jsonl", "wb") as fh:
            for q, hits in zip(queries, ranked):
                fh.write(
                    orjson.dumps(
                        {
                            "query_id": q.query_id,
                            "group": q.group,
                            "relevant": sorted(q.relevant),
                            "ranked": [[d, s] for d, s in hits],
                        }
                    )
                    + b"\n"
                )

    record = {
        "run_name": run_name,
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg,
        "corpus_stats": corpus_stats,
        "label_coverage": {
            k: v for k, v in coverage.items() if k != "unresolved_examples"
        },
        "label_policy": policy_stats,
        "results": summary,
    }
    (out_root / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out_root / 'metrics.json'}")
    print(f"rankings under {RUNS_DIR / run_name}")
    return 0


def cmd_toolbench(args: argparse.Namespace) -> int:
    from .toolbench_run import run

    return run(args.config)


def main() -> int:
    p = argparse.ArgumentParser(prog="ez-tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="corpus/query/label statistics").set_defaults(
        func=cmd_inspect
    )
    r = sub.add_parser("run", help="index, retrieve and evaluate")
    r.add_argument("--config", required=True)
    r.set_defaults(func=cmd_run)

    t = sub.add_parser(
        "toolbench", help="replicate ToolGen Table 1 on ToolBench's retrieval split"
    )
    t.add_argument("--config", required=True)
    t.set_defaults(func=cmd_toolbench)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
