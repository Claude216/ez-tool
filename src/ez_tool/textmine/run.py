"""Drive the section 7 replication over a config grid.

Each trial is one (arm, corpus, norm, seed): warm start, then greedy CNB
acquisition to the budget, recording classifier recall, false alarm and found
recall after every label. Full traces land on beegfs; a small summary lands in
results/ next to the phase 1 metrics.

Arms exist because section 7's warm start is underdetermined -- see
`active.WarmStart`. Running them together is cheap and makes the sensitivity
visible instead of hidden in a default.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from ..paths import RESULTS_DIR, RUNS_DIR, require_beegfs
from . import active, corpus, features

CHECKPOINTS = (24, 40, 50, 100, 200, 350, 500, 600)


def budget_for(cfg: dict, name: str) -> int:
    """Budget may be one number or one per corpus.

    Figure 4's panels do not share an x-axis: each corpus is drawn to its own
    upper bound, so the cap is per corpus here too.
    """
    b = cfg["budget"]
    return int(b[name]) if isinstance(b, dict) else int(b)


def _summarise(traces: list[active.Trace]) -> dict:
    """Median and quartiles across repeats, per label count."""
    n = min(len(t.n_labeled) for t in traces)
    out: dict = {"n_labeled": traces[0].n_labeled[:n]}
    for metric in ("classifier_recall", "false_alarm", "found_recall"):
        a = np.asarray([getattr(t, metric)[:n] for t in traces], dtype=float)
        out[metric] = {
            "median": np.median(a, axis=0).tolist(),
            "q25": np.percentile(a, 25, axis=0).tolist(),
            "q75": np.percentile(a, 75, axis=0).tolist(),
        }
    out["n_random_picks"] = [t.n_random_picks for t in traces]
    return out


def _checkpoints(summary: dict, budget: int) -> dict:
    """Median metrics at a handful of budgets, for the committed summary."""
    labels = summary["n_labeled"]
    index = {n: i for i, n in enumerate(labels)}
    rows = {}
    for n in sorted({*CHECKPOINTS, budget}):
        if n not in index:
            continue
        i = index[n]
        rows[str(n)] = {
            m: round(summary[m]["median"][i], 4)
            for m in ("classifier_recall", "false_alarm", "found_recall")
        }
    return rows


def run(cfg: dict) -> dict:
    require_beegfs()
    runs_dir = RUNS_DIR / "textmine" / cfg["run_name"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[str, bool], tuple[corpus.Corpus, np.ndarray]] = {}
    report: dict = {"run_name": cfg["run_name"], "config": cfg, "results": {}}

    for arm in cfg["arms"]:
        warm = active.WarmStart(**arm["warm"])
        for name in cfg["corpora"]:
            key = (name, bool(arm["include_title"]))
            t0 = time.time()
            if key not in cache:
                loaded = corpus.load(
                    name, include_title=key[1], klass=cfg.get("klass_override")
                )
                cache[key] = (
                    loaded,
                    features.build(loaded.texts, top=cfg["top_features"]).X,
                )
            c, X = cache[key]

            budget = budget_for(cfg, name)
            for norm in cfg["norms"]:
                traces = [
                    active.run_trial(
                        X,
                        c.y,
                        budget=budget,
                        warm=warm,
                        seed=cfg["seed"] + r,
                        norm=bool(norm),
                        alpha=cfg["alpha"],
                    )
                    for r in range(cfg["repeats"])
                ]
                summary = _summarise(traces)
                tag = f"{arm['name']}__{name}__norm{int(bool(norm))}"
                np.savez_compressed(
                    runs_dir / f"{tag}.npz",
                    n_labeled=np.asarray(summary["n_labeled"]),
                    **{
                        m: np.asarray(
                            [getattr(t, m)[: len(summary["n_labeled"])] for t in traces]
                        )
                        for m in ("classifier_recall", "false_alarm", "found_recall")
                    },
                )
                report["results"][tag] = {
                    "arm": arm["name"],
                    "corpus": name,
                    "norm": bool(norm),
                    "include_title": key[1],
                    "warm": asdict(warm),
                    "n_pos": c.n_pos,
                    "n_docs": len(c.y),
                    "random_picks_median": float(
                        np.median(summary["n_random_picks"])
                    ),
                    "budget": budget,
                    "checkpoints": _checkpoints(summary, budget),
                }
            print(
                f"  {arm['name']:<16} {name:<12} "
                f"{len(cfg['norms']) * cfg['repeats']:>3} trials  "
                f"{time.time() - t0:6.1f}s"
            )

    out = RESULTS_DIR / "textmine" / f"{cfg['run_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsummary -> {out}\ntraces  -> {runs_dir}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    print(f"config: {args.config}")
    run(cfg)


if __name__ == "__main__":
    main()
