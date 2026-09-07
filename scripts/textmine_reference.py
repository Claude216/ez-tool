"""Reference outputs from the untouched vendored ezr.py.

The vendored file is byte-identical to commit 500d1d4 and uses PEP 695 `type`
statements, so it needs Python 3.12 while the project venv is 3.11. This
script runs under 3.12, dumps what ezr computes to JSON, and
`ez_tool.textmine.validate` (3.11 + numpy) asserts the port reproduces it.

Run on a compute node:
  scripts/textmine_reference.sh
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ez_tool" / "textmine" / "vendor"))

import ezr  # noqa: E402  (needs the path above)

SLR = Path("/mnt/beegfs/lli66/ez-tool/data/raw/slr")
OUT = Path("/mnt/beegfs/lli66/ez-tool/runs/textmine/reference")
CORPORA = ("Hall", "Wahono", "Radjenovic", "Kitchenham")

# Small enough to store elementwise; used for a full matrix comparison.
FULL_MATRIX = "Kitchenham"


def prepared(name: str):
    """ezr's own pipeline, on ezr's own default text column (`abstract`)."""
    p = ezr.tmPrepare(str(SLR / f"{name}.csv"))
    return p, ezr.tmData(p)


def features_ref(name: str) -> dict:
    p, data = prepared(name)
    xs = data.cols.xs
    X = [[r[c.at] for c in xs] for r in data.rows]
    col_sums = [sum(row[j] for row in X) for j in range(len(xs))]
    return {
        "vocab": list(p.tfidf),
        "n_rows": len(X),
        "n_features": len(xs),
        "col_sums": col_sums,
        "row_sums": [sum(r) for r in X],
        "n_empty_rows": sum(1 for r in X if not any(r)),
        "X": X if name == FULL_MATRIX else None,
    }


def cnb_ref(name: str, n_subsets: int = 4, size: int = 40) -> list[dict]:
    """Weights and recall for fixed labelled subsets, both norm settings."""
    p, data = prepared(name)
    key = data.cols.klass.at
    xs = data.cols.xs
    rng = random.Random(0)
    n = len(data.rows)
    pos = [i for i, r in enumerate(data.rows) if r[key] == "yes"]

    subsets = [sorted(rng.sample(range(n), size)) for _ in range(n_subsets - 1)]
    # One subset guaranteed to contain positives, so the `yes` class exists,
    # and the all-negative case above exercises the single-class path.
    subsets.append(sorted(rng.sample(pos, 5) + rng.sample(range(n), size - 5)))

    out = []
    for idx in subsets:
        rows = [data.rows[i] for i in idx]
        for norm in (0, 1):
            ezr.the.textmine.norm = norm
            ws = ezr.cnb(data, rows)
            preds = [ezr._tm_best(ws, data, r) for r in data.rows]
            out.append(
                {
                    "idx": idx,
                    "norm": norm,
                    "klasses": sorted(ws),
                    "w": {k: [ws[k][c.at] for c in xs] for k in sorted(ws)},
                    "n_pred_yes": sum(1 for g in preds if g == "yes"),
                    "recall_int": ezr._tm_recall(ws, data, key),
                }
            )
    ezr.the.textmine.norm = 0
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ezr defaults: {ezr.the.textmine}")
    for name in CORPORA:
        ref = {"features": features_ref(name)}
        if name in ("Hall", "Kitchenham"):
            ref["cnb"] = cnb_ref(name)
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(ref))
        f = ref["features"]
        print(
            f"{name:<12} rows={f['n_rows']:>5} feats={f['n_features']} "
            f"empty_rows={f['n_empty_rows']:>4} -> {path.name} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )


if __name__ == "__main__":
    main()
