"""Run the experiment through the vendored ezr itself, not the port.

Imports `ezr.py` @ 500d1d4 unmodified and drives `tmActive` via its own `the.*`
configuration, capturing the table it prints. Runs under Python 3.12.

This exists to check the port end to end: `validate.py` proves the components
agree (features, CNB weights, predictions), and this proves the whole
active-learning trajectory agrees when both are pointed at the same setup.

It is not the production path, because vendored ezr structurally cannot
produce parts of figure 4:

* no false alarm anywhere in the commit -- figure 4's entire bottom row;
* no found recall, so no comparison to FASTREAD's X95;
* `tmPrepare` hardcodes `tmTokenize(f)`, i.e. the `abstract` column and the
  `label` class column, so no title text and no Kitchenham `abs` (132) labels;
* `_tm_warm` always seeds positives from the known relevant set, and the one
  config that makes it a uniform draw (`yes=0`) leaves CNB with a single class,
  where `tmActive`'s `cnbLikes(..., "yes")` raises KeyError.

Everything in that list is why `ez_tool.textmine` exists. Everything not in it,
ezr can do, and this script makes it do so.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ez_tool" / "textmine" / "vendor"))

import ezr  # noqa: E402

SLR = Path("/mnt/beegfs/lli66/ez-tool/data/raw/slr")
OUT = Path("/mnt/beegfs/lli66/ez-tool/runs/textmine/ezr_native")
ROW = re.compile(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")


_PREPARED: dict[str, object] = {}


def prepared(name: str):
    """`tmPrepare` is the intended input to `tmActive`; cache it per corpus.

    Passing a raw path instead makes `_tm_setup` fall through to
    `Data(csv(src))`, i.e. ezr's generic reader parsing free-text abstracts as
    columns, which fails.
    """
    if name not in _PREPARED:
        _PREPARED[name] = ezr.tmPrepare(str(SLR / f"{name}.csv"))
    return _PREPARED[name]


def run_one(name: str, budget: int, norm: int, yes: int, no: int,
            valid: int, seed: int) -> dict:
    ezr.the.textmine.norm = norm
    ezr.the.textmine.yes = yes
    ezr.the.textmine.no = no
    ezr.the.textmine.valid = valid
    ezr.the.learn.budget = budget
    random.seed(seed)

    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        ezr.tmActive(prepared(name))
    elapsed = time.time() - t0

    rows = [(int(m[1]), float(m[2]), float(m[3]))
            for line in buf.getvalue().splitlines() if (m := ROW.match(line))]
    return {
        "corpus": name, "budget": budget, "norm": norm,
        "warm": {"yes": yes, "no": no}, "valid": valid, "seed": seed,
        "seconds": round(elapsed, 1),
        "n_labeled": [r[0] for r in rows],
        "recall_pct_median": [r[1] for r in rows],
        "recall_pct_iqr": [r[2] for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+",
                    default=["Kitchenham", "Hall", "Wahono", "Radjenovic"])
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--valid", type=int, default=5)
    ap.add_argument("--yes", type=int, default=20)
    ap.add_argument("--no", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--norms", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--tag", default="agreement")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for name in args.corpora:
        for norm in args.norms:
            r = run_one(name, args.budget, norm, args.yes, args.no,
                        args.valid, args.seed)
            out.append(r)
            print(f"  {name:<12} norm={norm} budget={args.budget} "
                  f"valid={args.valid}  {r['seconds']:>7.1f}s  "
                  f"steps={len(r['n_labeled'])}  "
                  f"final_pd={r['recall_pct_median'][-1] if r['n_labeled'] else 'n/a'}")
    path = OUT / f"{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
