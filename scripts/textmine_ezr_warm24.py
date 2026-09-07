"""Vendored ezr with a 24-sample uniform warm start, continuing to budget.

`tmActive` cannot run this configuration: `--textmine.yes=0 --textmine.no=24`
gives the uniform draw, but when that draw holds no positive, `cnb` builds a
"no" class only and `cnbLikes(ws, ..., "yes")` raises KeyError. Measured, 70%
of Hall draws and 52% of Kitchenham draws have no positive.

So the loop below is `tmActive`'s, calling ezr's own `_tm_warm`, `cnb`,
`cnbLikes` and `_tm_recall` in ezr's own order, with exactly one deviation --
what to do when the model has never seen a positive:

  skip   drop the trial, as ezr effectively does by crashing. Survivor-biased:
         only warm starts that happened to catch a positive are kept.
  step1  pick uniformly at random until the first positive turns up, and
         charge each pick to the budget. This is FASTREAD's step 1.

`vendor/ezr.py` is not modified. Runs under Python 3.12.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "ez_tool" / "textmine" / "vendor"))

import ezr  # noqa: E402

SLR = Path("/mnt/beegfs/lli66/ez-tool/data/raw/slr")
OUT = Path("/mnt/beegfs/lli66/ez-tool/runs/textmine/ezr_warm24")
# Kitchenham's `abs` column is the 132 that table 6 reports; ezr's default
# `label` is the 45 that survived content review. Passed through ezr's own
# tokenizer argument, so nothing is patched.
KLASS = {"Kitchenham": "abs"}


def prepared(name: str):
    f = str(SLR / f"{name}.csv")
    klass = KLASS.get(name, "label")
    return ezr.tmTfidf(ezr.tmStem(ezr.tmNostop(ezr.tmTokenize(f, klass=klass))))


def one_trial(data, key, pos, idx, budget, mode, rng):
    """tmActive's loop, with the missing-`yes` case made explicit."""
    lab = ezr._tm_warm(pos, idx)
    pool = idx - lab
    trail, n_random = [], 0
    while True:
        ws = ezr.cnb(data, [data.rows[i] for i in lab])
        trail.append((len(lab), ezr._tm_recall(ws, data, key)))
        if len(lab) >= budget or not pool:
            break
        if "yes" in ws:
            pick = max(pool, key=lambda i: ezr.cnbLikes(ws, data, data.rows[i], "yes"))
        elif mode == "step1":
            pick = rng.choice(sorted(pool))
            n_random += 1
        else:
            return None, 0  # `skip`: this is where ezr raises KeyError
        lab.add(pick)
        pool.discard(pick)
    return trail, n_random


def run(name: str, budget: int, valid: int, mode: str, seed: int) -> dict:
    p = prepared(name)
    ezr.the.textmine.yes = 0   # no seeded positives
    ezr.the.textmine.no = 24   # 24 uniform draws
    ezr.the.textmine.norm = 0
    data, key, pos, idx = ezr._tm_setup(p)
    random.seed(seed)
    rng = random.Random(seed)

    t0 = time.time()
    trails, randoms = [], []
    for _ in range(valid):
        trail, nr = one_trial(data, key, pos, idx, budget, mode, rng)
        if trail is not None:
            trails.append(trail)
            randoms.append(nr)
    elapsed = time.time() - t0

    if not trails:
        return {"corpus": name, "mode": mode, "survived": 0, "attempted": valid}
    n = min(len(t) for t in trails)
    xs = [trails[0][s][0] for s in range(n)]
    med = [statistics.median([t[s][1] for t in trails]) for s in range(n)]
    return {
        "corpus": name, "mode": mode, "budget": budget,
        "attempted": valid, "survived": len(trails),
        "n_positives": len(pos), "n_docs": len(data.rows),
        "seconds": round(elapsed, 1),
        "median_random_picks": statistics.median(randoms) if randoms else 0,
        "n_labeled": xs, "recall_pct_median": med,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+",
                    default=["Kitchenham", "Hall", "Wahono", "Radjenovic"])
    ap.add_argument("--budget", type=int, default=290)
    ap.add_argument("--valid", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--modes", nargs="+", default=["skip", "step1"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for mode in args.modes:
        for name in args.corpora:
            r = run(name, args.budget, args.valid, mode, args.seed)
            out.append(r)
            at50 = next((v for x, v in zip(r.get("n_labeled", []),
                                           r.get("recall_pct_median", [])) if x == 50), None)
            print(f"  {mode:<6} {name:<12} survived {r['survived']:>2}/{r['attempted']}"
                  f"  pd@50={at50}  pd@{args.budget}="
                  f"{r.get('recall_pct_median', ['-'])[-1]}"
                  f"  rand_picks={r.get('median_random_picks', 0)}"
                  f"  {r.get('seconds', 0):.0f}s")
    (OUT / "warm24.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'warm24.json'}")


if __name__ == "__main__":
    main()
