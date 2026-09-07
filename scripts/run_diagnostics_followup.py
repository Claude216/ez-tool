#!/usr/bin/env python
"""Follow-up diagnostics: hard negatives (A) and test-set support (B).

    python scripts/run_diagnostics_followup.py

COMPUTE NODE ONLY. Statistics only -- no classifier, no retriever, no ezr.py.
Writes results/diagnostics/followup.json and figures into the same figures dir
as the first round.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ez_tool.diagnostics import d4_hardneg, d5_support, g1, plot  # noqa: E402
from ez_tool.paths import RESULTS_DIR  # noqa: E402

OUT_DIR = RESULTS_DIR / "diagnostics"
FIG_DIR = OUT_DIR / "figures"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t = time.time()
    data = g1.load()
    print(f"loaded {len(data.instances):,} G1 instances in {time.time() - t:.1f}s")

    print("follow-up A: hard negatives ...")
    d4 = d4_hardneg.run(data)

    print("follow-up B: test-set support ...")
    d5 = d5_support.run(data)

    figures = [
        plot.pool_ranks(d4, FIG_DIR),
        plot.overlap_tiers(d4, FIG_DIR),
        plot.support_overlay(d5, FIG_DIR),
        plot.near_duplicates(d5, FIG_DIR),
    ]

    d4.pop("_arrays")
    d5.pop("_arrays")
    d5["stabletoolbench"].pop("_arrays", None)

    report = {
        "provenance": {
            "training_data": "ToolGen repackaging of ToolBench retrieval split, "
                             "HF reasonwang/ToolGen-Datasets data.tar.gz blob "
                             "f80aa7422a39bc69b1d7c963f064fff8d94988f9dd153b12790087749cc2e7e0",
            "evaluation_set": "StableToolBench solvable_queries @ "
                              "aa4ed9f4737ad98bd706663f01d63623c3427812",
            "instance_count_discrepancy": "closed as unresolved -- the original "
                                          "ToolBench instruction files are not available",
        },
        "followup_a_hard_negatives": d4,
        "followup_b_test_support": d5,
        "figures": [str(f.relative_to(RESULTS_DIR)) for f in figures],
    }
    out = OUT_DIR / "followup.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out}")
    for f in figures:
        print(f"  figure {f}")
    print(f"  A: {d4['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
