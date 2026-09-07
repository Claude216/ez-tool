#!/usr/bin/env python
"""Run the ToolBench CNB feasibility diagnostics and write stats + figures.

    python scripts/run_diagnostics.py

COMPUTE NODE ONLY -- reads 350 MB of JSON from beegfs. Statistics only:
nothing here trains a classifier, builds an index, or imports ezr.py.

Writes results/diagnostics/diagnostics.json and results/diagnostics/figures/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ez_tool.diagnostics import d1_labels, d2_leakage, d3_sparsity, g1, plot  # noqa: E402
from ez_tool.paths import RESULTS_DIR  # noqa: E402

OUT_DIR = RESULTS_DIR / "diagnostics"
FIG_DIR = OUT_DIR / "figures"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t = time.time()
    data = g1.load()
    print(f"loaded {len(data.instances):,} G1 instances in {time.time() - t:.1f}s")

    report: dict = {
        "source": {
            "g1_dir": str(g1.G1_DIR),
            "files": ["train.json", "test_G1_instruction.json",
                      "test_G1_tool.json", "test_G1_category.json"],
            "label_field": "relevant APIs (list of [tool_name, api_name])",
        },
        "step0": data.stats,
    }

    print("diagnostic 1 ...")
    report["d1"] = d1_labels.run(data)

    print("diagnostic 2 ...")
    d2 = d2_leakage.run(data)

    print("diagnostic 3 ...")
    d3 = d3_sparsity.run(data)

    figures = []
    for level in d1_labels.LEVELS:
        figures.append(plot.support_histogram(level, report["d1"][level], FIG_DIR))
    figures.append(plot.overlap_distribution(d2, FIG_DIR))
    figures.append(plot.rank_cdf(d2, FIG_DIR))
    figures.append(plot.nonzero_features(d3, FIG_DIR))
    figures.append(plot.query_length(d3, FIG_DIR))

    # Arrays are for the figures only; the JSON keeps the summary statistics.
    d2.pop("_arrays")
    d3.pop("_arrays")
    report["d2"] = d2
    report["d3"] = d3
    # Histograms are large and already drawn; keep the JSON readable.
    for level in d1_labels.LEVELS:
        report["d1"][level].pop("histogram", None)
    report["figures"] = [str(f.relative_to(RESULTS_DIR)) for f in figures]

    out = OUT_DIR / "diagnostics.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out}")
    for f in figures:
        print(f"  figure {f}")
    for level in d1_labels.LEVELS:
        print(f"  D1 {level}: {report['d1'][level]['verdict']}")
    print(f"  D2: {d2['verdict']}")
    print(f"  D3: {d3['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
