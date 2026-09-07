"""Compare the port's trajectory against vendored ezr's, end to end.

`validate.py` checks components: identical features, identical CNB weights,
identical predictions on fixed labelled subsets. This checks the whole active
learning run -- warm start, acquisition, retraining, 250-odd steps of it.

The two cannot be compared trial by trial: ezr draws its warm start from the
module-level `random`, the port from a numpy Generator, so trial k is not the
same 40 papers on both sides. What is comparable is the median over 20 trials,
which is the quantity figure 4 plots.

ezr reports recall as a truncated whole percent (`int(100 * tp / n_pos)`), so
the port's float recall is floored the same way before comparing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..paths import RUNS_DIR

NATIVE = RUNS_DIR / "textmine" / "ezr_native"
PORT_RUN = "textmine_ezr_parity"
ARM = "code"


def _port(corpus: str, norm: int) -> tuple[np.ndarray, np.ndarray]:
    path = RUNS_DIR / "textmine" / PORT_RUN / f"{ARM}__{corpus}__norm{norm}.npz"
    with np.load(path) as z:
        # Match ezr's truncation to a whole percent, per trial, before the median.
        pct = np.floor(z["classifier_recall"] * 100 + 1e-9)
        return z["n_labeled"], np.median(pct, axis=0)


def main(tag: str = "full_b290") -> int:
    native = json.loads((NATIVE / f"{tag}.json").read_text())
    print(f"{'corpus':<12}{'norm':>5}{'steps':>7}{'median |diff|':>15}"
          f"{'max |diff|':>12}{'ezr final':>11}{'port final':>12}")
    worst = 0.0
    for run in native:
        x_e = np.asarray(run["n_labeled"])
        y_e = np.asarray(run["recall_pct_median"])
        x_p, y_p = _port(run["corpus"], run["norm"])

        n = min(len(x_e), len(x_p))
        if not np.array_equal(x_e[:n], x_p[:n]):
            print(f"{run['corpus']:<12}{run['norm']:>5}  x-axis mismatch "
                  f"(ezr {x_e[0]}..{x_e[n-1]}, port {x_p[0]}..{x_p[n-1]})")
            continue
        d = np.abs(y_e[:n] - y_p[:n])
        worst = max(worst, float(d.max()))
        print(f"{run['corpus']:<12}{run['norm']:>5}{n:>7}{np.median(d):>15.1f}"
              f"{d.max():>12.1f}{y_e[n-1]:>11.1f}{y_p[n-1]:>12.1f}")

    print(f"\nworst single-step gap across all runs: {worst:.1f} percentage points")
    print("Both sides are medians over 20 independent random warm starts, so a "
          "few points of sampling noise is expected; a systematic offset is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
