"""Figures for the section 7 replication.

Three of them:

* `fig4`  -- the paper's figure 4: classifier recall and false alarm against
  labels acquired, four corpora, red for Rennie normalisation on and blue for
  off, median over repeats with the interquartile band shaded.
* `metrics` -- the same runs drawn with both recall definitions on one axis,
  since FASTREAD's X95 is a count of papers a human read and is only
  comparable to `found_recall`.
* `warm` -- classifier recall per warm-start arm, because section 7.4 and
  ezr disagree about what the warm start was.

FASTREAD's X95 is drawn as a vertical rule where the budget reaches it, and
annotated off-axis where it does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..paths import RESULTS_DIR, RUNS_DIR  # noqa: E402
from . import fastread  # noqa: E402

NORM_COLOR = {True: "tab:red", False: "tab:blue"}
ARM_COLOR = {
    "paper": "tab:blue",
    "stratified": "tab:orange",
    "code": "tab:green",
    "paper_abstract_only": "tab:purple",
}


def _load(runs_dir: Path, tag: str) -> dict[str, np.ndarray]:
    with np.load(runs_dir / f"{tag}.npz") as z:
        return {k: z[k] for k in z.files}


def _band(ax, x, a, color, label=None):
    """Median line with the 25th-75th percentile shaded, as figure 4 does."""
    ax.plot(x, np.median(a, axis=0), color=color, lw=1.6, label=label)
    ax.fill_between(
        x,
        np.percentile(a, 25, axis=0),
        np.percentile(a, 75, axis=0),
        color=color,
        alpha=0.18,
        lw=0,
    )


def _x95(ax, name: str, xmax: int, y: float = 0.5) -> None:
    x = fastread.x95(name)
    if x <= xmax:
        ax.axvline(x, color="0.35", ls="--", lw=1.0)
        ax.annotate(
            f"FASTREAD X95={x}",
            (x, y),
            fontsize=6,
            rotation=90,
            va="center",
            ha="right",
            color="0.35",
        )
    else:
        ax.annotate(
            f"FASTREAD X95={x} (off axis)",
            (0.97, 0.05),
            xycoords="axes fraction",
            fontsize=6,
            ha="right",
            color="0.35",
        )


def _stated_budget(ax, budget: int, xmax: int) -> None:
    if budget <= xmax:
        ax.axvline(budget, color="0.75", ls=":", lw=1.0)


def _left_edge(ax, x0: int, y0: float) -> None:
    """Mark the value reached immediately after the warm start.

    Section 7.4: "Numerical labels at the left edge of each recall panel mark
    the recall achieved immediately after the warm start." That number is what
    distinguishes a seeded warm start from a uniform draw.
    """
    ax.annotate(
        f"{y0:.2f}",
        (x0, y0),
        xytext=(3, 4),
        textcoords="offset points",
        fontsize=7,
        color="0.2",
    )


def fig4(report: dict, runs_dir: Path, out: Path, arm: str = "paper") -> Path:
    corpora = report["config"]["corpora"]
    # No shared x-axis: each corpus is drawn to its own upper bound.
    fig, axes = plt.subplots(2, len(corpora), figsize=(3.1 * len(corpora), 5.0))
    for j, name in enumerate(corpora):
        xmax = report["results"][f"{arm}__{name}__norm0"]["budget"]
        x0 = None
        for norm in (False, True):
            d = _load(runs_dir, f"{arm}__{name}__norm{int(norm)}")
            x = d["n_labeled"]
            x0 = int(x[0])
            label = "normalised (Rennie)" if norm else "not normalised"
            _band(axes[0, j], x, d["classifier_recall"], NORM_COLOR[norm], label)
            _band(axes[1, j], x, d["false_alarm"], NORM_COLOR[norm], label)
            if not norm:  # annotate the blue curve only, to avoid overlap
                _left_edge(axes[0, j], x0, float(np.median(d["classifier_recall"][:, 0])))
                _left_edge(axes[1, j], x0, float(np.median(d["false_alarm"][:, 0])))
        r = report["results"][f"{arm}__{name}__norm0"]
        axes[0, j].set_title(f"{name}\n{r['n_pos']} of {r['n_docs']} relevant",
                             fontsize=9)
        _x95(axes[0, j], name, xmax)
        for ax in (axes[0, j], axes[1, j]):
            _stated_budget(ax, 50, xmax)
            ax.set_ylim(0, 1)
            # Start at the warm start: there is no model before it.
            ax.set_xlim(x0, xmax)
            ax.grid(alpha=0.25, lw=0.5)
        axes[1, j].set_xlabel("labels acquired")
    axes[0, 0].set_ylabel("classifier recall")
    axes[1, 0].set_ylabel("false alarm")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"Section 7 figure 4, replicated (arm: {arm}; median and IQR of "
        f"{report['config']['repeats']} repeats)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def metrics_fig(report: dict, runs_dir: Path, out: Path, arm: str = "paper") -> Path:
    corpora = report["config"]["corpora"]
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.1 * len(corpora), 2.9),
                             sharey=True)
    for j, name in enumerate(corpora):
        xmax = report["results"][f"{arm}__{name}__norm0"]["budget"]
        ax = axes[j]
        d = _load(runs_dir, f"{arm}__{name}__norm0")
        x = d["n_labeled"]
        _band(ax, x, d["classifier_recall"], "tab:blue", "classifier recall")
        _band(ax, x, d["found_recall"], "tab:green", "found recall (Yu's |LR|/|R|)")
        _left_edge(ax, int(x[0]), float(np.median(d["classifier_recall"][:, 0])))
        ax.axhline(fastread.TARGET_RECALL, color="0.35", lw=0.8, ls="-.")
        _x95(ax, name, xmax, y=0.75)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("labels acquired")
        ax.set_ylim(0, 1)
        ax.set_xlim(int(x[0]), xmax)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("recall")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Two different recalls. FASTREAD's X95 counts papers a human read, so "
        "it is comparable only to the green curve.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def warm_fig(report: dict, runs_dir: Path, out: Path) -> Path:
    corpora = report["config"]["corpora"]
    arms = [a["name"] for a in report["config"]["arms"]]
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.1 * len(corpora), 2.9),
                             sharey=True)
    for j, name in enumerate(corpora):
        ax = axes[j]
        for arm in arms:
            d = _load(runs_dir, f"{arm}__{name}__norm0")
            _band(ax, d["n_labeled"], d["classifier_recall"],
                  ARM_COLOR.get(arm, "0.4"), arm)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("labels acquired")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, report["results"][f"code__{name}__norm0"]["budget"])
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("classifier recall")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Warm-start sensitivity (no normalisation). Section 7.4 says 24 drawn "
        "at random; ezr draws 20 known-positive plus 20 random.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="textmine_paper")
    ap.add_argument("--arm", default="paper")
    args = ap.parse_args()

    report = json.loads(
        (RESULTS_DIR / "textmine" / f"{args.run_name}.json").read_text()
    )
    runs_dir = RUNS_DIR / "textmine" / args.run_name
    out_dir = RESULTS_DIR / "textmine" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        fig4(report, runs_dir, out_dir / f"{args.run_name}_fig4.png", args.arm),
        metrics_fig(report, runs_dir, out_dir / f"{args.run_name}_recalls.png", args.arm),
        warm_fig(report, runs_dir, out_dir / f"{args.run_name}_warmstart.png"),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
