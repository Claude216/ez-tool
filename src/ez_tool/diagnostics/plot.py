"""Figures for the ToolBench CNB feasibility diagnostics.

One file per plot, written to ``results/diagnostics/figures``. Each figure
carries the number the report cites, so a figure can be read without the
report and vice versa.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LEVEL_COLOR = {"category": "tab:blue", "tool": "tab:orange", "api": "tab:green"}


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def support_histogram(level: str, row: dict, out_dir: Path) -> Path:
    """Instances per class, log-log. The tail is the whole point of the figure.

    Bins are log-spaced: an integer-width bar on a log axis spans half a decade
    at x=1 and a rounding error at x=100, which would draw the starved tail as
    a mountain regardless of what the data says.
    """
    hist = {int(k): v for k, v in row["histogram"].items()}
    vals = np.repeat(np.array(sorted(hist)), [hist[k] for k in sorted(hist)])
    edges = np.logspace(0, np.log10(vals.max() + 1), 40)
    edges = np.unique(np.concatenate([[0.9], edges]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=edges, color=LEVEL_COLOR[level], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("instances per class (log)")
    ax.set_ylabel("number of classes (log)")
    ax.set_title(
        f"G1 {level}-level support\n"
        f"{row['classes']:,} classes, effective (exp H) = {row['perplexity']:.1f}, "
        f"median = {row['median']:.0f}"
    )
    ax.axvline(5, color="red", ls="--", lw=1)
    ax.annotate(
        f"5 instances/class\n{row['share_classes_le_5']:.0%} of classes at or below\n"
        f"holding {row['mentions_in_classes_le_5']:.1%} of labels",
        xy=(5, 1),
        xytext=(5.6, 1.4),
        color="red",
        fontsize=8,
        va="bottom",
    )
    ax.grid(alpha=0.3, which="both")
    return _save(fig, out_dir / f"d1_support_{level}.png")


def overlap_distribution(d2: dict, out_dir: Path) -> Path:
    """Query-to-documentation Jaccard: ground truth against random APIs."""
    true = d2["_arrays"]["overlap_true"]
    rand = d2["_arrays"]["overlap_rand"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 0.6, 90)
    ax.hist(rand, bins=bins, color="tab:grey", alpha=0.75, label="random non-relevant API")
    ax.hist(true, bins=bins, color="tab:red", alpha=0.65, label="ground-truth API")
    ax.set_xlabel("Jaccard overlap, query terms vs API documentation terms")
    ax.set_ylabel("queries")
    ax.set_title(
        f"Lexical leakage: means {true.mean():.3f} vs {rand.mean():.3f} "
        f"({d2['ratio_of_means']:.1f}x)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / "d2_overlap_distribution.png")


def rank_cdf(d2: dict, out_dir: Path) -> Path:
    """Where a zero-parameter overlap ranker puts ground truth."""
    fig, ax = plt.subplots(figsize=(7, 4))
    n_api = d2["n_apis"]
    for key, color, label in (
        ("rank_jaccard", "tab:red", "Jaccard (length-normalised)"),
        ("rank_expected", "tab:blue", "raw |Q & D| overlap"),
    ):
        r = np.sort(d2["_arrays"][key])
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=color, label=label)

    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.axvline(5, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"rank of ground-truth API among all {n_api:,} G1 APIs (log)")
    ax.set_ylabel("fraction of queries at or above rank")
    ax.set_title(
        "Parameter-free overlap ranker\n"
        f"top-5: {d2['rank_jaccard']['top5']:.1%} (Jaccard) vs "
        f"{d2['rank_expected']['top5']:.1%} (raw); chance top-5 = {5 / n_api:.2%}"
    )
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return _save(fig, out_dir / "d2_rank_cdf.png")


def nonzero_features(d3: dict, out_dir: Path) -> Path:
    """Non-zero features per query as EZR's vocabulary size varies."""
    rows = d3["sparsity"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["top_n"] for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs, [r["mean_nonzero"] for r in rows], color="tab:green", alpha=0.8)
    for x, r in zip(xs, rows):
        ax.annotate(
            f"{r['frac_zero']:.2%} zero",
            xy=(x, r["mean_nonzero"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(xs, labels)
    ax.set_xlabel("vocabulary size (the.textmine.top)")
    ax.set_ylabel("mean non-zero features per query")
    ax.set_title(
        "Query representation under EZR's pipeline\n"
        f"queries hold {d3['query_terms_after_pipeline']['median']:.0f} distinct "
        "terms at the median after stop-word removal and stemming"
    )
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir / "d3_nonzero_features.png")


def query_length(d3: dict, out_dir: Path) -> Path:
    """The length distribution EZR's top=100 default was not tuned for."""
    lens = np.asarray(d3["_arrays"]["query_nonzero_all"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lens, bins=np.arange(0, lens.max() + 2), color="tab:purple", alpha=0.8)
    ax.axvline(float(np.median(lens)), color="red", ls="--", lw=1)
    ax.set_xlabel("terms per query after tokenize -> stop words -> stem")
    ax.set_ylabel("queries")
    ax.set_xlim(0, 80)
    ax.set_title(
        f"G1 query length after EZR's pipeline (median {np.median(lens):.0f}); "
        "EZR was tuned on 150-250 word abstracts"
    )
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / "d3_query_length.png")


# --- follow-up round -------------------------------------------------------

POOL_LABEL = {
    "P0": "P0\nall 10,355",
    "P1": "P1\nsame category",
    "P2": "P2\nsame tool",
    "P3": "P3\nsize-matched\nrandom (control)",
}


def pool_ranks(d4: dict, out_dir: Path) -> Path:
    """Top-1/top-5 per candidate pool, against each pool's own chance rate."""
    pools = list(d4["pools"])
    xs = np.arange(len(pools))
    top1 = [d4["pools"][p]["top1"] for p in pools]
    top5 = [d4["pools"][p]["top5"] for p in pools]
    ch5 = [d4["pools"][p]["chance_top5"] for p in pools]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs - 0.2, top5, 0.38, label="top-5", color="tab:red", alpha=0.85)
    ax.bar(xs + 0.2, top1, 0.38, label="top-1", color="tab:blue", alpha=0.85)
    ax.plot(xs, ch5, "k_", markersize=26, markeredgewidth=2.5,
            label="top-5 by chance")
    for x, p in zip(xs, pools):
        v = d4["pools"][p]
        ax.annotate(
            f"{v['top5']:.0%}\n({v['lift_top5']:.1f}x)",
            xy=(x - 0.2, v["top5"]), ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(xs, [POOL_LABEL[p] for p in pools], fontsize=8)
    ax.set_ylabel("fraction of instances")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Ground-truth rank under harder candidate pools\n"
        "P1 sits BELOW its size-matched control P3 "
        f"({d4['pools']['P1']['top5']:.1%} vs {d4['pools']['P3']['top5']:.1%})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir / "d4_pool_ranks.png")


def overlap_tiers(d4: dict, out_dir: Path) -> Path:
    """Query-to-documentation overlap as the negatives get harder."""
    keys = [
        ("ground_truth", "ground-truth API", "tab:red"),
        ("same_tool_non_relevant", "other API, same tool", "tab:orange"),
        ("random_same_category", "random API, same category", "tab:olive"),
        ("random_any_category", "random API, any category", "tab:grey"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ys = np.arange(len(keys))
    means = [d4["overlap"][k]["mean"] for k, _, _ in keys]
    ax.barh(ys, means, color=[c for _, _, c in keys], alpha=0.9)
    for y, k in zip(ys, [k for k, _, _ in keys]):
        m = d4["overlap"][k]["mean"]
        lab = f"{m:.4f}"
        if k != "ground_truth":
            lab += f"   ({d4['overlap']['ground_truth']['mean'] / m:.2f}x below GT)"
        ax.annotate(lab, xy=(m, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_yticks(ys, [lab for _, lab, _ in keys], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.21)
    ax.set_xlabel("mean Jaccard overlap with the query")
    ax.set_title(
        "Round one's 12.4x leakage ratio decays as negatives get harder\n"
        "12.4x (any category) -> 7.4x (same category) -> 2.1x (same tool)"
    )
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_dir / "d4_overlap_tiers.png")


def support_overlay(d5: dict, out_dir: Path) -> Path:
    """Training support of a tool, per training instance vs per test instance.

    Bars are each series' own share of instances per bin, not a density:
    with log-spaced bins a density normalisation divides the narrow low-support
    bins by a tiny width and draws the tail as the tallest thing on the page.
    """
    train = np.asarray(d5["_arrays"]["train_tool_support_per_instance"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(0, np.log10(max(train.max(), 2) + 1), 26)

    series = [("training instances", train, "tab:grey")]
    for split, color in (
        ("test_G1_instruction", "tab:blue"),
        ("test_G1_tool", "tab:orange"),
        ("test_G1_category", "tab:green"),
    ):
        series.append((split, np.asarray(d5["_arrays"][split]), color))

    for i, (label, vals, color) in enumerate(series):
        w = np.full(len(vals), 1.0 / len(vals))
        if i == 0:
            ax.hist(vals, bins=bins, weights=w, color=color, alpha=0.55,
                    label=f"{label} (n={len(vals):,})")
        else:
            ax.hist(vals, bins=bins, weights=w, histtype="step", lw=1.8,
                    color=color, label=f"{label} (n={len(vals)})")

    ax.axvline(3, color="red", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("training instances listing this instance's ground-truth tool (log)")
    ax.set_ylabel("share of that series' instances")
    ax.set_title(
        "Test tools are NOT unseen: every G1 test split's tools appear in training\n"
        "zero-support test instances: 0 of 600; red line marks the <=3 threshold"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / "d5_support_overlay.png")


def near_duplicates(d5: dict, out_dir: Path) -> Path:
    """How close the nearest training query is to each test query."""
    nd = d5["near_duplicates"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, color in (
        ("test_G1_instruction", "tab:blue"),
        ("test_G1_tool", "tab:orange"),
        ("test_G1_category", "tab:green"),
        ("stabletoolbench_solvable_G1", "tab:red"),
    ):
        v = np.sort(np.asarray(d5["_arrays"]["nearest"][label]))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1.8,
                label=f"{label} (median {np.median(v):.2f})")
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("Jaccard to the most similar TRAINING query")
    ax.set_ylabel("fraction of test queries at or below")
    ax.set_title(
        "Train/test near-duplication in the ToolGen retrieval split\n"
        f"{nd['stabletoolbench_solvable_G1']['frac_above_0.5']:.0%} of the "
        "StableToolBench solvable queries have a training query at Jaccard >= 0.5"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir / "d5_near_duplicates.png")
