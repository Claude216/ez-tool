"""FASTREAD's published numbers, for the comparison figure 4 draws.

FASTREAD is treatment HUTM in table 2 of Yu, Kraft & Menzies, "Finding Better
Active Learners for Faster Literature Reviews" (arXiv 1612.03224) -- the one
treatment that holds rank 1 on all four corpora. Values below are the median
and IQR of X95 over their 30 simulations, plus WSS@95.

X95 is the number of studies a human reviewed to reach 95% recall, where
recall is |LR|/|R| -- papers actually found. It is therefore comparable to
`metrics.found_recall` and *not* to `metrics.classifier_recall`, which is what
section 7's figure 4 plots. That mismatch is the reason every run here records
both curves.

One caveat on Kitchenham: Yu's X95 counts reaching 95% of |R| = 45, the
content-review level. The EZR paper's table 6 uses the 132 that passed
title-and-abstract screening. The two Kitchenham targets are not the same set,
so its bar is the least comparable of the four.
"""

from __future__ import annotations

# corpus -> (X95 median, X95 iqr, WSS@95 median)
HUTM: dict[str, tuple[int, int, float]] = {
    "Hall": (350, 120, 0.91),
    "Wahono": (670, 230, 0.85),
    "Radjenovic": (680, 180, 0.83),
    "Kitchenham": (630, 110, 0.58),
}

TARGET_RECALL = 0.95


def x95(corpus: str) -> int:
    return HUTM[corpus][0]
