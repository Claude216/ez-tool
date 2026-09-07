"""Assert the numpy port reproduces the vendored ezr implementation exactly.

`scripts/textmine_reference.sh` runs the byte-identical vendored `ezr.py`
under Python 3.12 and dumps what it computes; this reads that dump back and
checks the 3.11 + numpy path against it. Nothing else in the replication is
trusted until this passes.

Feature extraction is checked on ezr's own input -- the `abstract` column
alone, ezr's default label column -- because that is the only text vendored
ezr can be made to read without editing it. The title switch adds a prefix to
the same string before the identical pipeline, so validating there validates
both.

CNB is checked on fixed labelled subsets under both `norm` settings, including
one all-negative subset that exercises the single-class path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import cnb, corpus, features, metrics

REFERENCE_DIR = Path("/mnt/beegfs/lli66/ez-tool/runs/textmine/reference")
W_TOL = 1e-12


class ValidationError(AssertionError):
    pass


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValidationError(msg)


def validate_features(name: str, ref: dict) -> list[str]:
    f = ref["features"]
    # ezr reads `abstract` only, and its own default label column.
    c = corpus.load(name, include_title=False, klass="label")
    got = features.build(c.texts, top=f["n_features"])

    _check(
        got.vocab == f["vocab"],
        f"{name}: vocab differs; first mismatch at "
        f"{next((i for i, (a, b) in enumerate(zip(got.vocab, f['vocab'])) if a != b), None)}",
    )
    _check(got.X.shape == (f["n_rows"], f["n_features"]), f"{name}: X shape")
    _check(
        got.X.sum(axis=0).tolist() == f["col_sums"], f"{name}: column sums differ"
    )
    _check(got.X.sum(axis=1).tolist() == f["row_sums"], f"{name}: row sums differ")

    n_empty = int((got.X.sum(axis=1) == 0).sum())
    _check(n_empty == f["n_empty_rows"], f"{name}: empty-row count differs")

    notes = [f"features ok (vocab, col sums, row sums; {n_empty} empty rows)"]
    if f["X"] is not None:
        _check(np.array_equal(got.X, np.asarray(f["X"], dtype=np.int32)),
               f"{name}: full X matrix differs")
        notes.append("full X matrix identical")
    return notes


def validate_cnb(name: str, ref: dict) -> list[str]:
    c = corpus.load(name, include_title=False, klass="label")
    X = features.build(c.texts).X
    notes, tie_evidence = [], []

    for case in ref["cnb"]:
        labeled = np.zeros(len(c.y), dtype=bool)
        labeled[case["idx"]] = True
        got = cnb.fit(X, c.y, labeled, norm=bool(case["norm"]))

        _check(sorted(got) == case["klasses"],
               f"{name}: classes {sorted(got)} != {case['klasses']}")
        for k in case["klasses"]:
            _check(
                np.allclose(got[k], np.asarray(case["w"][k]), rtol=W_TOL, atol=0.0),
                f"{name}: {k} weights differ (norm={case['norm']}, "
                f"max |rel| = {np.max(np.abs(got[k] - np.asarray(case['w'][k])) / (np.abs(case['w'][k]) + 1e-300)):.3e})",
            )
        # A subset with no positives has no `yes` weights, so every tie rule
        # predicts all-negative and tells us nothing. Only two-class cases
        # discriminate, and there the default must reproduce ezr exactly.
        matching = [
            tie
            for tie in (cnb.NO, cnb.YES)
            if int((p := cnb.predict_yes(X, got, tie=tie)).sum()) == case["n_pred_yes"]
            and metrics.ezr_recall_int(p, c.y) == case["recall_int"]
        ]
        _check(
            bool(matching),
            f"{name}: no tie rule reproduces ezr (norm={case['norm']}, "
            f"want n_pred_yes={case['n_pred_yes']}, recall={case['recall_int']})",
        )
        if len(case["klasses"]) > 1:
            _check(
                matching == [cnb.YES],
                f"{name}: two-class case is reproduced by {matching}, not "
                f"only the {cnb.YES!r} default (norm={case['norm']})",
            )
            tie_evidence.append(case["norm"])

    notes.append(f"cnb ok on {len(ref['cnb'])} subsets x weights, recall, prediction count")
    notes.append(
        f"tie rule pinned to {cnb.YES!r} by {len(tie_evidence)} two-class cases"
    )
    return notes


def main() -> int:
    if not REFERENCE_DIR.exists():
        raise SystemExit(
            f"no reference at {REFERENCE_DIR}; run scripts/textmine_reference.sh first"
        )
    failures = 0
    for name in corpus.DATASETS:
        ref = json.loads((REFERENCE_DIR / f"{name}.json").read_text())
        print(f"\n{name}")
        try:
            for note in validate_features(name, ref):
                print(f"  ok   {note}")
            if "cnb" in ref:
                for note in validate_cnb(name, ref):
                    print(f"  ok   {note}")
        except ValidationError as e:
            failures += 1
            print(f"  FAIL {e}")
    print(f"\n{'PASS' if not failures else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
