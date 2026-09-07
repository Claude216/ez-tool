r"""Diagnostic 3 -- feature sparsity under EZR's vocabulary.

EZR's text settings were tuned for 150-250 word abstracts at
``the.textmine.top=100``. A row in the A3 design is a *query*, an order of
magnitude shorter. A query whose terms all fall outside the vocabulary is a
row of zeros: CNB scores it identically under every class, so its prediction
is whatever ``argmax`` returns first. Those rows are unclassifiable by
construction, and no amount of classifier tuning reaches them.

The vocabulary is built from the query corpus, because in A3 the queries are
the rows and ``tmTfidf`` ranks terms over the rows it is given.

A second failure mode lives on the tool side. ``\b[a-zA-Z]+\b`` treats ``_``
as a word character, so a snake_case identifier contains no word boundary and
tokenizes to nothing whatsoever -- not to fragments, to the empty list.
"""

from __future__ import annotations

import statistics

from ez_tool.diagnostics import textpipe
from ez_tool.diagnostics.d2_leakage import api_doc_text
from ez_tool.diagnostics.g1 import G1Data

VOCAB_SIZES = (100, 500, 1000, 5000, None)
ZERO_FEATURE_LIMIT = 0.10  # the decision rule's ~10% threshold


def _label(n: int | None) -> str:
    return "all" if n is None else str(n)


def sparsity_table(query_docs: list[list[str]]) -> list[dict]:
    """Non-zero feature counts per query at each vocabulary size."""
    sets = [set(d) for d in query_docs]
    rows = []
    for n in VOCAB_SIZES:
        vocab, _ = textpipe.top_terms(query_docs, n)
        keep = set(vocab)
        nz = [len(s & keep) for s in sets]
        total = len(nz) or 1
        rows.append(
            {
                "top_n": _label(n),
                "vocab": len(keep),
                "mean_nonzero": statistics.fmean(nz),
                "median_nonzero": statistics.median(nz),
                "frac_zero": sum(1 for v in nz if v == 0) / total,
                "frac_le_2": sum(1 for v in nz if v <= 2) / total,
            }
        )
    return rows


def tokenizer_damage(names: list[str]) -> dict:
    """How many identifiers EZR's tokenizer empties or mangles.

    "Damaged" means EZR's token list differs from the list obtained once
    underscore/camelCase boundaries are inserted -- i.e. content was lost.
    """
    total = len(names) or 1
    empty = damaged = empty_split = 0
    for name in names:
        plain = textpipe.tokenize(name)
        split = textpipe.tokenize(name, split_identifiers=True)
        if not plain:
            empty += 1
        if plain != split:
            damaged += 1
        if not split:
            empty_split += 1
    return {
        "n": len(names),
        "frac_empty_ezr": empty / total,
        "frac_damaged_ezr": damaged / total,
        "frac_empty_after_split": empty_split / total,
    }


def stemmer_collisions(cache: dict, top: int = 10) -> dict:
    """How often EZR's suffix stripper merges distinct surface forms.

    The suffix list ends in ``e`` and ``y``, so stripping is aggressive enough
    to produce non-words ("nam", "countr") and to collapse words that mean
    different things. Reported because it motivates replacing the stemmer, not
    because the merge is always wrong.
    """
    groups: dict[str, set] = {}
    for surface, stem in cache.items():
        groups.setdefault(stem, set()).add(surface)
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    return {
        "surface_forms": len(cache),
        "stems": len(groups),
        "stems_merging_2plus": len(merged),
        "worst": [
            (k, sorted(v))
            for k, v in sorted(merged.items(), key=lambda x: -len(x[1]))[:top]
        ],
    }


def run(data: G1Data) -> dict:
    cache: dict = {}
    query_docs = [textpipe.prepare(i.query, cache) for i in data.instances]
    doc_docs = [
        textpipe.prepare(api_doc_text(d), cache) for d in data.docs.values()
    ]

    q_lengths = [len(d) for d in query_docs]
    rows = sparsity_table(query_docs)

    _, q_scores = textpipe.top_terms(query_docs, 100)
    _, d_scores = textpipe.top_terms(doc_docs, 100)

    api_names = [d.api for d in data.docs.values()]
    tool_names = sorted({d.tool for d in data.docs.values()})

    chosen = next(r for r in rows if r["top_n"] == "100")
    out = {
        "n_queries": len(query_docs),
        "query_terms_after_pipeline": {
            "mean": statistics.fmean(q_lengths),
            "median": statistics.median(q_lengths),
            "min": min(q_lengths),
            "max": max(q_lengths),
        },
        "sparsity": rows,
        "tokenizer_damage_api_names": tokenizer_damage(api_names),
        "tokenizer_damage_tool_names": tokenizer_damage(tool_names),
        "top30_query_terms": sorted(q_scores.items(), key=lambda x: -x[1])[:30],
        "top30_api_doc_terms": sorted(d_scores.items(), key=lambda x: -x[1])[:30],
        "stemmer_collisions": stemmer_collisions(cache),
        "_arrays": {"query_nonzero_all": q_lengths},
    }
    out["verdict"] = (
        f"REPRESENTATION INADEQUATE at EZR's default top=100: "
        f"{chosen['frac_zero']:.1%} of queries have zero non-zero features."
        if chosen["frac_zero"] > ZERO_FEATURE_LIMIT
        else (
            f"Adequate at EZR's default top=100: only {chosen['frac_zero']:.1%} "
            f"of queries have zero features."
        )
    )
    return out
