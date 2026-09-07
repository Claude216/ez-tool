"""Follow-up B -- how much training support the test instances' tools have.

Round one bounded the damage from starved tool classes at "one query in twenty",
because the 1,528 tools with <=3 training instances hold only 5.1% of all
labels. That bound holds only if the test set draws tools the way training does.
It has not been checked, and the three G1 test splits are constructed to
generalise along different axes:

    test_G1_instruction   unseen instructions, seen tools   -> support expected
    test_G1_tool          unseen tools                      -> support expected zero
    test_G1_category      unseen categories                 -> support expected zero

The last two are the evidence for a scope statement -- that A3 cannot answer on
two of the three G1 splits at all -- so they are measured rather than assumed.

The same measurement is repeated for StableToolBench's solvable queries, since
that is the set downstream work reports on.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ez_tool.diagnostics import textpipe
from ez_tool.diagnostics.d2_leakage import _binary_matrix
from ez_tool.diagnostics.g1 import G1Data, G1_TEST_SUBSETS, _norm
from ez_tool.paths import DATA_RAW

STB_DIR = (
    DATA_RAW / "StableToolBench" / "solvable_queries" / "test_instruction"
)
STB_G1 = ("G1_instruction", "G1_tool", "G1_category")


def _quantiles(vals: list[int]) -> dict:
    if not vals:
        return {k: None for k in ("min", "p25", "median", "p75", "max")}
    v = sorted(vals)

    def q(f):
        pos = f * (len(v) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"min": v[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": v[-1]}


def _profile(supports: list[int]) -> dict:
    n = len(supports) or 1
    return {
        "instances": len(supports),
        **_quantiles(supports),
        "frac_zero": sum(1 for s in supports if s == 0) / n,
        "frac_le_3": sum(1 for s in supports if s <= 3) / n,
        "frac_le_5": sum(1 for s in supports if s <= 5) / n,
        "frac_le_10": sum(1 for s in supports if s <= 10) / n,
    }


def train_support(data: G1Data) -> tuple[Counter, Counter]:
    """Training-instance counts per tool and per category."""
    tools: Counter = Counter()
    cats: Counter = Counter()
    for inst in data.instances:
        if inst.split != "train":
            continue
        tools[inst.apis[0][0]] += 1
        for c in {_norm(data.docs[k].category) for k in inst.apis}:
            cats[c] += 1
    return tools, cats


def run(data: G1Data) -> dict:
    tool_sup, cat_sup = train_support(data)

    out: dict = {
        "train_instances": sum(1 for i in data.instances if i.split == "train"),
        "train_tools": len(tool_sup),
        "train_categories": len(cat_sup),
        "splits": {},
        "_arrays": {
            # Per class, for the label-space view.
            "train_tool_support_per_class": list(tool_sup.values()),
            # Per training instance, which is the like-for-like comparison
            # against the per-test-instance numbers below.
            "train_tool_support_per_instance": [
                tool_sup[i.apis[0][0]] for i in data.instances if i.split == "train"
            ],
        },
    }

    for split in G1_TEST_SUBSETS:
        insts = [i for i in data.instances if i.split == split]
        t_sup = [tool_sup.get(i.apis[0][0], 0) for i in insts]
        c_sup = [
            max(
                cat_sup.get(c, 0)
                for c in {_norm(data.docs[k].category) for k in i.apis}
            )
            for i in insts
        ]
        out["splits"][f"test_G1_{split}"] = {
            "tool": _profile(t_sup),
            "category": _profile(c_sup),
            "distinct_tools": len({i.apis[0][0] for i in insts}),
            "tools_unseen_in_train": len(
                {i.apis[0][0] for i in insts if tool_sup.get(i.apis[0][0], 0) == 0}
            ),
        }
        out["_arrays"][f"test_G1_{split}"] = t_sup

    out["stabletoolbench"] = _stablebench(data, tool_sup, cat_sup, out)
    nd = nearest_training_query(data)
    out["_arrays"]["nearest"] = nd.pop("_best")
    out["near_duplicates"] = nd
    return out


def nearest_training_query(data: G1Data) -> dict:
    """Jaccard between each test query and its most similar training query.

    Measured because the support numbers imply this release is a random split
    rather than ToolBench's unseen-tool / unseen-category holdouts. If it is,
    train and test can contain paraphrases of one another, and a high similarity
    here is direct evidence of that.
    """
    cache: dict = {}
    sets = [
        set(textpipe.prepare(i.query, cache, split_identifiers=True))
        for i in data.instances
    ]
    vocab_index = {t: i for i, t in enumerate(sorted(set().union(*sets)))}
    M = _binary_matrix(sets, vocab_index)
    lens = np.array([len(x) for x in sets], dtype=np.float32)

    is_train = np.array([i.split == "train" for i in data.instances])
    tr = np.flatnonzero(is_train)
    Tt = M[tr].T.tocsc()
    tr_len = lens[tr]

    res: dict = {"_best": {}}
    # The StableToolBench solvable set is the operationally relevant one: it is
    # what downstream work scores on, so its exposure to the training pool is
    # what a reported number would actually inherit.
    stb_ids = set()
    for name in STB_G1:
        for rec in json.loads((STB_DIR / f"{name}.json").read_text()):
            stb_ids.add(rec.get("query_id"))

    groups = {f"test_G1_{s}": [s] for s in G1_TEST_SUBSETS}
    groups["stabletoolbench_solvable_G1"] = list(G1_TEST_SUBSETS)

    train_texts = {
        data.instances[j].query.strip().lower() for j in tr
    }

    for label, splits in groups.items():
        stb_only = label.startswith("stable")
        te = np.flatnonzero(
            np.array([
                i.split in splits and (not stb_only or i.query_id in stb_ids)
                for i in data.instances
            ])
        )
        inter = np.asarray((M[te] @ Tt).todense(), dtype=np.float32)
        union = lens[te][:, None] + tr_len[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        best = jac.max(axis=1)
        exact = sum(
            1 for j in te
            if data.instances[j].query.strip().lower() in train_texts
        )
        res[label] = {
            "instances": int(len(te)),
            "exact_text_duplicates_in_train": exact,
            "max_jaccard_to_any_training_query": {
                "median": float(np.median(best)),
                "p75": float(np.percentile(best, 75)),
                "p95": float(np.percentile(best, 95)),
                "max": float(best.max()),
            },
            "frac_above_0.5": float((best >= 0.5).mean()),
            "frac_above_0.7": float((best >= 0.7).mean()),
        }
        res["_best"][label] = best
    return res


def _stablebench(data: G1Data, tool_sup: Counter, cat_sup: Counter, out: dict) -> dict:
    """The 474 G1 solvable queries, matched back to the training set.

    Matching is by query id first, then by exact query text, and is reported
    rather than assumed -- an id that also appears in training would mean the
    evaluation set is not held out.
    """
    by_id = {i.query_id: i for i in data.instances}
    by_text = {}
    for i in data.instances:
        by_text.setdefault(i.query.strip().lower(), i)
    train_ids = {i.query_id for i in data.instances if i.split == "train"}
    train_texts = {
        i.query.strip().lower() for i in data.instances if i.split == "train"
    }

    res: dict = {"files": {}, "_arrays": {}}
    for name in STB_G1:
        records = json.loads((STB_DIR / f"{name}.json").read_text())
        supports, cats = [], []
        matched_id = matched_text = unmatched = 0
        in_train_id = in_train_text = 0

        for rec in records:
            raw = rec.get("relevant APIs") or []
            pairs = [(_norm(p[0]), _norm(p[1])) for p in raw if len(p) == 2]
            if not pairs:
                unmatched += 1
                continue
            tool = pairs[0][0]
            supports.append(tool_sup.get(tool, 0))

            known = [p for p in pairs if p in data.docs]
            cats.append(
                max(
                    (cat_sup.get(_norm(data.docs[p].category), 0) for p in known),
                    default=0,
                )
            )

            qid, text = rec.get("query_id"), str(rec.get("query") or "").strip().lower()
            if qid in by_id:
                matched_id += 1
                if qid in train_ids:
                    in_train_id += 1
            elif text in by_text:
                matched_text += 1
                if text in train_texts:
                    in_train_text += 1
            else:
                unmatched += 1

        n = len(records) or 1
        res["files"][name] = {
            "records": len(records),
            "tool": _profile(supports),
            "category": _profile(cats),
            "matched_by_query_id": matched_id,
            "matched_by_query_text": matched_text,
            "unmatched": unmatched,
            "match_rate": (matched_id + matched_text) / n,
            "also_in_toolgen_train_by_id": in_train_id,
            "also_in_toolgen_train_by_text": in_train_text,
        }
        res["_arrays"][name] = supports

    pooled = [s for a in res["_arrays"].values() for s in a]
    res["pooled_G1"] = _profile(pooled)
    res["_arrays"]["pooled"] = pooled
    return res
