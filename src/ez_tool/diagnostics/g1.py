"""Load G1 (single-tool) instances and their API documentation.

Source of truth is ToolGen's re-release of ToolBench's retrieval split::

    data/raw/toolgen_data/data/retrieval/G1/train.json   88,395 instances
    data/raw/toolgen_data/data/retrieval/G1/test.json       600 instances

Each record is ``{api_list, query, "relevant APIs", query_id}``. The label
lives in ``relevant APIs`` as a list of ``[tool_name, api_name]`` pairs -- a
*list*, not a scalar, even in G1. ``api_list`` is the candidate pool the
generator was shown; its union over all instances is the API documentation
corpus used here, which keeps labels and documents keyed identically and
avoids the name-normalisation mismatches that ``corpus.tsv`` introduces.

Nothing else in the repo reads these files, so the loader owns its own
name-normalisation rather than sharing ``data/toolbench.py``'s docid path.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

from ez_tool.paths import DATA_RAW

G1_DIR = DATA_RAW / "toolgen_data" / "data" / "retrieval" / "G1"

# The three official G1 test subsets, held out of train.json.
G1_TEST_SUBSETS = ("instruction", "tool", "category")


def _norm(s: str) -> str:
    """Fold the whitespace/case noise that separates otherwise equal names.

    ``relevant APIs`` and ``api_list`` disagree on capitalisation and stray
    spaces for a small number of APIs; without folding, those instances look
    like missing labels rather than the naming artefact they are.
    """
    return " ".join(str(s or "").split()).strip().lower()


@dataclasses.dataclass(slots=True)
class Instance:
    query_id: int
    split: str  # "train" or one of G1_TEST_SUBSETS
    query: str
    # Ground truth, normalised. Each entry is a key into ApiPool.docs.
    apis: tuple[tuple[str, str], ...]


@dataclasses.dataclass(slots=True)
class ApiDoc:
    category: str
    tool: str
    api: str
    description: str
    required: tuple[tuple[str, str], ...]  # (name, description)
    optional: tuple[tuple[str, str], ...]


@dataclasses.dataclass(slots=True)
class G1Data:
    instances: list[Instance]
    docs: dict[tuple[str, str], ApiDoc]  # (tool, api) -> documentation
    # Display names, keyed the same way, for report tables.
    stats: dict


def _params(entries) -> tuple[tuple[str, str], ...]:
    """Keep parameter names and descriptions; never defaults.

    Defaults carry base64 payloads and example values -- indexing them would
    measure the corpus's junk, not its vocabulary.
    """
    out = []
    for e in entries or []:
        if isinstance(e, dict):
            out.append((str(e.get("name", "") or ""),
                        str(e.get("description", "") or "")))
    return tuple(out)


def load(root: Path | None = None) -> G1Data:
    """Read every G1 instance and the API pool they draw from."""
    root = root or G1_DIR
    stats: Counter = Counter()

    files = [("train", root / "train.json")]
    files += [(s, root / f"test_G1_{s}.json") for s in G1_TEST_SUBSETS]

    instances: list[Instance] = []
    docs: dict[tuple[str, str], ApiDoc] = {}
    seen_qids: set[int] = set()

    for split, path in files:
        records = json.loads(path.read_text())
        stats[f"records_{split}"] = len(records)

        for rec in records:
            # api_list first: the pool must exist before labels resolve.
            for a in rec.get("api_list") or []:
                key = (_norm(a.get("tool_name")), _norm(a.get("api_name")))
                if not key[0] or not key[1]:
                    stats["api_list_unnamed"] += 1
                    continue
                docs.setdefault(key, ApiDoc(
                    category=str(a.get("category_name", "") or ""),
                    tool=str(a.get("tool_name", "") or ""),
                    api=str(a.get("api_name", "") or ""),
                    description=str(a.get("api_description", "") or ""),
                    required=_params(a.get("required_parameters")),
                    optional=_params(a.get("optional_parameters")),
                ))

            qid = rec.get("query_id")
            query = str(rec.get("query") or "").strip()
            raw = rec.get("relevant APIs")

            if qid is None or not query:
                stats["dropped_no_query_or_id"] += 1
                continue
            if qid in seen_qids:
                stats["dropped_duplicate_qid"] += 1
                continue
            if not raw:
                stats["dropped_empty_relevant_apis"] += 1
                continue

            apis, bad = [], False
            for pair in raw:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    bad = True
                    break
                key = (_norm(pair[0]), _norm(pair[1]))
                if not key[0] or not key[1]:
                    bad = True
                    break
                apis.append(key)
            if bad or not apis:
                stats["dropped_malformed_relevant_apis"] += 1
                continue

            seen_qids.add(qid)
            instances.append(Instance(qid, split, query, tuple(dict.fromkeys(apis))))

    # A label whose API never appears in any api_list has no documentation and
    # cannot participate in the leakage diagnostic. Count them before dropping.
    kept, unresolved = [], Counter()
    for inst in instances:
        missing = [k for k in inst.apis if k not in docs]
        if missing:
            unresolved["instances_with_unresolved_api"] += 1
            unresolved["unresolved_api_mentions"] += len(missing)
        resolved = tuple(k for k in inst.apis if k in docs)
        if not resolved:
            unresolved["dropped_all_apis_unresolved"] += 1
            continue
        kept.append(dataclasses.replace(inst, apis=resolved))

    stats.update(unresolved)
    stats["instances_kept"] = len(kept)
    stats["api_docs"] = len(docs)
    return G1Data(instances=kept, docs=docs, stats=dict(stats))


def levels(data: G1Data, inst: Instance) -> dict[str, tuple[str, ...]]:
    """Ground-truth label sets at each level of the hierarchy."""
    apis = inst.apis
    tools = tuple(dict.fromkeys(t for t, _ in apis))
    cats = tuple(dict.fromkeys(_norm(data.docs[k].category) for k in apis))
    return {
        "api": tuple(f"{t}\t{a}" for t, a in apis),
        "tool": tools,
        "category": cats,
    }
