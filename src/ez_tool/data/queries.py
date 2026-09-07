"""Load StableToolBench solvable queries and turn them into qrels.

Each query carries a ``relevant APIs`` field holding ``[tool_name, api_name]``
pairs -- on average 2.1 to 3.0 per query depending on group. Those pairs are
the ground truth for retrieval.

Queries also carry a small ``api_list`` (mean 4-7 entries): the candidate pool
the original benchmark hands the agent. We deliberately ignore it and retrieve
over the full corpus, which is the realistic agent setting.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import orjson

from ..paths import GROUPS, QUERIES_DIR
from .corpus import doc_id
from .text import normalize


@dataclasses.dataclass(slots=True)
class Query:
    query_id: str
    group: str
    text: str
    relevant: set[str]  # doc_ids

    @property
    def normalized(self) -> str:
        return normalize(self.text)


def load_queries(
    groups: tuple[str, ...] = GROUPS, root: Path | None = None
) -> list[Query]:
    root = root or QUERIES_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run scripts/fetch_data.sh on a compute node."
        )

    queries: list[Query] = []
    for group in groups:
        path = root / f"{group}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing group file: {path}")
        for raw in orjson.loads(path.read_bytes()):
            relevant = {
                doc_id(str(pair[0]).strip(), str(pair[1]).strip())
                for pair in raw.get("relevant APIs") or []
                if len(pair) >= 2
            }
            queries.append(
                Query(
                    query_id=f"{group}:{raw['query_id']}",
                    group=group,
                    text=raw["query"],
                    relevant=relevant,
                )
            )
    return queries


def apply_label_policy(
    queries: list[Query], corpus_ids: set[str], policy: str
) -> tuple[list[Query], dict]:
    """Reconcile labels annotated on the 2023 environment with the 2024 corpus.

    ``toolenv2404_filtered`` dropped RapidAPI endpoints that had died by 2024,
    but ``relevant APIs`` was annotated against the older, larger environment.
    24.3% of labels therefore point at documents that no longer exist and are
    unreachable by any retriever whatsoever.

    ``keep_all``   leave them in. Comparable across methods, but recall is
                   capped at 0.757 and no method can ever reach 1.0.
    ``drop_dead``  discard unreachable labels, then discard queries left with
                   none (126 of 765). Recall regains a true ceiling of 1.0, at
                   the cost of a smaller and slightly re-weighted query set.
    """
    if policy not in ("keep_all", "drop_dead"):
        raise ValueError(f"unknown label_policy: {policy!r}")

    stats = Counter()
    kept: list[Query] = []
    for q in queries:
        reachable = q.relevant & corpus_ids
        stats["labels_total"] += len(q.relevant)
        stats["labels_dead"] += len(q.relevant - corpus_ids)

        if policy == "keep_all":
            kept.append(q)
            continue

        if not reachable:
            stats["queries_dropped"] += 1
            continue
        kept.append(dataclasses.replace(q, relevant=reachable))

    stats["queries_in"] = len(queries)
    stats["queries_out"] = len(kept)
    return kept, {"policy": policy, **dict(stats)}


def check_label_coverage(queries: list[Query], corpus_ids: set[str]) -> dict:
    """How many labels actually resolve to a corpus document.

    This is the load-bearing sanity check for the whole pipeline: labels that
    do not resolve are unreachable by any retriever, so they silently cap
    recall and make every method look worse by the same unknown amount. Run it
    before trusting any metric.
    """
    stats = Counter()
    unresolved_examples: list[str] = []

    for q in queries:
        stats["queries"] += 1
        if not q.relevant:
            stats["queries_without_labels"] += 1
        for did in q.relevant:
            stats["labels"] += 1
            if did in corpus_ids:
                stats["labels_resolved"] += 1
            else:
                stats["labels_unresolved"] += 1
                if len(unresolved_examples) < 10:
                    unresolved_examples.append(did.replace("\t", " :: "))

    labels = stats["labels"] or 1
    return {
        **dict(stats),
        "resolution_rate": stats["labels_resolved"] / labels,
        "unresolved_examples": unresolved_examples,
    }
