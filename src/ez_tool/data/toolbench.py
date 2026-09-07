"""Load ToolBench's retrieval split, as ToolGen's Table 1 uses it.

This is a *different* dataset from StableToolBench's solvable queries handled
in ``queries.py``. It exists so our BM25 can be placed next to ToolGen's
published numbers on the data those numbers were actually computed on.

Layout, from ``reasonwang/ToolGen-Datasets/data.tar.gz``::

    data/retrieval/
        corpus_G123.tsv                      49,936 docs (multi-domain)
        G{1,2,3}/corpus.tsv                  per-domain corpus (in-domain)
        G{1,2,3}/test_{stage}_{split}.query.txt
        G{1,2,3}/qrels.test_{stage}_{split}.tsv
        G{1,2,3}_toolid_to_full_tool_id.json  in-domain docid -> G123 docid

Two properties of the released files will silently corrupt any loader that
does not handle them, and ToolGen's own code only survives them because it
keys everything into dicts:

* **The corpora contain duplicate rows.** ``G1/corpus.tsv`` parses to 432,799
  rows but holds only 10,439 distinct docids -- roughly 41 copies of each.
* **The query files contain duplicate rows.** ``test_G1_instruction.query.txt``
  has 457 rows but only 200 distinct qids; a query is repeated once per
  relevant API. Evaluating the raw rows would weight multi-API queries higher
  and report a query count 2.3x too large.

Both are deduplicated by id here, matching their behaviour exactly.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# The six (stage, split) pairs behind ToolGen's I1/I2/I3 columns.
TOOLBENCH_SPLITS = (
    ("G1", "instruction"),
    ("G1", "tool"),
    ("G1", "category"),
    ("G2", "instruction"),
    ("G2", "category"),
    ("G3", "instruction"),
)

# ToolGen reports I1/I2/I3; StableToolBench renamed the same subsets G1/G2/G3.
SUBSET_OF_STAGE = {"G1": "I1", "G2": "I2", "G3": "I3"}


@dataclasses.dataclass(slots=True)
class ToolBenchQuery:
    query_id: str
    stage: str
    split: str
    text: str
    relevant: set[str]

    @property
    def group(self) -> str:
        return f"{self.stage}_{self.split}"


def document_text(doc: dict) -> str:
    """ToolGen's document concatenation, reproduced verbatim.

    From ``evaluation/retrieval/eval_bm25.py::process_retrieval_ducoment``.
    The schemas go in as raw ``json.dumps`` output -- defaults, type names and
    JSON punctuation included -- which is what their BM25 actually indexes.
    """
    return (
        (doc.get("category_name", "") or "")
        + ", "
        + (doc.get("tool_name", "") or "")
        + ", "
        + (doc.get("api_name", "") or "")
        + ", "
        + (doc.get("api_description", "") or "")
        + ", required_params: "
        + json.dumps(doc.get("required_parameters", ""))
        + ", optional_params: "
        + json.dumps(doc.get("optional_parameters", ""))
        + ", return_schema: "
        + json.dumps(doc.get("template_response", ""))
    )


def load_corpus(
    root: Path, stage: str, setting: str
) -> tuple[list[str], list[str], dict]:
    """Return ``(doc_ids, texts, stats)`` for one retrieval setting.

    ``in_domain`` ranks within that stage's own corpus; ``multi_domain`` ranks
    over the pooled 49,936-document corpus, which is the setting our
    StableToolBench runs correspond to.
    """
    if setting == "multi_domain":
        path = root / "corpus_G123.tsv"
    elif setting == "in_domain":
        path = root / stage / "corpus.tsv"
    else:
        raise ValueError(f"unknown setting: {setting!r}")

    df = pd.read_csv(path, sep="\t")
    stats = Counter({"rows_in_file": len(df)})

    by_id: dict[str, str] = {}
    for docid, content in zip(df["docid"], df["document_content"]):
        key = str(docid)
        if key in by_id:
            stats["duplicate_rows"] += 1
            continue
        by_id[key] = document_text(json.loads(content))

    stats["documents"] = len(by_id)
    doc_ids = list(by_id)
    return doc_ids, [by_id[d] for d in doc_ids], dict(stats)


def load_queries(
    root: Path, stage: str, split: str, setting: str
) -> tuple[list[ToolBenchQuery], dict]:
    """Load one split's queries and qrels, deduplicated by id."""
    qpath = root / stage / f"test_{stage}_{split}.query.txt"
    lpath = root / stage / f"qrels.test_{stage}_{split}.tsv"

    qdf = pd.read_csv(qpath, sep="\t", names=["qid", "query"])
    ldf = pd.read_csv(lpath, sep="\t", names=["qid", "unused", "docid", "label"])
    stats = Counter({"query_rows": len(qdf), "qrel_rows": len(ldf)})

    # Later rows overwrite earlier ones for a repeated qid, exactly as their
    # `ir_test_queries[row.qid] = row.query` does. The text is identical
    # across duplicates, so the choice does not matter.
    texts: dict[str, str] = {}
    for qid, text in zip(qdf["qid"], qdf["query"]):
        texts[str(qid)] = text

    relevant: dict[str, set[str]] = {}
    for qid, docid in zip(ldf["qid"], ldf["docid"]):
        relevant.setdefault(str(qid), set()).add(str(docid))

    if setting == "multi_domain":
        # In-domain docids are local to a stage; the pooled corpus uses global
        # ids, so every label has to be remapped before it can be matched.
        remap_path = root / f"{stage}_toolid_to_full_tool_id.json"
        remap = {str(k): str(v) for k, v in json.loads(remap_path.read_text()).items()}
        remapped: dict[str, set[str]] = {}
        for qid, docids in relevant.items():
            out = set()
            for d in docids:
                if d in remap:
                    out.add(remap[d])
                else:
                    stats["labels_unmapped"] += 1
            remapped[qid] = out
        relevant = remapped

    queries = [
        ToolBenchQuery(
            query_id=qid,
            stage=stage,
            split=split,
            text=text,
            relevant=relevant.get(qid, set()),
        )
        for qid, text in texts.items()
    ]
    stats["queries"] = len(queries)
    stats["labels"] = sum(len(q.relevant) for q in queries)
    return queries, dict(stats)
