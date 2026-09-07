"""Put the StableToolBench corpus onto the ezr text-mining substrate.

Section 7's method is a per-topic active learner: one topic, thousands of
candidate documents, a human labelling them one at a time. Tool selection is
shaped differently, and the difference decides the design:

|                  | SLR (section 7)      | StableToolBench          |
|------------------|----------------------|--------------------------|
| topics           | 1 per corpus         | 765 queries              |
| candidates       | 1,704-8,911          | 47,058 APIs              |
| relevant each    | 48-132 (1 in 86)     | ~2.8 (1 in 16,800)       |
| labels available | one per paper read   | none at query time       |

Two consequences. First, `_tm_warm(yes=20)` cannot be transplanted: there are
only about three relevant APIs per query, so a warm start of twenty positives
would hand the learner every answer. Second, `retrieval.base.Retriever` is a
zero-shot protocol -- `index` then `retrieve`, no labels -- while CNB needs
labelled examples for the query in front of it.

So CNB is used here as a *feedback re-ranker*, not a retriever: BM25 supplies a
shortlist, an agent tries entries from it and each attempt reveals whether that
API was relevant, and CNB reorders what remains. See `select.py`. This module
is the part common to any framing -- turning APIs into the same TF-IDF
substrate the SLR corpora use, and turning qrels into label vectors.

Feature width is the one parameter that cannot carry over. 100 global features
describe a single SLR topic adequately; they cannot separate 765 unrelated
queries over 47,058 APIs, so `top_features` here is thousands and the matrix
is sparse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.corpus import load_corpus
from ..data.queries import apply_label_policy, load_queries
from ..paths import DATA_PROCESSED, GROUPS
from . import features

CACHE = DATA_PROCESSED / "textmine_toolbench"


@dataclass(frozen=True)
class ToolCorpus:
    doc_ids: list[str]
    X: object  # scipy CSR, (n_apis, n_features)
    vocab: list[str]
    at: dict[str, int]  # doc_id -> row

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)


def build(
    top_features: int = 5000,
    include_parameters: bool = True,
    backfill_description: bool = False,
    cache: bool = True,
) -> ToolCorpus:
    """One row per API, text through the same pipeline the SLR corpora use.

    Defaults mirror the `with_parameters` BM25 variant -- argument names and
    descriptions indexed, blank descriptions not backfilled -- so any
    difference against the BM25 baseline is the method, not the document text.
    """
    tag = f"top{top_features}_p{int(include_parameters)}_b{int(backfill_description)}"
    path = CACHE / f"{tag}.npz"
    if cache and path.exists():
        with np.load(path, allow_pickle=True) as z:
            from scipy.sparse import csr_matrix

            X = csr_matrix(
                (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
            )
            doc_ids = [str(d) for d in z["doc_ids"]]
            vocab = [str(v) for v in z["vocab"]]
        return ToolCorpus(doc_ids, X, vocab, {d: i for i, d in enumerate(doc_ids)})

    docs, _ = load_corpus()
    texts = [
        d.text(
            backfill_description=backfill_description,
            include_parameters=include_parameters,
        )
        for d in docs
    ]
    t0 = time.time()
    X, vocab = features.build_sparse(texts, top=top_features)
    print(
        f"  built {X.shape[0]} x {X.shape[1]} ({X.nnz} nonzero, "
        f"{X.nnz / X.shape[0]:.1f} terms/doc) in {time.time() - t0:.1f}s"
    )
    doc_ids = [d.doc_id for d in docs]
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=X.data,
            indices=X.indices,
            indptr=X.indptr,
            shape=np.asarray(X.shape),
            doc_ids=np.asarray(doc_ids, dtype=object),
            vocab=np.asarray(vocab, dtype=object),
        )
    return ToolCorpus(doc_ids, X, vocab, {d: i for i, d in enumerate(doc_ids)})


def labelled_queries(
    corpus: ToolCorpus,
    groups: tuple[str, ...] = GROUPS,
    label_policy: str = "drop_dead",
) -> tuple[list, dict]:
    """Queries with their relevant APIs resolved to corpus rows."""
    queries = load_queries(groups)
    kept, stats = apply_label_policy(queries, set(corpus.doc_ids), label_policy)
    return kept, stats


def relevant_rows(corpus: ToolCorpus, query) -> np.ndarray:
    """Row indices of the APIs labelled relevant for this query."""
    return np.asarray(
        sorted(corpus.at[d] for d in query.relevant if d in corpus.at), dtype=int
    )


def main() -> None:
    corpus = build()
    queries, stats = labelled_queries(corpus)
    n_rel = np.asarray([len(q.relevant) for q in queries])
    print(f"\ncorpus  : {corpus.n_docs} APIs, {len(corpus.vocab)} features")
    print(f"queries : {len(queries)} after {stats['policy']}")
    print(
        f"relevant: mean {n_rel.mean():.2f}, min {n_rel.min()}, max {n_rel.max()} "
        f"-> positive rate 1 in {corpus.n_docs / n_rel.mean():,.0f}"
    )
    empty = sum(1 for i in range(corpus.n_docs)
                if corpus.X.indptr[i + 1] == corpus.X.indptr[i])
    print(f"APIs with no indexed term: {empty}")


if __name__ == "__main__":
    main()
