"""Complementary Naive Bayes re-ranking on pseudo-relevance feedback.

The transplant of section 7's method that keeps the comparison against BM25
honest. CNB is a supervised learner and this benchmark supplies no labels at
query time, so the labels are *presumed* rather than known:

* **pseudo-positives** -- BM25's top `n_pos` for the query. Some are wrong;
  BM25's R@1 is 0.229, so the top result is right about a quarter of the time.
* **pseudo-negatives** -- `n_neg` documents drawn uniformly from the corpus.
  With roughly two relevant APIs in 47,058, a random draw is overwhelmingly
  irrelevant. This is exactly Yu et al.'s presumptive sampling, which §7.2
  credits FASTREAD with, applied here without a human.

No qrels are read, so this satisfies `Retriever` and is directly comparable to
BM25 on the same queries, unlike the feedback re-ranker in
`textmine.select`, which is handed ground truth on every attempt.

`fusion` decides how much of BM25's evidence survives. `cnb` discards it and
ranks on the classifier alone; `rrf` fuses the two rankings, so a document has
to satisfy both. Since a handful of noisy pseudo-positives is thin evidence
against corpus-wide term statistics, `rrf` is the default.

`score` matters more than it looks. In complementary Bayes a class's weights
are built from the documents *outside* it, so `w_yes` is a function of the
negatives alone -- here, of 200 randomly drawn APIs that say nothing about the
query. Ranking on `X @ w_yes` (`score="yes"`, which is what ezr's `tmActive`
uses to pick its next paper) therefore just prefers documents avoiding common
vocabulary, and scores 0.124 R@5 against BM25's 0.478. The pseudo-positives
only enter through `w_no`, so the discriminative quantity is the decision
margin `X @ w_yes - X @ w_no` -- the same comparison `_tm_best` makes when it
classifies. That is the default here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from scipy.sparse import csr_matrix, vstack

from ..textmine import cnb, features
from .bm25 import BM25Retriever


class CnbPrfRetriever:
    name = "cnb_prf"

    def __init__(
        self,
        *,
        pool: int = 100,
        n_pos: int = 5,
        n_neg: int = 200,
        top_features: int = 5000,
        norm: bool = False,
        alpha: float = 1.0,
        fusion: str = "rrf",
        score: str = "margin",
        k_rrf: float = 60.0,
        seed: int = 1,
        seed_source: str = "bm25",
        length_norm: bool = True,
        balance: bool = True,
        rounds: int = 1,
        bm25: dict | None = None,
    ) -> None:
        if fusion not in ("rrf", "cnb"):
            raise ValueError(f"unknown fusion: {fusion!r}")
        if score not in ("margin", "yes"):
            raise ValueError(f"unknown score: {score!r}")
        if seed_source not in ("bm25", "query"):
            raise ValueError(f"unknown seed_source: {seed_source!r}")
        self.pool = pool
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.top_features = top_features
        self.norm = norm
        self.alpha = alpha
        self.fusion = fusion
        self.score = score
        self.seed_source = seed_source
        self.length_norm = length_norm
        self.balance = balance
        self.rounds = rounds
        self.k_rrf = k_rrf
        self.seed = seed
        self._bm25 = BM25Retriever(**(bm25 or {}))
        self._doc_ids: list[str] = []

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        self._bm25.index(doc_ids, texts)
        self._doc_ids = list(doc_ids)
        self._at = {d: i for i, d in enumerate(self._doc_ids)}
        self._X, self._vocab = features.build_sparse(
            list(texts), top=self.top_features
        )
        self._vocab_at = {w: j for j, w in enumerate(self._vocab)}

    def _query_vector(self, text: str) -> csr_matrix:
        """The query itself as a row in the corpus feature space."""
        words = features.stem(features.drop_stopwords(features.tokenize([text])))[0]
        counts: dict[int, int] = {}
        for w in words:
            j = self._vocab_at.get(w)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        idx = np.asarray(sorted(counts), dtype=np.int32)
        vals = np.asarray([counts[j] for j in idx], dtype=np.int32)
        return csr_matrix(
            (vals, idx, np.asarray([0, len(idx)], dtype=np.int32)),
            shape=(1, self._X.shape[1]),
        )

    def _rank_scores(self, v: np.ndarray) -> np.ndarray:
        """Score every document, optionally per unit length.

        The decision margin carries a constant offset on every feature -- the
        `log(den_yes / den_no)` term -- which cancels when classifying one
        document but not when ranking many. With one query document against
        thousands of negatives that constant is large, so `X @ v` is dominated
        by document length and the ranking degenerates to "longest API first"
        (measured: R@100 = 0.006, i.e. worse than chance at the top).

        Dividing by the document's term count removes it. This is Rennie et
        al.'s length normalisation, the companion to the weight normalisation
        `norm` already exposes.
        """
        s = np.asarray(self._X @ v).ravel()
        if not self.length_norm:
            return s
        lengths = np.asarray(self._X.sum(axis=1)).ravel()
        return s / np.maximum(lengths, 1.0)

    def _margin(self, w: dict) -> np.ndarray | None:
        if cnb.YES not in w:
            return None
        v = w[cnb.YES]
        if self.score == "margin" and cnb.NO in w:
            v = v - w[cnb.NO]
        return v

    def _cold_scores(self, text: str, rng) -> np.ndarray | None:
        """Cold start: the query is the only positive, no BM25 involved.

        The negatives are drawn uniformly from the corpus, so the classifier is
        fitted on {this query} vs {a sample of everything}. Ranking is over the
        whole corpus, which makes this a retriever rather than a re-ranker --
        if it loses, BM25's pseudo-labels were not the problem.
        """
        n_docs = self._X.shape[0]
        neg = rng.choice(n_docs, size=min(self.n_neg, n_docs), replace=False)
        qv = self._query_vector(text)
        Xn = self._X[neg]
        if self.balance and qv.nnz:
            # One query document against `n_neg` corpus documents is a 1-vs-N
            # imbalance, and complementary Bayes builds `w_yes` from the
            # negatives. Unscaled, a query term contributes only log(1 + alpha)
            # = 0.69 while -log(neg_count + alpha) reaches past -5, so corpus
            # frequency swamps query membership: measured margins put
            # "football" at -1.46 for a football query while an unrelated rare
            # token scores +0.74. Scaling the positive class to the same total
            # mass as the negative class restores the balance the weights
            # assume.
            scale = float(Xn.sum()) / float(qv.sum())
            qv = qv.astype(np.float64) * scale
        small = vstack([qv, Xn], format="csr")
        y = np.zeros(small.shape[0], dtype=bool)
        y[0] = True
        w = cnb.fit(small, y, np.ones(small.shape[0], dtype=bool),
                    alpha=self.alpha, norm=self.norm)
        v = self._margin(w)
        return None if v is None else self._rank_scores(v)

    def retrieve(
        self, queries: Sequence[str], top_k: int
    ) -> list[list[tuple[str, float]]]:
        if self.seed_source == "query":
            return self._retrieve_cold(queries, top_k)
        pools = self._bm25.retrieve(queries, top_k=max(self.pool, top_k))
        rng = np.random.default_rng(self.seed)
        n_docs = len(self._doc_ids)
        out: list[list[tuple[str, float]]] = []

        for hits in pools:
            rows = np.asarray([self._at[d] for d, _ in hits], dtype=int)
            if rows.size == 0:
                out.append([])
                continue

            y = np.zeros(n_docs, dtype=bool)
            labeled = np.zeros(n_docs, dtype=bool)
            pos = rows[: self.n_pos]
            y[pos] = True
            labeled[pos] = True
            neg = rng.choice(n_docs, size=min(self.n_neg, n_docs), replace=False)
            labeled[neg] = True
            # A presumptive negative that is also a pseudo-positive stays
            # positive: `y` is not cleared, only `labeled` is widened.

            w = cnb.fit(self._X, y, labeled, alpha=self.alpha, norm=self.norm)
            if cnb.YES not in w:
                out.append(list(hits)[:top_k])
                continue
            Xr = self._X[rows]
            s = np.asarray(Xr @ w[cnb.YES]).ravel()
            if self.score == "margin" and cnb.NO in w:
                s = s - np.asarray(Xr @ w[cnb.NO]).ravel()

            if self.fusion == "cnb":
                order = np.argsort(-s)
            else:
                cnb_rank = np.empty(len(rows), dtype=float)
                cnb_rank[np.argsort(-s)] = np.arange(len(rows))
                fused = 1.0 / (self.k_rrf + np.arange(len(rows))) + 1.0 / (
                    self.k_rrf + cnb_rank
                )
                order = np.argsort(-fused)

            ranked = [(hits[j][0], float(s[j])) for j in order[:top_k]]
            out.append(ranked)
        return out

    def _retrieve_cold(
        self, queries: Sequence[str], top_k: int
    ) -> list[list[tuple[str, float]]]:
        """Cold start, optionally bootstrapped from its own ranking.

        With `rounds > 1` the top `n_pos` of the previous round become
        pseudo-positives for the next -- pseudo-relevance feedback seeded by
        CNB instead of by BM25, which is the direct test of whether BM25's
        weak R@1 was what held the re-ranker back.
        """
        rng = np.random.default_rng(self.seed)
        n_docs = self._X.shape[0]
        out: list[list[tuple[str, float]]] = []

        for text in queries:
            s = self._cold_scores(text, rng)
            if s is None:
                out.append([])
                continue
            for _ in range(max(0, self.rounds - 1)):
                pos = np.argpartition(-s, self.n_pos)[: self.n_pos]
                y = np.zeros(n_docs, dtype=bool)
                labeled = np.zeros(n_docs, dtype=bool)
                y[pos] = True
                labeled[pos] = True
                labeled[rng.choice(n_docs, size=min(self.n_neg, n_docs),
                                   replace=False)] = True
                v = self._margin(
                    cnb.fit(self._X, y, labeled, alpha=self.alpha, norm=self.norm)
                )
                if v is None:
                    break
                s = self._rank_scores(v)
            top = np.argpartition(-s, min(top_k, n_docs - 1))[:top_k]
            top = top[np.argsort(-s[top])]
            out.append([(self._doc_ids[i], float(s[i])) for i in top])
        return out
