"""BM25 baseline over the API corpus.

Uses ``bm25s`` (sparse scipy-backed scoring) with Snowball stemming. On 47k
short documents this indexes in seconds and retrieves the full 765-query set
in well under a minute on CPU, so the baseline needs no GPU at all.
"""

from __future__ import annotations

from typing import Sequence

import bm25s
import Stemmer


class BM25Retriever:
    name = "bm25"

    #: bm25s/sklearn default. `\w` includes the underscore, so `get_all` stays
    #: one token, while `-` and `/` are separators. Single characters are
    #: dropped by the `\w\w+` requirement.
    DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"

    #: Keeps intra-word hyphens and slashes joined, approximating what
    #: `nltk.word_tokenize` does to `climate-news` and `v1/quote`.
    NLTK_LIKE_TOKEN_PATTERN = r"(?u)[\w\-/]+"

    def __init__(
        self,
        *,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stemmer: bool = True,
        stopwords: str = "en",
        token_pattern: str | None = None,
    ) -> None:
        self.method = method
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.token_pattern = token_pattern or self.DEFAULT_TOKEN_PATTERN
        self._stemmer = Stemmer.Stemmer("english") if stemmer else None
        self._doc_ids: list[str] = []
        self._retriever: bm25s.BM25 | None = None

    def _tokenize(self, texts: Sequence[str], *, return_ids: bool):
        return bm25s.tokenize(
            list(texts),
            stopwords=self.stopwords,
            stemmer=self._stemmer,
            token_pattern=self.token_pattern,
            return_ids=return_ids,
            show_progress=False,
        )

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        self._doc_ids = list(doc_ids)
        # Corpus side is tokenized to integer ids, which is what BM25.index wants.
        tokens = self._tokenize(texts, return_ids=True)
        self._retriever = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self._retriever.index(tokens, show_progress=False)

    def retrieve(
        self, queries: Sequence[str], top_k: int
    ) -> list[list[tuple[str, float]]]:
        if self._retriever is None:
            raise RuntimeError("index() must be called before retrieve()")

        # Queries stay as token *strings*: retrieve() maps them against the
        # frozen index vocabulary and silently drops terms the corpus never
        # saw, which is the correct BM25 behaviour for out-of-vocabulary query
        # terms. Tokenizing queries to ids would instead build a second,
        # misaligned vocabulary.
        tokens = self._tokenize(queries, return_ids=False)
        k = min(top_k, len(self._doc_ids))
        idx, scores = self._retriever.retrieve(
            tokens, k=k, show_progress=False, n_threads=0
        )

        results: list[list[tuple[str, float]]] = []
        for row in range(idx.shape[0]):
            results.append(
                [
                    (self._doc_ids[int(idx[row, c])], float(scores[row, c]))
                    for c in range(idx.shape[1])
                ]
            )
        return results
