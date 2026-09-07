"""ToolGen's BM25 stack, reproduced for replication.

Their ``BM25Indexer`` (``evaluation/utils/retrieval.py``) is
``rank_bm25.BM25Okapi`` over ``nltk.word_tokenize(document.lower())`` -- no
stemming, no stopword removal, and a tokenizer that leaves code-shaped
identifiers intact (``get_all``, ``climate-news`` and ``v1/quote`` each stay a
single token).

This exists purely so their published numbers can be reproduced on their own
terms. ``bm25.BM25Retriever`` is the stack we use for our own results.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

#: Same pattern bm25s uses, so the two libraries can be fed identical tokens.
_REGEX_PATTERN = re.compile(r"(?u)[\w\-/]+")
_HAS_WORD_CHAR = re.compile(r"\w")


class BM25OkapiRetriever:
    name = "bm25_okapi_toolgen"

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer: str = "nltk",
    ):
        # `tokenizer='regex'` feeds BM25Okapi the same tokens bm25s would see,
        # isolating the scoring library from the tokenizer.
        if tokenizer not in ("nltk", "regex", "nltk_nopunct"):
            raise ValueError(f"unknown tokenizer: {tokenizer!r}")
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer == "regex":
            return _REGEX_PATTERN.findall(text.lower())
        tokens = word_tokenize(text.lower())
        if self.tokenizer == "nltk_nopunct":
            # Drop tokens carrying no word character -- ',', '{', '"', ':' and
            # friends. These documents are json.dumps output, so nltk emits
            # 174.4 tokens/doc against the regex tokenizer's 71.2, and the
            # excess is almost entirely JSON punctuation. Removing exactly
            # that, and nothing else, isolates its contribution.
            tokens = [t for t in tokens if _HAS_WORD_CHAR.search(t)]
        return tokens

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        self._doc_ids = list(doc_ids)
        tokenized = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def retrieve(
        self, queries: Sequence[str], top_k: int
    ) -> list[list[tuple[str, float]]]:
        if self._bm25 is None:
            raise RuntimeError("index() must be called before retrieve()")

        k = min(top_k, len(self._doc_ids))
        results: list[list[tuple[str, float]]] = []
        for query in queries:
            scores = self._bm25.get_scores(self._tokenize(query))
            # argpartition then sort only the top k: full sorts of a 50k-doc
            # score vector per query dominate the runtime otherwise.
            top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
            top = top[np.argsort(-scores[top])]
            results.append([(self._doc_ids[int(i)], float(scores[i])) for i in top])
        return results
