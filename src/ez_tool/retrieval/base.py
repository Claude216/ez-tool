"""The seam every tool-selection method plugs into.

Phase 1 is BM25. The method under investigation implements the same protocol,
so the evaluation harness, metrics, and run records stay untouched and the
comparison is apples to apples by construction.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Rank corpus documents against queries."""

    name: str

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Prepare whatever structure the method needs over the corpus."""

    def retrieve(
        self, queries: Sequence[str], top_k: int
    ) -> list[list[tuple[str, float]]]:
        """Return, per query, ``top_k`` ``(doc_id, score)`` pairs, best first."""
