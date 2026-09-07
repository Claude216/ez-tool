"""Text normalization shared by corpus documents and queries.

API identifiers are code-shaped rather than prose: ``getUserById``,
``get_all_climate_change_news``, ``v1/stock-quote``. A plain whitespace
tokenizer keeps those as single opaque terms, which costs BM25 most of its
lexical signal on exactly the field that identifies the endpoint. Splitting
them into words has to happen identically on both sides of the index.
"""

from __future__ import annotations

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[_\-/.:]+")
_NON_WORD = re.compile(r"[^\w\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, split camelCase and separator-joined identifiers."""
    text = _CAMEL.sub(" ", text)
    text = _SEPARATORS.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip().lower()
