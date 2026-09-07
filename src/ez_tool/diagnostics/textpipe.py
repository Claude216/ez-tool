r"""EZR's text pipeline, re-implemented rather than imported.

``ezr.py`` reaches its tokeniser only through ``tmCsv`` -> ``tmTokenize`` ->
``tmNostop`` -> ``tmStem`` -> ``tmTfidf``, all of which read a CSV with
``abstract``/``label`` columns and consult the module-level ``the.textmine.top``.
Reproducing the behaviour here keeps the diagnostics free of that global state
and of the CSV round-trip, and lets the vocabulary size vary per call.

Every step mirrors the vendored source at
``src/ez_tool/textmine/vendor/ezr.py`` (``tmTokenize`` .. ``tmTfidf``):

* ``re.findall(r'\b[a-zA-Z]+\b', text.lower())``, keeping ``len(w) > 2``
* stop words from ``resources/text/stop_words.txt`` (197 entries)
* suffix stripping from ``suffixes.txt`` (71 entries), one round, guarded by
  ``len(c) >= 2 and len(c) >= len(w) * .5``
* top-N terms by ``sum over docs of tf * log(N_docs / df)``

Two deliberate departures, both documented at their site: deterministic
tie-breaks where EZR leans on hash order, and :func:`tokenize`'s
``split_identifiers`` flag, which exists only to give the tokenizer-damage
measurement something to compare against. With the flag off this is EZR.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log
from pathlib import Path

_RESOURCES = (
    Path(__file__).resolve().parents[1] / "textmine" / "vendor" / "resources" / "text"
)

WORD_RE = re.compile(r"\b[a-zA-Z]+\b")
# Underscore/hyphen/dot/slash/digit boundaries, plus lowerUPPER camel humps.
_SNAKE_RE = re.compile(r"[_\-/.0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _load(name: str) -> set[str]:
    text = (_RESOURCES / name).read_text()
    return {w.strip().lower() for w in text.splitlines() if w.strip()}


STOP_WORDS = _load("stop_words.txt")

# EZR sorts by length alone and leaves equal-length ties to set iteration order.
# The tie is harmless -- a word ends with at most one suffix of a given length --
# so the alphabetic tiebreak makes this reproducible without ever changing which
# suffix matches.
SUFFIXES = sorted(_load("suffixes.txt"), key=lambda s: (-len(s), s))


def _stem1(w: str, cache: dict, n: int = 1) -> str:
    """EZR's ``_tm_stem1``, cache quirk included.

    The recursion bottoms out in ``cache.setdefault(c, c)``, so a stem that was
    itself seen as a token earlier resolves to whatever *it* was stemmed to
    then. The mapping is therefore order-dependent, and reproducing that
    matters: a cleaner one-shot stripper yields a different vocabulary.
    """
    if w in cache or n <= 0:
        return cache.setdefault(w, w)
    for s in SUFFIXES:
        if w.endswith(s) and len(w) > len(s) + 2:
            c = w[: -len(s)]
            if len(c) >= 2 and len(c) >= len(w) * 0.5:
                return cache.setdefault(w, _stem1(c, cache, n - 1))
    return cache.setdefault(w, w)


def tokenize(text: str, *, split_identifiers: bool = False) -> list[str]:
    r"""Lowercase alphabetic tokens of length > 2.

    ``\b[a-zA-Z]+\b`` treats ``_`` as a word character, so an identifier like
    ``get_weather_forecast`` has no word boundary anywhere inside it and yields
    *nothing at all*. ``split_identifiers`` inserts the boundaries first.
    """
    s = str(text or "")
    if split_identifiers:
        s = _CAMEL_RE.sub(" ", s)
        s = _SNAKE_RE.sub(" ", s)
    return [w for w in WORD_RE.findall(s.lower()) if len(w) > 2]


def prepare(text: str, cache: dict, *, split_identifiers: bool = False) -> list[str]:
    """tokenize -> drop stop words -> stem, i.e. EZR's first three stages."""
    return [
        _stem1(w, cache)
        for w in tokenize(text, split_identifiers=split_identifiers)
        if w not in STOP_WORDS
    ]


def top_terms(
    docs: list[list[str]], n: int | None
) -> tuple[list[str], dict[str, float]]:
    """EZR's ``tmTfidf``: rank terms by summed tf*idf, keep the first ``n``.

    ``n=None`` keeps every term -- the ``all`` column of the sparsity table.
    The aggregation sums over documents, so it rewards terms that are both
    frequent and widespread; the two pulls partly cancel, which is why
    boilerplate survives it. Ties are broken alphabetically (EZR leaves them
    to dict order).

    EZR spells the aggregation as a loop over documents per term, which is
    O(V*D) and does not finish at ToolBench's scale. Since ``log(N/df)`` does
    not depend on the document, that sum collapses exactly to
    ``ctf(w) * log(N/df(w))`` -- same values, one pass.
    """
    ctf: Counter = Counter()
    df: Counter = Counter()
    for d in docs:
        c = Counter(d)
        ctf.update(c)
        df.update(c.keys())

    total = len(docs) or 1
    scored = sorted(
        ((w, ctf[w] * log(total / d)) for w, d in df.items()),
        key=lambda x: (-x[1], x[0]),
    )
    if n is not None:
        scored = scored[:n]
    return [w for w, _ in scored], dict(scored)
