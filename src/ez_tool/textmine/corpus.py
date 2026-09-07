"""The four SLR corpora from Yu et al., as used in section 7 of the EZR paper.

Each row is one candidate paper carrying a human relevance judgement. Row and
positive counts produced here reproduce Table 6 of the EZR paper exactly.

Three details are worth stating, because each silently changes the numbers:

* ``Kitchenham.csv`` carries *two* label columns. ``abs`` (132 yes) records
  what passed title-and-abstract screening; ``label`` (45 yes) records what
  survived full content review, and the 45 are a subset of the 132. Table 6
  reports 132, and section 7.4 speaks of "108 of the 132 relevant papers", so
  the paper screened on ``abs`` -- while vendored ezr defaults to ``label``.
  Despite the name, ``abs`` holds a label, not an abstract.
* Column order is not shared across files: Radjenovic puts ``year`` before
  ``abstract``, Kitchenham adds ``authors``/``source``. Everything below is
  addressed by header name.
* ``document_title`` is a separate column that ezr's ``tmTokenize`` never
  reads -- it tokenizes ``abstract`` alone. Section 7.4 says each row "was a
  paper's title and abstract", so including the title is a switch here and an
  ablation in the config, not a fixed choice.

CSV parsing mirrors ezr's hand-rolled ``_tm_cells``/``tmCsv`` rather than the
stdlib ``csv`` module, so the text feeding the model is the same text the
vendored reference implementation sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import DATA_RAW

SLR_DIR = DATA_RAW / "slr"

TEXT_COL = "abstract"
TITLE_COL = "document_title"


@dataclass(frozen=True)
class Spec:
    """Expected shape of one corpus, asserted at load time."""

    klass: str  # header name of the relevance column
    n_rows: int
    n_pos: int


# n_pos values are Table 6 of the EZR paper. Kitchenham's 132 comes from the
# `abs` column; its `label` column would give 45, which is Yu's Table 1 figure.
DATASETS: dict[str, Spec] = {
    "Hall": Spec(klass="label", n_rows=8911, n_pos=104),
    "Wahono": Spec(klass="label", n_rows=7002, n_pos=62),
    "Radjenovic": Spec(klass="label", n_rows=6000, n_pos=48),
    "Kitchenham": Spec(klass="abs", n_rows=1704, n_pos=132),
}


@dataclass(frozen=True)
class Corpus:
    name: str
    texts: list[str]
    y: np.ndarray  # bool, True where the paper is relevant

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())


def _cells(s: str) -> list[str]:
    """Split one CSV line on commas, respecting quotes. Ports ezr `_tm_cells`.

    Deliberately line-at-a-time, exactly as ezr does. That would mangle a
    quoted field containing a newline; `load` asserts the row count instead of
    assuming none exists.
    """
    r: list[str] = []
    c: list[str] = []
    q = 0
    for ch in s:
        if ch == '"' and (not c or q):
            q ^= 1
        elif q < 1 and ch == ",":
            r.append("".join(c))
            c = []
        else:
            c.append(ch)
    r.append("".join(c))
    return r


def _thing(txt: str) -> str | int | float:
    """Coerce to number where possible. Ports ezr `thing`.

    Kept because ezr applies it to every cell before tokenizing, and it is not
    a no-op on text: a title of "007" becomes 7, hence "7" once stringified.
    """
    txt = txt.strip()
    for f in (int, float):
        try:
            return f(txt)
        except ValueError:
            pass
    return {"true": 1, "false": 0}.get(txt.lower(), txt)


def read_rows(path: Path) -> tuple[list[str], list[list[str | int | float]]]:
    """Header plus body, using ezr's reader semantics."""
    header: list[str] | None = None
    body: list[list[str | int | float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            raw = _cells(line)
            if not any(x.strip() for x in raw):
                continue
            row = [_thing(x) for x in raw]
            if header is None:
                header = [str(x) for x in row]
            else:
                body.append(row)
    if header is None:
        raise ValueError(f"{path} is empty")
    return header, body


def load(
    name: str,
    include_title: bool = True,
    root: Path | None = None,
    klass: str | None = None,
) -> Corpus:
    """Load one corpus, asserting it matches the shape Table 6 reports.

    `klass` overrides the relevance column. Only validation uses it, to read
    Kitchenham through ezr's default `label` (45 positive) instead of the
    `abs` column (132) the paper reports; the Table 6 check is then skipped.
    """
    spec = DATASETS[name]
    if klass is not None:
        spec = Spec(klass=klass, n_rows=spec.n_rows, n_pos=-1)
    path = (root or SLR_DIR) / f"{name}.csv"
    header, body = read_rows(path)

    for col in (TEXT_COL, spec.klass):
        if col not in header:
            raise ValueError(f"{path} has no '{col}' column; header={header}")
    at_text, at_klass = header.index(TEXT_COL), header.index(spec.klass)
    at_title = header.index(TITLE_COL) if include_title else None

    texts, labels = [], []
    for row in body:
        body_text = str(row[at_text])
        if at_title is not None:
            # Section 7.4: "each row was a paper's title and abstract".
            body_text = f"{row[at_title]} {body_text}"
        texts.append(body_text)
        labels.append(str(row[at_klass]).strip().lower() == "yes")

    y = np.asarray(labels, dtype=bool)
    if spec.n_pos < 0:
        if len(texts) != spec.n_rows:
            raise ValueError(f"{name}: got {len(texts)} rows, expected {spec.n_rows}")
        return Corpus(name=name, texts=texts, y=y)
    if len(texts) != spec.n_rows or int(y.sum()) != spec.n_pos:
        raise ValueError(
            f"{name}: got {len(texts)} rows / {int(y.sum())} positive, "
            f"expected {spec.n_rows} / {spec.n_pos} (Table 6). "
            "A mismatch means the file or the label column changed."
        )
    return Corpus(name=name, texts=texts, y=y)
