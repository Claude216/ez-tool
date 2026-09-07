"""Build the retrieval corpus from the StableToolBench tool environment.

The environment nests API endpoints inside tool files:

    toolenv2404_filtered/<Category>/<tool>.json
        tool_name, tool_description, api_list[{name, description, ...}]

A retrieval unit is one *API* -- the thing an agent actually calls -- keyed by
the ``(tool_name, api_name)`` pair, because that is exactly how the query
labels reference it and because ``api_name`` alone is not unique: 29.2% of
APIs share a name with an API under a different tool ('Search' occurs in 216
tools).

What text a document carries is a config choice, because disclosure
granularity and index granularity are not the same thing. Anthropic's tool
search indexes "tool names, descriptions, argument names, and argument
descriptions" -- the parameter schema is hidden from the model but visible to
the retriever. ``include_parameters`` selects between mirroring that (index
the schema) and the stricter setting where only what the model is shown up
front is indexed.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import orjson

from ..paths import TOOLENV_DIR
from .text import normalize


def doc_id(tool_name: str, api_name: str) -> str:
    """Stable identifier for one callable. Mirrors the label format."""
    return f"{tool_name}\t{api_name}"


@dataclasses.dataclass(slots=True)
class ApiDoc:
    doc_id: str
    category: str
    tool_name: str
    tool_description: str
    api_name: str
    api_description: str
    parameter_text: str
    toolgen_text: str

    def text(
        self,
        *,
        backfill_description: bool,
        include_parameters: bool = False,
        toolgen_style: bool = False,
    ) -> str:
        """Indexed text for this unit.

        ``backfill_description`` substitutes the parent tool's description when
        the API's own description is blank (9.5% of the corpus). Parent
        descriptions are frequently uninformative -- 'API for TheClique
        company' -- so this is an ablation switch, not a default assumption.

        ``include_parameters`` appends argument names and their descriptions,
        matching what Anthropic's tool search actually indexes. Parameter
        *defaults* are never included: they hold payloads such as inline
        base64 images that would swamp the term statistics.

        ``toolgen_style`` reproduces ToolGen's document format verbatim
        (``evaluation/retrieval/eval_bm25.py``), which dumps the parameter and
        response schemas as raw JSON -- defaults, type names and JSON
        punctuation included. It overrides the other switches, and exists to
        separate "different corpus text" from "different query set" when
        comparing against their published BM25 row.
        """
        if toolgen_style:
            return normalize(self.toolgen_text)
        desc = self.api_description.strip()
        if not desc and backfill_description:
            desc = self.tool_description.strip()
        parts = [self.tool_name, self.api_name, desc]
        if include_parameters and self.parameter_text:
            parts.append(self.parameter_text)
        return normalize(" ".join(parts))


def iter_tool_files(root: Path | None = None) -> Iterator[Path]:
    root = root or TOOLENV_DIR
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run scripts/fetch_data.sh on a compute node."
        )
    yield from sorted(root.glob("*/*.json"))


def load_corpus(root: Path | None = None) -> tuple[list[ApiDoc], dict]:
    """Return deduplicated API documents plus a stats dict for the run record."""
    docs: list[ApiDoc] = []
    seen: set[str] = set()
    stats = Counter()

    for path in iter_tool_files(root):
        stats["tool_files"] += 1
        try:
            tool = orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError:
            stats["tool_files_unparseable"] += 1
            continue

        tool_name = (tool.get("tool_name") or "").strip()
        tool_desc = (tool.get("tool_description") or "").strip()
        category = path.parent.name

        for api in tool.get("api_list") or []:
            stats["apis_seen"] += 1
            api_name = (api.get("name") or "").strip()
            if not tool_name or not api_name:
                stats["apis_unnamed"] += 1
                continue

            did = doc_id(tool_name, api_name)
            if did in seen:
                # 533 exact (tool, api) collisions exist in the released env.
                stats["apis_duplicate"] += 1
                continue
            seen.add(did)

            api_desc = (api.get("description") or "").strip()
            if not api_desc:
                stats["apis_empty_description"] += 1

            # Argument names and descriptions only -- never `default`, which
            # carries payloads like inline base64 images.
            params: list[str] = []
            for group in ("required_parameters", "optional_parameters"):
                for p in api.get(group) or []:
                    if not isinstance(p, dict):
                        continue
                    pname = (p.get("name") or "").strip()
                    pdesc = (p.get("description") or "").strip()
                    if pname:
                        params.append(pname)
                        stats["parameters_seen"] += 1
                    if pdesc:
                        params.append(pdesc)
                        stats["parameters_with_description"] += 1
            if params:
                stats["apis_with_parameters"] += 1

            # ToolGen's exact concatenation, from eval_bm25.py.
            toolgen_text = (
                f"{category}, {tool_name}, {api_name}, {api_desc}"
                f", required_params: {json.dumps(api.get('required_parameters', ''))}"
                f", optional_params: {json.dumps(api.get('optional_parameters', ''))}"
                f", return_schema: {json.dumps(api.get('template_response', ''))}"
            )

            docs.append(
                ApiDoc(
                    doc_id=did,
                    category=category,
                    tool_name=tool_name,
                    tool_description=tool_desc,
                    api_name=api_name,
                    api_description=api_desc,
                    parameter_text=" ".join(params),
                    toolgen_text=toolgen_text,
                )
            )

    stats["docs"] = len(docs)
    return docs, dict(stats)
