"""Canonical filesystem locations.

Bulk data lives on beegfs, which is mounted on compute nodes only. The repo
under $HOME holds code and small result files; $HOME has a 40 GB quota shared
with everything else the user owns, so nothing large may land there.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Overridable so the pipeline can be pointed at a scratch copy without edits.
BEEGFS_ROOT = Path(os.environ.get("EZ_TOOL_DATA", "/mnt/beegfs/lli66/ez-tool"))

DATA_RAW = BEEGFS_ROOT / "data" / "raw"
DATA_PROCESSED = BEEGFS_ROOT / "data" / "processed"
INDEX_DIR = BEEGFS_ROOT / "indexes"
RUNS_DIR = BEEGFS_ROOT / "runs"

# Small enough to be worth version-controlling alongside the code.
RESULTS_DIR = REPO_ROOT / "results"

# Raw benchmark inputs, as downloaded by scripts/fetch_data.sh.
TOOLENV_DIR = DATA_RAW / "toolenv2404_filtered"
QUERIES_DIR = DATA_RAW / "StableToolBench" / "solvable_queries" / "test_instruction"

# The six official StableToolBench evaluation groups.
GROUPS = (
    "G1_instruction",
    "G1_tool",
    "G1_category",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, INDEX_DIR, RUNS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def require_beegfs() -> None:
    """Fail loudly on a login node rather than halfway through a run."""
    if not BEEGFS_ROOT.parent.exists():
        raise RuntimeError(
            f"{BEEGFS_ROOT.parent} is not visible. beegfs is mounted on compute "
            "nodes only -- run this on a compute node (see CLAUDE.md)."
        )
