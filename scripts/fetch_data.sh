#!/bin/bash
# Download the StableToolBench corpus and solvable queries onto beegfs.
#
# COMPUTE NODE ONLY -- beegfs is not mounted on login nodes.
set -euo pipefail

# Never let temp files land in the shared /tmp.
export TMPDIR="${TMPDIR:-/home/lli66/tmp}"
mkdir -p "$TMPDIR"

RAW="${EZ_TOOL_DATA:-/mnt/beegfs/lli66/ez-tool}/data/raw"
mkdir -p "$RAW"
cd "$RAW"

# The tool environment: 12,303 tools / 47,591 APIs across 49 categories.
if [ ! -d toolenv2404_filtered ]; then
  echo ">> fetching toolenv2404_filtered (12 MB)"
  curl -fsSL -o toolenv2404_filtered.tar.gz \
    "https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz"
  tar xzf toolenv2404_filtered.tar.gz
fi

# The 765 solvable queries with their `relevant APIs` annotations.
if [ ! -d StableToolBench ]; then
  echo ">> cloning StableToolBench (solvable_queries)"
  git clone --depth 1 https://github.com/THUNLP-MT/StableToolBench.git
fi

echo ">> done"
du -sh "$RAW"/* 2>/dev/null || true
