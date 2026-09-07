#!/bin/bash
#SBATCH --job-name=ez-tool-bm25
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/bm25_%j.out
#SBATCH --error=logs/bm25_%j.err
#
# BM25 is CPU-only -- no GPU partition needed. Submit from a login node:
#   sbatch scripts/slurm/run_bm25.sbatch
#   sbatch --export=ALL,CONFIG=configs/my_ablation.yaml scripts/slurm/run_bm25.sbatch

set -euo pipefail

CONFIG="${CONFIG:-configs/bm25_baseline.yaml}"
PROJECT_DIR=/home/lli66/ez-tool
VENV=/mnt/beegfs/lli66/venvs/ez-tool

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

export HF_HOME=/mnt/beegfs/lli66/cache/huggingface
export EZ_TOOL_DATA=/mnt/beegfs/lli66/ez-tool

# Never let temp files land in the shared /tmp on a compute node.
export TMPDIR=/home/lli66/tmp
mkdir -p "$TMPDIR"

echo "node   : $(hostname)"
echo "config : $CONFIG"

"$VENV/bin/python" -m ez_tool.cli run --config "$CONFIG"
