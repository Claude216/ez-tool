#!/bin/bash
#SBATCH --job-name=ez-tool-tm-parity
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/tmparity_%j.out
#SBATCH --error=logs/tmparity_%j.err
#
# Runs the experiment twice: once through vendored ezr itself (python 3.12,
# pure python, slow) and once through the port (3.11 + numpy), on the same
# configuration, then compares the recall trajectories.

set -euo pipefail

PROJECT_DIR=/home/lli66/ez-tool
VENV=/mnt/beegfs/lli66/venvs/ez-tool

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

export EZ_TOOL_DATA=/mnt/beegfs/lli66/ez-tool
export TMPDIR=/home/lli66/tmp
# uv lives on beegfs and is not on PATH in a batch shell.
export PATH=/mnt/beegfs/lli66/bin:$PATH
export UV_CACHE_DIR=/mnt/beegfs/lli66/cache/uv
export UV_PYTHON_INSTALL_DIR=/mnt/beegfs/lli66/venvs/uv-python
export PYTHONHASHSEED=0
mkdir -p "$TMPDIR"

echo "node : $(hostname)"

PY312="$(uv python find 3.12)"
echo "--- vendored ezr, native (python 3.12) ---"
"$PY312" scripts/textmine_ezr_native.py \
  --corpora Kitchenham Hall Wahono Radjenovic \
  --budget 290 --valid 20 --tag full_b290

echo "--- port, parity config ---"
"$VENV/bin/python" -m ez_tool.textmine.run --config configs/textmine_ezr_parity.yaml

echo "--- comparison ---"
"$VENV/bin/python" -m ez_tool.textmine.compare
