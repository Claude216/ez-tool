#!/bin/bash
#SBATCH --job-name=ez-tool-textmine
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/textmine_%j.out
#SBATCH --error=logs/textmine_%j.err
#
# Section 7 replication. CPU-only; the corpora total 27 MB. Submit from a
# login node:
#   sbatch scripts/slurm/run_textmine.sbatch
#   sbatch --export=ALL,CONFIG=configs/my_arm.yaml scripts/slurm/run_textmine.sbatch
#
# The vendored-ezr reference dump that validate.py checks against is produced
# separately by scripts/textmine_reference.sh -- it needs Python 3.12.

set -euo pipefail

CONFIG="${CONFIG:-configs/textmine_paper.yaml}"
PROJECT_DIR=/home/lli66/ez-tool
VENV=/mnt/beegfs/lli66/venvs/ez-tool

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

export EZ_TOOL_DATA=/mnt/beegfs/lli66/ez-tool

# Never let temp files land in the shared /tmp on a compute node.
export TMPDIR=/home/lli66/tmp
mkdir -p "$TMPDIR"

echo "node   : $(hostname)"
echo "config : $CONFIG"

"$VENV/bin/python" -m ez_tool.textmine.validate
"$VENV/bin/python" -m ez_tool.textmine.run --config "$CONFIG"
RUN_NAME="$("$VENV/bin/python" -c \
  "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CONFIG")"
"$VENV/bin/python" -m ez_tool.textmine.plot --run-name "$RUN_NAME"
