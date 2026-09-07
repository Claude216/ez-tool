#!/bin/bash
#SBATCH --job-name=ez-fig4
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/lli66/ez-tool/logs/fig4_%A_%a.out
#SBATCH --error=/home/lli66/ez-tool/logs/fig4_%A_%a.err
#
# Figure 4 reproduction: recall + false-alarm curves from the patched ezr.
# One array task per (corpus, klass column, norm).  Submit from a login node:
#   sbatch scripts/slurm/run_fig4.sbatch
set -euo pipefail

export TMPDIR=/home/lli66/tmp
mkdir -p "$TMPDIR" /home/lli66/ez-tool/logs

# _tm_best breaks score ties by dict order, which is hash-seed dependent.
export PYTHONHASHSEED=0

PY=/mnt/beegfs/lli66/venvs/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12

echo "node : $(hostname)"
echo "cell : $SLURM_ARRAY_TASK_ID"
exec "$PY" /home/lli66/tmp/fig4/curves.py "$SLURM_ARRAY_TASK_ID"
