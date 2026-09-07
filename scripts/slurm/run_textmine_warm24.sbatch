#!/bin/bash
#SBATCH --job-name=ez-tool-warm24
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/warm24_%j.out
#SBATCH --error=logs/warm24_%j.err
set -euo pipefail
cd /home/lli66/ez-tool
export TMPDIR=/home/lli66/tmp
export PATH=/mnt/beegfs/lli66/bin:$PATH
export UV_CACHE_DIR=/mnt/beegfs/lli66/cache/uv
export UV_PYTHON_INSTALL_DIR=/mnt/beegfs/lli66/venvs/uv-python
export PYTHONHASHSEED=0
mkdir -p "$TMPDIR" logs
echo "node: $(hostname)"
exec "$(uv python find 3.12)" scripts/textmine_ezr_warm24.py --budget 290 --valid 20
