#!/bin/bash
# Regenerate the vendored-ezr reference dump. Compute node only.
#
# ezr.py is kept byte-identical to commit 500d1d4, which means PEP 695 syntax
# and therefore Python 3.12 -- the project venv is 3.11. uv supplies the 3.12
# interpreter; ezr itself is dependency-free.
set -euo pipefail

export TMPDIR=/home/lli66/tmp
# uv lives on beegfs and is not on PATH in a batch shell.
export PATH=/mnt/beegfs/lli66/bin:$PATH
export UV_CACHE_DIR=/mnt/beegfs/lli66/cache/uv
export UV_PYTHON_INSTALL_DIR=/mnt/beegfs/lli66/venvs/uv-python
mkdir -p "$TMPDIR"

# _tm_best resolves score ties by dict order, which is set-iteration order and
# therefore hash-seed dependent. Pin it so the reference is reproducible.
export PYTHONHASHSEED=0

PY312="$(uv python find 3.12)"
echo "python : $PY312"
exec "$PY312" /home/lli66/ez-tool/scripts/textmine_reference.py
