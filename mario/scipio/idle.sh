#!/bin/bash
# Run a python script the way every leg-1 process must run: GPU hidden, threads pinned to 1, Idle priority.
#   idle.sh [098|095] script.py args...
# 095 = .venv-095's interpreter, read-only: numba's cache and .pyc files are sent away from the venv, nothing is installed.
V=${1:-098}; shift
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
export CUDA_VISIBLE_DEVICES=-1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$HERE/scratch/numba-$V" CADENCE_BACKEND=cpu
mkdir -p "$HERE/scratch/numba-$V"
PY="P:\projects\cadence-arcade\.venv-$V\Scripts\python.exe"
exec cmd //c "start /low /b /wait $PY $*"
