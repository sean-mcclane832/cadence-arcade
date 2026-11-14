#!/bin/bash
# stuckmap.py on .venv-099: GPU hidden, threads pinned to 1, stuck limit 225, Idle priority.  stuckmap.sh play [run ...]
export CUDA_VISIBLE_DEVICES=-1 CADENCE_BACKEND=cpu NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MARIO_STUCK=225
cd "$(dirname "$0")"
PY="P:\projects\cadence-arcade-scipio\.venv-099\Scripts\python.exe"
exec cmd //c "start /low /b /wait $PY stuckmap.py $*"
