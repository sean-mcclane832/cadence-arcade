#!/bin/bash
# Run a python script on the Scipio venv (.venv-099, Cadence 0.99.0): GPU hidden, CPU backend, threads pinned to 1,
# BelowNormal priority, stuck limit 225. Runs in the caller's directory.
#   py099.sh script.py args...
export CUDA_VISIBLE_DEVICES=-1 CADENCE_BACKEND=cpu NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MARIO_STUCK=225
PY="P:\projects\cadence-arcade-scipio\.venv-099\Scripts\python.exe"
exec cmd //c "start /belownormal /b /wait $PY $*"
