#!/bin/bash
# Relaunch the three unfinished plateau runs after the 2026-11-06 shutdown, from their last checkpoints, at Idle priority.
export MARIO_STUCK=40  # the plateau test was pre-registered at the old 2.7 s stuck limit; keep it for these runs (2026-11-06)
cd /p/projects/cadence-arcade/mario/coldstart || exit 1
PY=/p/projects/cadence-arcade/.venv-concepts/Scripts/python.exe
for n in LivyB CiceroLowA CiceroLowB; do
  nohup $PY ctrain.py --name $n --init brains0/${n}_ck0.npz --out runs/$n --sel8 --stop-chunk 200 --stop-at "" --hours 400 --dagger-every 6 --episodes 1 --workers 1 --coach base --resume >> runs/$n/relaunch.log 2>&1 &
done
sleep 5; powershell -NoProfile -ExecutionPolicy Bypass -File /p/agent/tools/throttle-training.ps1 idle
