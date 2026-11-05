#!/bin/bash
# Pilot queue (COACHFREE.md): at most 2 processes at a time, Idle priority (set inside b1.py).
cd "$(dirname "$0")"
PY=../../.venv-concepts/Scripts/python.exe
while [ ! -f runs/firstsight/firstsight.md ]; do sleep 20; done; sleep 10
for L in 6-1 6-2; do $PY b1.py calib --level $L --ckpt CatoA --n 4 > runs/calib_$L.out 2>&1; done
for L in 6-1 6-2; do
  $PY b1.py run --arm b1 --level $L --out runs/pilot_$L/b1 > runs/pilot_$L.b1.out 2>&1 &
  $PY b1.py run --arm anch --level $L --out runs/pilot_$L/anch > runs/pilot_$L.anch.out 2>&1 &
  wait
done
$PY b1.py run --arm reload --level 6-1 --out runs/pilot_6-1/reload --forget 0 > runs/pilot_6-1.reload.out 2>&1 &
$PY b1.py run --arm reload --level 6-2 --out runs/pilot_6-2/reload --forget 0 > runs/pilot_6-2.reload.out 2>&1 &
wait
echo PILOT_DONE
