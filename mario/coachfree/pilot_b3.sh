#!/bin/bash
# B3 arms after the B1 pilot queue (2 processes max).
cd "$(dirname "$0")"
PY=../../.venv-concepts/Scripts/python.exe
while ! grep -q PILOT_DONE runs/pilot.out 2>/dev/null; do sleep 30; done
$PY b3.py --level 6-1 --out runs/pilot_6-1/b3 > runs/pilot_6-1.b3.out 2>&1 &
$PY b3.py --level 6-2 --out runs/pilot_6-2/b3 > runs/pilot_6-2.b3.out 2>&1 &
wait
echo B3_DONE
