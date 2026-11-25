#!/bin/bash
# Endless gauntlet for RomulusB (11/25): unlimited lives, 30 tries per level, skip capped levels, stop after 7-4.
cd "$(dirname "$0")/../scipio" && source ./flags.sh && cd ../coldstart || exit 1
P=../../.venv-101/Scripts/python.exe
for ck in "$@"; do
  $P -c "import ctypes,sys,runpy;ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000);sys.argv=sys.argv[1:];runpy.run_path('cgauntlet.py',run_name='__main__')" \
    cgauntlet.py --name RomulusB_ck$ck --brain runs/RomulusB/RomulusB_ck$ck.npz --senses fabius --lives 0 --retries many \
    --max-tries 30 --on-cap skip --clip clears --first 1-1 --last 7-4 --mode greedy \
    --label "RomulusB keeps trying" --out runs/RomulusB/endless/RomulusB_ck$ck 2>&1 | grep -v -i "gym\|upgrade\|migration"
done
