#!/usr/bin/env bash
# First start of Scipio runs (vault Plans/Scipio Design 2026-11-12). One name per call is the launch rule: start the next
# only when the previous has finished its first DAgger round (chunk 4, the RAM peak) AND >= 6 GB RAM is free.
#   NAMES=ScipioA ./launch.sh            WORKERS=1 NAMES=ScipioFast ./launch.sh
# BelowNormal priority (training one step below the scheduled jobs). Resumes are the training guard's job (--resume).
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../coldstart" || exit 1
. ../scipio/flags.sh
NAMES=${NAMES:?set NAMES, e.g. NAMES=ScipioA}
WORKERS=${WORKERS:-2}
for name in $NAMES; do
  case $name in ScipioA|ScipioB|ScipioFast|Twin|MarcellusA|MarcellusB|TwinB|CamillusA|CamillusB|MarcellusC|TwinC|AgrippaA|AgrippaB) ;; *) echo "unknown run $name"; exit 1 ;; esac
  if powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? { \$_.CommandLine -match 'ctrain.py --name $name ' }) { exit 1 }"; then :; else
    echo "skip $name: already running"; continue
  fi
  if [ -s runs/$name/history.jsonl ]; then echo "skip $name: runs/$name already has chunks (resume through the guard)"; continue; fi
  mkdir -p runs/$name
  cmd //c "start /belownormal /b $SCIPIO_PY ctrain.py --name $name --init ../scipio/brains0/${name}_ck0.npz --out runs/$name $(common_for $name) --workers $WORKERS --stop-chunk $SCIPIO_TARGET >> runs\\$name.out 2>&1"
  echo "started $name (workers $WORKERS, stop chunk $SCIPIO_TARGET) $(date '+%F %T')"
done
