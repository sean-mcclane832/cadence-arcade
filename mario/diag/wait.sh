#!/bin/sh
# wait.sh RUN_DIR N : block until N solo evals are logged (or the run crashed/finished)
d=/p/projects/cadence-arcade/mario/runs/$1
while :; do
  n=$(grep -c "SOLO" "$d/log.txt" 2>/dev/null); n=${n:-0}
  [ "$n" -ge "$2" ] && break
  grep -q "Traceback" "$d.out" 2>/dev/null && { echo CRASH; tail -5 "$d.out"; break; }
  grep -q " done$\|stopping" "$d/log.txt" 2>/dev/null && break
  sleep 20
done
grep "SOLO\|taught" "$d/log.txt" | cut -c1-200 | tail -${3:-4}
