#!/bin/sh
# waitst.sh N RUN... : block until every run's log has >= N chunk lines (or done/crash), then print summaries
t0=$(date +%s); while [ $(( $(date +%s) - t0 )) -lt ${MAXW:-570} ]; do
  ok=1
  for r in "${@:2}"; do d=/p/projects/cadence-arcade/mario/runs/$r
    n=$(grep -c " chunk [0-9]* (" $d/log.txt 2>/dev/null); n=${n:-0}
    grep -q "done\. best\|Traceback" $d/log.txt /p/projects/cadence-arcade/mario/runs/$r.out 2>/dev/null && continue
    [ "$n" -lt "$1" ] && ok=0
  done
  [ $ok = 1 ] && break; sleep 30
done
for r in "${@:2}"; do echo "== $r"; grep -h "chunk [0-9]* (\|BEST\|dagger\|done\|Trace\|throttle" /p/projects/cadence-arcade/mario/runs/$r/log.txt /p/projects/cadence-arcade/mario/runs/$r.out 2>/dev/null | sed 's/fit {.1-1.: //' | cut -c7-200 | tail -${TAILN:-14}; done
