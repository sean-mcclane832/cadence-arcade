#!/usr/bin/env bash
# Follow a training run live: play its newest checkpoint through the levels with live.py (game + brain windows),
# then go again with the newest checkpoint (the same one if training has not saved a newer one), until the run's last checkpoint has been shown.
#   ./follow-run.sh CiceroLowB [last_chunk=200] [extra live.py options]
cd /p/projects/cadence-arcade/mario/coldstart || exit 1
NAME=${1:-CiceroLowB}; LAST=${2:-200}; shift 2 2>/dev/null
PY=/p/projects/cadence-arcade/.venv-concepts/Scripts/python.exe
shown=""
while true; do
  ck=$(ls runs/$NAME/ | grep "^${NAME}_ck[0-9]*\.npz$" | sort | tail -1)
  [ -z "$ck" ] && { echo "no checkpoints for $NAME"; exit 1; }
  # no newer checkpoint yet: replay the newest one, so the windows stay up between chunks
  shown=$ck; n=$((10#$(echo "$ck" | sed 's/.*_ck\([0-9]*\)\.npz/\1/')))
  echo "$(date +%T) showing $ck"
  "$PY" live.py --brain "runs/$NAME/$ck" --name "$NAME chunk $n (training)" --tries 1 --out "runs/$NAME/live_ck$(printf %04d $n).json" "$@" || exit 0
  [ "$n" -ge "$LAST" ] && [ "$ck" = "$prev" ] && { echo "last checkpoint shown twice"; exit 0; }
  prev=$ck
done
