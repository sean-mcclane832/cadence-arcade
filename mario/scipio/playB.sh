#!/bin/bash
# Leg 2c run B plays, one at a time (2 workers each). Resumable: playstudy skips finished episodes.
cd "P:/projects/cadence-arcade-scipio/mario/scipio"
for t in all11dag_0 all11dag_1 all11dag_2 all11dag_3 all11_0 all11_1; do
  [ -f scratch/play/mlp_${t}.json ] && continue
  until [ -f scratch/tiny/mlp_${t}_E20.json ]; do sleep 15; done
  MARIO_STUCK=225 ./idle.sh 098 playstudy.py mlp scratch/tiny/mlp_${t}_E20.pt $t --mode none > scratch/playB_${t}.out 2>&1
  echo "$t $(grep -E '^\{' scratch/playB_${t}.out | head -1 | cut -c1-80) $(grep -E '^seconds|Traceback' scratch/playB_${t}.out)"
done
echo ALLDONE
