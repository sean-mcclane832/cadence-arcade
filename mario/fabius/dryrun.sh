#!/usr/bin/env bash
# Trainer dry run before the Fabius launch (deep review 11/8, fix 1): each arm 1 chunk, then --resume to chunk 2,
# on the smoke piles of 1-1, 4-1, 6-4, DAgger every chunk. Logs in coldstart/runs/dry_<arm>/.
cd /p/projects/cadence-arcade/mario/coldstart || exit 1
PY=../../.venv-095/Scripts/python.exe
common="--levels 1-1,4-1,6-4 --levels-file data/fab_smoke/levels.json --senses fabius --coach smart --sel8 --p-seq 1.0 --K 8 --chunk 20 --dagger-every 1 --episodes 1 --workers 2 --stop-at= --hours 400"
run() {  # name init extra
  "$PY" ctrain.py --name dry_$1 --init ../fabius/brains_k8/$1_ck0.npz --out runs/dry_$1 $common $2 --stop-chunk 1 > runs/dry_$1.out 2>&1 &&
  "$PY" ctrain.py --name dry_$1 --init ../fabius/brains_k8/$1_ck0.npz --out runs/dry_$1 $common $2 --stop-chunk 2 --resume >> runs/dry_$1.out 2>&1
  echo "$1 exit $?"
}
run FabiusA "--pred --mask mix10" &
run Cato2A "" &
wait
