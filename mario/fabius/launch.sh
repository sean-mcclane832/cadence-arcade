#!/usr/bin/env bash
# The Fabius run (vault Plans/Fabius Build 2026-11-08): FabiusA/B (recall + 104 prediction slots, mask mix10, fabius/TINY.md) and
# Cato2A/B (control: same eyes, no recall, no slots), worlds 1-6 (24 levels), smart coach, stuck 225, ordered rows only
# (--p-seq 1.0), burn-in K 8 = recall capacity, chunk 500, DAgger every 4, SEL8 every chunk, stop at chunk 200.
# Pre-registered primary: mean of sel_mean_all over chunks 181-200 per run. Gate: fabius/gate.py at chunks 10 and 25.
# Started BelowNormal (throttle-training states apply). One play worker per run.
cd /p/projects/cadence-arcade/mario/coldstart || exit 1
[ -f data/fab/levels.json ] || { echo "data/fab/levels.json missing: write it when the pile rebuild ends"; exit 1; }
PY="..\..\.venv-095\Scripts\python.exe"
LEVELS=$(for w in 1 2 3 4 5 6; do for s in 1 2 3 4; do printf "%s-%s," $w $s; done; done); LEVELS=${LEVELS%,}
common="--levels $LEVELS --levels-file data/fab/levels.json --senses fabius --coach smart --sel8 --p-seq 1.0 --K 8 --chunk 500 --dagger-every 4 --episodes 2 --workers 1 --stop-chunk 200 --stop-at= --hours 2000"
RESUME=${RESUME:-}   # RESUME=--resume ./launch.sh to continue
export MARIO_STUCK=225
for name in FabiusA FabiusB Cato2A Cato2B; do
  case $name in Fabius*) extra="--pred --mask mix10" ;; *) extra="" ;; esac
  mkdir -p runs/$name
  cmd //c "start /belownormal /b $PY ctrain.py --name $name --init ../fabius/brains_k8/${name}_ck0.npz --out runs/$name $common $extra $RESUME > runs\\$name.out 2>&1"
  echo "started $name"
done
