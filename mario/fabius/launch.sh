#!/usr/bin/env bash
# The Fabius run (vault Plans/Fabius Build 2026-11-08): FabiusA/B (recall + 104 prediction slots, mask mix10, fabius/TINY.md) and
# Cato2A/B (control: same eyes, no recall, no slots), worlds 1-6 (24 levels), smart coach, stuck 225, ordered rows only
# (--p-seq 1.0), burn-in K 8 = recall capacity, chunk 500, DAgger every 4 (1 episode per level, DAgger episodes end
# after 40 stuck decisions; SEL keeps 225), SEL8 every chunk, stop at chunk 100 (Sean 11/8: not a multi-week run).
# Pre-registered primary: mean of sel_mean_all over chunks 81-100 per run. Gate: fabius/gate.py at chunks 10 and 25.
# Started BelowNormal (throttle-training states apply). Two play workers per run: the four runs take ~21 GB
# (11/8 20:07: 0.8 GB of 32 free with other apps open; Sean freed memory and asked for 2 workers again, 20:15).
cd /p/projects/cadence-arcade/mario/coldstart || exit 1
[ -f data/fab/levels.json ] || { echo "data/fab/levels.json missing: write it when the pile rebuild ends"; exit 1; }
PY="..\..\.venv-095\Scripts\python.exe"
LEVELS=$(for w in 1 2 3 4 5 6; do for s in 1 2 3 4; do printf "%s-%s," $w $s; done; done); LEVELS=${LEVELS%,}
common="--levels $LEVELS --levels-file data/fab/levels.json --senses fabius --coach smart --sel8 --p-seq 1.0 --K 8 --chunk 500 --dagger-every 4 --episodes 1 --dagger-stuck 40 --workers 2 --stop-chunk 100 --stop-at= --hours 2000"
RESUME=${RESUME:-}   # RESUME=--resume ./launch.sh to continue
NAMES=${NAMES:-FabiusA FabiusB Cato2A Cato2B}   # NAMES="FabiusA FabiusB" to start only some arms
export MARIO_STUCK=225
for name in $NAMES; do
  if powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? { \$_.CommandLine -match 'ctrain.py --name $name ' }) { exit 1 }"; then :; else
    echo "skip $name: already running"; continue
  fi
  # --recycle 100 (11/9): Fabius plays on the GPU and leaks VRAM per checkpoint; a fresh worker about once a chunk
  case $name in Fabius*) extra="--pred --mask mix10 --recycle 100" ;; *) extra="" ;; esac
  mkdir -p runs/$name
  cmd //c "start /belownormal /b $PY ctrain.py --name $name --init ../fabius/brains_k8/${name}_ck0.npz --out runs/$name $common $extra $RESUME >> runs\\$name.out 2>&1"
  echo "started $name"
done
