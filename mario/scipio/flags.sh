# Scipio's training recipe (vault Plans/Scipio Design 2026-11-12, registered 11/14). Sourced by scipio/launch.sh and by
# P:/agent/tools/training-guard.sh, so a first start and every resume use the same flags (ctrain --resume checks them).
# Run from mario/coldstart of the worktree P:/projects/cadence-arcade-scipio (branch scipio).
SCIPIO_PY="..\..\.venv-099\Scripts\python.exe"        # Cadence 0.99.0 (upstream 352587ac), worktree venv
SCIPIO_LEVELS=$(for w in 1 2 3 4 5 6; do for s in 1 2 3 4; do printf "%s-%s," $w $s; done; done); SCIPIO_LEVELS=${SCIPIO_LEVELS%,}
SCIPIO_COMMON="--levels $SCIPIO_LEVELS --levels-file ../scipio/levels-fab.json --senses fabius --coach smart --sel8 --p-seq 1.0 --K 0 --chunk 500 --dagger-every 4 --episodes 1 --dagger-stuck 100 --last-all 100 --sel-every 2 --horizon 250 --stop-at= --hours 2000"
SCIPIO_TARGET=100                                        # 2026-11-19: Camillus, registered read at chunk 100 (Marcellus ended at 100 11/19)
export CUDA_VISIBLE_DEVICES=-1 CADENCE_BACKEND=cpu MARIO_STUCK=225 CADENCE_COMMIT=352587ac
# Marcellus (vault Plans/Marcellus Design 2026-11-16): Scipio's recipe + the gated DAgger tie rule. Registered read at chunk 50.
MARCELLUS_COMMON="$SCIPIO_COMMON --tie-stuck 10"
# Camillus (vault Plans/Camillus Design 2026-11-19): Scipio's memory brain on the same tie-gate recipe.
common_for() { case $1 in Marcellus*|Camillus*) echo "$MARCELLUS_COMMON" ;; *) echo "$SCIPIO_COMMON" ;; esac; }
