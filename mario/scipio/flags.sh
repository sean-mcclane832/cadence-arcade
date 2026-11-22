# Scipio's training recipe (vault Plans/Scipio Design 2026-11-12, registered 11/14). Sourced by scipio/launch.sh and by
# P:/agent/tools/training-guard.sh, so a first start and every resume use the same flags (ctrain --resume checks them).
# Run from mario/coldstart of the worktree P:/projects/cadence-arcade-scipio (branch scipio).
SCIPIO_PY="..\..\.venv-101\Scripts\python.exe"        # Romulus worktree (P:/projects/cadence-arcade-romulus): Cadence 0.101.0 (v0.101.0, 6a5f8081)
SCIPIO_LEVELS=$(for w in 1 2 3 4 5 6; do for s in 1 2 3 4; do printf "%s-%s," $w $s; done; done); SCIPIO_LEVELS=${SCIPIO_LEVELS%,}
SCIPIO_COMMON="--levels $SCIPIO_LEVELS --levels-file ../scipio/levels-fab.json --senses fabius --coach smart --sel8 --p-seq 1.0 --K 0 --chunk 500 --dagger-every 4 --episodes 1 --dagger-stuck 100 --last-all 100 --sel-every 2 --horizon 250 --stop-at= --hours 2000"
SCIPIO_TARGET=8                                          # Romulus trial (chunks 0-8, read by a session before 11/30); the launch target comes with the registered design
export CUDA_VISIBLE_DEVICES=-1 CADENCE_BACKEND=cpu MARIO_STUCK=225 CADENCE_COMMIT=6a5f8081
# Marcellus (vault Plans/Marcellus Design 2026-11-16): Scipio's recipe + the gated DAgger tie rule. Registered read at chunk 50.
MARCELLUS_COMMON="$SCIPIO_COMMON --tie-stuck 10"
# Camillus (vault Plans/Camillus Design 2026-11-19): Scipio's memory brain on the same tie-gate recipe.
# Agrippa (vault Plans/World 2 Plan 2026-11-19): Marcellus's recipe + one level added every 5 chunks (1-1 alone first).
AGRIPPA_COMMON="$MARCELLUS_COMMON --add-every 5"
# Romulus (vault Plans/Romulus Design 2026-11-21): Marcellus's recipe + wall starts (2 per level per round, 45 / 90 decisions
# before the last round's furthest failed attempt) + own rows sampled 4x the coach's. SCIPIO_TARGET 8 = the trial (chunks 0-8).
ROMULUS_COMMON="$MARCELLUS_COMMON --wall-back 45,90 --own-weight 4"
common_for() { case $1 in Romulus*) echo "$ROMULUS_COMMON" ;; Marcellus*|Camillus*) echo "$MARCELLUS_COMMON" ;; Agrippa*) echo "$AGRIPPA_COMMON" ;; *) echo "$SCIPIO_COMMON" ;; esac; }
