# Overnight brief: Mario concepts / death clinic / gauntlet (2026-10-31 night)

You are running Sean's approved overnight Cadence Mario run. Work hands-on: read every result, adapt the next step, keep a notebook. Sean is fine with the token cost (don't swap in unattended driver scripts to save tokens).

## Read first
- Plan: P:/vault/Agent/Plans/Mario Concepts Goals Dying 2026-10-31.md (including "Sean's answers and additions").
- P:/projects/cadence-arcade/mario/LEARNING_FIX.md (the RESUME block at the top, E23-E26 at the end). Recipes, brains, SEL/TEST/HELD start sets, steady.py usage are all there.
- Cadence changelog for 0.91.2 / 0.91.3 (recall fixes): P:/projects/cadence (upstream clone, read-only, NEVER commit there). Note anything that should change how recall is configured.

## Environment
- Python: P:/projects/cadence-arcade/.venv-concepts (Cadence 0.91.3, smoke-tested). `.venv-mario` (0.91.1) is the showcase env: don't modify it.
- Work in mario/concepts/ (new scripts, run dirs under mario/concepts/runs/). Reuse mario/*.py by import or copy; don't change behavior of existing files the showcase depends on.
- Notebook: mario/concepts/CONCEPTS.md. Hypothesis + falsifier BEFORE each run, timestamps, results, corrections in place. Same style as LEARNING_FIX.md.
- CPU: full PC is yours unless hoi4.exe or javaw.exe is running (check with tasklist every ~30 min). If either runs, drop to 2 workers and no GPU.
- Save every checkpoint. 3 h no-progress rule per experiment. All training stops by 08:00.

## Order of work
0. Control on 0.91.3 (30-45 min). Check that brains saved under 0.91.1 load and act the same under 0.91.3 (compare actions on a fixed set of frames; if they differ, say how much). Re-evaluate the showcase brains on HELD under 0.91.3: U3ck83, one_11_13, 1-1_U3ck3, 1-2_12n59, U3ck28. **Also test U3ck83 and st_U3 ck_0044 on 1-2 HELD+TEST** (never done). If either flags 1-2 even once, that one brain has cleared every World 1 level: verify with a clip and log it as a milestone.
1. Phase 1: RAM concept map (ceiling test, labeled CHEAT everywhere). ~13x16 cells around Mario, channels solid/enemy/pit/powerup/empty, from RAM. Feed it to ONE Cadence brain (with or without the screen, your call, pre-register which). Win: 1-2 HELD >= 20/50 or 2-1 first-sight max x > 1,000. Lose: no better than control on both.
2. Phase 2: death clinic (3a) + keep-only-survivals (3b) on World 1, on the best input from phase 1. Rewind to 1/2/4 s before each solo death, coach labels there; keep its own attempts only when they pass a spot the previous checkpoint died at, weight 0.5, capped at 20% of lessons. Win: 1-2 own-path critical error < 15% (from 27%). Lose: still >= 25%. Don't use 2-1 for coach labels (the coach can't clear it); World 2 is a test set only.
3. Phase 3 only if phase 1 won and time remains: learned concepts INSIDE one Cadence brain (predict the map from the screen as a side task / extra readout of the same brain). Sean does not want a separate concept module ("removes the point of using a cadence brain"). If Cadence's API can't do a second readout on one brain, write that down and stop phase 3; don't build a separate module.
Phases 4-5 are for a later night.

Scoreboard every phase: HELD solo clears on 1-1 / 1-2 / 1-3 / 1-4, first sight 2-1 / 2-2 (max x, flags), 1-2 own-path critical error.

## Gauntlet tester (Sean's request, must do)
Every ~90 min, and after each phase: real start of 1-1, FIVE lives (Sean, updated), on a flag continue straight to 1-2, then 1-3, 1-4, 2-1, ... A death restarts the current level from its start with the same brain (no learning between lives); no extra lives earned; game over on the 5th death. Log deaths per level, where each happened (level + x), lives left. **Clip every gauntlet, every time** (mp4 via the existing record/make_video code), even short ones.
- Main gauntlet: a single brain the whole way (the run's current best joint brain, plus U3ck83 as the standing champion). Only a single-brain gauntlet may be called "one brain".
- Optional second: a relay (best specialist per level), always labeled "relay of specialists".
- Log each in CONCEPTS.md and in a gauntlet table on the page: time, brain, levels cleared, where it died (level + x), clip link.

## Showcase page and docs (Sean's request)
Public page: cadence-arcade docs/mario/index.html (Crys from the Cadence Discord shares it). Keep it always current; honest numbers; RAM results always marked as RAM/cheat and never the headline.
- Add a "Meet the brains" roster so each brain is distinct: name, a plain short nickname (e.g. "the all-rounder", "the Bowser specialist"), what it was trained on, which run/checkpoint, Cadence version, HELD score per level, what's special about it. At least: U3ck83, st_U3 ck_0044, one_11_13, 1-1_U3ck3, U3ck28, 1-2_12n59, plus any new ones tonight.
- Headline feature: one brain (st_U3 ck_0044 / ck_0083) clears 1-1, 1-3 AND 1-4 with the same weights (ck_0044 HELD 25/15/37, ck_0083 15/45/33). Upgrade to "every World 1 level" only if step 0 shows a 1-2 flag. Show its clips prominently.
- Gauntlet section with the clips and table.
- Commit and push to cadence-arcade as you go (full freedom on that repo; clean commits; end messages with
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>). Never touch upstream Pragma repos.

## Reporting
- Text Sean (mcp__outbound__message_me, if available; else `python P:/agent/tools/notify_cli.py "text"`) only for: a milestone (one brain clears all of World 1, a phase win, a gauntlet getting past 1-4), or a stall/crash. Short, plain text, a few emoji, no links.
- Final answer: under 250 words: scoreboard per phase, gauntlet results (furthest level), what's pushed, what's next, anything broken. Put the RESUME block at the top of CONCEPTS.md.
