# mario/ progress (overnight 2026-10-29)

Goal: ONE Cadence 0.91.1 brain playing Super Mario Bros. World 1-1 from the screen, as far as
possible (ideally the flag), with a replay page.

## Stages
- [x] Feasibility: nes-py 8.2.1 + gym-super-mario-bros 7.4.0 build on Windows with MSVC 2019
  Build Tools (vcvars64 + DISTUTILS_USE_SDK=1). ~776 emulator frames/s on one core. ROM ships in
  the pip package; nothing ROM-related is in git.
- [x] Cadence 0.91.1 speed: 6,656 inputs, one region of 256, 32-row lesson ~0.15 s on CPU after a
  one-off ~35 s numba compile.
- [x] Coach (emulator look-ahead planner, game.py) clears 1-1 from every unjittered start
  (6/6, ~270 decisions); ~1.2 s per label per worker on the shared machine.
- [x] Baselines (random 594 median x, always-run 312, jump rhythm 311; 0 flags).
- [x] Round 0 (3,259 coach screens, 306 lessons): brain only runs, dies at first Goomba (x 312).
  Raised lessons to 2,000-4,000 per round, resumed with --resume --extra-teach at 02:11.
- [ ] (running, runs/main/log.txt) Imitation (teaching="normalized", 32-row lessons) + DAgger rounds on the brain's own path.
- [ ] Record the brain's run, build page, publish to docs/mario/.

## Doom precedent
Upstream muellerberndt/cadence #118 "Build and qualify the full native Doom expert" is an open
roadmap epic (M5), with no posted playing results. Its first skill is Basic combat, it allows only
rendered information to enter the player, and it says a frame/action classifier or host planner is
a component control, not the complete expert. No public Cadence Doom demo was found.

## Notes
- Venv: `P:/projects/cadence-arcade/.venv-mario` (uv, py3.11, cadence-net 0.91.1 from the local
  clone, non-editable, numpy<2 for old gym).
- Build: see README for the vcvars command.
