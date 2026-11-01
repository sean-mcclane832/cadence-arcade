# Brain viz

Shows what a Cadence brain actually does while it plays Mario: brain-cam videos and an interactive page (`docs/brain/`). Every number comes from the real brain on the real emulator. Nothing is decorative.

## Files

- `record_brain.py`: plays one solo attempt (same loop as the solo eval in `dagger.py`), then settles every decision again inside `cadence.recording.record_settlements` to get the real neuron states, then runs the occlusion "poke" (180 patches of 4x4 blocks blacked out, plus one run with the body sense zeroed). Writes `viz/out/<name>.npz` (not committed).
- `brain_video.py`: npz -> side-by-side MP4 (game | input, body sense, neurons by region, buttons, settle curve).
- `export_page.py`: npz -> `docs/brain/data/` (json + gzipped uint8 bin + webp frame sheets).

## Regenerate (from `mario/`, with the `.venv-mario` Python, Cadence 0.91.1)

```
python viz/record_brain.py --brain brains/U3ck83.npz --level 1-1 --start 0  --name u3ck83-1-1-clear
python viz/record_brain.py --brain brains/U3ck83.npz --level 1-1 --start 6  --name u3ck83-1-1-death
python viz/record_brain.py --brain brains/U3ck83.npz --level 1-4 --start 0  --name u3ck83-1-4-clear
python viz/record_brain.py --brain brains/U3ck83.npz --level 1-4 --start 15 --name u3ck83-1-4-death
python viz/brain_video.py --name u3ck83-1-1-clear --out ../docs/brain/clips/u3ck83-1-1-clear.mp4   # same for the others
python viz/export_page.py u3ck83-1-1-clear u3ck83-1-4-clear u3ck83-1-1-death u3ck83-1-4-death
```

Each record takes 1 to 5 minutes on 1-2 cores (most of it is the poke). Preview the page with `python -m http.server` from `docs/`, then open http://localhost:8000/brain/. URL options: `?run=u3ck83-1-4-death&k=160&sal=1`.

## What the recording showed (and how it differs from what you might expect)

- The connectome has four regions: sensory 2,000 (1,664 eye blocks + 42 body numbers x 8 copies), association 256, prefrontal 256, motor 7.
- Each `predict` call makes two records: a 0-step check, then the real settle. Every decision here settled in exactly 32 steps.
- `predict` reads no live memory, so the prefrontal (working memory) region is 0 at every decision. The page and videos show it dark and say why.
- The recording runs Cadence's NumPy path (the docs warn it can differ at round-off). The recorded choice matched the pressed button at 100% of decisions in all four runs, and every replay ended at the same x and flag as the eval.
- Activations are in [-0.1, 1]. The page stores them at 8 bits (q = round((a + 0.1) / 1.1 * 255)).

## Link line for the Mario page

```html
<p><a href="../brain/">Brain scope</a>: scrub through a solo run and watch the brain's real neuron activity, or poke it to see what part of the screen each move depended on.</p>
```
