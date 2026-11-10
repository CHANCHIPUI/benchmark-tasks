# Set a word in wood type in a chase so that it prints the right way round

A task file for the `yam_arena` embodiment (yam-isaac-lab-arena, Isaac Lab-Arena 0.3.1).

Five loose wood-type sorts (K, E, L, P and a spare J) lie face up on a bench mat. The robot is asked to set the word KELP
in a chase of four fixed bays so that it prints correctly. Type faces are cut mirror-reversed and the paper meets them
face to face, so the forme must be the mirror image of the word: seen from above with the letters' heads to the far
side, P, L, E, K from the left. Laid out to read KELP from above, it prints PLEK. The E lies turned end for end, and the
spare J's reversed face looks like an L.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the word KELP in the chase so that it prints correctly, with the top of every letter toward the far side." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to one
of: `reading` (K, E, L, P set in bays 1 to 4 as they lie, E still turned; K and P are passed between the arms on the
mat: 0), `eturned` (the right order with E left turned: 0.75), `jforl` (the right order with the J in place of the L:
0.75), `held` (the right order, E still held at the end: 0.75), `nothing` (0).

`task_score` is 1.0 once `success()` has held for 1 s, otherwise `score()` at the end. A callback run once per control
step (`_monitor`) judges the four bays from the sorts' poses and both grippers' TCPs, and latches a sort falling off the
table; only a reset clears it. If `TASK_SCORE_DIR` is set, it also writes a per-step trace (`trace.jsonl`): every sort's
position, tilt and heading, the TCPs, both arms' joint positions, and which bay conditions fail.

## Runtime and seeds

- Demo: 841 steps (42.05 s simulated) to success, about 4 minutes including start-up.
- Dream-1: up to 2,400 steps (120 s simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, scorer and scripted demo.
- `assets.py`: the sorts' and the bench's geometry; writes their USD files into `assets/` when the task loads.
- `usdkit.py`, `scenekit.py`: helpers that write textured meshes with separate box colliders and load them as rigid bodies.
- `make_textures.py`: draws the textures in `textures/` (run once; needs Pillow and NumPy).
- `fonts/`: Oswald (variable font, used at Bold) and its licence.

## Sources

- Wood type: 18-line (18 picas, 76.2 mm body) poster sorts, type-high (0.918 in, 23.3 mm), end-grain hard maple at about
  0.72 g/cm3; set widths fitted to each letter with 3 to 4 mm shoulders. Meshes and textures our own (`assets.py`,
  `make_textures.py`).
- Letters: Oswald, Copyright 2016 The Oswald Project Authors, SIL Open Font License 1.1 (`fonts/OFL.txt`).
- Chase, furniture (2-pica, 8.5 mm), imposing surface and bench mat: our own meshes, sized from common letterpress shop
  equipment.
- No third-party images.
