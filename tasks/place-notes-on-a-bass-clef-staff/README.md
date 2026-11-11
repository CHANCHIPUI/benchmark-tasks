# Write four named notes on a staff board printed with a bass clef, one note head in each bar

A task file for the `yam_arena` embodiment (yam-isaac-lab-arena, Isaac Lab-Arena 0.3.1).

A music teaching board lies flat between the arms: a five-line staff with a bass clef at its left end and bar lines that
make four bars. Four loose note heads (black wooden discs) wait in two shallow trays along the board's near edge. The robot is asked to write the notes D, F,
C and E, one in each bar from left to right, on the staff's lines and spaces. The instruction does not mention the clef.
In bass clef, D is on line 3, F on line 4 (between the clef's dots), C in space 2 and E in space 3. Read as the more
common treble clef, every note lands at least two staff steps away, and scores nothing.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the note heads in the trays to write the notes D, F, C and E on the staff, one note in each bar from left to right. Put each note on one of the five lines or in one of the four spaces between them." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to one
of: `treble` (the treble reading: 0), `reverse` (the right pitches in reverse order: 0), `ascending` (the letters in
pitch order, C D E F: 0), `onetreble` (three right, E on its treble place: 0.75), `held` (right, but bar 3's head still
held at the end: 0.75), `chord` (D and F both in bar 1, bar 2 empty: 0.5), `gap` (right, except F 12 mm above line 4:
0.75), `barline` (right, except D 2 cm from bar 1's left bar line: 0.75), `nothing` (0).

`task_score` is 1.0 once `success()` has held for 1 s, otherwise `score()` at the end. A callback run once per control
step (`_monitor`) judges the four bars from the heads' poses and both grippers' TCPs. Nothing latches: heads may be
lifted and placed again. If `TASK_SCORE_DIR` is set, it also writes a per-step trace (`trace.jsonl`): every head's
position, tilt and speed, the TCPs, both arms' joint positions, and which bar conditions fail.

## Runtime and seeds

- Demo: 708 steps (35.4 s simulated) to success, about 3 minutes including start-up.
- Dream-1: up to 2,400 steps (120 s simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, scorer and scripted demo.
- `assets.py`: the geometry, and the USD files it writes into `assets/` when the task loads.
- `scenekit.py`, `usdkit.py`: helpers that load textured USD bodies and write meshes, materials and box or hull colliders.
- `steptime.py`: records simulated time and the slowest control steps.
- `make_textures.py`: draws `textures/` (run once; needs Pillow).
- `fonts/Bravura.otf`: the music font the clef is drawn from.

## Sizes and sources

- Staff board: sized like a large classroom staff board, with the staff enlarged (4.5 cm line spacing, 8 mm lines) so it
  reads at camera resolution. White melamine-faced board, matte black print.
- Note heads: 40 mm by 18 mm beech discs, black lacquer, like large wooden game counters (beech about 0.71 g/cm3, 16 g).
- Trays: shallow beech trays, 4.8 by 18 cm inside, with a 1 cm rim.
- Textures: our own artwork, drawn by `make_textures.py`. The bass clef is the F-clef glyph (U+E062) of Bravura,
  SIL Open Font License 1.1 (`fonts/OFL.txt`).
