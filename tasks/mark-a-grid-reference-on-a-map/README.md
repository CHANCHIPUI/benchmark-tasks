# Mark a grid square and a six-figure grid reference on a topographic map with two markers

A task file for the `yam_arena` embodiment (yam-isaac-lab-arena, Isaac Lab-Arena 0.3.1).

A printed topographic map lies taped to the table between the arms, north to the far side, with a 1 km grid of 10 cm
squares whose two-digit numbers are printed on the margins. The robot is asked to put the red marker in grid square 3234
and the blue marker on grid reference 346 332. By the map-reading convention the easting (read along the bottom or top
edge) comes before the northing (read up the side), a four-figure reference names the square whose south-west corner is
the crossing of its two lines, and a six-figure reference adds tenths of a square estimated by eye. The number ranges
overlap, so a reference read northing first still lands on the map, in the wrong place.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the red marker in grid square 3234 and the blue marker on grid reference 346 332." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to one
of: `northfirst` (both references read northing first: red in square 3432, blue 1.5 cm from where 346 332 lands read that
way: 0), `crossing` (red on the crossing of lines 32 and 34: 0.5), `southwest` (red in square 3133, south-west of that
crossing: 0.5), `offpoint` (blue in the middle of square 3433, 2.9 cm from its point: 0.75), `held` (blue still held on
its point at the end: 0.5), `nothing` (0).

`task_score` is 1.0 once `success()` has held for 1 s, otherwise `score()` at the end. A callback run once per control
step (`_monitor`) judges both markers from their poses and both grippers' TCPs, and counts how long each has been still.
If `TASK_SCORE_DIR` is set, it also writes a per-step trace (`trace.jsonl`): each marker's base centre and tilt, the TCPs,
the score so far, and which conditions fail.

## Runtime and seeds

- Demo: 283 steps (14.15 s simulated) to success, about 70 s of wall time including start-up.
- Dream-1: up to 2,400 steps (120 s simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, scorer and scripted demo.
- `layout.py`: where the grid lies on the table, and the two references.
- `assets.py`: the board's and the markers' geometry; writes their USD files into `assets/` when the task loads.
- `usdkit.py`, `scenekit.py`: helpers that write textured meshes with separate colliders and load them as rigid bodies.
- `make_textures.py`: draws the map in `textures/` (run once; needs Pillow and NumPy). `map_north_up.png` is the same map
  north up, for people to read.
- `fonts/`: Oswald (variable font, used at Bold) and its licence.

## Sources

- Map: our own invented terrain and layout (`make_textures.py`); it copies no published sheet or map style.
- Grid numbers: Oswald, Copyright 2016 The Oswald Project Authors, SIL Open Font License 1.1 (`fonts/OFL.txt`).
- Markers: our own meshes, sized as weighted plastic map markers (25 mm across, 40 mm tall, 40 g).
- No third-party images.
