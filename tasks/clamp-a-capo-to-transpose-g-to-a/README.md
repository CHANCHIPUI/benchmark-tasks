# Move a guitar's capo so that a singer's G chord shapes sound in A

A task file for the `yam_arena` embodiment (yam-isaac-lab-arena, Isaac Lab-Arena 0.3.1).

A full-size acoustic guitar lies on its back on the bench, strings up, its headstock toward the robot's right. A red
cam-lever capo is clamped on the neck at the 4th fret, left there from the previous song. The singer wants the next song
in A but plays it with G chord shapes. A capo raises the strings one semitone per fret, and A is two semitones above G,
so the capo belongs at the 2nd fret: the space between the 1st and 2nd fret wires, counted from the nut. That space has
no dot; the first dot is in the 3rd. The capo must be released (press its lever's tail), moved, and clamped again (press
its handle flat). The 1st fret, the 3rd, the 4th where it starts, any fret counted from the body end, or the right fret
left unclamped, leaves the song in the wrong key and scores 0.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The singer wants the next song in A, but she only knows how to play it with G chord shapes. Set the capo on her guitar for it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to one
of:
- `failure`: slid to the 2nd fret but left open, lever up (0);
- `fret3`, `fret1`, `fret6`: clamped at the 3rd, 1st or 6th fret (0);
- `wire`: clamped on top of the 2nd fret wire (0.5);
- `skew`: clamped in the 2nd fret space, turned 15 degrees (0.5);
- `held`: clamped in the 2nd fret space, then gripped again without the hand going 6 cm from the capo, and held to the end (0);
- `nothing`: the capo left at the 4th fret (0);
- `grip`: a firm grip on the clamped capo, then the demo (1.0);
- `drag`: the clamped capo gripped and dragged along the neck without opening the lever (0);
- `side`: the demo, but the capo slid off the neck sideways, along, and back on (1.0).

`task_score` is 1.0 once `success()` has held for 1 s, otherwise `score()` at the end. Two callbacks run once per
control step:
- `_run_capo` is the capo's cam and latch. A lever pressed to within 4 degrees of flat closes it: the jaw's drive
  clamps the neck with 25 N. A lever turned past 16 degrees up, with its tail pressed flat, opens it: the jaw drops 10 mm.
- `_monitor` measures the capo from its pose, its joints and both grippers' TCPs:
  - where its bar's centre line crosses the 1st and 6th strings;
  - whether it is clamped (lever flat, pad on the strings, jaw on the neck's back, level);
  - its angle to square;
  - how far it has moved in the last 1 s;
  - the TCPs' distance.

Two failures latch for the rest of the trial and only a reset clears them: the capo moved more than 5 mm along the neck
while its lever is closed (within 5 degrees of flat, whatever the capo's tilt or height), and the guitar moved (the
guitar is fixed in simulation). If `TASK_SCORE_DIR` is set, the callback also writes a per-step trace (`trace.jsonl`).

## Runtime and seeds

- Demo: 320 steps (16.0 s simulated): success from 15.05 s, held 1 s. About 80 s of wall time including start-up.
- Dream-1: up to 1,200 steps (60 s simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, the capo's latch, the scorer and the scripted demo.
- `assets.py`: the geometry, and the USD files it writes into `assets/` when the task loads (the guitar as one fixed
  body; the capo as an articulation of frame, jaw and lever).
- `scenekit.py`, `usdkit.py`: helpers that load textured USD bodies and articulations, set joint targets, and write
  meshes, materials and box or hull colliders.
- `steptime.py`: records simulated time and the slowest control steps.
- `make_textures.py`: draws `textures/` (run once; needs Pillow).

## Sizes and sources

- Guitar: sized from a full-size dreadnought steel-string acoustic: 650 mm scale, 104 cm long, 14 frets clear of the
  body, 43 mm nut, lower bout 39.7 cm. Frets by the equal-temperament rule, 650 x (1 - 2^(-n/12)) mm from the nut. Dots
  8 mm across (usually 6 mm) so they show at camera resolution. The body is 10 cm deep throughout.
- Capo: a cam-lever capo sized from common 6-string capos. It has a 66 mm bar with a rubber pad, a padded jaw under
  the neck, a rocker lever, and a 25 N clamp; it weighs 47 g.
- Textures: our own artwork, drawn by `make_textures.py` (rosewood, spruce with soundhole, rosette and pickguard,
  mahogany, beech). No third-party assets.
