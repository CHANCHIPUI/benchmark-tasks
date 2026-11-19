# Hand-tap an M6 tool hole in a robot tool adapter plate

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

A round aluminium tool adapter plate for a robot wrist (ISO 9409-1-50-4-M6 on its robot face, ISO 9409-1-63-4-M6 on
its tool face) is held in a drill-press vise with one pre-drilled 5 mm tool hole ringed in blue. The robot takes a
T-handle tap wrench with an M6 plug tap out of its stand, starts the tap square in the marked hole, taps right through
with a back-off of at least a quarter turn after every turn in, stops at the red band on the tap, turns the tap back
out and stands the wrench back in its stand.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c "import json; print(json.load(open('task.json'))['instruction'])")" \
  --scorer task_score --max-steps 9000 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. `TASK_DEMO=nothing` plays no motion
(scores 0); `TASK_DEMO=start` plays only the start and the first six half turns.

The demo, open loop with the right arm: clamp the wrench's knurled body (the T-bar passes between the fingers), lift it
out of the stand, set the point in the countersink and push about 4 N, then 22 half turns in (each 188 deg clockwise,
back off, run forward, regrasp), 24 half turns out, lift the wrench out and stand it back. The first two half turns
use a firm clamp so the hand holds the tap upright while the thread's axis forms; later ones a light clamp, so the
T-bar drives against the fingers as a couple and the hand does not lever the tap.

## How `score()` reaches the score

- `tap_usd.py` writes the scene's USD files into `build/` when the task loads. The wrench and tap are one body on a
  fixed-base chain of three slides and three hinges (x, y, z; tilt about x and y at the tap's point; turn about the
  tap's axis). With no drives the wrench is free: it is carried, dropped or stood like any rigid body.
- The countersink and the drilled hole's wall are a separate body (eight convex wedges); the plate's own collider leaves
  an 8 mm square clearance round the hole, so the tap never touches the hole's wall once the thread rule holds it. A PhysX contact view between
  it and the wrench, read after every physics step, gives the push (mean over each 50 ms control step).
- The thread rule, every control step:
  - Bite: the point seated in the countersink (within 0.5 mm of the hole's axis, 1.5 to 4 mm below the plate's face),
    pushed 3 N or more, the tap within 10 deg of the hole's axis, and turning clockwise. The countersink body is then
    moved away and the drives take over.
  - First turn: slide drives hold the point on the hole's axis and on the helix (1 mm per turn of the tap's angle);
    the tilt only sticks (it yields to 0.05 N m, so the hand sets it and the tap does not fall over when let go).
  - The thread's axis is the tap's axis averaged over its first full turn (weighted by angle). After that, stiff
    drives hold the tap on that axis (200 N m/rad) and on the helix along it.
  - Turning resistance: a damper on the turning hinge whose force is capped at the cutting torque: 0.8 N m plus
    0.8 N m per turn of new thread since the last back-off of 90 deg or more, 0.2 N m through thread already cut,
    0.3 N m backing off the fresh cut.
  - Snap (latched 0): while in the thread after its first turn, a bending moment of 1.5 N m at the plate's face (from
    the drives' forces on the tap, at each control step) or 5 N m of twist; forcing past 24 mm of thread counts too.
  - Chip not broken (latched 0): more than 360 deg of new thread since the last back-off.
  - Out: turned back past the bite, the drives let go. While its point is still in the hole's mouth the tap rests on
    the thread's first turn (it cannot drop deeper, lifts out freely, and its tilt sticks as in the first turn).
- The end-state table (task.json, "scoring") is read every step; 1.0 needs its state (through and square, tap out,
  wrench in its stand within 15 deg of vertical, both hands 3 cm clear, the tap's tip moving under 1 cm/s over each
  50 ms step) for 20 steps (1 s), the
  same 1 s the harness holds `success()` for. Any latch gives 0 from then on; only a reset clears it.

## Runtime and seeds

- Demo: about 5,400 steps (270 s simulated); about 42 minutes on an RTX 3070 Ti laptop GPU at 640 x 360 (about 0.45 s
  a step), including start-up.
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop.

## Real-world kit

- Tap: HSS M6 x 1.0 D3 straight-flute plug hand tap (Regal Cutting Tools 020169AS), with a 2 mm red paint-marker band
  whose lower edge is 13.5 mm above the point.
- Wrench: Starrett 93B T-handle tap wrench. Its body and T-bar sizes are estimates (20 mm knurled body, 6.4 x
  100 mm bar).
- Vise: Wilton 11744 4 in drill-press vise, bolted to the table; two 150 x 32 x 3 mm hardened parallels.
- Adapter plates: 6061-T6, turned from 80 mm round bar, as drawn in task.json's setup; four trials per plate.
- Stand: grey PLA block 40 x 40 x 60 mm with a 12 mm hole chamfered to 24 mm, taped to the table.
- Gauges: M6 x 1.0 6H GO thread plug gauge; M6 x 60 mm DIN 976 threaded rod, 75 mm engineer's square, feeler gauges.

## Estimates

Estimates: the cutting torques (0.8 N m, plus 0.8 N m per turn cut since a back-off), the 3 N bite push, the
1.5 N m bending and 5 N m twist snap limits, and the wrench's body and T-bar sizes.

## Assets

All meshes, materials and colliders are generated by `tap_usd.py` (no third-party assets).
