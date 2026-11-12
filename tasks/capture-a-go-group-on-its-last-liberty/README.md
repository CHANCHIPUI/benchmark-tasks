# Capture a white Go group on its last liberty, then put only the captured stones in Black's lid

A task file for the `yam_arena` embodiment (yam-isaac-lab-arena, Isaac Lab-Arena 0.3.1).

A 9x9 Go teaching board lies flat between the arms, with a beginner's capture problem in its near-right corner. Black's
bowl (three black stones) and its upturned lid stand to the right of the board, and White's bowl and lid are on the far
side. Columns A to J run from the robot's left (Go boards skip I), and rows 1 to 9 from the robot's edge. X is Black:

```
     E  F  G  H  J
 5   .  .  X  .  .
 4   .  .  O  .  .
 3   .  X  O  X  X
 2   .  .  X  O  O
 1   .  .  .  X  .
```

The white pair H2-J2 has one liberty left, the corner point J1. A black stone there captures it, and the two stones
must then come off the board into Black's lid. The white pair G3-G4 touches H2 only diagonally, so it is a separate group
with two liberties (F4, H4) and stays. G1, the hole beside H2, is not a liberty.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "It is Black's turn. Play one black stone that captures white stones, then put the captured stones in Black's lid." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to one
of: `failure` (J1, then all four white stones into the lid: 0), `g1` (the stone on G1: 0), `h4` (the stone on H4: 0),
`leave` (J1, captured stones left on the board: 0.5), `table` (J1, captured stones onto the table: 0.75), `early` (J2
taken before J1 is played: 0), `two` (J1 and a second stone on H4: 0), `liftback` (a live stone lifted and set back:
0), `nothing` (0).

`task_score` is 1.0 once `success()` has held for 1 s, otherwise `score()` at the end. A callback run once per control
step (`_monitor`) reads the 13 loose stones' poses and both grippers' TCPs, latches the failures that stay failed (a
live stone leaving its point; a captured stone leaving its point before a stone is on J1), and judges the end state. If
`TASK_SCORE_DIR` is set, it also writes a per-step trace (`trace.jsonl`): every stone's position and tilt, the TCPs,
both arms' joint positions and the scorer's flags. `../checks/scorer_offline.py` checks the scorer against made-up
sequences of poses, without the simulator.

## Runtime and seeds

- Demo: see `scores.txt` for the steps to success and the time per step.
- Dream-1: up to 2,400 steps (120 s simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, scorer and scripted demo.
- `assets.py`: the geometry, and the USD files it writes into `assets/` when the task loads.
- `scenekit.py`, `usdkit.py`: helpers that load textured USD bodies and write meshes, materials and colliders.
- `steptime.py`: records simulated time and the slowest control steps.
- `make_textures.py`: draws `textures/` (run once; needs Pillow).

## Sizes and sources

- Board and stones: sized from a 9x9 Go demonstration set (a 42 cm board with 4 cm stones, sold for introducing the game
  to groups), laid flat. The grid is 50 mm with 5 mm lines; the stones are single-convex like common melamine beginner
  stones (flat underside, domed top), 40 mm by 12 mm, 16 g (melamine about 1.5 g/cm3).
- Bowls and lids: turned chestnut Go bowls, squat for the large stones, 17 cm across.
- Textures: our own artwork, drawn by `make_textures.py` (procedural wood grain and the printed grid). No third-party
  assets.
