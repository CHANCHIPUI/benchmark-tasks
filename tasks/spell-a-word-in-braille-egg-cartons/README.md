# Spell "hid" in braille with balls in three egg cartons, one letter per carton, for a reader on the robot's side

A task file for the `yam_arena` embodiment (yam-isaac-lab-arena, Isaac Lab-Arena 0.3.1).

Three empty half-dozen egg cartons stand in a row between the arms, each one braille cell of two columns of three cups.
Ten red wooden balls wait in two racks, one beside each arm. The robot is asked to spell "hid" in braille for a reader
on its own side of the table, one letter per carton, a ball in a cup being a raised dot. The instruction gives the dot
numbering (1-2-3 down the left column, 4-5-6 down the right, top row farthest from the robot) but not the letters: h is
dots 1-2-5, i is 2-4 and d is 1-4-5, 8 balls in all. Each of these letters' mirror image is another letter (j, e, f),
so a cell built with its columns swapped spells something else and scores nothing.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Spell the word \"hid\" in braille for a reader sitting on your side of the table, one letter in each egg carton. A ball in a cup is a raised dot. In each carton, the top row is the one farthest from you, dots 1, 2 and 3 run down the left column, and dots 4, 5 and 6 run down the right column. The word reads from left to right." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. To check the scorer, set `TASK_DEMO` to one
of these; each builds the cells shown, from the racks, with the same moves:

| `TASK_DEMO` | Cells built (cartons 1, 2, 3) | Score |
|---|---|---|
| `jef` | every cell's columns swapped: j, e, f | 0 |
| `fej` | the whole row mirrored: f, e, j | 0 |
| `across` | built for a reader across the table (turned 180 degrees) | 0 |
| `rows` | dots numbered across the rows: 1-3-4, 4-5, 1-3-5 | 0 |
| `upside` | every cell upside down | 0 |
| `dih` | the right letters in reverse order | 0.33 |
| `stray` | hid, with a ninth ball laid on carton 1's middle ridge (it comes to rest on the two seated balls) | 0.67 |
| `held` | hid, with the last ball still held over its cup at the end | 0.67 |
| `nothing` | nothing | 0 |

`task_score` is 1.0 once `success()` has held for 1 s, otherwise `score()` at the end: 1/3 per correct cell, at most
2/3. A callback run once per control step (`_monitor`) judges the three cells from the ball positions and both grippers'
TCPs (`judge`). Stillness is measured from each ball's positions over the last 0.25 s, because the simulator's reported
speed of a ball resting in a cup can stay at a few cm/s while the ball does not move. If `TASK_SCORE_DIR`
is set, it also writes a per-step trace (`trace.jsonl`): each cell's filled dots and stray balls, every ball's position
and speed, and the TCPs.

## Runtime and seeds

- Demo: 816 steps (40.8 s simulated) to success, about 3 minutes including start-up.
- Dream-1: up to 2,400 steps (120 s simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, scorer and scripted demo.
- `assets.py`: the cartons', racks' and balls' geometry; writes their USD files into `assets/` when the task loads.
- `usdkit.py`, `scenekit.py`: helpers that write textured meshes with separate colliders and load them as rigid bodies.
- `make_textures.py`: draws the textures in `textures/` (run once; needs Pillow and NumPy).

## Sources

- Egg cartons: half-dozen moulded-pulp carton bases (about 156 x 106 mm, cups at 50 mm pitch), lids cut off as in the
  classroom activity. Meshes and pulp texture our own (`assets.py`, `make_textures.py`).
- Balls: 40 mm craft-store beech balls (about 0.72 g/cm3, 24 g), painted. Racks: pine boards with drilled sockets.
- No third-party images or meshes.
