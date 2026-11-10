# Set 2,605 on a soroban

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1), on the station scene: [task.py](task.py).

A cleared 5-rod teacher's soroban lies flat on the bench, facing the robot. The robot must set 2,605 on it with the ones digit on the rod under the right-hand unit dot (rod 4 of 5, counting from the robot's left). On each rod the bead above the beam is worth 5 and each of the four below it is worth 1, and a bead counts only when it is pushed against the beam. So the 6 needs the 5-bead and one 1-bead, the 5 needs the 5-bead alone, the tens rod stays cleared for the zero, and the rightmost rod (the tenths) stays cleared too.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set 2,605 on the soroban. The rod with the right-hand dot on the beam is the ones place." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. The demo is open loop. With closed fingers, the left arm pushes two 1-beads on rod 1 up to the beam from the groove behind the second one, pushes one 1-bead on rod 2 the same way, then presses on rod 2's 5-bead and slides it down to the beam. The right arm slides rod 4's 5-bead to the beam. `TASK_DEMO` picks a variant for checking the scorer, each run with the same command:

| `TASK_DEMO` | What the arms do | Score |
|---|---|---|
| `success` (default) | As above | 1.0 |
| `ones` | Every bead counted as 1: 2, 3 and 3 1-beads on rods 1, 2 and 4 | 0.3 |
| `shifted` | The rightmost rod taken as the ones place: 2, 6, 0, 5 on rods 2 to 5 | 0 |
| `nozero` | The zero skipped: 2, 6, 5 on rods 2, 3, 4 | 0 |
| `halfway` | As the demo, but rod 4's 5-bead is left halfway | 0.6 |
| `redo` | A 5-bead set on the spare rod 5 and slid back first, then the demo | 1.0 |
| `force` | Rod 4's 5-bead pressed hard and driven 2 cm past its stop at the beam: the stop holds (0.0 mm past it), and rod 4 reads 5 | 0.3 |
| `nothing` | Hovers | 0 |

The run prints `task_score` at the end. `score()` latches a failure for the rest of the trial; `success()` is the 1.0 state held, with no bead moving more than 1 mm, for 1 s. With `TASK_SCORE_DIR` set, each step's rod readings and bead travel are logged to `steps.jsonl` there.

## Runtime and seeds

- Demo: 372 steps (18.6 s simulated), ends on success; about 2 minutes of wall time including start-up.
- Dream-1: up to 2,400 steps (2 minutes simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, scorer and scripted demo.
- `soroban_usd.py`: the layout, the meshes and the soroban's articulation (frame plus 25 beads on sliding joints), written as USD into `build/` each time the task file loads.
- `usdkit.py`, `scenekit.py`: helpers for meshes with separate colliders and for loading USD bodies and articulations.
- `make_textures.py`, `textures/`: the textures, drawn by that script.
- `steptime.py`: records step timing when `TASK_STEP_FILE` is set.

## Sources

All meshes are built in `soroban_usd.py` from boxes, cylinders and lathed profiles. The textures (`wood.png`, `wood_v.png`, `rubber.png`) are drawn by `make_textures.py`. No third-party assets are used.

Sizes: a commercial 5-rod teacher's soroban, 320 x 280 x 50 mm and 800 g, with a wooden frame, yellow plastic beads and a white beam. The bead size (40 mm across, 24 mm thick) and the rod spacing (44 mm) are measured from its product photo; teacher's sorobans in general have beads about 4.5 cm across. The unit dots sit every third rod, as on standard sorobans.
