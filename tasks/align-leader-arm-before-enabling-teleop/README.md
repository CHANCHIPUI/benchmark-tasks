# Match a leader arm to its follower before enabling teleoperation, then steer the follower round a glass onto a dot

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1), on the station scene: [task.py](task.py).

A leader-follower teleoperation rig stands on the bench. The yellow leader, a 0.75-scale replica of the black follower, is folded in front of the robot's left arm. The follower is parked beyond the robot's reach, reaching out to the right, with a glass standing in its sweep and a green dot beyond the glass. Once ENABLE is pressed, the follower copies the leader's joint angles, so it moves straight to wherever the leader is. The robot has to pose the leader to match the follower before pressing ENABLE, then steer the follower round the inside of the glass to the dot.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the yellow leader arm to put the black follower arm's tip on the green dot. Press ENABLE to start teleoperation: from then on, the follower copies the leader's joint angles." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. The demo is open loop. The left arm grips the leader's knob and drags it on a folded arc round the leader's base and out to the follower's pose. The right arm presses ENABLE while the left arm holds the knob. The left arm then folds the leader in, swings it round, and reaches out to the dot's pose; the follower follows, passing inside the glass. `TASK_DEMO` picks a variant for checking the scorer, each run with the same command:

| `TASK_DEMO` | What the arms do | Score |
|---|---|---|
| `success` (default) | As above | 1.0 |
| `enable_first` | Press ENABLE at once, with the leader still folded | 0 |
| `goal_then_enable` | Pose the leader at the dot's pose, then press ENABLE | 0 |
| `straight` | Match, press ENABLE, then drag the knob straight to the dot's pose | 0 |
| `stop` | Match, press ENABLE, then let go and stop | 0.5 |
| `nothing` | Both arms hover | 0 |

The run prints `task_score` at the end. `score()` latches a failure for the rest of the trial and returns 0.5 after a clean enable; `success()` is the 1.0 state, which the harness holds for 1 s. With `TASK_SCORE_DIR` set, each step's joint angles, the gap at the enable, the tip's distance from the dot and the glass's pose are logged to `steps.jsonl` there.

## Runtime and seeds

- Demo: 560 steps (28.0 s simulated), ends on success; about 2.5 minutes of wall time including start-up.
- Dream-1: up to 2,400 steps (2 minutes simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, the rig's controller (every control step it detects the enable, aims the follower's drives, and moves the leader's stick-slip friction set points), the scorer and the scripted demo.
- `rig_usd.py`: the layout, the arms' kinematics and the USD files (both arms, the ENABLE box, the glass, the mats and cables), written into `build/` each time the task file loads.
- `usdkit.py`, `scenekit.py`: helpers for meshes with separate colliders, and for loading USD bodies and articulations (with drive settings and a reset that puts joints back at their start).
- `make_textures.py`, `textures/`: the textures, drawn by that script. `fonts/`: Oswald, for the ENABLE label.
- `steptime.py`: records step timing when `TASK_STEP_FILE` is set.

## Sources

All meshes are built in `rig_usd.py` from extruded outlines, boxes, cylinders and lathed profiles. The textures are drawn by `make_textures.py`. The ENABLE label uses Oswald, Copyright 2016 The Oswald Project Authors, SIL Open Font License 1.1 (`fonts/OFL.txt`). No other third-party assets are used.

Sizes: the follower is sized as a small desktop SCARA-type arm (links of 14 and 16 cm between axes); the leader is a 0.75-scale replica, as low-cost leader arms often are. The glass is a common 300 ml tumbler (7.4 cm across, 10 cm tall). The ENABLE box is a small single-button control enclosure with a 30 mm illuminated push button.
