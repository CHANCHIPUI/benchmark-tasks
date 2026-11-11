# Turn a telescope mount so its N leg points north

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1), on the station scene: [task.py](task.py).

A hand compass lies on the lid of a red steel toolbox on the robot's left. A tabletop equatorial telescope mount stands on the robot's right, pinned at its centre so that it can only turn; one of its three legs is yellow and marked N. The robot must turn the mount so that its N leg points north, using the compass. Steel pulls a compass needle off north: on the lid the needle points 50 degrees anticlockwise of north, and it reads within 2.5 degrees of north only once the compass is 25 cm or more from the box (within 1 degree from 30 cm). The instruction does not mention steel.

North is at a bearing of 115 degrees (clockwise from straight ahead, seen from above), and declination is taken as zero. The magnetic field is computed, not simulated (physics engines have no magnetism): Earth's horizontal field, 20 uT toward north, plus the field of one point dipole for the toolbox, 0.33 A m2 at its centre ([field.py](field.py)). Every control step the needle's joint is driven toward the field's direction in the plane of the compass face, with a 0.35 s lag, like a liquid-damped compass. The arms and the mount are not magnetic in the model.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Turn the telescope mount so that its N leg points north. Use the compass." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. The demo is open loop. North is fixed in the setup, so it turns the mount to it directly, but it moves the compass first so that the video shows the reading change: the left arm grips the compass on the lid, carries it 30 cm clear of the toolbox and sets it down, and the needle swings from bearing 65 to bearing 115. The right arm then grips a leg 10 cm from the axis and swings the mount 70 degrees anticlockwise along waypoints 10 degrees apart, lets go and rises. `TASK_DEMO` picks a variant for checking the scorer, each run with the same command:

| `TASK_DEMO` | What the arms do | Score |
|---|---|---|
| `success` (default) | As above | 1.0 |
| `lid` | The N leg turned to the compass's reading on the lid (bearing 65) | 0 |
| `above5` | The N leg turned to the reading 5 cm above the lid (bearing 100, 15 degrees off) | 0.5 |
| `ahead` | The N leg turned straight ahead (bearing 0) | 0 |
| `right` | The N leg turned to the robot's right (bearing 90, 25 degrees off) | 0 |
| `held` | As the demo, but the gripper keeps hold of the leg to the end | 0 |
| `nothing` | The arms only park | 0 |

The run prints `task_score` at the end. With `TASK_SCORE_DIR` set, each step's mount bearing, error, release and stillness, and the compass's position, reading and needle state are logged to `trace.jsonl` there.

## Runtime and seeds

- Demo: 499 steps (24.95 s simulated), ends on success; about 2 minutes of wall time including start-up. The slowest 30 control steps took 6.2 s at 320 x 176 and at 640 x 360 (about 0.21 s a step).
- Dream-1: up to 2,400 steps (2 minutes simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: objects, the needle drive, the scorer and the scripted demo.
- `field.py`: the computed magnetic field.
- `scene_usd.py`: the layout, the meshes, the compass's articulation (housing and needle joint) and the mount's (a pin and the mount on a revolute joint), written as USD into `build/` each time the task file loads.
- `usdkit.py`, `scenekit.py`: helpers for meshes with separate colliders and for loading USD bodies and articulations.
- `make_textures.py`, `textures/`: the textures, drawn by that script.
- `steptime.py`: records step timing when `TASK_STEP_FILE` is set.

## Sources

All meshes are built in `scene_usd.py` from boxes, cylinders, prisms, tubes and lathed profiles, sized from common products: a small single-compartment steel toolbox (about 32 x 16 x 11 cm), a 7.5 cm liquid-filled hand compass, and a tabletop equatorial mount scaled to the table. The textures (`paint_red.png`, `brushed.png`, `cast.png`) are drawn by `make_textures.py`. No third-party assets are used.
