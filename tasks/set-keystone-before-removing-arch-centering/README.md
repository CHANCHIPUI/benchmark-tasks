# Set the keystone in a stone arch before taking its wooden centering out from under it

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

Instruction: "Take the wooden centering out from under the stone arch."

A bench model of a dry-laid stone arch stands on its timber centering, every stone laid except the keystone, which stands on the table in front of it. Without the keystone each half of the arch leans on the centering: drawn out, the arch falls. A mason sets the keystone first (wide end up, turned to fit the gap, pressed straight down) and only then strikes the centering. The instruction does not mention the keystone.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the wooden centering out from under the stone arch." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. `TASK_DEMO` picks a variant (see `demo()` in `task.py`): `failure` draws the centering first, `seat_only` sets the keystone and leaves the centering in, `unturned` lowers the keystone onto the gap without its quarter turn and then draws the centering, `nothing` waits, and `keyed` is a physics check that starts with the keystone seated, draws the centering and runs to the step limit.

The run prints `task_score`. `success()` and `score()` read the poses of the eight voussoirs, the keystone and the centering, and points on both grippers' finger and palm collision meshes, every control step. A trial-long tracker latches a disturbance (any voussoir more than 5 mm from where it rested at the first step), after which the score is 0 for the rest of the trial; only a reset clears it. Otherwise 1.0 once the keystone is seated, the centering is out, both grippers are at least 2 cm from every stone and everything is still, held for 2 s (`SUCCESS_HOLD_S = 2`, which ends the trial); 0.5 at the run limit with the keystone seated; else 0. With `TASK_SCORE_DIR` set, every step's state is written to `steps.jsonl` there.

## Runtime and seeds

- Demo: 462 steps (23.1 s simulated), about 0.3 s of wall time per step plus start-up.
- Dream-1: up to 1,200 steps (1 minute simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: the objects, the scorer and the scripted demo.
- `arch_usd.py`: the stones, abutments and centering (meshes, convex colliders, masses), written as USD into `build/` when the task loads.
- `make_textures.py`: draws the textures in `textures/`.
- `scenekit.py`, `usdkit.py`: helpers that add USD bodies to a task file, and write meshes, materials and colliders.
- `steptime.py`: records each step's wall time.

## Sources

Every mesh is built in `arch_usd.py` from the arch's geometry: a 120 degree segmental arch of nine stones with radial joints (intrados radius 14.5 cm, ring 3.5 cm, span 25.1 cm, rise 7.25 cm), as small segmental door and window heads are built, at about 1:4. The centering is a solid "turning piece", the timber centre used for small segmental arches. All textures (three sandstones, pine, beech) are drawn by `make_textures.py`. No third-party assets are used.
