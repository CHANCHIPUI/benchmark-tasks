# Take the binder clip off a stack of old invoices before feeding the stack into a desktop shredder

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

Instruction: "Shred these old invoices."

A personal paper shredder stands on an office desk. In front of it, three old invoices, folded as they came by post, stand in an oak letter rack, held together by a black binder clip on their top edge. Personal shredders take staples and small paper clips, but a binder clip must come off first: fed in, it reaches the cutters and jams or damages them. The instruction does not mention the clip. The robot must pinch the clip's handles to open it, lift it off and put it down away from the shredder, then lift the stack out of the rack and feed it edge first into the 23 x 1.6 cm slot.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Shred these old invoices." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. `TASK_DEMO` picks a variant for checking the scorer (see `demo()` in `task.py`): `failure` feeds the stack with the clip on, `nothing` does nothing, `cliponly` puts the clip in the dish and stops, `clipslot` drops the clip into the slot, `clipontop` leaves the clip on the shredder's top and then feeds the stack, `byclip` lifts the clip by its body (the stack comes with it), and `fingers` pushes the stack in with the fingertips inside the slot. `pulltest` (debug) holds the stack and pulls the clip straight up with a force rising 5 N/s, to measure its grip.

The run prints `task_score`. `score()` latches a failure (the shredder started with the clip on the stack, a clip entered the slot, or a fingertip entered the slot) and then returns 0 for the rest of the trial. Otherwise it returns 1.0 once the stack is shredded, the clip is off the stack and put down at rest away from the shredder and the grippers, and every fingertip is at least 3 cm from the slot (`success()` then holds for 1 s and ends the trial); and 0.3 with the clip off the stack and the stack still on the desk or in the shredder. With `TASK_SCORE_DIR` set, every step's state is written to `steps.jsonl` there.

The shredder runs in `_run_shredder()` once per control step:

- **The clip.** The clip is an articulation: a body with the fixed jaw and the handles, and a moving jaw on a prismatic joint whose drive presses it against the paper with up to 15 N (its spring). It holds the stack by friction. While a gripper's fingers are closed on the folded-up handle pair, across it, the drive opens the jaw by 4 mm, as squeezing a real clip's handles opens it. The clip counts as off once it has been lifted clear of the paper's edge.
- **The feed.** When the stack's lowest corner is 1.5 cm into the slot, the shredder starts: from then on the stack's velocity is set every control step so that it follows a steady 4 cm/s descent, squared up to the slot over the first five steps, with a force cancelling its weight. It never jumps, so a clip on it is carried along by its grip. Once its upper edge is 2 cm below the top, it has passed the cutters: it stops below the dark throat and strips appear in the bin's window.
- **The jam.** If any part of a clip goes below the top surface inside the slot's outline, the feed stops for good, the stack is held where it is, and the red jam light comes on.

## Runtime and seeds

- Demo: 476 steps (23.8 s simulated) to full score, about 2.5 minutes of simulation (about 0.3 s per step on the slowest 30 steps at 320 x 176) plus start-up.
- Dream-1: up to 1,200 steps (1 minute simulated).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop and deterministic.

## Files

- `task.py`: the objects, the shredder's and the clip's logic, the scorer and the scripted demo.
- `shred_usd.py`: the shredder, rack, invoices, clip and dish: meshes, colliders and the clip's joint, written as USD into `build/` when the task loads.
- `make_textures.py`: draws the textures in `textures/`.
- `scenekit.py`, `usdkit.py`: helpers that add USD bodies and articulations to a task file, and write meshes and materials.
- `steptime.py`: records each step's wall time.

## Sources

All meshes are built in `shred_usd.py` from boxes, cylinders, prisms and tubes. Both textures (oak and the printed invoice face, a generic layout with no names or logos) are drawn by `make_textures.py`. The word "INVOICE" on the invoice face is set in the DejaVu Sans font, which is under the Bitstream Vera Fonts license (the DejaVu Fonts License, https://dejavu-fonts.github.io/License.html); DejaVu's own changes are in the public domain. The font file is not included: `make_textures.py` loads it from the system, and only the rendered images are in `textures/`. No other third-party assets are used.

Sizes: the shredder's head and slot follow personal 10-sheet shredders with a 220 to 230 mm feed opening; its bin is made shallower than a real one (the machine is 12 cm tall instead of about 30 cm) so that the slot is within the arms' reach. The invoices are A4 sheets folded in three (210 x 99 mm). The clip is a standard 41 mm binder clip. The rack is a small oak desk letter rack, and the dish a desk clip dish.
