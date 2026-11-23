# Fasten a 4-ft LED wrap light to the ceiling with a partner robot: keep it flush while each robot drives its own screw

Two bimanual YAM stations stand side by side under a plywood panel that serves as the ceiling, its underside 580 mm
above the table. The bare steel housing of a Commercial Electric WR484080LSL 4-ft LED wrap light rests with its back
up on two padded stands, 100 mm below the ceiling, under its pencil outline. Its lens and channel cover are off, as in
the maker's step 1. The bare housing is 1,218 mm long and weighs about 3.7 kg (estimate). Two pilot holes are drilled
in the ceiling, one at each end, above a diagonal pair of the housing's keyhole slots. Each station has a cordless
screwdriver standing on a holder, with a #8 x 3/4 in screw on its magnetic bit. The robots use each screwdriver in
push-to-start mode: the motor runs while the bit is pressed against a screw and stops when the push is released (its
on/off button is not used), so a hand drives a screw by holding the body still and pushing up; the body never turns.

Each robot is told, separately:

> With the robot beside you, raise the light off its stands and hold it flat against the ceiling inside its pencil
> outline, with a slot lined up under each pilot hole. Keep it flush while each of you drives the screw on your
> screwdriver up through the slot at your own end into its pilot hole until it is tight. Then let go.

Neither robot can finish it alone. Each reaches only its own end: the far end's slots are 910-960 mm from the nearer
of a station's two shoulders, beyond the YAM's 750 mm reach. The screws are driven up through the slots while the
light is held against the ceiling, the hold-and-drive method of the how-to guides below. Until a screw bites, only the
hands hold the raised light up. After the first screw bites, the light pivots on it unless the far end is held, so
each station holds its own end flush while its other hand drives.

## Why it matters

- Electricians (821,000 in the US, O*NET 47-2111.00) install fixtures as core work: "Work from ladders, scaffolds,
  or roofs to install, maintain, or repair electrical wiring, equipment, or fixtures."
- How-to guides describe this method:
  - Hunker, "How To Hang Fluorescent Light Fixtures": "You may want a helper to hold the fluorescent fixture flush
    with the ceiling while you drill in the screws."
  - Warehouse-Lighting, "How to Install a Fluorescent Light Fixture": "Flush Mount: Hold the fixture against the
    ceiling. Drill pilot holes into the studs (or larger holes for toggle bolts) and screw the housing securely into
    place."
- A maker's sheet for a 4-ft LED wrap, NICOR's WPC4: "place fixture against mounting surface and install four (4)
  screws into marked locations/anchor bolts."
- Home Depot's "How to Install a Light Fixture": "Use a second ladder and another person to help you. However, keep in
  mind that coordinating your movements can be difficult with a very heavy fixture."
- The maker of this light: "NOTE: Two people are recommended for the installation of this light fixture." Its own
  guide drives the screws in first, leaving a 3/8 in gap, and then hangs the light's keyholes on their heads.
  Hold-and-drive is the method in the how-to guides above and in NICOR's sheet; this task uses it, driving each screw
  up through a keyhole's narrow part.

Here each robot is one of the two people, at one end. The two policies cannot talk, and each sees only its own
cameras. What makes it hard:
- blind alignment: each slot must be lined up under a pilot hole hidden above the housing's back;
- keeping one end undisturbed while the other is fastened: the end not being fastened must stay flush and lined up
  under its pilot hole while the other robot pushes and turns its screw;
- precise position control overhead: each end flush with the ceiling within 2 mm;
- tool use: a powered screwdriver whose screw must go up through a 6 mm slot, pushed hard enough to start it.

## Files

| File | What it is |
|---|---|
| `task.py` | the yam_pair task file: objects, the screw model (task code), the monitor, `score()`, `report()`, `demo()` |
| `geometry.py` | every size, place and constant; each estimate is marked "estimate" |
| `build_assets.py` | writes `assets/housing.usda`, `assets/driver.usda` and `assets/screws.usda` (needs `pxr` only) |
| `scripted.py` | the scripted demo: one generator script per station, in its own frame |
| `task.json` | the submission fields, with both stations' instructions |
| `yam-pair-kit/` | the four-arm Inspect Robots plugin (`yam_pair` embodiment, `dream_pair` and `yam_pair_scripted` policies, `pair_task_score`, `yam-pair-video`) |
| `dev/` | offline reach and clearance checks (`layout_check.py`, `r2_check.py`, `build_check.py`, `holder_check.py`) and their outputs, smaller offline reach checks (`pick_check.py`, `reach_scan*.py`, `ws_sample.py`), `fetch_replay.py` (replays a station's script offline to its screwdriver pick) and `hold_torque.py` (static joint torques while an outer arm holds its end); `run.sh`, `checks.sh`, `summary.py` and `results.py` run the demo and the scorer checks in an Isaac Lab-Arena environment (`run.sh` wraps the `inspect-robots run` command below), `media.sh` renders a plain video and `setup.png` from a run's frames, and `overlay.py` re-renders `demo.mp4` with the status lines; `demo_d43.txt`, `repeat.txt` and `checks.txt` hold their results, and `runs/` each run's per-step monitor lines |
| `media/` (uploaded media, not in the repository) | `model_run.mp4` (the Dream-1 run), `demo.mp4` (the scripted demo) and `setup.png` |

All assets are built from scratch in `build_assets.py` and `task.py`. No third-party meshes, scans or textures are
used. The robots and stations are yam_arena's and yam-pair-kit's.

## Run a trial

This task runs on the four-arm `yam_pair` embodiment in `yam-pair-kit`: two bimanual YAM stations, each with its own
policy, cameras, state and instruction. Install the kit once into the Isaac Lab-Arena venv:

```bash
source <ARENA_DIR>/yam.env
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e ./yam-pair-kit
inspect-robots list | grep -E "yam_pair|dream_pair|pair_task_score"
python build_assets.py                      # once
```

Each robot's own instruction is in `task.json` (`instructions.a`, `instructions.b`). Both are the same text, and so is
`instruction`, the label the trial is logged under.

```bash
INSTR="$(python -c 'import json;print(json.load(open("task.json"))["instruction"])')"
# Dream-1 on both stations: two independent sessions, queried concurrently every step
inspect-robots run --embodiment yam_pair -E task=task.py -E overview=true --policy dream_pair \
    --instruction "$INSTR" --scorer pair_task_score --max-steps 6000 --store-frames --log-dir logs/dream
# the scripted demo
inspect-robots run --embodiment yam_pair -E task=task.py -E overview=true --policy yam_pair_scripted \
    --instruction "$INSTR" --scorer pair_task_score --max-steps 6000 --store-frames --log-dir logs/demo
yam-pair-video logs/demo demo.mp4           # 2x3 grid of the robots' cameras, plus the overview
```

The layout is fixed in the task file: `side_by_side`, 1.0 m apart. The overview camera is for the video only; neither
robot sees it. `success()` is never true, so a model's trial runs to its step limit and is scored on its end state.
The scripted demo ends when both scripts are done. The scorer checks below run the demo with `CLP_VARIANT=<name>` in
the environment. `dev/checks.sh` runs them all, one after another, and `dev/summary.py` tabulates them.

## The scripted demo

`yam_pair_scripted` runs `demo()` in `task.py`. Each station has its own generator script in `scripted.py`. A script
works in its own station's frame, reads object poses (privileged) and shares no state with the other. Times are from
the demo run.

1. **Take hold** (0-8 s). A's outer hand and both of B's hands pinch the end walls' rubber trim from below. Meanwhile
   A's inner hand takes A's screwdriver off its peg, gripping it 150 mm below the bit tip.
2. **Raise and press** (8-18 s). The three hands lift the housing off its stands in 5 s. Each hand runs a closed loop
   on the housing's measured height at its grip, because the arms sag under the load. Three millimetres under the
   ceiling, each outer hand lines its slot up under the pilot hole to within 0.4 mm. Then the three hands press it
   flush.
3. **A's screw** (20-54 s).
   - A's inner hand brings the screw under the slot, raises it through the narrow part, and sets its tip on the
     pilot hole, servoing on the screw's tip.
   - It pushes until the screw bites (41.7 s), then follows it in. It keeps pressing until the screw stops turning
     (46.6 s).
   - It lowers the driver and parks it, still in hand, clear of everything.
4. **B's screw** (48-98 s; B starts while A parks its driver).
   - B's inner hand passes its share of the load to B's outer hand over 3 s and lets go.
   - It takes B's screwdriver off the taller peg, gripping it 165 mm below the bit tip, and drives B's screw the same
     way: bite at 88.7 s, tight at 91.2 s.
   - B's outer hand holds B's end flush throughout. A's end is already screwed, and A's outer hand stays in place
     under it until B's screw is tight.
5. **Let go** (93-110 s). A's outer hand opens at about 93 s, 1.5 s after B's screw is tight, and B's at about 99 s,
   after B parks its driver; both go home. The light hangs on its two screws from 99.7 s. The trial ends at 110 s
   (step 2,200), when both scripts are done.

## Results

**Demo: 1.0.** Four runs of the demo all give 1.0, identical step for step: no line of the per-step monitor
differs. Both ends are fastened and tight, and the light hangs on its own from 99.7 s. There is no drop and no contact
between the stations.

| End | Bite | Tight | Clamp at the end | Head gap | Edge gap | Driver rotation in the gripper while driving |
|---|---|---|---|---|---|---|
| A | 41.7 s, (0.7, 1.0) mm off the hole, 5.2 N push | 46.6 s | 138 N | 0 (in contact) | 0.0 mm | 78 degrees |
| B | 88.7 s, (-0.6, -0.7) mm off the hole, 5.3 N push | 91.2 s | 131 N | 0 (in contact) | 0.0 mm | 43 degrees |

**Scorer checks.** Each check is the demo with one change (`CLP_VARIANT`).

| Check | Change from the demo | Expected | Score | End state |
|---|---|---|---|---|
| `nothing` | both stations hold still | 0 | **0** | on its stands, nothing fastened |
| `alone` | B holds still; A raises its end and holds it | 0 | **0** | A's end is raised, but its near edge stays 70 mm under the ceiling (the far end is still on its stand); nothing fastened |
| `drop` | raise and press flush, then the three hands holding it open | 0 | **0** | dropped at 17.4 s (a corner moving down at 0.43 m/s) |
| `unseated` | both screws stop after 6 of their 10.4 turns | 0.5 | **0.5** | both fastened; after the let-go the light settles onto both heads (edge gaps 8.6 and 3.7 mm), so neither end is tight; it hangs |
| `one_tight` | A seats its screw; B's stops after 6 turns | 0.75 | **0.75** | A tight (clamp 161 N); B fastened, its end sagged onto the head, 6.2 mm below the ceiling; it hangs |
| `one_end` | B never drives; all hands keep hold to the end | 0.25 | **0.25** | A tight; B's end held flush but not fastened |
| `still_held` | the demo, but both outer hands keep pressing to the end | 0.25 | **0.25** | both ends tight, but a gripper still touches the light |
| `slip` | B's gripper is commanded to close only to 33.3 mm on the 36 mm grip, so it barely squeezes | 0.25 | **0.25** | B's driver slips out of its fingers on the way to the slot, so B's screw is never driven; A's end is fastened |

**Per-arm peak loads in the demo.** The table gives each joint's peak torque from the arm's drives, with the motors'
peak-torque ratings (DM4340 joints 1-3, DM4310 joints 4-6), and each hand's peak net contact force on the housing.
That force includes sideways force: B's outer hand reaches 42 N late in its hold while its J2 peak stays at
17.8 N m.

| Arm | J1 N m | J2 N m | J3 N m | J4 N m | J5 N m | J6 N m | Hand on housing (net contact force), N |
|---|---|---|---|---|---|---|---|
| peak rating | 27 | 27 | 27 | 7 | 7 | 7 | |
| A outer (left) | 8.5 | 21.2 | 13.2 | 3.9 | 1.9 | 1.4 | 27.5 |
| A inner (right) | 1.6 | 11.3 | 12.1 | 4.2 | 1.5 | 2.2 | 0.0 |
| B inner (left) | 12.1 | 16.7 | 11.1 | 3.7 | 3.3 | 1.8 | 23.1 |
| B outer (right) | 12.6 | 17.8 | 10.3 | 3.6 | 3.2 | 2.3 | 42.0 |

- The highest joint load is A's outer shoulder (J2), at 79% of its peak rating.
- Sustained load: A's outer arm holds A's end alone for about 38 s (8-46.6 s), with 14-20 N of contact force, and B's
  outer arm holds B's end alone for about 47 s (51-98 s). The per-step log keeps only running peaks, and they bound
  these holds: J2 stays at or below 21.2 N m (A) and 17.8 N m (B). A's peak comes during the raise and B's at the
  hand-over, at the start of its hold, and neither rises after. A static estimate (`dev/hold_torque.py`,
  `dev/hold_torque.txt`) gives 11.2 N m at J2 from the arm's own weight alone, and 19.2 N m with 15 N pushing up. So J2
  holds about twice the DM4340's rated continuous torque (9 N m in most listings; sellers give 7-12 N m), within its
  27 N m peak. Counted above the arm's own weight, the hold adds about 8 N m, within that rating. The wrist J4 peaks at
  3.6-4.2 N m, against the DM4310's 3 N m rated and 7 N m peak. A slower policy than the demo would hold for
  minutes, not 40 s; the reset section gives the temperature rule.
- The peak push of bit on screw is 8.0 N (A) and 5.4 N (B).
- A's inner hand never touches the housing: it takes its screwdriver before the raise.
- Each end carries about half of the 3.7 kg housing (1.85 kg), within the YAM's 2 kg nominal payload. The outer
  hands' net contact force peaks at 27.5 N (A) and 42 N (B) (table), which includes sideways force.

**Media** (`media/`):
- `demo.mp4`: the demo, with the six station cameras in a 2 x 3 grid plus the overview, with status lines from the
  run's log burnt into the overview (`dev/overlay.py`): the light's state, and for each station its phase (ending
  with "has let go of its end") and its screw's state (on the bit, pushed on the pilot hole, has bitten, driving or
  motor stopped with the head's distance from the housing, tight). The screwdriver's body does not turn while it
  drives, and the 8 mm screw head is too small to see turning at this size, so the status lines show the driving.
  110 s, H.264, 5.3 MB. Both videos start at step 1: at step 0 the overview camera has not yet been placed.
- `setup.png`: the overview and both stations' top cameras at the start.
- `model_run.mp4`: the Dream-1 run, in the same layout as the demo. 300 s.

## Dream-1 run

Dream-1 (dreamlabs/dream-1-beta), one session per robot through `dream_pair`, each with its own three cameras, joint
state and the task's instruction, ran to the 6,000-step limit and scored 0. Robot B lifted its screwdriver off its
holder at about 30 s, dropped it onto the table twice (at about 57 s and 130 s), picked it up again each time, and
never raised it toward the light. Robot A's inner hand
brushed the light once at 9 s (2.4 N at most); no other hand touched it, the other screwdriver stayed in its holder,
and the light never left its stands. The robots never touched each other. `media/model_run.mp4` shows the whole run
(the overview beside both robots' three cameras). An earlier run, with an earlier wording of the instruction, also
scored 0.

## Scoring

Scored on the end state, except that a drop at any time scores 0.

- **Dropped:** at any time either end of the light comes down onto a stand, the table, a robot or the floor without a
  gripper holding that end. Real trial: the video. Sim: any corner of the housing moving down at 0.3 m/s or faster,
  more than 25 mm below where it is flush. A screw that has only just bitten leaves its end 17.5 mm low, so settling
  onto a short screw is not a drop.
- **End fastened:** that end's screw passes up through the narrow part of the housing's slot into its pilot hole and is
  driven in, so that end can't come down past its head. Real trial: look up into the open housing. Sim: the screw has
  bitten and its shank lies in the slot's narrow part.
- **End tight:** two checks.
  - Seated: a 0.5 mm feeler gauge blade does not go under the screw's head, and the end does not shift when pulled
    sideways, across the light, with 10 N (about 1 kg on a luggage scale), each way. Sim: head within 0.3 mm of the
    housing and clamping it with 50 N or more.
    - Why the pull test works: the head clamps the housing's back against the ceiling. With a clamp C at the head and
      the end's share of the light's weight W (about 18 N), the back presses on the ceiling with C - W. The shank has
      about 0.9 mm of play each way across the 6 mm slot (estimate; measure the real slot), so the end can slide that
      far unless friction holds it: μ1 C at the head (zinc head on the painted steel back) plus μ2 (C - W) at the
      ceiling (painted steel on plywood). With C at the sim's 50 N threshold and friction estimates μ1 0.15-0.3 and
      μ2 0.25-0.5, a seated end holds at least 0.15 x 50 + 0.25 x (50 - 18) = 15.5 N. A screw whose head only
      touches the housing carries just W, so its end slides at no more than 0.3 x 18 = 5.4 N. The 10 N pull lies
      between, with margin both ways. The result depends on the clamp and on friction, not on how hard the screw turns
      in the plywood.
    - Other checks considered: a downward pull can't tell a loose screw from a tight one, because the head holds the
      end either way; a torque check depends on the running torque in each hand-cut plug.
  - Flush: a 2 mm shim, pushed flat against the ceiling at the long edge of the back nearest the screw and in line with
    it, does not go in. Sim: that edge point within 2 mm of the ceiling. A's screw is 90 mm from the near long side;
    B's is 90 mm from the far long side.
- **Hangs on its own:** for the last 3 s neither robot touches the light. Real trial: the video. Sim: no contact over
  0.05 N between the housing and any gripper or fingertip.

| Score | End state |
|---|---|
| 1.0 | both ends fastened and tight; it hangs on its own |
| 0.75 | both ends fastened, it hangs on its own, one end tight |
| 0.5 | both ends fastened, it hangs on its own, neither end tight |
| 0.25 | at least one end fastened and not dropped, but less than 0.5: only one end fastened, or a robot still touches it in the last 3 s |
| 0 | dropped at any time, or no end fastened (never raised, held up but never screwed, or screws driven anywhere but through the slots) |

**Real trial**, right after the run, in this order:
1. Check the video for a drop and for the last 3 s.
2. Look up into the housing at each slot.
3. Try the 2 mm shim at each end's nearest edge.
4. Try the 0.5 mm feeler gauge blade under each screw's head.
5. At each end, draw a short pencil line (or stick a strip of masking tape and cut it through at the joint) up the
   end wall's outer face and on along the ceiling, in line with the light. Hook the luggage scale (or a cord loop on
   it) under the lower edge of the near long side wall, from inside, about 50 mm from the end, and pull straight
   toward the robots, across the light, to 10 N. Then hook the far long side wall the same way and pull away from the
   robots. The end is seated if the two halves of the line stay in line; a slide shows as a sideways step of up to
   about 1 mm.
6. A run the spotter stops is scored on its state at the stop (a light still in a hand does not hang on its own).

## Real-world build

Positions in mm from the midpoint between the stations: x forward from the arms, y toward station A, heights above
the table.

| Item | Details | Price |
|---|---|---|
| Light | Commercial Electric WR484080LSL 4-ft 8,000 lm LED wrap light, 4000K: Home Depot Internet # 313512347, store SKU 1005578285 (the 2023 manual calls it WR484080LSLV); HD Supply sells it as Precision P232165. 47.96 x 14.13 x 2.82 in and 10.4 lb with its lens (spec sheet). Lens and channel cover off (maker's step 1); the bare housing is estimated at 1,218 x 280 x 55 mm and 3.7 kg. | $101.67 (Home Depot, 2025-09-01); $139.99 (HD Supply, 2026-02-12). The prices are about 15 and 9 months old. |
| Edge trim | push-on rubber U-channel edge trim, 8 mm across, cut to four 80 mm pieces, on the end walls' lower edges at the grip spots. It protects the gripper pads from the sheet-steel edge. | about $8 |
| Ceiling panel | 23/32 in plywood, one 4 x 8 ft sheet (for example Home Depot "23/32 in. x 4 ft. x 8 ft. RTD Southern Yellow Pine Wood Sheathing Plywood"): panel 2,000 x 600 mm, feet, and offcuts for the plugs and caps | about $42 |
| Frame, stands, holders | four 2 x 4 x 8 ft studs: four legs to the panel's corners (580 mm under it), two stands (post plus 300 mm padded crossbar, top 425 mm). Two holder risers, each cut from a 4 x 4 post (89 x 89 mm, as in the simulation). Each riser is topped with a 22 mm square hardwood peg, 30 mm tall, turned 37 degrees about vertical so that its faces are square to the line from the inner arm's base. The peg tops are 190 mm up for A's driver and 205 mm for B's, and each driver stands freely on its peg. The 22 mm peg is narrower than the driver's 36 mm grip, so the fingers, closed on the driver, pass either side of it. A cup or recess around the driver's base would be in their way: B grips 30 mm above the base, and its finger bodies reach down to the base. | 4 x $3.44; a 4 x 4 x 8 ft post about $12 |
| Screwdrivers | 2 x Bosch GO (Gen 3, 0 601 9H2 201; a Gen 2 also works): 3.6 V, 306-396 rpm, 7 torque settings, push-to-start; each with its clutch at the lowest setting. The robots use each one in push-to-start mode: the motor runs while the bit is pressed against a screw and stops when the push is released (its on/off button is not used), so a hand drives a screw by holding the body still and pushing up; the body never turns. | £67.55 each incl. VAT (Bosch UK), about $88 |
| Screws and bits | #8 x 3/4 in pan-head square-drive wood screws (box of 100); #2 square bits. The light comes with Phillips wood screws; square drive cams out less when driven overhead. | about $10 |
| Plugs and caps | all cut from the panel's own sheet, so every plug is the same plywood (its offcuts give about 100). Each 50 mm disc is cut with a 2-1/8 in hole saw with its pilot bit removed, guided by a 2-1/8 in hole in a scrap board clamped over the offcut, so the disc has no centre hole (a circle cutter in a drill press also works). The disc is trimmed or sanded to fit the 2 in-saw hole; the cap's two screws fix its place. Before the cap goes on, the pilot hole is drilled at the disc's centre with a 1/8 in (3.2 mm) bit, square to the face in a drill press or drill guide, through the disc's full 18 mm (the screw's tip goes about 18 mm in, to the cap). The disc is then glued to the middle of a 100 mm square cap. | offcuts; the two hole saws about $25 |
| Foam, clamps, cap screws | 10 mm foam strip for the stands; four C-clamps for the feet; a few 1 in wood screws for the plug caps | about $30 |
| Scoring tools, small parts | digital luggage scale (0-50 kg, 10 g steps) and a cord loop; a kitchen scale (1 g steps); a pencil; 0.5 mm feeler gauge; a 2 mm plastic shim; torque screwdriver, 0.1-0.6 N m, 1/4 in hex (for example Wiha TorqueVario 0.1-0.6 N m), used only for the measurements before trials; 22 mm square hardwood stock for the pegs; screws for the frame; a drill guide | torque screwdriver $159.90 (Wiha US); the luggage scale about $12; the kitchen scale about $15; the rest about $25 |

**Total about $630, or about $470 without the torque screwdriver.**
- At each screw spot ((370, +508) for A, (470, -508) for B), the panel has a 2 in (51 mm) hole. A capped plug sits in
  it, its disc flush with the underside and its cap resting on the panel's top face. Two short wood screws through
  the cap into the panel's top hold it down, so the screw's push can't lift it, and the cap stops the hanging light
  from pulling it down.
- The pencil outline marks the housing's place: long sides at x 280 and 560, ends at y +/-609.
- To place the pilot holes: hold the housing on the outline and mark the middle of the two diagonal slots' narrow
  parts through it. Cut the 2 in holes centred on the marks, then fit the plugs. The coordinates above are estimates:
  the real slots decide.

**Measure before trials:**
- the bare housing's mass and size, and its keyhole slots;
- each screwdriver's slip torque at its lowest clutch setting: drive a screw until the clutch slips, then find the
  torque that turns it further with the torque screwdriver;
- each screwdriver's push-to-start force (press it on a kitchen scale);
- the torque and push that start and seat one screw in a plug (torque screwdriver, kitchen scale);
- the 10 N sideways pull, once with a loose screw and once with a seated one at the same end, to confirm that the
  loose end slides and the seated one holds, and that a loose end's slide is visible on the mark;
- each outer arm's coil temperature rise: hold the demo's outer-arm pose with a 2 kg load for 60 s and log T_Rotor.

Update the estimates in `geometry.py` with these measurements.

**Safety:**
- The light is never connected to power. Wiring (in the maker's step 4) is left out, as are the channel cover and
  lens (step 5) and the other two screws, which one hand can drive once the light hangs on the diagonal pair: the
  task covers mounting only.
- The padded stands stay under the light the whole trial. A drop from flush falls 100 mm onto them, or about 170 mm at
  an end that pivots on the far screw.
- A spotter stands by and stops the run if the light starts to come down, or if a shoulder coil passes 80 °C. A run
  the spotter stops is scored on its state at the stop.
- A drop is unlikely to ruin the light. Only the bare sheet-steel housing falls (lens and channel cover off), at most
  about 170 mm onto foam, and the light is never powered, so damage to its LEDs or driver would not affect later
  trials. The likeliest damage is a screw pulled out of its plug, which is replaced at every reset anyway, or a bent
  edge at a slot, which can be bent back. A spare light costs about $102.

## Simulation fidelity

Rigid PhysX, 240 Hz physics (12 substeps per 20 Hz control step). The screw model is task code in `task.py`
(`_substep()` runs before every physics substep; `on_step()` after every control step).

- **Screws.** Both screws are one articulation fixed to the world. Each screw is on its own chain of x, y and z
  prismatic joints and a spin revolute, all at its pilot hole.
  - Before it bites, the chain is free and the screw rides on the driver's bit. The bit has no collider: its contact
    with the head and its magnet are a stiff spring in task code between the bit's tip and the head. It pushes the
    head up with up to 60 N and pulls it back with at most 3 N per axis (the magnet), while the head is within 6 mm
    of the bit tip (released beyond 18 mm). The push also acts back on the driver.
  - The shank and head collide with the housing, so the shank must pass up through the slot's opening. The z joint's
    upper limit stops the tip at the ceiling.
- **Bite.** All of these hold for 0.1 s:
  - the tip is at the ceiling within 1.5 mm of its pilot hole;
  - the driver is within 8 degrees of vertical;
  - the bit pushes the head up with 5 N or more (estimate: the push that starts a screw in a 1/8 in pilot hole).

  Then the pilot hole takes the tip: x and y are drawn onto the hole's axis and held there. The z joint becomes the
  thread: its drive target follows the spin angle at 1.69 mm per turn (15 threads per inch, estimate), through a stiff
  implicit drive.
- **Driver.** A rigid body: a 36 mm grip, a collar and a #2 square bit; 0.31 kg.
  - After the bite, the head pushes back on the bit while the bit is pressed into it.
  - The motor runs only while the bit sits in the head (within 1.5 mm and 6 degrees) and pushes it with 3 N or more
    (push-to-start, estimate).
  - It turns the screw's spin joint at up to 360 rpm. The torque the screw needs is the thread drag (0.25 N m,
    estimate) plus the clamp through the thread, at 11.5% efficiency. That torque is capped by the 0.6 N m clutch
    (the lowest setting, estimate).
  - The equal and opposite torque is applied to the driver body. The driver is held only by finger contact friction,
    so it slips in the hand if the grip can't take that torque.
- **Seating.** It takes about 10.4 turns to go from the tip at the ceiling to the head on the housing's back (a
  1.5 mm collider; the real sheet is about 0.6 mm). After that the head clamps the housing harder with every turn,
  and the clutch would slip at about 150 N of clamp. In the demo each hand stops pressing once the screw stops
  turning, at 130-140 N. A real wood screw clamps much harder.
- **Not simulated:** electrics, the lens, the screw's thread surfaces and the wood itself. The bit's magnet and the
  thread are task-code drives, as above.
- **Estimates:**
  - the bare housing's size, mass and keyholes;
  - the torques, push forces and clutch setting;
  - the edge trim's size and friction (1.1 / 1.0);
  - the screw's thread pitch.
- **Scoring in the sim.** "Hangs on its own" sees only gripper and fingertip contact, so a forearm or a held driver
  against the light would not count as touching it. The drop check does not look for grippers, so lowering the light
  in hand faster than 0.3 m/s, more than 25 mm below flush, would count as a drop.

The scripted demo reads object poses (privileged); a model under test sees only its own station's three cameras.

## Reset between trials

1. Support the light, then back both screws out with the screwdrivers in reverse.
2. Set the housing back on its stands, with its back up, centred at x 420, y 0.
3. Swap in two fresh plugs, pre-drilled as above: unscrew each cap, lift the used plug out, drop a new capped plug
   into the hole and screw its cap down.
4. Put a new screw on each screwdriver's bit and stand each on its holder's peg.
5. Let the arms rest at least 10 minutes between trials. Each outer arm's shoulder holds about twice its rated
   continuous torque for 38-47 s in the demo, and longer under a slower policy (see Results). Before the next trial,
   read each joint motor's coil temperature, which the DM motors report over CAN (T_Rotor), and start only when
   every coil reads 50 °C or less. During a trial the spotter stops the run if a shoulder coil passes 80 °C. These
   are conservative choices, not maker limits: the DM-J4340's manual gives no number, and the manual for its smaller
   sibling, the DM-J4310, disables the motor above 100 °C (coil) or 120 °C (driver).

The reset itself takes about 3 minutes, plus the rest. Two plugs and two screws are used per trial; nothing else is
used up.

## Sources

- Commercial Electric WR484080LSL Use and Care Guide (Home Depot, thdstatic PDF 33f09c44): "NOTE: Two people are
  recommended for the installation of this light fixture."; step 1 (lens and channel cover off). Step 3, in full:
  "Position the fixture and mark the location of the mounting holes." / "Using a 1/8 in. drill bit, drill pilot holes
  at the marked locations." / "If mounting to wall studs, install the four wood screws (AA) directly into the mounting
  surface leaving a 3/8 in. gap between the screw head and the mounting surface." Step 4 then hangs the keyholes on
  the screw heads, slides the fixture and tightens the screws.
- WR484080LSL spec sheet (thdstatic PDF 934eb83f): "Weight (±0.5 lb) 10.41lbs"; "47.96 in x 14.13 in x 2.82 in".
- Home Depot product page 313512347 (Wayback, 2025-09-01): "10.4 lb", $101.67. HD Supply P232165 (Wayback,
  2026-02-12): $139.99.
- Hunker, "How To Hang Fluorescent Light Fixtures": "You may want a helper to hold the fluorescent fixture flush with
  the ceiling while you drill in the screws."
- Warehouse-Lighting.com, "How to Install a Fluorescent Light Fixture": "Flush Mount: Hold the fixture against the
  ceiling. Drill pilot holes into the studs (or larger holes for toggle bolts) and screw the housing securely into
  place."
- NICOR, WPC4 Surface Mount LED Wrap installation instructions (nicorlighting.com, 2025-10-07): "If mounting directly
  to surface, place fixture against mounting surface and install four (4) screws into marked locations/anchor
  bolts."
- Home Depot, "How to Install a Light Fixture" (2024): "Use a second ladder and another person to help you."
- O*NET 47-2111.00 Electricians: 821,000 employed (2025); "Work from ladders, scaffolds, or roofs to install, maintain,
  or repair electrical wiring, equipment, or fixtures."
- Bosch GO data sheet (Gen 2, Farnell 3270282): 0-360 rpm, 2.5 / 5 N m, "push machine forward or press 'on' button to
  begin". Bosch UK, GO Gen 3 06019H2201: £67.55 incl. VAT, "7 torque settings and maximum performance mode".
- Wiha US, SoftFinish TorqueVario 0.1-0.6 N m, 1/4 in hex, SKU 28415: $159.90 (wihatools.com, 2026-11-22).
- I2RT YAM store page (Wayback, 2026-10-22): "Nominal Payload: 2 kg | Max. workspace range: 750 mm".
- Damiao DM-J4340 listings: 9 N m rated, 27 N m peak (search snippets; see the quotes file). DM-J4340-2EC manual
  (foxtech): speed, position, torque and temperature reported over CAN, including the coil temperature T_Rotor; no
  temperature limit given. DM-J4310-2EC manual: disabled above 100 °C (motor) or 120 °C (drive) (search snippet).
- Home Depot, "23/32 in. x 4 ft. x 8 ft. RTD Southern Yellow Pine Wood Sheathing Plywood": $41.99 (search snippet).
- Exact quotes, URLs and fetch dates: `research/quotes_product.md`.

## Licences

Everything in this folder is the team's own work, apart from the short quotations from the cited sources.
`yam-pair-kit/` is the team's plugin. Isaac Lab-Arena, Isaac Lab, yam_arena, inspect-robots and the YAM URDF are used
as installed and are not included. No third-party meshes, scans, textures or fonts are used.
