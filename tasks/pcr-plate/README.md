# Load a capped 96-well PCR plate onto a thermal cycler block, A1 back left; slide the cover forward, pull the lever down

An Applied Biosystems GeneAmp PCR System 9700 thermal cycler with its 96-well sample block stands on the table, front
toward the arms, powered off and cold. Its heated cover is slid back and its lever is up, so the block is open. To the
robot's left of it, a MicroAmp Optical 96-Well Reaction Plate (N8010560), capped with a MicroAmp 96-Well Full Plate Cover
(N8010550), sits in its Splash-Free 96-Well Support Base with well A1 at the back left. The instruction is:
**"Load the capped PCR plate onto the thermal cycler's block with well A1 at the back left, then close the heated
cover."**

To do it, the robot has to:
- lift the plate out of its base by its two long edges and carry it over the block without turning it;
- lower it square so that all 96 of its conical wells enter the block's 96 conical wells together (9 mm pitch),
  with A1 at the back left, and let go low, so the plate settles into the block instead of skating on it;
- slide the heated cover completely forward to its stop, and only then pull the lever down to its stop, which lowers
  the cover's platen onto the plate;
- do the closing only once the plate is down in the block, and never while still holding the plate.

## Why it matters

PCR is among the most common procedures in molecular biology, and every run starts with a plate loaded into a thermal
cycler, by hand in most research labs. The 9700's manual gives the procedure this task follows: place the plate on the
block "so that the well numbered A1 is located at the upper left corner", "Slide the heated cover forward" (with the
note "ensure that the cover is pulled completely forward"), then "Pull the lever down to engage the heated cover and
the sample tray". A forensic lab's 9700 SOP has the same steps ("Slide the cover forward over the samples and press the
lever down to seat the tubes into the block").

The task tests:
- **Blind alignment of a 96-feature array.** Once the plate is over the block its own deck hides both its wells and
  the block's. Half a pitch off, its wells stand on the ridges between the block's wells: the plate sits about 12 mm
  high and closing the cover on it would crush it. Only a plate lowered square and let go low settles in.
- **A two-stage instrument closure in its documented order**, gated on the part being engaged: a sliding carriage
  pushed home to its end stop, then a lever pulled through its travel.
- **Orientation.** A1 is shown by the printed labels and the cut corner; the plate starts the right way round, so a
  placement that turns it is an error.

The closest tasks we know of load tubes or buckets into a centrifuge rotor, seat a wafer on a chuck, or open and close
a cycler's lid (AutoBio); none has the robot mate a 96-well plate with a block blind and then close the cycler.

## Files

| File | What it is |
|---|---|
| `task.py` | the yam_arena task file: objects, the cycler's mechanism (end stop, interlock, latch, cam), the monitor and scoring |
| `geometry.py` | every dimension and the layout (estimates marked "est.") |
| `build_assets.py` | writes `assets/*.usda` from `geometry.py` (needs `pxr`, not the simulator) |
| `scenekit.py` | USD bodies and articulations for task files |
| `demokit.py`, `demo.py` | the scripted demo and its variants |
| `dev/` | build-time checks (reach maps, gate scripts, logs) |
| `NOTES.md` | build notes |

All assets are built from scratch in `build_assets.py` (no third-party meshes or scans). The robot is yam_arena's.

## Run a trial

```
source <ARENA_DIR>/yam.env
python build_assets.py                                    # once
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    --scorer task_score --max-steps 3600 --policy ...
```

## Dream-1 run

One trial with Dream-1 (`dreamlabs/dream-1-beta`):

```
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --scorer task_score --max-steps 3600 --store-frames --log-dir dream1 \
    --instruction "Load the capped PCR plate onto the thermal cycler's block with well A1 at the back left, then close the heated cover."
```

Dream-1 scored **0**. Neither arm took the plate: it stayed in its support base the whole trial, the block stayed
empty, and the cover stayed slid back (no visible movement) with the lever up. The left arm reached over the open
block (its camera looking into the wells), brought its nearly closed fingers up to the cover's handle bar without
taking it, and then moved round the cycler's front and left side; near step 3,000 its camera looked at the plate in
its base, but it did not go to it. Near the end the right arm came over the cycler's right side, beside the lever.
From about step 850 to 3,250 nearly every action row it sent lay outside the action bounds (2,364 of 3,600 rows).
The trial ran to the 3,600-step limit (180 s at 20 Hz): 6 min 42 s on an RTX 5090 (6 min 56 s with Isaac Sim's
start), mean inference latency 0.58 s per request (round trip; 0.33 s on the server), 75 requests of 48 rows. The run
used one continuous Dream-1 session (no retries). Video: `model_run.mp4`.

## Scripted demo

```
python demo.py task.py demo_out [--no-record] [--variant ridges|nudge_ridges]
```

The left arm does the whole task (the cycler stands in front of it):
1. It takes the plate from its base by the two long edges (pads on the semi-skirt, 80 mm commanded on the 85.5 mm
   plate), lifts it straight out of the base's wells, moves it back beside the cycler, raises it and carries it in
   from the side, its deck 17 mm above seated (the tips 4.7 mm over the block's top), corrects the plate's measured
   offset over the block and lowers it until the wells are in the block's wells, letting go 1.5 mm above seated.
2. Closed fingertips tap both ends of the plate down onto the block.
3. The hand takes the cover's handle bar with a 45-degree forward-down grasp and pulls the cover forward to its stop.
4. It pulls the lever down against its spring, the bar gripped firmly: through the first 45 degrees the hand turns
   with the lever; for the rest it holds its orientation and the bar turns between the pads. Under the load (up to
   20 N at the bar) the arm gives by up to about 5 cm, so the command runs 12 degrees past the stop and leads the bar
   3 cm downward; the lever reaches its over-centre point and latches.
5. It lets go, backs off and goes home.

Results (final code; the demo is recorded, the two checks of zero 2 are the same script with the changes listed):

| Run | What the script does | Score |
|---|---|---|
| demo | as above | **1.0**, success at step 1018 (51 s of the 3,600-step limit); no zero condition (highest deck corner on any step the cover or lever moved: 0.01 mm over seated) |
| check: ridges | lets go of the plate half a pitch off diagonally (on the ridges, 0.4 mm over them), then pulls the cover | 0 (zero 2 as the cover slides forward: plate 12.3 mm high) |
| check: nudge, then ridges | first pulls the cover 3 mm forward and lets go, then lets go of the plate on the ridges, then pulls the cover | 0 (zero 2 as the cover slides forward again) |

Media: `demo.mp4` (the top camera large, the left and right arm cameras stacked beside it); `setup.png` is the first
top-camera frame.

## Scoring

Read every control step (20 Hz). **The run scores 0 (latched)** if:
1. the plate falls more than 1 cm in free fall (its height drops more than 1 cm below its highest point since a hand
   last held it), comes to rest upside down, or comes to rest (0.5 s, not held) with its footprint's centre over
   neither its support base's top nor the block's top;
2. the cover is slid forward or the lever pulled down at any step (more than 0.3 mm or 0.5 degrees past where it last
   was, so a slow push counts too) while the plate's footprint overlaps the block's top with any deck corner 5 mm or
   more above its seated height (a plate on the ridges sits about 12 mm high);
3. the cover or the lever moves while a hand holds the plate (both its pads pressing on it with 0.3 N or more).

Otherwise, from the final 1 s (every step of it must reach the rung):
- **engaged:** all four deck corners within 1 mm of seated height; **oriented:** A1 within half a pitch of the block's
  back-left well (turned: A1 in the front-right well);
- **0.25:** engaged but turned 180 degrees, cover in any state;
- **0.5:** engaged and oriented, cover not fully forward (more than 1 mm from its stop);
- **0.75:** engaged and oriented, cover fully forward, lever not down to its stop (more than 2 degrees from it); or
  everything done but a gripper still within 1 cm of the plate, cover or lever;
- **1.0:** engaged and oriented, cover fully forward, lever down, both grippers (body and fingers) 1 cm or more from the
  plate, cover and lever. The trial ends once this has held 1 s;
- **0:** every other end state (on the ridges, outside the block, untouched).

## Real-world build

Positions in cm from the midpoint between the arm bases, x forward, y to the robot's left; heights above the table.

| Item | Details |
|---|---|
| Thermal cycler | Applied Biosystems GeneAmp PCR System 9700 with its 96-well sample block module (about 26 H x 28 W x 41 D cm; base 8.6 kg, block module 3.2 kg). Used units sell for about US$100-250; a for-parts unit is fine only if its cover slide, lever and 96-well block work; it stays powered off and cold. Front toward the robot, front edge at x 14, centred at y +16 (in front of the left arm). Cover slid back, lever up. |
| Plate | MicroAmp Optical 96-Well Reaction Plate N8010560: polypropylene, semi-skirted, 96 x 0.2 mL wells, 9 mm pitch, 127.8 x 85.5 mm. Empty. |
| Cover | MicroAmp 96-Well Full Plate Cover N8010550, pressed onto the plate. |
| Support base | MicroAmp Splash-Free 96-Well Support Base 4312063, centred at x 20, y +38 (to the robot's left of the cycler, beside its front half), the plate in it with A1 at the back left. |
| Cameras | the rig's own, plus a levelled close-up camera on a stand against the cycler's right side, its lens at seated height + 5 mm, looking across the block along the plate's long edges; nothing stands on the cycler's left side, where the plate is carried in. The image row at the lens height is marked once by sighting a plate set on 5 mm spacers on the block. |

**Measured on the unit before the trials** (estimates in the sim, `geometry.py`): block top height (sim 16.8 cm),
the cover's travel (sim 16.0 cm), the lever's travel (sim 90 degrees) and the force to pull it down on a seated plate,
with a spring scale hooked on the handle bar. The measured values go into this table.

- **The sim's lever:** the cam lowers the platen 5.8 mm (1 mm into a seated plate's caps, about 40 N on it); until it
  latches, a return stiffness pulls the lever back toward up (0.9 N m per radian: 0.7 N m at 45 degrees and 1.4 N m
  at the stop, i.e. 10 and 20 N at the bar, 7 cm from the pivot).
- **Estimate of the real lever force:** about 10-20 N at the bar, peaking just before it latches. Basis: no figure is
  published for the 9700; heated lids press a plate with roughly 100-250 N (VWR's UNO 96 HPL lid test: 120 N, a
  competitor's maximum, and 250 N, its own). About 100 N built up over the platen's few millimetres of travel is
  0.25-0.5 J over the lever's quarter turn, 0.16-0.32 N m (2-5 N at the bar) on average; the cam, the cover's
  springs, the over-centre catch and friction put the peak near the stop at several times that. It is an estimate,
  not a measurement.
- **The arm's pull:** the YAM's wrist joints (DM4310 motors, 7 N m peak) limit it to roughly 45 N at a bar 15 cm from
  them (the shoulder's DM4340s give 27 N m); in the sim, with the same effort limits, the arm pulls the lever down
  against the 20 N spring. The estimated 10-20 N is well under that.

Real-trial scoring:
- **Zero 2:** on every frame where the cover starts moving forward, no deck corner shows above the marked line
  (seated + 5 mm, at the camera's lens height).
- **After the trial:** lift the lever and slide the cover back; check the plate is seated (it doesn't rock when a corner
  is pressed) and oriented (photo of the cut corner and the A1 label); a photo of the cover fully forward and the lever
  down is taken before opening.
- **From the video:** drops, set-downs elsewhere, upside-down landings, and the cover or lever moved while a gripper holds the plate; hands-off (both grippers
  1 cm or more from plate, cover and lever) from its final frames.

## Simulation fidelity

Rigid PhysX bodies, 240 Hz physics, 20 Hz control.
- The block and the support base are fixed triangle meshes with 96 conical wells each, 0.3 mm radial clearance round
  the plate's wells; the plate seats with its well tips on the wells' bottoms, its deck 3.2 mm over the block.
- The plate (with its cap mat) is one rigid 25 g body: 96 convex well hulls, the deck, the semi-skirt (3 mm thick
  colliders: the real ~1 mm wall would let the pads sink in) and the caps. Plate flex is not simulated.
- The cover is a prismatic carriage with an end stop. The lever is a revolute joint on the cover's front top edge;
  it is blocked until the carriage is within 1 mm of its stop (a joint limit the task changes), the carriage is locked
  once the lever moves, and within 3 degrees of its stop the lever latches (over-centre). Until it latches, the
  lever's drive pulls it back toward up like a spring (0.9 N m per radian: 0.7 N m at 45 degrees, 1.4 N m at the stop;
  10 and 20 N at the bar; an estimate of the cam's resistance, see above), so a lever let go part-way springs back up. The platen is a sprung
  prismatic joint whose target follows the lever's angle (the cam); its force on the plate is not fed back into the
  lever (the lever's spring stands in for it).
- The 9700's outer size is the maker's; its inner sizes (panel, surround, block height, cover and lever) are estimates
  from the manual's drawings and stated in `geometry.py`.

## Reset between trials

Lift the lever, slide the cover back, lift the plate out by its edges (rock it gently if it sticks), and put it back
in the support base with A1 at the back left. Replace the plate or its cover (about $3) if it cracks or the cover lifts.
Nothing is used up. A for-parts 9700 is fine only if its cover slide, lever and 96-well block work (it stays off).

## Sources

- GeneAmp PCR System 9700 96-Well Sample Block Module User's Manual (Applied Biosystems, cms_041143): plate placement
  (A1 at the upper left), "Slide the heated cover forward", lever, MicroAmp plate configurations.
- GeneAmp 9700 base module manual (cms_041034): heated cover and lever.
- Maine State Police Crime Laboratory, "Operation of 9700 Thermal Cyclers" (SOP).
- Thermo Fisher product pages: MicroAmp Optical 96-Well Reaction Plate N8010560 (semi-skirted, 0.2 mL), MicroAmp 96-Well
  Full Plate Cover N8010550, Splash-Free Support Base 4312063.
- LabWrench listing of the 9700 (26 x 28 x 41 cm; 8.6 kg base, 3.2 kg block module); eBay listings (used units
  US$100-250).
- SBS/ANSI 96-well footprint (127.8 x 85.5 mm, A1 offsets 14.4 / 11.2 mm); USA Scientific semi-skirted plate drawing
  (17-degree well taper, ~20 mm height).
