# Pull the comb from a set agarose gel without tearing a well, then set the gel on its tray in the gel box to run DNA

A 1% agarose gel has set in its 7 x 7 cm UV-transparent tray on the bench, an 8-well 1.5 mm comb still standing in it.
Behind it stands an empty, dry Bio-Rad Mini-Sub Cell GT base (black cathode at +y, red anode at -y); to the left, a
folded lab tissue. Instruction: **"Take the comb out of this gel without damaging the wells, then put the gel on its tray
into the gel box, ready to run DNA."** The robot must hold the tray down and draw the 8 rigid teeth (5.54 x 1.5 mm)
straight out of the soft gel, square across their 1.5 mm thickness to a few tenths of a millimetre and slowly; put the
comb down off the gel; take the tray by its two open ends (the base leaves no room beside it), carry it level, turn it
and set it flat on the gel stage with the wells at the black end (not stated); let go only once it rests there.

## Why it matters

Agarose gels are the everyday check on DNA, hand-poured with a comb in research and teaching labs (Addgene). Bio-Rad's
Mini-Sub Cell GT manual (tape casting): "5. Carefully remove the comb from the solidified gel. [...] 7. Place the tray
onto the leveled Sub-Cell base so that the sample wells are near the cathode (black)"; buffer goes in after (step 8),
so the tank is empty. IBI: "a gentle wiggling, upward motion"; Hoefer: "Partially lift and slightly tilt the comb at one end and then slowly withdraw it", "lift very slowly to prevent the gel from
breaking" and "Pulling the comb straight up creates a vacuum in the wells that may lift the gel out of the tray" (so both a straight pull and an end-first pull, tilted up to 20 degrees, are allowed);
Bio-Rad lists a deformed well as a cause of ragged bands. The tray's slots locate the comb loosely (1 mm play per
side here), so the robot keeps the teeth square. A lab protocol: "Don't let the gel slide off the tray!" Orientation is
scored (wells at red: at most 0.5) but is not the claimed difficulty; the tray starts across the tank, so either
90-degree turn costs the same. We found no gel-comb task in public lab-robotics suites (AutoBio, LabUtopia).

## Files

`task.py` (objects, the comb's hold and release, monitor, scoring), `geometry.py` (dimensions, layout, thresholds;
estimates marked "est."), `build_assets.py` (writes `assets/*.usda`; needs `pxr`), `scenekit.py`, `demokit.py`,
`demo.py` (demo and scorer checks), `dev/` (reach and clearance gates, run logs), `NOTES.md`. All assets are built from
scratch (no third-party meshes); the robot is yam_arena's.

```
source <ARENA_DIR>/yam.env && python build_assets.py
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    --scorer task_score --max-steps 3600 --policy ...
python demo.py task.py demo_out [--no-record] [--variant nohold|sideways|red|fast|yank]
```

## Dream-1 run

Dream-1 (dreamlabs/dream-1-beta) scored **0** with the exact instruction, `object_jitter=0`, 20 Hz, 320 x 176 cameras and
the 3,600-step limit (one run, not retried). After its 32 pre-roll steps it raised both arms over the bench and held
them near their start poses for the rest of the run: neither hand reached the comb or the tray, the comb stayed in the
gel and nothing reached the tank. 2,918 of its 3,600 action rows were beyond the arm's limits and were clipped. 75
inference requests, mean latency 0.63 s (median 0.40 s); 69 of them needed a second attempt, which opened a new API
session. 3,600 steps took 19 min 48 s with the GPU shared with another simulation. Video: `model_run.mp4`.

## Scripted demo

1. The right hand presses its two open fingertips on both side walls' top edges, 1 cm past the tray's middle (hand
   pitched 40 degrees away from the comb), holding the tray on the bench.
2. The left hand settles its measured tool point over the comb's bar to about 0.1 mm and its pads parallel to the bar
   (each round backs off and comes back, since the arm's joints stick about a millimetre short), and closes on it.
3. It draws the comb up at 3 mm/s with a +-0.5 mm, 4 Hz vertical wiggle until the hold gives (F_REL), holds its
   command while the hold fades, then draws the teeth out at 4 mm/s with the command at most about 1.6 mm ahead of the
   hand (a +-0.15 mm wiggle) until they are 2 mm clear, lifts the comb at 50 mm/s and lets it fall flat on the tissue.
4. The right hand takes the tray by its open ends (pads on the end faces, a light pinch), lifts it 2 cm, raises it over
   the rim and carries it across, turning it 90 degrees clockwise; it corrects the measured offset over the stage,
   lowers it the last 15 mm at 10 mm/s with the fingers beside the tray's ends over the buffer chambers, opens 5 mm,
   rises clear and goes home.

| Run (final code; checks change one thing) | Score |
|---|---|
| demo | **1.0**, success at step 1012 (51 s; limit 3,600 steps = 180 s); no zero condition; widest tooth offset 0.19 mm (d_tear 0.4), fastest tooth in the gel 9.2 mm/s (v_tear 30), tilts 0.25 / 0.10 deg, released at 1.09 N, end pinch 1.7 N, gel slide 0.3 mm |
| check `nohold`: the tray is never held | 0: the tray and gel (0.53 N, lighter than the 1 N hold) rose with the comb, 3 mm by step 560 (zero 1); as the unheld tray began to tip, 3 wells tore at ~520 and all 8 by 560 (zero 4) |
| check `sideways`: hand commanded 2 mm across the teeth over the pull's first second | 0: the teeth were pushed 0.40-0.50 mm across their wells, all 8 torn (zero 4); the rest of the demo ran on |
| check `fast`: pulled at 60 mm/s | 0: the teeth left the gel at 38-39 mm/s, all 8 wells torn (zero 4) |
| check `yank`: pulled at 150 mm/s | 0: the teeth left the gel at 52-55 mm/s (the hold and the arm limit it), all 8 torn (zero 4) |
| check `red`: the left arm carries the tray, turning it 90 degrees counter-clockwise | 0.5: comb out, seated, wells at the red end, no defect |

Media: `demo.mp4` (top camera large, both arm cameras beside it); `setup.png` (first top-camera frame, with a close-up
of the comb standing in the gel from a separate photo run with the left wrist camera).

## Scoring

Read every control step (20 Hz); wells 1-8 from the robot's left at the start. **0 for the run (latched)** if:
1. the tray rises, or the gel lifts off its floor, 3 mm or more while a tooth is in its well;
2. the gel slides 10 mm or more along the tray or leaves it (cracking is not simulated);
3. the tray falls 3 mm or more in free fall (below its highest point since a hand last held it: both pads of a hand on
   tray or gel with 0.3 N or more), tips past 30 degrees, comes to rest (0.5 s, not held) anywhere but the bench or the
   tank, or any part of the robot, the tray, gel or comb touches an electrode guard or wire (the tissue counts as elsewhere);
4. two or more wells are torn;
5. a pad or the comb presses the gel's top surface with F_DENT or more, or a pad presses a gel end face with F_END or
   more (contact normal forces resolved in the gel's frame).

**Torn well** (per well, latched; while its tooth tip is in its well, and on the step it leaves): the tooth's offset
across the comb's thickness from its well's centre line, anywhere along its length in the gel, exceeds d_tear; the
comb's tilt about its long axis exceeds theta_t; its end-first tilt exceeds theta_e (tearing the wells whose teeth are
still in); the tip's rise over one step relative to the gel exceeds v_tear (before or after release); or the tooth is
pushed 1 mm below its cast depth. All in the gel's frame.

**From the final 1 s** (lowest rung of its steps): **comb out** = every tip 5 mm or more above the gel's top (or off
the gel's footprint) and the comb touching none of gel, tray, tank; **seated** = every bottom corner within 1 mm of the
stage, footprint inside the stage outline, run axis along the tank within 5 degrees, both grippers 3 cm or more from
tray, gel and comb; **wells at black** = the wells end is nearer the black electrode; **defects** = one torn well, a
gel slide of 3-10 mm.

| End state (comb out) | No defect | One defect | Both |
|---|---|---|---|
| Seated, wells at black | 1.0 | 0.75 | 0.5 |
| Seated, wells at red | 0.5 | 0.25 | 0.25 |
| Not seated (on the bench, held or within 3 cm, on a wall, off the stage, askew) | 0.25 | 0.25 | 0.25 |

Anything else scores 0 (comb still in, any zero condition, nothing done). The trial ends once 1.0 has held for 1 s.

**Thresholds (provisional, set by the pilot).** The sim's tear, release and dent thresholds are estimates, to be set by
a pilot on real gels made to the recipe (filmed at 60 fps against a mm card): each midway between the largest
perturbation that left every well intact and the smallest that tore one; a rule whose perturbation never tears a well
is removed from the sim scorer. The real trial does not use them: it is scored from the wells, the gel and the video,
as below.

| `geometry.py` | Estimate | Pilot condition |
|---|---|---|
| `D_TEAR` | 0.4 mm | bar pushed 0.5 / 1 mm sideways |
| `THETA_T` | 4 deg | pulls rocked about the long axis |
| `THETA_E` | 20 deg | end-first pulls at 10 / 20 / 30 deg |
| `V_TEAR` | 30 mm/s | pulls jerked out in about 0.2 s |
| `PUSH_DOWN` | 1 mm | a press-down first |
| `F_REL` | 1.0 N (above the 0.53 N tray + gel) | comb pulled slowly on a 0.1 g scale |
| `F_DENT`, `F_END` | 3 N, 8 N | 10 mm pad on the top, 68 mm pad on an end face |
| `GEL_FRICTION` | 0.21 (tan 12 deg) | slide angle on an inclinometer |

The demo's end pinch (1.7 N in the sim) stays under F_END/2; 10 carries on real gels at that pinch must leave no
end-face dent.

## Real world

Layout and parts as in the task's setup (positions in cm from the midpoint between the arm bases): base centred at
x 38, y 0, long axis along y; tray centred at x 22.5, y -6, run axis along x, comb end toward the robot; tissue at
x 33, y +22. [M] sizes (sim estimates in `geometry.py`): stage length (about 7 cm between its casting-gate slots; the
manual casts only the 7 x 7 tray on it) and height, inner width, guards, tray outside and slots, comb bar and play.
Finger pads in nitrile covers, wiped each trial. Cost: about $500 new for a base kit with 3 trays and 3 combs; a used
Mini-Sub GT base, or any horizontal mini gel box that takes 7 cm trays, is the cheaper route.

Scoring (spec): **torn wells** from an overhead photo with a raking light over black card and a mm ruler (a crack of 1
mm or more from a well, a broken wall, a chip of 1 mm or more, a holed floor); a doubtful well gets 5 ul of 6x loading
dye from a P10 or P20 pipette under 3 mm of 1x TAE (a streak within 60 s = torn; drain and dry after). **Dents** 1 mm or
more in the same photo and a low side photo. **Slide** with a ruler. **Seated** without moving the tray: a 1 mm shim
from the chamber under each corner, a side photo at stage level against a ruler, an overhead photo. **Video** (60 fps
side camera with a mm card): drops, tilt, lift during the pull, guard contact, final hand distance.

## Simulation fidelity

Rigid PhysX, 240 Hz physics, 20 Hz control. The gel is a rigid 70 x 70 x 5 mm slab loose on the floor (friction 0.21)
with 8 pockets 0.5 mm wider than the teeth per side. **The comb's hold**: an invisible, frictionless catch rests on
each end of the bar, on a stiff vertical slide in the tray (an articulation joint); the pull it carries is read off the
slide, and at F_REL for 0.1 s the task code fades it over 0.3 s and parks it inside the side wall. An unheld tray and
gel (0.53 N) rise with the comb. The sim has no suction on the gel and the comb's hold acts on the tray, so the gel can't lift off the floor in the
sim; the real trial checks gel lift on the side video. The tank is fixed colliders (maker's outer size, estimated inner sizes); the guards and wire risers are a
separate body so contact with them is read. The YAM's static joint friction leaves the hand about a millimetre short of
a command and makes slow moves stick-slip, hence the demo's settling, wiggle and capped lead.

## Reset

Lift the tray out by its ends, discard the gel, wash and dry tray and comb (no dried agarose on the teeth), wipe the pad
covers. Pour the next gel: 1x TAE from 50x stock; 1 g agarose per 100 ml, melted in a microwave (swirl every 30 s) or
on a stirring hot plate, cooled to 55-60 C; 20 ml into the taped tray with the comb in; set 40 min; never chilled. With
3 trays and combs in rotation, one trial every 15 min; about $0.30 per trial.

Sources: Bio-Rad Mini-Sub Cell GT manual; Hoefer HE33 and HE-PLUS manuals; IBI QS-710 manual; Addgene, MIT OCW 7.15
and EcoEvoDevo lab gel protocols; Thermo Fisher electrophoresis troubleshooting.
