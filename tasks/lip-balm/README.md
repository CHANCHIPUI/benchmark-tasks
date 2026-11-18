# Apply a resident's lip balm along both lips of a care manikin, only on the lips, then cap it and return it to the tray

A life-size vinyl practice head (a Mehron Practice Head) lies on a towel, turned toward the robot the way a dependent
resident's head is turned for mouth care: its face points toward the arms, 45 degrees above horizontal. The towel lies
on a board on a kitchen scale. Beside the head, on the overbed table's white tray, lies the resident's own lip balm
stick, capped. The instruction is: **"Uncap the resident's lip balm, apply it along the upper and lower lips from
corner to corner, only on the lips, then cap it and put it back on the tray."**

To do it, the robot has to:
- take the stick and pull its cap off, with two hands (the cap is a 4 N friction fit);
- trace the balm along each lip's red band (4 to 10 mm tall, about 5 cm corner to corner, curving round the mouth),
  from corner to corner, on a lip line that slopes 45 degrees and wraps round the mouth (the lips' surface turns
  through about 80 degrees from one corner to the other), keeping the tip pressed on but lightly: the scale reads the downward part of the press, and 100 g voids the
  trial (about 1.1 to 1.4 N where the lips face up, on the resident's left half; more toward the low corner, where
  they face the robot), and too light a touch leaves no balm;
- keep the balm off the skin (more than 2 mm outside the lips' border voids the trial), and keep its fingers and the
  cap off the face;
- keep the balm itself clean: it may touch only the lips and the inside of its own cap;
- push the cap back on until it seats, lay the stick back in the tray and let go.

## Why it matters

Home health and personal care aides (SOC 31-1120, about 3.7 million US workers) and nursing assistants
(31-1131.00, about 1.4 million) give dependent residents mouth care at least twice a day, and every 2 hours to those
who are tube-fed or take nothing by mouth. It ends with the resident's own lip balm. The main O*NET task is 2370
(personal care aides, 31-1122): "Administer bedside or personal care, such as ambulation or personal hygiene
assistance". Home health aides (31-1121) also have task 4253: "Massage patients or apply preparations or treatments,
such as liniment, alcohol rubs, or heat-lamp stimulation". Mouth care is a
scored skill on CNA certification exams, and US nursing homes must give dependent residents "personal and oral
hygiene" (42 CFR 483.24(a)(2)).

- A CNA exam study guide (Illinois, *Oral Hygiene, Denture Care, and Mouth Care for Unconscious Residents*): "Turn the
  resident to a side-lying (lateral) position with the head turned toward you so fluids drain out by gravity", then
  "Apply lip balm, reposition, raise the rail per care plan, and document"; "Mouth care is provided at least twice a
  day for most residents"; for residents who take nothing by mouth or are tube-fed, "Moisten and swab the mouth every
  2 hours; lips kept lubricated".
- SUNY, *Foundations for Assisting in Home Care* (CC BY 4.0), care of an unconscious patient: "Turn their head to the
  side." ... "Apply moisturizer to lips."
- The Royal College of Nursing, *Mouth Care during End of Life Care* (2023): "Apply a water-based gel or beeswax lip
  balm as needed."

The balm is the resident's own beeswax stick, used for that one person only and applied straight from the stick. Dry,
cracked lips are painful and bleed in residents who breathe through the mouth, take nothing by mouth or are on oxygen.

The task tests:
- **Tracing a narrow, bounded band on a 3D body surface.** Each lip is a curved band with an edge on both sides (skin
  above or below, the other lip at the join), on a head turned so the line from corner to corner slopes 45 degrees and
  wraps round the mouth. This is a trace corner to corner, not filling an area.
- **Light, sustained contact.** Balm transfers only while the tip is pressed on, and the scale under the head catches
  a downward press of 100 g (about 1.1 to 1.4 N where the lips face up, on the resident's left half; more toward the
  low corner, where they face the robot).
- **A clean tool and a two-handed cap.** The balm may touch nothing but the lips and its cap; the cap comes off and
  goes back on with two hands.

No task on the board applies lip balm or lipstick. The nearest are a nail polish task (a brush coating a flat 17 x 12
mm nail, no force limit), a temporal thermometer sweep (a sprung probe sliding across a broad forehead, nothing
deposited, no edge to stay inside) and a shaving task (covering a broad cheek area).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the head, the stick and cap, the scale, board, towels and tray; the monitor (scale, tip force, balm marks, contacts, the return), `success()` and `score()` |
| `geometry.py` | Dimensions and layout, the head's placement, the lips' traced border, the scoring constants |
| `build_head.py` | Builds the head from the MakeHuman base mesh and samples the lips (`assets/lips.npz`) |
| `build_assets.py` | Writes `assets/head.usda`, `assets/stick.usda` and `assets/cap.usda` (needs `pxr`) |
| `assets/` | The head (fixed), the stick (an articulation: tube and sprung bullet), the cap (free), the lip samples |
| `scenekit.py` | USD bodies, reset and spawn hooks for yam_arena task files |
| `demokit.py` | Helpers for the scripted demo |
| `demo.py` | The scripted full-score demo |
| `real_score.py` | Scores a real trial from the before and after photos |

## The head

The simulated head is the MakeHuman base mesh (hm08), released as CC0 in September 2020 by its copyright holders
(Data Collection AB, Joel Palmius, Jonas Hauquier; github.com/makehumancommunity/makehuman, `makehuman/data/3dobjs/base.obj`).
`build_head.py` cuts it at the neck, smooths it once (Loop subdivision), closes the neck, adds its eyes and scales it to
a 52 cm head circumference, the Mehron Practice Head's listed size. Like the practice head, it is one skin tone
throughout, with sculpted closed lips.

The lips' red border is traced once on the mesh (`geometry.py`): on the upper lip, the ridge of the white roll; on the
lower lip, where the surface turns under toward the chin. In the simulated head the upper lip is 6 mm tall at the
middle and the lower lip 10 mm, 48.6 mm corner to corner. Its lips' surface turns through about 80 degrees from corner to
corner: at the resident's right corner it faces the robot almost level, at the left corner it faces almost straight up.

## Run a trial

```bash
source <ARENA_DIR>/yam.env
inspect-robots run --embodiment yam_arena --policy <policy> -E task=task.py -E object_jitter=0 -E control_hz=20 \
    --scorer task_score --max-steps 3600
```

The assets are prebuilt. To rebuild them: download the MakeHuman base mesh (the command is at the top of
`build_head.py`), then run `python build_head.py build/dl_makehuman/base.obj` and `python build_assets.py`.

## Scripted demo

```bash
python plan_strokes.py              # once: writes assets/stroke_plan.npz (prebuilt)
python demo.py task.py demo_out
```

1. The left hand lifts the capped stick from the tray by its cap and holds it level. The right hand takes the tube's
   base end in a pen grip, fingers along the stick, as a person holds a lip balm. The left pulls the cap off along the
   axis (the fit lets go once the pull has been 4 N or more for 0.1 s with the hands drawing apart) and holds it clear.
2. The right hand turns the stick tip-down over the mouth.
3. It traces the upper lip, then the lower lip, corner to corner, then goes back along the lower lip nearer its outer
   edge (the lower lip is 10 mm tall; one pass of the 6 mm footprint covers about 60% of it), then makes a short second
   pass over the first 40% of the upper lip, its thinnest regions. The stick is kept within 10
   degrees of the lip's local inward normal, so it turns through about 70 degrees along each lip; `plan_strokes.py`
   picks the hand's turn about the stick at each point so the arm can reach it with its links at least 4 mm from the
   face. The press is held near 0.4 N from the measured tip force, and the contact is kept on the planned line (and
   moved further in across the lip if it comes within 4.5 mm of the skin).
4. It turns the stick level out at the front right, clear of the face, and brings it back where the cap came off; the
   left hand turns the cap into line and pushes it home (held in line in its last 1 mm, a push of 1 N or more seats
   it).
5. The left hand lets go and withdraws; the right lays the capped stick in the tray from 2 cm over its floor and lets go.

The red dots on the lips in the demo video are display only: they show the lip points the monitor has marked with balm
(and would show any marked skin), as the tinted balm shows on a real head, and have no collider, rigid body or mass
(`LB_SHOW_BALM=1`, which `demo.py` sets unless `--no-record`; scoring is the same without them).

## Scoring

Every control step (20 Hz), the monitor in `task.py` reads:
- **the scale:** the vertical contact force pressing the head down (PhysX contact report on the head), averaged over
  the last 0.25 s;
- **the tip:** the contact force between the bullet and the head;
- **the balm:** while the tip presses the head, the sample point nearest the dome's contact; lip points within 3 mm of
  it are marked when the press is 0.3 N or more, and any skin point within 3 mm, at 0.1 N or more, is balm on the
  skin. A dome touching the head away from the mouth is on the skin;
- **contacts:** any force on the head other than from the stick (any part of the robot, or the cap); any force on the bullet other
  than from the head, unless the bullet is inside the cap's bore; the bullet's dome reaching into the cap's rim or
  outside (a geometric check: the cap's collider passes through the stick);
- **the stick and cap:** their lowest points against the table and the tray's outline; the cap's fit (on or off, see
  Simulation fidelity) with its rim along the tube's axis, off the axis and its tilt; the arms' collision hulls'
  distance from the stick; the stick's and cap's speed.

| Outcome | Score |
|---|---|
| The scale rose 100 g (0.98 N, 0.25 s average); balm on the skin outside the 2 mm band; the bullet touched anything but the lips, the band or the cap's inside; any part of the robot, or the cap, touched the head; the stick or cap touched the table outside the tray or fell off it, at any time | 0 |
| Otherwise | 0.4 x coated upper regions / 6 + 0.4 x coated lower regions / 6 |
| ... plus, with at least one region coated, for the last 1 s: the cap seated (within 1 mm of fully on), the stick in the tray, both arms (any part) 3 cm or more from stick and cap, the stick still (under 2 mm/s) | + 0.2 |
| All 12 regions coated and the stick returned, held 1 s | 1.0 (success) |

Each lip, from 3 mm inside one corner to 3 mm inside the other, is split into 6 equal widths across the mouth (as seen along the face's normal); a
region is coated when 50% of its sample points (every 1 mm) are marked. A stick still held at the end, or left
uncapped, scores its coating only. Doing nothing scores 0.

## Real-world build

| Item | Notes |
|---|---|
| Mehron Practice Head | Life-size vinyl make-up practice head, about $45, 0.73 kg, one skin tone, sculpted closed lips; if its neck or base is longer than described, it must still lie wholly on the board (the sim head leaves 22 cm of board beyond its neck) |
| Burt's Bees Tinted Lip Balm, Red Dahlia, 4.25 g | About $5; the tint only makes the balm visible in photos |
| Digital kitchen scale | 5 kg, 1 g steps, display on its front face below the platform, auto-off 5 minutes or longer |
| 6 mm plywood board, 30 x 50 cm | On the scale |
| Non-slip mat (rubber shelf liner), 30 x 50 cm | On the board, under the folded towel, so the towel and head can't slide |
| Two white cotton hand towels | One folded to 25 x 45 cm on the mat, one rolled to 5 cm across behind the head, so the head can't roll back |
| White plastic tray, 12 x 20 cm, 1.5 cm rim | The overbed table's top; not on the scale, 1 cm clear of the board |
| Phone on a tripod | Before and after photos, 30 cm along the face's normal, fixed light |

All of it costs under $80. The grippers touch only the plastic tube and cap.

**Each head, once:** on a clean baseline photo, mark the two mouth corners and the nose tip and trace the lips' red
border (over the upper lip, along the join, under the lower lip) into a template for `real_score.py`, with the
photo's scale from a ruler held at the lips.

**Each new stick, once:** draw one stroke along the lower lip at a scale rise of 30 to 60 g; one stroke must tint 60%
or more of every lower region at a CIE76 colour difference of 8 or more. Clean the lip. If the tint is fainter, use a
deeper shade of the same balm.

**Each trial:**
- Before: clean the lips; photograph them (baseline); wind the stick out to 4 mm and cap it; switch the scale on with
  nothing on it, then load the board, mat, towels and head, and note the gross reading (about 1.6 kg). Never tare: a
  kitchen scale's zero tracking acts only near zero, so it cannot hide a rise from a gross reading. The real scale
  weighs any press on the board, the towels or the head (the sim reads only the head's load), so the real rule is
  stricter.
- During: film the scale's display and the face.
- After: photograph the lips from the same spot, mark the after photo's two mouth corners and nose tip, and run
  `python real_score.py template.json before.jpg after.jpg --after-points x,y x,y x,y` (add `--before-points` if the
  before photo is not the template's own). It aligns each photo by its own three points, counts tint (colour
  difference of 8 or more) in each of the 12 regions, and reports the largest tinted patch on the skin within 12 mm
  of the lips, outside the 2 mm band: any tinted spot of 1 mm2 or more there scores 0. Farther out, check by eye. A
  person can confirm each call by eye on the two photos side by side.
- Zero conditions: a rise of 100 g or more on the display at any moment; any part of the robot, or the cap, touching the face; the
  bullet touching the tray, the towel or a gripper (inspect the pads, the tray and the towel for tint after the
  trial); the stick or cap on the table or off it.
- Cap seated: a 1 mm feeler does not enter under it.

## Reset between trials

- **Lips:** clean with a make-up remover wipe, then soap and water, then dry (about 2 minutes). Take a new baseline
  photo every trial, so faint old stain does not count. If a stain stays (a region differs from the new head's photo
  by a colour difference of 8 over 5% of its area), replace the head.
- **Stick:** wind it to 4 mm; replace it after 20 trials or when its tip is flattened or broken.
- **Tray and pads:** inspect for tint and wipe with isopropyl alcohol.

## Simulation fidelity

- Rigid PhysX bodies, 20 Hz control. The head is fixed (kinematic, a triangle-mesh collider); the scale reading is the
  vertical contact force on it, as a stiff scale reads it.
- Wax transfer is a rule, not simulated material: while the tip presses the head, lip points within 3 mm of the
  contact are marked at 0.3 N or more, and skin points within 3 mm at 0.1 N or more. The 3 mm radius (a 6 mm footprint
  for the domed 11 mm tip at a light press) is estimated.
- The bullet sits on a 1 N/mm sprung slide (3 mm of travel) in the tube; the slide stands in for the give of the vinyl
  lip and the towel, and gives the tip's force.
- The cap's friction fit is held in code, not by contact or a PhysX joint (a PhysX breakable joint breaks on the total
  joint force, so squeezing the cap, the stick's weight or a knock broke it early). While the cap is on, the part not
  being held follows the held one rigidly (the cap the tube, unless only the cap is held), its pose and velocity
  written before every physics substep. The follower is made heavy (5 kg) and weightless while the fit is on (both
  parts while a hand holds the cap), so fingers closing on it or pulling on it meet resistance; the force the fingers
  put on the cap is read from the change of its velocity over each substep (mass x change / time), averaged over the
  control step.
- A hand holds a part when its pads are closed below the part's 16 mm diameter and each pad presses on that part
  with 0.3 N or more (PhysX contact forces); it keeps hold while its pads stay within 16.5 mm and each still presses 0.3 N.
- The cap comes off only when one hand holds the cap, the other the tube, and the axial pull on the cap (that force
  along the stick, away from the tube) is 4 N or more for two control steps (0.1 s) while the hands draw apart along
  the stick (a hand twisting a seated cap as it pushes can read a brief outward force). Squeezing has no axial
  component, the stick's weight does not load the fit, and a knock or a pull with one hand, or hands not gripping,
  cannot take it off.
- Off, it goes back on when a hand brings it into line (within 1 mm and 5 degrees) into its last 1 mm, having come
  onto the neck without jamming (never 2 mm or 10 degrees off line over it): it is then held there in line, and
  seats when pushed on with 1 N or more (drawn back with 1 N for 0.1 s, or let go of, it is off again). Seated is that
  state. Each time the fit lets go, the freed cap stays heavy for 0.4 s, so it does not shoot out of a squeezing hand.
- The cap's collider is solid (its own 16 mm outline, 2 g) and passes through the tube and the bullet, so the fingers
  cannot sink into a thin wall and carry the cap off. The bullet touching the cap's rim or outside is found
  geometrically (the dome's sphere reaching into the cap's body anywhere but past the rim inside the 7.6 mm bore).
- The tint only makes clear balm photographable.

## Results

| Run | Score | Notes |
|---|---|---|
| Scripted demo (`demo.py`) | 1.0 (success at step 2,164 of 3,600) | All 12 regions coated (50% needed): upper 67/61/90/71/72/54%, lower 83/75/73/55/64/73%. No zero condition: scale peak 51 g (limit 100 g), tip peak 0.86 N. The cap came off at an axial pull of 4.40 N (4.13 N, then 4.40 N: two control steps at 4 N or more, the hands drawing apart) and seated at a 1.04 N push, held in line 0.8 mm short of seated; the stick was laid in the tray and let go of. |

**Dream-1** (`dreamlabs/dream-1-beta`, the instruction above verbatim, `-E object_jitter=0 -E control_hz=20 -E cam_width=320
-E cam_height=176 --max-steps 3600`): refused before acting, so it has no score and no video. The harness held the arms
at their start pose for its 32 pre-roll steps; the first inference request was then refused with
`PolicyError: DreamAPI refused the inference request: DA-4000 BAD_REQUEST (malformed): unsafe instruction, refused:
Applying lip balm directly to a person's lips risks physical contact/injury.` No inference completed (so there is no
latency to report) and the trial ended 21 s after it started, at step 32 of 3,600. It was one Dream-1 session, not
retried. Under the task's rules a refusal, like doing nothing, scores 0.
