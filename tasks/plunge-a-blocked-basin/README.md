# Blocked basin

Instruction: **Clear the blockage so this basin drains normally, then leave the drain unobstructed.**

This is a household pressure-plunging task. The cup must contact the bowl, and a leaky cup or overflow dissipates pressure. The score measures restored drainage through the waste and removal of the cup from the opening. It does not measure grip, arm count, stroke count, pressure peaks or a prescribed order.

## Reproducing the scene

Coordinates are metres in the standard station frame; all heights below are above its unchanged tabletop.

- Ceramic basin: nominal external dimensions 350 × 280 × 140 mm, 95 mm bowl depth. Centre `(0.18, 0)`, bowl floor at 65 mm and rim at 160 mm. The normal back apron is on two white-painted steel supports, each 40 × 60 × 20 mm, leaving a cavity beneath the bowl.
- Waste: centre `(0.18, -0.04)`, bore 32 mm. The neck is a 13 mm-long clear PMMA tube (32 mm bore, 38 mm outside diameter) from 52 to 65 mm. The waste fitting is a chrome 1 1/4 in (32 mm) slotted grid basin waste (e.g. Wirquin slotted grid basin waste). Its cross grate has 3 mm-wide ribs just inside the neck top and is held by one central screw; the robot cannot remove it, and the person removes it only to reset the mat. The transparent PMMA inspection chamber has a 44 mm interior diameter, 50 mm exterior diameter and a floor at 19 mm. The transparent PMMA outlet tube has a 32 mm bore and 38 mm exterior diameter; its centre is 38 mm above the table. Its lower outer surface passes 1 mm above the receiver wall and the outlet mouth empties inside the receiver.
- Mat: a wet 6.0 ± 0.5 g mat of black Kanekalon modacrylic braiding hair and Ivory soap solution, 32 mm across and 8 mm thick, centred 56 mm above the table: its lower face is level with the neck's lower end and its top is 5 mm below the neck's top. It bridges the neck below the grate. It is made and set to the simulated resistance as described in *Making and resetting the blockage*. Its upper face leaves the 32 mm neck after 8 mm of downward travel; the simulated resistance acts over the first 10 mm (`PLUG_LENGTH`), allowing for fibres dragging at the neck exit, and the bore counts as open after 10 mm. The guide's upper and lower stops correspond to the grate and chamber floor.
- Plunger: nominal mouth 120 mm and total length 160 mm; corrugated plastic body, rubber lip, total simulated mass 79 g. It starts at `(0.18, -0.095)` with its mouth offset from the waste. The lip has sixteen independently compliant sectors with a rounded 8 × 6 mm rubber cross section. Each sector samples the real lower-profile contact line at both angular ends and its midpoint; a single contacting corner cannot seal the whole sector.
- Overflow: a genuine 30 mm circular opening in the sloping back panel, centred approximately `(0.18, 0.1338, 0.139)`. As in an ordinary ceramic basin, the integral overflow channel runs down inside the back wall to the waste hole and joins the waste bore through the slotted waste body's slots, just below the grate and above the clear neck and the mat; the model's overflow branch therefore connects to the same pressure node as the waste above the mat. A 50 × 7 mm rubber cover with a normal pull ring starts at `(0.16, 0.285)`. Either cover face may seal it. This supplied rubber cover is the modelled flexible sealing surface; rigid grippers have no added sealing membrane. The score itself imposes no arm or grip requirement.
- Receiver: transparent PMMA shallow tray, outer diameter 340 mm, inner diameter 332 mm, overall height 18 mm; bottom at 4 mm. Five graduations represent 200 mL increments. Its 15.6 mm full-charge waterline remains below the 22 mm lower edge of the pipe mouth.
- Fixed tap: chrome-plated metal, 18 mm tube diameter, 70 mm above the basin rim and 60 mm spout reach; it remains off. The waste grate and cover pull ring are chrome-plated metal.
- Charge the initially empty system with a measured 1.00 L of clean water. Water filling the overflow branch is part of this charge. During each reset, pre-wet the cup and restore the upright offset start; no separate water dose is added. The waste remains unsealed and the mat remains blocked. After settling, the cup lip is submerged by approximately 25 mm of water. Reset the mat, charge, empty receiver, plunger and cover between trials. The only chemical is household bar soap.

The scripted demonstration produces ordinary arm joint commands. It never moves objects, sets blockage positions during a trial, or installs a grasp constraint.

## Mechanical and hydraulic approximation

`basin_scene.py` generates the original meshes. The fixed basin and plumbing use matching concave triangle-mesh colliders. Dynamic parts have separate convex colliders. The cover pull ring uses a 22 × 7 × 16 mm grasp-envelope box: the opposing outer faces grasped by broad finger pads are within approximately 1–2 mm of the metal-wire envelope, but collision fills the ring hole. The ring is not used for insertion or threading. The mat's one-dimensional guide represents confinement in the short neck. Once it exits, pressure equalises around it and its low damping represents motion in the wider chamber. The mat remains visible and receives collision forces.

The plastic bellows has an axial stiffness of 350 N/m and damping of 4 N·s/m. Lip segments have 250 N/m stiffness and 1.2 N·s/m damping. The effective displaced area is derived from the authored bellows profile, approximately 0.00656 m². Total pressure reaction uses the actual 112 mm contact-line area, 0.009852 m² (the rubber bead has a 120 mm outside diameter); 2 kPa therefore produces approximately 19.7 N. Handle and rim forces use the same area coefficients as the volume-change calculation, preserving their mechanical work relationship. The sealed overflow cover also receives pressure force and the corresponding torque about its centre of mass.

The wet mat uses 1.20 N breakaway and 0.50 N sliding resistance, plus 30 N·s/m resistance while confined. These are mild-blockage values, not manufacturer measurements; every real mat is set to them with a push gauge before its trial (below). Its 6 g wet mass and approximately 5 mL displaced volume include buoyancy. The corrugated part encloses approximately 335 mL over its 51 mm working height; integrating its profile gives the 0.00656 m² volume derivative. Including the fixed lower cavity, the uncompressed internal cup volume is approximately 394 mL. The nominal seated geometry at approximately 25 mm basin depth gives roughly 220–240 mL of combined air space above the local liquid surface and in the unfilled branch. The fixed 230 mL air-equivalent compliance is an engineering estimate, not a separately measured or calibrated reset charge. Initially the model holds one connected 1 L basin inventory; when the working cup encloses the waste, its liquid inventory is captured from the measured geometry without changing the outside waterline. Air mass, individual bubbles and the wetting transient are not simulated.

Pressure and reaction forces update at the native 240 Hz physics cadence using freshly updated rigid-body positions. All hydraulic velocity estimates come from actual position increments, including those used by the force predictor; solver velocity reports do not create displaced fluid volume. An incremental backward-Euler predictor includes the actual handle/rim mass and the authored bellows/lip spring and damping. All three pressure-loaded objects use the same predicted node pressure for that step. After the step, the measured rim/handle displacement and actual mat displacement correct the authoritative gas pressure and hydraulic transport. The predictor therefore cannot accumulate fictitious displaced air volume. This partitioned coupling has finite force-prediction error; it does not increase physical mass, air volume or damping, or clip normal pressure forces. `check_coupling.py` measures pressure/volume consistency, energy and time-step convergence with free, stiff-gripped and releasing-contact mechanics; the GPU records pressure prediction error, actual displacement balance and node work separately.

`hydraulics.py` separately tracks water outside the cup, finite liquid inside the cup, water filling the overflow branch, water occupying the space vacated by the mat, the small sump, transport to the outlet, collected water and exterior loss. Pipe transport takes 0.35 s. The approximately 4.56 mL sump below the outlet remains in the chamber. Air uses a nominal density of 1.2 kg/m³ and water 1,000 kg/m³. The finite cup inventory prevents an open air path from draining the surrounding basin through a perfectly sealed rim. Air venting never increases collected or spilled water. A lifted cup drains through its downward-facing mouth; its projected release area determines return to the basin, uncredited water in the receiver, or exterior spill. An upward-facing mouth retains water. There is no capacity clamp that moves water through a sealed wall. A normal overflow jet falls back into the bowl; exterior loss is counted only after the water branch fills and the jet trajectory clears the basin rim. Jet dispersion, free-surface sloshing, foam, ceramic fracture and full continuum fluid/soft-body dynamics are not simulated.

The outside free surface follows the bowl's volume/height relationship and excludes the occupied cup region. Merely reclassifying water into the cup leaves this level unchanged. The authored cup section is integrated; using the mean bowl floor below the mouth introduces at most 1.2 mL error when centred and 4.42 mL over the admitted 33 mm centre-offset range, checked against the actual floor triangles. Partial filling, gas/liquid interfaces and pipe transit are lumped approximations, not a full two-phase continuum solver. The pressure model, water amount and visible mat displacement determine drainage; there is no stroke counter or automatic clearing event.

## Making and resetting the blockage

The blockage is hand-made, so its resistance is not taken on trust: each mat is measured in the neck and adjusted to the simulated values before its trial.

**Materials**

| Item | Example product | Use |
|---|---|---|
| Fibre | Kanekalon modacrylic braiding hair, black (e.g. Outre X-Pression Pre-Stretched Braid, colour 1B) | 1.50 g per mat, cut into 25 mm lengths |
| Soap | Ivory Original bar soap | 5% solution: 5 g finely grated, dissolved in 95 g of water at 40 °C; use at 20–25 °C; make fresh each day |
| Scale | Any 0.01 g pocket scale | Fibre portions and wet mass |
| Push gauge | 10 N digital push-pull gauge, 0.01 N resolution, track mode (e.g. HF-10, about US$60) | Breakaway and sliding force |
| Push foot | 30 mm disc (3 mm acrylic or a 30 mm bottle cap) glued to the gauge's flat head, with a stop collar (a large washer taped to the gauge's extension rod) | Pushes the whole mat face; the collar halts the foot 5 mm below the neck top |
| Tweezers | 200 mm straight tweezers | Removing the used mat |

**Making and setting a mat (about 5 min, including the 3 min rest)**

1. Before the session, weigh 1.50 g portions of cut fibre, one per trial.
2. Soak one portion in the soap solution for 30 s. Squeeze it to 6.0 ± 0.5 g wet and roll it into a disc about 35 mm across.
3. With the grate off and the basin empty, press the disc into the neck until its top is flush with the neck's top. Let it rest 3 min.
4. Set the gauge to track mode and film its display with a phone. Push the foot steadily to the stop, 5 mm in 2–3 s. The mat's lower face is then level with the neck's lower end, which can be checked through the clear chamber.
5. From the video, read the first peak: the **breakaway force, 1.20 ± 0.15 N**. Read the steady value over the last 3 mm: the **sliding force, 0.55 ± 0.10 N**.
6. Adjust if needed and repeat from step 3. Breakaway low: lift the mat out with the tweezers and add 0.10 g of soaked fibre. Breakaway high: remove 0.10 g. Sliding alone high: re-soak for 10 s. Sliding alone low: squeeze to 5.5 g. After two adjustments that do not bring both readings into range, discard the mat and make a new one.
7. The stroke has just moved the mat, so let it rest again before the robot meets it: refit the grate, charge the water and start the trial 3 to 5 min after the stroke, about as long as it rested before its measured breakaway.

The targets are the simulated mat's values. Its static breakaway is 1.20 N. Its sliding resistance is 0.50 N plus 30 N·s/m while it is in the neck, which adds about 0.06 N at the 2 mm/s gauge stroke. The 30 mm foot loads nearly the whole 32 mm face, as water pressure does. Over the 8.04 cm² bore, 1.20 N is about 1.49 kPa; with the standing water above the mat, the plunger must add about 1.39 kPa to start it. The mat's weight in water is under 0.01 N, so the ±0.5 g wet-mass tolerance does not change these forces.

**Per-trial reset (about 8 min)**

1. Unscrew the grate. Lift the used mat out of the chamber through the neck with the tweezers and discard it.
2. Make, set and install a fresh mat as above.
3. Empty and dry the receiver, and recharge the basin with 1.00 L of clean water.
4. Pre-wet the plunger cup and stand it upright at its offset start. Put the loose cover back at its start. Start the trial 3 to 5 min after the gauge stroke.

## Real-trial containment and gripper protection

These items are not part of the simulated scene and do not change the robot's task.

- **Secondary spill containment.** Polyethylene-backed absorbent bench liner (e.g. Whatman Benchkote Plus) covers the tabletop under the basin, outlet and receiver. Absorbent socks (e.g. New Pig PIG Blue, 3 in × 4 ft) ring the liner outside the arms' reach. The 1 L charge is the only water present. Weighing the liner and socks on a kitchen scale before and after the trial gives the water that reached the table rather than the receiver (1 g = 1 mL), for the 50 mL failure condition.
- **Gripper protection.** A clear polyethylene sleeve (e.g. OP/TECH USA Rainsleeve) covers each wrist and gripper body and is taped to the forearm with PVC electrical tape. It keeps splashes out of the gripper mechanism. The finger pads and the wrist camera lens stay bare, as in the simulation, so the robot gains no added sealing surface. Wipe the pads dry after each trial.

## Score

`task.json` contains the full scoring text. In a real trial the received water is found by weighing the receiver on a kitchen scale (2 kg or more, 1 g resolution; tared empty; 1 g = 1 mL); its 200 mL graduations cannot resolve the 0.90 L threshold and are only a visual guide. Full credit requires at least 900 mL in the receiver, an open passage through the former obstruction, and at least 30 mm clearance between the whole cup and the 52 mm waste-fitting outer rim for one second. Partial credit requires the passage still open and at least 500 mL or 200 mL collected. More than 50 mL outside the basin/receiver permanently fails the trial. A closed cup over the opening cannot pass the clearance test merely because its hollow rim lies far from the centre.

`python check_liquid_inventory.py` checks finite-cup drainage, direct-transfer non-credit, exterior loss and retained water. `python check_physics.py` checks water conservation, idle/uncovered/covered pressure cases, possible overflow loss, and the whole-cup clearance regression. These offline checks do not establish robot feasibility; that requires the separately recorded GPU demonstration.

## Recorded validation

The scripted solution scored **1.0** in **628 control steps** (31.4 s simulated, limit 2,400) with object_jitter=0, evaluation seed 0 and reset seed 1415158438. The run used source commit 7404c8b, GPU physics and 640 × 360 cameras. Wall time was 320.52 s on an RTX 3070 Ti Laptop GPU; the slowest measured 30-step block took 16.10 s.

The final receiver contained 995.438 mL through the actual outlet, with 4.562 mL retained in the sump. The basin and cup were empty. The mat was 33 mm below its initial position, the whole-cup clearance was 39.577 mm, and exterior loss and uncredited tool transfers were 0. The maximum cup-overcapacity diagnostic, updated at 240 Hz with no hidden capacity-return branch, was 0 mL. Recorded total-water error was below 3.7e-12 mL; the cumulative air-volume balance residual was −6.31e-6 mL.

In the recorded 240 Hz interval from 6.667 to 15.833 s, applied node pressure peaked at 1.953 kPa and corrected pressure at 2.151 kPa; the net pressure reaction reached 16.42 N. Force-prediction error was 149 Pa RMS before opening and 33.8 Pa RMS after opening with a slow mat. The full-run peak difference was 943 Pa during the opening transition. Applied node work was 0.492104 J, air-energy change −0.0000675 J and integrated network pressure work 0.494371 J. Their signed residual (node minus air change minus network) was −0.002200 J, or 0.447% of node work. It changed by less than 2e-9 J during the held-still 20–25 s interval.

The uncovered-overflow control scored **0.0** in 602 steps at 320 × 176. It collected 0 mL, left the mat in the neck, and had no exterior loss or cup overcapacity. Its recorded applied-pressure peak was 1.125 kPa, below the approximately 1.392 kPa breakaway requirement. Pressure error was 31.6 Pa RMS, with 111 Pa peak. This high-leakage case has a larger signed work residual: −0.04595 J against 0.70943 J node work, or 6.48%; it stopped accumulating when motion stopped. These are finite partitioned-coupling errors, **not a proof of strict passivity**. The model preserves the task's pressure loading and failure contrast but does not claim uniformly sub-percent energy accuracy.

Dream-1 scored **0.0** after **2,400 steps** (120 s simulated), reaching the step limit with the blockage still in place. It collected only 5.515 mL of seepage, with no exterior loss or cup overcapacity. The complete run took 1,262.85 s on the same GPU at 320 × 176. It made 296 requests at eight executed steps per request; all requests belonged to one continuous session, with no expiry or retry. The recorded model source was commit 514837d; the scene and scorer were unchanged from the successful scripted demonstration.

## Dimensions, references and licences

All scene geometry and materials are original procedural work. No downloaded product photographs, proprietary CAD, logos, third-party textures or assets are included.

- [Milton compact ceramic basin](https://www.victorianplumbing.co.uk/milton-350-x-280-wall-hung-compact-basin-1-tap-hole): external dimensions, bowl depth and overflow. The mesh is a dimensioned reconstruction, not a manufacturer CAD model.
- [Manufacturer RoPü Mini page](https://www.rothenberger-industrial.com/de/produkte/saugglocke-ropue-mini-16-cm-o-12-cm-roi-1500001444/): 12 cm mouth, 16 cm length, plastic body and rubber seal. The [manufacturer catalogue](https://www.toolteam.com/de-DE/downloads/dl/file/id/20173/Rothenberger_Industrial_Gesamtkatalog.pdf) calls 16 cm the total length. Its instructions depict reciprocating the corrugated cup.
- [Retail mass cross-check](https://www.pevex.hr/pumpa-za-vodu-rothenberger-za-odstopavanje-ropu-mini): approximately 0.08 kg.
- [Home Depot sink-plunger procedure](https://www.homedepot.com/c/ab/best-toilet-plungers-and-sink-plungers/9ba683603be9fa5395fab90961943fd) and [university household-maintenance example](https://extensionpubs.unl.edu/publication/cc435/2011/pdf/view/cc435-2011.pdf): plunging and covering a sink overflow.
- [PhysX kinematic mesh support](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/RigidBodyCollision.html): concave triangle-mesh collision on kinematic bodies.

These pages are dimensional/procedural references, not redistributed visual assets.

## Running the scripted solution

From this task folder in the installed station harness:

```sh
TASK_DEMO=success inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 --instruction "Clear the blockage so this basin drains normally, then leave the drain unobstructed." --scorer task_score --max-steps 2400 --store-frames
```

`TASK_DEMO=failure` repeats the plunging motions with the overflow uncovered. The demonstration removes and puts down the cover, then lifts the cup and waits for the basin to drain; those choices are not added scoring conditions.

To reproduce the recorded model rollout in the installed station harness:

```sh
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 -P rows_per_request=8 --instruction "Clear the blockage so this basin drains normally, then leave the drain unobstructed." --scorer task_score --max-steps 2400 --store-frames
```
