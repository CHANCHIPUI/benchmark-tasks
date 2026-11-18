# Cleave a 30 mm test piece off a silicon wafer at a scribed nick

A task file for the `yam_arena` embodiment ([yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena), Isaac Lab-Arena 0.3.1).

The robot scribes a short initiation nick at the tick on a 100 mm silicon wafer's rim with a pen-style diamond scriber,
then breaks the wafer from the nick with 3-point wafer-cleaving pliers, touching the wafer only with the scriber's tip
and the pliers' jaws.

## Run a trial

Dream-1:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cleave this wafer at the tick on its rim to cut off a 30 mm wide test piece. With the diamond scriber, scribe a short nick at the tick, no more than 5 mm in from the rim. Then break the wafer from the nick with the cleaving pliers. Touch the wafer only with the scriber's tip and the pliers' jaws, and leave both pieces on the stage." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir runs/dream
```

Scripted demo (scores 1.0): the same command with `--policy yam_scripted`. `TASK_DEMO=nothing` plays no motion (scores 0).

## How `score()` reaches the score

- `cleave_usd.py` writes the scene's USD files into `build/` when the task loads. The wafer is one rigid body; its 26
  pre-cut pieces (13 lines 1 mm apart, within 6 mm of the target line) wait hidden on a shelf under the table.
- A PhysX rigid-contact view on the wafer and every piece, filtered by the scriber, the pliers' four links and the six
  finger links, is read after every physics step; the tracker uses each control step's mean and peak.
- Every control step the tracker latches 0 for a finger on the wafer, a tool striking it at over 0.3 m/s, the
  scriber's barrel or the pliers' body on its face, a scratch, a scribe over 6 N, 24 N on the jaw ridge with no nick,
  or a piece off the stage or face down. It records scribe strokes as nicks (with a chip flag for 3 to 6 N).
- The cleave rule: a 12 N clamp at the jaw ridge with a nick within 3 mm of the ridge's line swaps the wafer, at its
  pose and velocity, for the two pieces cut on the pre-cut line nearest the nick. The clamp is the holding gripper's
  squeeze on the grips (its finger drives' effort) less the return spring, times the lever (108 mm / 25.3 mm), while
  the jaws are at their closing stop on the wafer; PhysX lets two driven jaws pass through a thin body, so the stop
  (set at the wafer's thickness) stands in for the wafer's stiffness. The real pair needs no stop.
- The gate (fingertips 3 cm from every piece, tools off, pieces still) must hold for 1 s; each time it does, a reading
  of the end-state table is taken. `score()` is the last reading, or 0 after a latch. `success()` is the 1.0 state
  with the gate's conditions now, and the harness holds it for the same 1 s.

## Runtime and seeds

- Demo: about 900 steps (45 s simulated) to the success hold; about 9 minutes on an RTX 3070 Ti laptop GPU including
  start-up (about 0.5 s a step at 640 x 360).
- `object_jitter=0` keeps the layout fixed, so no seed is needed. The scripted demo is open loop.

## Real-world kit

- Wafer: 100 mm, 525 µm, (100), p-type, single-side polished test grade (for example University Wafer, US$10.90).
- Scriber: Electron Microscopy Sciences 62107-ST pen-style diamond scriber, straight.
- Pliers: SPI Supplies CleanBreak 07649-AB or Ted Pella CleanBreak 54361, 6 in, with 3-point plastic jaws; a nylon cord
  loop round the grips sets the rest opening (jaws 6 mm, grips 50 mm apart).
- Stage (with its two hold-down tabs), scriber stand, pliers' stand and tick gauge: printed (PETG, PLA); sizes in
  `task.json`'s setup. The 0.25 mm polyester pocket shim is cut to the wafer's outline plus 0.3 mm (the simulation
  draws the wafer 1 mm thick and the shim 0.5 mm). Files in `cad/`, all in mm:
  - `stage.stl`: plate with its notch, column and the two hold-down tabs;
  - `pocket_shim.svg`: the shim's cut lines (plate outline, wafer outline plus 0.3 mm, notch, the tabs' feet), 1:1;
  - `scriber_stand.stl`, `pliers_stand.stl`, `tick_gauge.stl`.
  `python cleave_usd.py --stl` writes the STL files from the same geometry as the simulation.
- Spare jaw inserts: the SPI pair comes with a second 3-point (crystalline) set; for the Ted Pella pair, confirm the
  3-point set when ordering replacement jaws.
- Pad covers: 1 in polyimide tape (3M 5413) on each gripper pad.

## Force thresholds and the bench check

Every force threshold in this task is an estimate until the bench check below has been run on the real kit; it is run,
and its values recorded in this section, before any real trial. The simulation's thresholds are these
fracture-mechanics estimates, not measurements: a scribe of 0.3 to 3 N makes a nick,
3 to 6 N also chips the edge and over 6 N shatters the wafer; 12 N at the jaw ridge cleaves from a nick within 3 mm,
and 24 N with no nick there shatters it (12 N is the 3-point-bend force for a 5 to 20 um scribe flaw with K_IC about
1 MPa m^0.5, 6 mm of wafer between pads 12 mm apart). They have not yet been measured on the real kit.

The bench check below confirms them on the real rig, with the bought scriber and pliers, before trials. The real trial
is scored by its outcomes (nick, chip, scratch, pieces and coupon width, seen under the loupe and measured with the
caliper), so these numbers set only the simulation. If a measured value differs, record it here next to the estimate.

With a kitchen scale (0.1 g to 5 kg), film its display to read the peaks:

1. Scribe force window: on 3 test wafers lying on a plate on the scale, scribe initiation strokes at about 0.3, 1, 3,
   4.5 and 7 N and sort each under the loupe (no nick, nick, chip, broken). Compare the transitions with 0.3 / 3 / 6 N.
2. Cleave force: scribe 5 wafers as specified, grip each at the nick with the bought pliers with the lower grip's end on
   the scale, press the upper grip down until it cleaves, read the peak. The median times the pair's lever ratio is the
   ridge force to compare with 12 N.
3. No-nick squeeze: squeeze 3 unscribed wafers at their rims to the pliers' full closure. None breaking is consistent
   with a no-nick threshold of at least twice the cleave force (24 N); if they break, record the median breaking ridge
   force and how they broke.

Measured values: not yet recorded.

## Sources

Every mesh is generated by `cleave_usd.py` from the real products' dimensions (wafer: SEMI M1 100 mm; scriber: EMS
62107-ST, 9 mm x 140 mm; pliers: 6 in CleanBreak-type). No third-party meshes or textures are used.
