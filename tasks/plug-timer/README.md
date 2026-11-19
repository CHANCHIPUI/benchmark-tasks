# Set a plug-in lamp timer: with a pen, push down only the 6:00-10:30 PM segments, 3.3 mm apart, then turn it to the time

A white 6-outlet power strip lies flat on the table, switched off. A GE 24-hour plug-in mechanical timer (model 15119)
is plugged into its third socket from the switch end, dial up, and a table lamp is plugged into the timer. Every one of
the dial's 48 half-hour segments is up and the pointer shows 9:00 AM. A used-up ballpoint pen stands point down in a
mug at the robot's right. The instruction is:
**"It is 2:30 PM. Set the plug-in timer so the lamp comes on at 6:00 PM and goes off at 11:00 PM: with the pen, push
down the ten segments from 6:00 PM to 10:30 PM and no others, and turn the dial clockwise until its pointer shows
2:30 PM."**

To do it, the robot has to:
- take the pen out of the mug and hold it as a tool, its 1 mm ball about 135 mm beyond the fingers;
- land the ball on each of the ten segments from 6:00 PM to 10:30 PM in turn and click it down. The segments are
  3.0 mm wide with 0.3 mm gaps, 3.3 mm apart, about half a fingertip's thickness; a ball 1.3 mm off a segment's
  centre line already bears on its neighbour;
- leave the 5:30 PM and 11:00 PM segments beside the run up, and push no segment down with a gripper;
- turn the dial clockwise by its ribbed side until the pointer is within a quarter hour of 2:30 PM;
- let go of everything and move clear.

## Why it matters

Plug-in 24-hour timers switch lamps, fish-tank lights and holiday lights on and off on their own. Police forces
advise putting lamps on plug-in timer switches whenever the house is empty, so it looks occupied (Avon and Somerset
Police; King County Sheriff's Office). Mechanical ones cost about $8 and are stocked by every home-improvement
retailer; a best-selling 48-segment model (BN-LINK) shows about 22,000 shopper ratings. The maker's instruction sheet (GE 15119 / 15131 / 15417,
Jasco) describes the act:
- "These segments, each representing 30 minutes, can be pushed down from the edge of the dial (try using the tip of a
  pen or pencil)."
- "Select a time period (or periods) you want the lamp turned on, then push down ALL the segments that fall on or
  within that time period."
- "Rotate the timer's dial clockwise until the pointer on the face of the dial points to the current time of day."

The instruction names the ten segments, so nothing is left to reading or arithmetic: what is tested is the physical
act. The difficulty is geometric. A held tool's 1 mm ball must land within 1.3 mm of each of ten targets 3.3 mm apart
round a dial 52 mm across, on a compliant arm that gives about 3 mm per newton at the pen's tip while it pushes about 3 N per
segment. The two ends of the run are decisive: a careless press at 6:00 PM or 10:30 PM takes the neighbour too, and
the lamp then comes on half an hour early or goes off half an hour late. A fingertip spans two segments.

Nearest tasks on the board: a held pin into a 1 mm hole (SIM tray), a rotary dial dragged with a pencil, a padlock
dial set to numbers, an alarm clock set with buttons, and caster brake tabs and DIP switches pressed into detents.
None selects among adjacent targets a few millimetres apart with a held tool.

## Files

| File | What it is |
|---|---|
| `task.py` | the yam_arena task file: objects, physics rate, the segment detents, the dial's drag and the switch (run before every physics substep), contact reading and scoring |
| `geometry.py` | every size and the layout (estimates marked "est.") and the scoring constants |
| `build_assets.py` | writes `assets/*.usda` and the dial's textures from `geometry.py` (needs `pxr` and PIL, not the simulator) |
| `scenekit.py` | USD bodies and articulations for task files, and the physics-rate hook |
| `demokit.py`, `demo.py` | the scripted demo and the scorer checks |
| `dev/` | build-time checks (reach scans, launch script, logs) |
| `NOTES.md` | build notes |

All assets are built from scratch in `build_assets.py` (no third-party meshes, scans or textures). The dial's figures
are drawn with the DejaVu Sans Condensed Bold font (DejaVu fonts licence: free to use, embed and redistribute). The
robot is yam_arena's.

## Run a trial

```
source <ARENA_DIR>/yam.env
python build_assets.py                                    # once
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
    --scorer task_score --max-steps 3600 --policy ...
```

No seeds are used: the layout is the same in every trial. A full 3,600-step trial takes about 30 minutes of wall time
on one GPU (480 Hz physics, three cameras).

## Dream-1 run

Dream-1 scored 0 with the task's exact instruction over the full 3,600 steps (one run, not retried). Its right arm reached the mug that holds the pen and nudged and turned it slightly, but the pen stayed in the mug; it never pushed a segment and never touched the timer. Its three-view video is attached with the submission.

## Scripted demo

```
python demo.py task.py demo_out [--no-record] [--variant neighbour|wrongtime|fingertip|missing|nothing]
```

The demo reads object poses from the simulator (privileged: the pen's pose, the dial's angle, the timer's pose). In
order:
1. **Pick the pen (right arm, while the left arm sets the time).** The hand comes down along the pen as it leans in the
   mug (its top toward the robot, about 36 degrees), fingers closing across it in its lean plane 135 mm above the
   ball (fully closed, about 20 N a finger), and lifts it straight up out of the mug.
2. **Set the time (left arm).** Fingers pointing down astride the dial, the pads on its ribbed side 2 mm above the
   timer's face (not touching a segment); it closes and rolls the wrist clockwise, by feedback on the dial's angle
   with a 2 degree lead, until the pointer shows 2:30 PM; it opens, lifts and parks with its wrist camera 15 cm
   from the dial (the close-up view in the video). Either order gives the same end state; setting the time first puts
   the ten targets on the dial's left half, where they are pressed.
3. **Press the ten segments (right arm).** The pen tipped 24 degrees, its top toward the robot's right. The order
   alternates ends: 10:30, 6:00, 10:00, 6:30 PM ... For each: the hand moves smoothly to put the ball 2.5 mm over the
   segment's centre line, 23 mm from the dial's axis (the two end segments 0.45 mm toward the inside of the run), and
   waits for the arm's sway to die down; the ball's position is averaged over 0.3 s and the hand corrected smoothly by
   the mean error until it is within 0.2 mm. Then straight down. On the segment the hand holds the ball where it
   landed (feedback on its position) and leans back toward the pen's top as it goes down: the arm gives about
   3 mm per newton, so the hand goes down 20 to 35 mm further to push the segment the 1 mm that latches it (about
   3.3 N). It lifts from where the hand actually is. An end segment whose landing is more than 0.55 mm toward its
   forbidden neighbour is lifted off at once and redone; a press whose ball slips off its segment is stopped and
   redone.
4. **Close-up and park.** After the last press the left arm brings its wrist camera to 10 cm from the dial, 45 degrees
   above it, from beyond its far side, while the pen waits just over the dial, 20 mm or more from the timer; the
   success hold ends the trial there.

**Result (final code, recorded run `dev/out_d13`, log `dev/d13.log`): 1.0, success at step 1,988 (99 s of the 3,600-step
limit).** All ten segments 6:00 to 10:30 PM down and no others (`task.lamp_test`: on 6:00 PM, off 11:00 PM); the
pointer at 2:31 PM (0.17 degrees past the 2:30 PM mark; the limit is 3.75); no gripper touched a segment; slide switch
at TIMER ON; strip moved 0.25 mm; at the end both grippers 28 mm or more and the pen 56 mm from the timer.

**Margins** (from `task.py`'s record at the moment each segment latched: the ball's offset from the segment's centre
line, along the dial's circumference):

| Segment | Ball offset at the latch | Margin to the neighbour at that radius | Clearance |
|---|---|---|---|
| 6:00 PM (decisive) | 0.73 mm toward 5:30 PM | 1.17 mm | 0.44 mm from 5:30 PM, which was pushed down 0.02 mm at most (it latches at 1 mm) |
| 10:30 PM (decisive) | 0.58 mm toward 10:00 PM (inside the run) | 1.04 mm | 1.62 mm from 11:00 PM, which did not move |
| the eight inside the run | up to 2.3 mm, each toward a neighbour that is also one of the ten | 1.02 to 1.15 mm | on three presses (10:00, 9:00 and 7:30 PM) the ball slid about 2 mm along the dial onto the neighbour; the press was stopped and redone; the 7:30 PM one latched 8:00 PM first |

The neighbour margin is 1.3 mm at the segments' outer edge (half the 3.0 mm width plus the 0.3 mm gap, less the
ball's 0.5 mm radius) and 1.1 to 1.2 mm where the demo lands, 21 to 23 mm from the axis. The demo lands the two end
balls 0.45 mm toward the inside of the run and redoes a landing more than 0.55 mm toward a forbidden neighbour (neither
end needed it in this run). The arm's sway while hovering is about 0.5 mm at the ball. The dial was turned to 0.5 degrees
short of 2:30 PM; the ten presses then dragged it 0.7 degrees on.

**Scorer checks** (logs `dev/v_*_c.log` and `dev/v_neighbour2_d.log`):

Run with the demo's code (moves 0.6 s and 5-step settling to save time; no recording):

| Run | What the script does | Score |
|---|---|---|
| demo (recorded) | as above | **1.0**, success at step 1,988 |
| check: wrong time | the dial turned to 3:30 PM, the ten pressed as in the demo | 0.75 (all ten down and no others; pointer at 3:20 PM after the presses, outside 2:15 to 2:45 PM) |
| check: missing segments | the 8:00 and 8:30 PM segments skipped | 0.25 (8 of the ten down, none other; the lamp would be on 6:00 to 8:00 and 9:00 to 11:00 PM) |
| check: fingertip | the closed fingertips, tipped as the pen was, pressed onto each of the ten in turn | 0 (the fingertips span two or three segments and latched none; any segment a gripper pushes down would cap the score at 0.25) |
| check: neighbour caught | after the ten, a deliberate extra pen press on the 5:30 PM segment, which latches it | 0.5 (5:30 to 10:30 PM down, nothing else: the lamp comes on half an hour early; step 2,093) |

Media: `demo.mp4` (the top camera large, the left and right arm cameras stacked beside it; the left arm's camera
parks 15 cm from the dial after setting the time, so the presses are seen close up; the video ends with the left
arm camera's last 3 s, the dial enlarged and brightened, from 10 cm and 45 degrees above: the ten pushed segments sit
level with the dial on the lower left, the rest stand up, and the pointer is at 2:31 PM); `setup.png` is the first
top-camera frame; `dial_end_wrist.png` is the left arm camera's last frame and `dial_end_closeup.png` the dial
from it, enlarged 3 times and brightened.

## Scoring

Read every control step (20 Hz) from the joints and poses, and judged on the final 1 s (every step of it must reach
the level):
- **Down:** a segment pushed 1 mm or more latches, as the real detent clicks, and stays down (2 mm, level with the
  dial). "The ten" are 6:00 PM to 10:30 PM: segment i covers i/2 to i/2 + 0.5 h after midnight, so the ten are
  segments 36 to 45, and their neighbours 35 (5:30 PM) and 46 (11:00 PM).
- **Time set:** the pointer within 3.75 degrees (a quarter hour, 1.6 mm at the segments' edge) of 2:30 PM, and the dial
  never turned back counterclockwise more than 15 degrees (an hour) from the furthest clockwise angle it reached.
- **End conditions, held for the final 1 s:** both grippers (their bodies' and fingers' collision hulls) and the pen 20
  mm or more from the timer (its housing and its dial with the centre cap, as boxes); the slide switch at TIMER ON; the
  timer never raised more than 3 mm (out of its socket; latched).
- **Pen only:** a gripper (body or finger) pushing any segment down at any time caps the score at 0.25 (latched): in
  contact with it (PhysX contact forces between the 48 segments and both hands) while it goes down 0.5 mm or more or
  clicks down. A touch that does not push a segment down, such as a finger brushing the segments' outer ends while it
  turns the dial, does not count, nor does a touch on a segment that was already down.

| End state | Score |
|---|---|
| exactly the ten down, nothing else, time set (the lamp lights at 6:00 PM, goes out at 11:00 PM, stays out the rest of the day) | 1.0 |
| exactly the ten down, nothing else, time not set | 0.75 |
| one unbroken run that is the ten with one end moved by one segment (on at 5:30 or 6:30 PM, or off at 10:30 or 11:30 PM), nothing else | 0.5 |
| at least 8 of the ten down and at most one other | 0.25 |
| anything else, including nothing pressed or the timer pulled out of its socket | 0 |

`success()` is the 1.0 state; the trial ends once it has held for 1 s. The device output (`task.lamp_test`) is
computed from the down segments: the on and off times over one turn.

**Real trial.** The strip stays switched off during the run. At the end, a phone photo of the dial from 45 degrees
above **decides** the score: a raised segment stands 2 mm proud of the dial, and a segment counts as down only when it
is pushed fully down, level with the dial; one left part way down counts as up. The **lamp test confirms** it: switch
the strip on and turn the dial one full turn clockwise by hand, noting the pointer each time the lamp goes on or off
(if the two disagree, the photo decides). Where 120 V mains is not at hand, a continuity tester between the unplugged
timer's narrow (hot) blade and its outlet's narrow slot stands in for the lamp during the same hand-turned revolution.
The dial may only be turned clockwise, as the maker's sheet says: a counterclockwise turn of more than an hour (15
degrees, plain in the trial video) means the time is not set. A gripper pushing a segment down is judged from the trial
video (top and wrist cameras): a finger or gripper pressing on a segment's top as it goes down counts; fingers on the
dial's ribbed side or on the segments' outer ends while turning the dial do not, and a segment that ends down where
the pen never pressed it is checked frame by frame in the video. "20 mm clear" is checked in the final frames, and
"fully in its socket" with a ruler (raised less than 3 mm).

**The pen, the same every time.** Take a new BIC Round Stic (medium, 1.0 mm), cap off, and write on scrap paper until
a 10 cm stroke at normal writing pressure leaves no visible line; wipe the ball with a dry tissue. Before each trial,
draw one 10 cm stroke: if it marks, write it dry again (or replace it with another prepared pen). Keep the cap off and
the pen point down in the mug between trials. A new pen with ink also works: its marks don't change the score;
wipe the dial with a damp cloth and dry it at each reset.

## Real-world build

Positions in mm from the midpoint between the arm bases, x forward, y to the robot's left; heights above the table.

| Item | Details |
|---|---|
| Table | the station's wooden bench top |
| Power strip | a white 6-outlet power strip, outlets in a row with an on/off switch at one end, 280 x 50 x 38 mm, 0.4 kg (estimates), switched off. Flat, sockets up, its length along y, the switch end toward the robot's right, centred at x 300, y 5; its cord straight back off the table. |
| Timer | GE 24-Hour Plug-In Mechanical Timer, model 15119 (Jasco; white plastic, listed 2.4 x 2.4 x 2.3 in, 3.2 oz, 48 captive 30-minute segments, OUTLET ON / TIMER ON slide switch, one polarized outlet; about US$7). Plugged into the strip's **third socket from the switch end**, dial up, centred at x 300, y 0; its top edge (the slide switch, at TIMER ON) faces +x, so the pointer on the fixed white centre points away from the robot; its own outlet faces +y. |
| Dial | black, on a ribbed side 52 mm across that stands 8 mm proud of the timer's face; 48 segments round its top edge, each 3.0 mm wide with 0.3 mm gaps (3.3 mm apart) and 5 mm long; up, a segment stands 2 mm proud of the dial, its top 80 mm above the table; pushed down it is level with the dial (estimates). Start: every segment up, the pointer at 9:00 AM. |
| Lamp | IKEA LAMPAN table lamp (white, 290 mm tall, base 130 mm, shade 190 mm) with a 470 lm LED bulb, its own switch on, at x 550, y 300; its cord behind the strip to the timer's outlet. |
| Pen | a used-up BIC Round Stic ballpoint (1.0 mm ball, cap off): its ink has run out, so it leaves no marks on the dial. 150 mm long, 8 mm across, 5 g (estimates). Point down in the mug, its top leaning on the near rim. |
| Mug | ceramic, 95 mm tall, 80 mm across, 0.3 kg (estimates), handle toward the robot's right, at x 220, y -200. |

Nothing else is on the table. The layout is the same in every trial, as in the screenshot. With the strip switched
off no part is live during the run, and the robot touches only a consumer device made to be handled.

## Simulation fidelity

Rigid PhysX, 480 Hz physics (24 substeps per 20 Hz control step; yam_arena's default is 240 Hz), GPU dynamics.
- **Strip and timer:** one floating articulation: the root body is the strip with the timer's housing and fixed white
  centre (0.476 kg, real friction on the table), so the timer is fixed in its socket; the dial is a revolute joint
  about its axis; each of the 48 segments is a 2 mm prismatic joint along the dial's axis, a child of the dial (so the
  segments turn with it), with a wedge collider 3.0 mm wide and 5 mm long at the outer end; the slide switch is a
  prismatic joint along the top edge. 51 links, self-collision off. Contact offset 0.2 mm, rest offset 0.
- **Segment detent (task code):** before every physics substep, `task.py` holds an up segment on its upper stop by a
  3 N/mm spring (0.3 N preload); a segment pushed 1 mm (about 3.3 N) latches: its spring target moves past the lower
  stop, so it snaps down the remaining 1 mm and stays there. Pushing a segment back up is not modelled (the reset does
  it by hand). Segments are 5 g for the solver (the real parts are lighter).
- **Dial drag (task code):** a stiff spring (5 N m/rad) holds the dial to a rest angle that `task.py` drags along once
  the spring's torque passes 0.03 N m, so the dial turns under a hand and stays where it is left (PhysX joint friction
  creeps).
- **Slide switch (task code):** a spring to whichever stop (TIMER ON or OUTLET ON) is on the switch's side of its
  middle.
- **Not simulated:** the clock motor and the switch contacts (the score reads the segment, dial and switch joints;
  `task.lamp_test` turns the down segments into the lamp's on and off times); the lamp, its bulb and both cords are
  visual only (the lamp's base and shade collide).
- **Pen:** one rigid body, 5 g: a 1.0 mm ball, a tapered point and an 8 mm barrel (all colliding); friction 0.6 / 0.5.
  The mug is a free 0.3 kg body (24 wall boxes and a floor).
- **Estimates:** the strip's size and mass, the dial and segment sizes, travel and heights (the timer's case 32 mm
  deep on the strip), the detent's 3 N, the dial's 0.03 N m drag, the pen's size and mass, and the mug's.
- **Demo:** the scripted demo reads object poses from the simulator (privileged); a model under test sees only the
  cameras.

## Reset between trials

Switch the strip off. Push every down segment back up by hand, turn the dial clockwise until the pointer shows
9:00 AM, check the slide switch is at TIMER ON and the timer is pushed fully into its socket, and stand the pen point
down in the mug, leaning on its near rim, with the mug on its mark. Nothing is consumed; about 1 minute.

## Sources

- GE 24-Hour Plug-In Mechanical Timer instruction sheet, models 15119 / 15131 / 15417 (Jasco Products, 2009), steps 1-3
  and figures 1-3.
- GE 15119 retail listings (Amazon B00P76URH8, Manuals+): 2.4 x 2.4 x 2.3 in, 3.2 oz, 125 VAC / 15 A, 30-minute
  increments.
- IKEA LAMPAN table lamp, article 200.469.88: 29 cm tall, base 13 cm, shade 19 cm, cord 130 cm.
- BIC Round Stic ballpoint, medium point (1.0 mm).
- Avon and Somerset Police, "Lighting" crime-prevention advice ("lights on timer switches" to make a home look occupied),
  https://www.avonandsomerset.police.uk/crime-prevention-advice/home-protection/lighting/
- King County Sheriff's Office, home lighting advice, https://kingcounty.gov/depts/sheriff/police-precincts/safety/home/lighting.aspx
- HomeTips, timer light switches buying guide (plug-in receptacle timers about $8),
  https://www.hometips.com/buying-guides/timer-light-switches.html
- BN-LINK 24-hour plug-in mechanical timer listing: 4.5 of 5 stars from 21,904 ratings, #9 in Plug-In Timers
  (retail mirror of the Amazon listing).
