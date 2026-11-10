"""On the station table: three used lithium-ion camera batteries (40 x 20.5 x 56 mm, 80 g, black, grey label) stand in
a row near the robot with their contact end up and the four gold contacts bare. Their three black snap-on terminal
covers (40 x 20.5 x 10.5 mm open-bottom sleeves, 5 g) lie loose on the bench, open side down: two on the left, one on
the right. Further out between the arms, a grey polypropylene collection box (20 x 30 x 12 cm, open top, labelled
"used batteries") is fixed to the bench. The batteries and covers are loose; the box is fixed.

The instruction names only the box. Lithium-ion terminals that touch metal or another battery short and spark, so
collection programs require each battery's terminals to be covered before it goes in a collection container: the
robot must push a cover onto each battery's contact end first. A battery in the box without its cover fails the trial.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import assets as a  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

a.build()

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- layout (station frame, metres: x forward, y to the robot's left) ---------------------------------------------
BATTERY_XY = {"battery_left": (0.15, 0.18), "battery_middle": (0.15, 0.0), "battery_right": (0.15, -0.18)}
COVER_XY = {"cover_1": (0.28, 0.24), "cover_2": (0.20, 0.27), "cover_3": (0.28, -0.24)}
BOX_XY = (0.33, 0.0)
RIM_Z = TOP + a.BOX_H

BATTERIES = list(BATTERY_XY)
COVERS = list(COVER_XY)

OBJECTS = [
    *[UsdBody(n, usd="assets/battery.usda", pos=(x, y, TOP + a.BAT_H / 2 + 0.0002), mass=a.BAT_KG,
              size=(a.BAT_W, a.BAT_T, a.BAT_H), friction=a.FRICTION, solver_iters=32, torsional_patch=0.008)
      for n, (x, y) in BATTERY_XY.items()],
    *[UsdBody(n, usd="assets/cover.usda", pos=(x, y, TOP + a.COVER_H / 2 + 0.0002), mass=a.COVER_KG,
              size=(a.COVER_W, a.COVER_T, a.COVER_H), friction=a.FRICTION, solver_iters=32, torsional_patch=0.006)
      for n, (x, y) in COVER_XY.items()],
    UsdBody("collection_box", usd="assets/box.usda", pos=(*BOX_XY, TOP), mass=None,
            size=(a.BOX_L, a.BOX_W, a.BOX_H), friction=(0.5, 0.4)),
]

# -- scoring ------------------------------------------------------------------------------------------------------
SEAT_TOL = 0.003  # m: a cover's centre from its fully pushed-on position, in the battery's frame
TILT_TOL_DEG = 10.0  # between the cover's axis and the battery's
BARE_STEPS = 5  # 0.25 s at 20 Hz: a battery in the box this long without a seated cover fails the trial
STILL = 0.01  # m/s
CREDIT_IN_BOX, CREDIT_COVERED = 0.3, 0.1


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _cover_offset(battery, cover) -> tuple[float, float]:
    """(distance in m from the fully pushed-on position, tilt in degrees) of a cover relative to a battery."""
    rb, rc = _rot(battery.quat_xyzw), _rot(cover.quat_xyzw)
    rel = rb.T @ (np.asarray(cover.pos) - np.asarray(battery.pos)) - a.SEAT
    tilt = math.degrees(math.acos(float(np.clip(rb[:, 2] @ rc[:, 2], -1.0, 1.0))))
    return float(np.linalg.norm(rel)), tilt


def _seated(battery, cover) -> bool:
    dist, tilt = _cover_offset(battery, cover)
    return dist <= SEAT_TOL and tilt <= TILT_TOL_DEG


def _covered(objects, name) -> bool:
    return any(_seated(objects[name], objects[c]) for c in COVERS)


def _in_box(battery) -> bool:
    """Its centre is inside the box's inner walls (top view) and lower than the rim."""
    x, y, z = (float(v) for v in battery.pos)
    return abs(x - BOX_XY[0]) < a.INNER_HX and abs(y - BOX_XY[1]) < a.INNER_HY and z < RIM_Z


def _still(battery) -> bool:
    return float(np.linalg.norm(battery.lin_vel)) < STILL


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(failed=False, why=[], bare={n: 0 for n in BATTERIES}, step=0)


_reset_state()


def _fail(why: str) -> None:
    _STATE["failed"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def _trace(record: dict) -> None:
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "trace.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


@on_step
def _monitor(environment) -> None:
    """Once per control step: latch a failure the moment it happens. Only a reset clears it."""
    from yam_arena.scene import object_states

    st = object_states(environment, BATTERIES + COVERS)
    _STATE["step"] += 1
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2)}
    for n in BATTERIES:
        b = st[n]
        covered, inside = _covered(st, n), _in_box(b)
        _STATE["bare"][n] = _STATE["bare"][n] + 1 if inside and not covered else 0
        if _STATE["bare"][n] >= BARE_STEPS:
            _fail(f"{n} in the box without a seated cover")
        if float(b.pos[2]) < TOP:
            _fail(f"{n} fell off the table")
        best = min((_cover_offset(b, st[c]) + (c,) for c in COVERS), key=lambda r: r[0])
        rec[n] = {"pos": [round(float(v), 4) for v in b.pos], "in_box": inside, "covered": covered,
                  "cover": best[2], "off_mm": round(best[0] * 1000, 1), "tilt": round(best[1], 1)}
    for c in COVERS:
        r = _rot(st[c].quat_xyzw)
        tilt, yaw = math.degrees(math.acos(float(np.clip(r[2, 2], -1, 1)))), math.degrees(math.atan2(r[1, 0], r[0, 0]))
        rec[c] = [round(float(v), 4) for v in st[c].pos] + [round(tilt, 1), round(yaw, 1)]
    rec["failed"] = _STATE["why"]
    _trace(rec)


def success(objects) -> bool:
    """All three batteries in the box, each with a seated cover, all three still, and no failure so far; held 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    return bool(not _STATE["failed"] and all(_in_box(objects[n]) and _covered(objects, n) and _still(objects[n])
                                             for n in BATTERIES))


def score(objects) -> float:
    """0 after any failure (a battery in the box without a seated cover for 0.25 s, or a battery off the table).
    Otherwise 0.3 for each battery in the box with a seated cover and 0.1 for each one outside the box with a seated
    cover (success scores 1.0)."""
    if _STATE["failed"]:
        return 0.0
    total = 0.0
    for n in BATTERIES:
        if _covered(objects, n):
            total += CREDIT_IN_BOX if _in_box(objects[n]) else CREDIT_COVERED
    return round(total, 3)


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
TRAVEL = TOP + 0.165  # a held battery's bottom is then 14.7 cm up, over the 12 cm rim
COVER_TRAVEL = TOP + 0.10
COVER_GRIP = TOP + 0.0155  # pad tips 2.5 mm above the table on a cover lying open side down
BAT_GRIP = TOP + 0.018  # pad tips 5 mm above the battery's base
OVER_NECK = TOP + a.BAT_H + 0.003 + 0.0155  # a held cover's rim 3 mm above the battery's top
ENGAGE = TOP + a.BAT_H - 0.0005 + 0.0155  # a held cover's rim 0.5 mm into the neck's lead-in chamfer
PRESS = TOP + a.BAT_H + a.COVER_TOP + 0.013 - 0.002  # closed tips 2 mm into a seated cover's top
BOX_RELEASE = TOP + a.BOX_WALL + 0.010 + 0.018  # a held battery's base 1 cm above the box floor
WIDE = -0.020  # finger joint: pads 40 mm apart
LET_GO = -0.016  # pads 32 mm apart: lets go of a cover
HOLD_COVER = -(a.COVER_T - 0.004) / 2  # pads 4 mm narrower than the cover: a light grip (about 10 N a finger)
HOLD_BATTERY = -(a.BAT_T - 0.008) / 2  # pads 8 mm narrower than the battery (about 20 N a finger)
DROP_XY = {"battery_left": (0.30, 0.08), "battery_middle": (0.29, 0.0), "battery_right": (0.30, -0.09)}
# Where the hand is aimed past each battery when it puts a cover on. The arms sag a little under load, and the held
# cover comes to rest these few millimetres short of the aim; measured (cover minus battery, while held 3 mm above
# the neck) in a logged run of this plan, and subtracted here.
COVER_AIM = {"battery_left": (-0.0004, -0.0005), "battery_middle": (-0.0006, -0.0005), "battery_right": (-0.0005, 0.0007)}


def _put_cover_on(cover: str, battery: str) -> list[Waypoint]:
    cx, cy = COVER_XY[cover]
    bx, by = BATTERY_XY[battery]
    ax, ay = bx + COVER_AIM[battery][0], by + COVER_AIM[battery][1]
    return [
        Waypoint(1.6, (cx, cy, COVER_TRAVEL), WIDE, f"to {cover}"),
        Waypoint(1.4, (cx, cy, COVER_GRIP), WIDE, "descend to the cover"),
        Waypoint(0.8, (cx, cy, COVER_GRIP), HOLD_COVER, "grip the cover"),
        Waypoint(1.0, (cx, cy, COVER_TRAVEL), HOLD_COVER, "lift the cover"),
        Waypoint(1.8, (ax, ay, COVER_TRAVEL), HOLD_COVER, f"over {battery}"),
        Waypoint(1.4, (ax, ay, OVER_NECK), HOLD_COVER, "lower over the contacts"),
        Waypoint(0.6, (ax, ay, OVER_NECK), HOLD_COVER, "settle"),
        Waypoint(1.0, (ax, ay, ENGAGE), HOLD_COVER, "into the lead-in"),
        Waypoint(0.6, (ax, ay, ENGAGE), LET_GO, "let go"),
        Waypoint(0.8, (ax, ay, OVER_NECK + 0.012), LET_GO, "rise"),
        Waypoint(0.5, (ax, ay, OVER_NECK + 0.012), FINGER_CLOSED, "close"),
        Waypoint(1.0, (ax, ay, PRESS), FINGER_CLOSED, "press the cover home"),
        Waypoint(0.8, (ax, ay, COVER_TRAVEL), FINGER_CLOSED, "rise"),
    ]


def _into_box(battery: str) -> list[Waypoint]:
    bx, by = BATTERY_XY[battery]
    dx, dy = DROP_XY[battery]
    return [
        Waypoint(0.6, (bx, by, COVER_TRAVEL), WIDE, "open"),
        Waypoint(1.4, (bx, by, BAT_GRIP), WIDE, f"descend to {battery}"),
        Waypoint(0.8, (bx, by, BAT_GRIP), HOLD_BATTERY, "grip the battery"),
        Waypoint(1.5, (bx, by, TRAVEL), HOLD_BATTERY, "lift"),
        Waypoint(2.0, (dx, dy, TRAVEL), HOLD_BATTERY, "over the box"),
        Waypoint(1.5, (dx, dy, BOX_RELEASE), HOLD_BATTERY, "lower into the box"),
        Waypoint(0.6, (dx, dy, BOX_RELEASE), WIDE, "release"),
        Waypoint(1.2, (dx, dy, TRAVEL), WIDE, "rise"),
    ]


def demo():
    """The right arm covers the right battery and puts it in the box, then parks. The left arm does the same for the
    left battery, then the middle one. Variants (TASK_DEMO), to check the scorer: failure puts all three in the box
    bare; partial covers all three and boxes only the left one; inout puts the right one in bare and takes it out
    again before doing everything properly; nothing does nothing."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"left": [Waypoint(4.0, (0.25, 0.30, TRAVEL), WIDE, "hold")]}
    cover = mode != "failure"
    right: list[Waypoint] = []
    if mode == "inout":
        bx, by = BATTERY_XY["battery_right"]
        dx, dy = DROP_XY["battery_right"]
        right += _into_box("battery_right")[:6]  # lowered into the box, still held
        right += [
            Waypoint(1.0, (dx, dy, BOX_RELEASE), HOLD_BATTERY, "hold it in the box"),
            Waypoint(1.5, (dx, dy, TRAVEL), HOLD_BATTERY, "lift it out"),
            Waypoint(2.0, (bx, by, TRAVEL), HOLD_BATTERY, "back over its place"),
            Waypoint(1.5, (bx, by, BAT_GRIP + 0.002), HOLD_BATTERY, "set it down"),
            Waypoint(0.6, (bx, by, BAT_GRIP + 0.002), WIDE, "let go"),
            Waypoint(1.0, (bx, by, COVER_TRAVEL), WIDE, "rise"),
        ]
    if cover:
        right += _put_cover_on("cover_3", "battery_right")
    if mode != "partial":
        right += _into_box("battery_right")
    right.append(Waypoint(1.5, (0.20, -0.28, TRAVEL), WIDE, "park"))

    left: list[Waypoint] = []
    for c, b in (("cover_1", "battery_left"), ("cover_2", "battery_middle")):
        if cover:
            left += _put_cover_on(c, b)
        if mode != "partial" or b == "battery_left":
            left += _into_box(b)
    left.append(Waypoint(1.5, (0.20, 0.28, TRAVEL), WIDE, "park"))
    return {"left": left, "right": right}
