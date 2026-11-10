"""On the station table, in front of the arms: a cleared 5-rod teacher's soroban lying flat on a black rubber desk mat.

The soroban (32 cm wide, 28 cm deep, 5 cm thick) faces the robot: its rods run away from the robot, with the beam
across them nearer the far side. Its frame is lacquered brown hardwood and its beam is white, with two black unit
dots on rods 1 and 4 (counting from the robot's left). Each rod carries five yellow plastic beads (40 mm across,
24 mm thick): one 5-bead between the beam and the far bar, and four 1-beads between the beam and the near bar. A bead
counts when it is pushed against the beam. At the start every 5-bead rests against the far bar and every 1-bead
against the near bar, so the soroban shows 0. The soroban is taped down; its beads slide along their rods and stay
where they are pushed.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import soroban_usd as S  # noqa: E402
import steptime  # noqa: E402
import scenekit  # noqa: E402
from scenekit import UsdBody, on_reset  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = S.build()

MAT = UsdBody("desk_mat", usd=USD["mat"], pos=(0.0, 0.0, TOP), size=S.MAT_SIZE)
SOROBAN = UsdBody("soroban", usd=USD["soroban"], pos=(0.0, 0.0, TOP), size=(0.28, 0.32, 0.05), articulation=True,
                  joint_pos=((".*", 0.0),), joint_damping=1.0, friction=(0.6, 0.5), solver_iters=16)
OBJECTS = [MAT, SOROBAN]

# -- scoring --------------------------------------------------------------------------------------------------------
AT_BEAM = 0.005        # m: a bead is at the beam with this much empty rod, or less, between it and the beam
AWAY = 0.020           # m: ... and away with this much or more; in between it is undecided
STILL = 0.001          # m: no bead moves more than this during the 1 s hold
DAMAGE = 0.004         # m: a bead forced this far past its stop at the beam or the frame damages the soroban
FRAME_MOVED = 0.01     # m: the frame moving this far fails the trial (only possible in a real trial)
TARGET = {1: 2, 2: 6, 3: 0, 4: 5, 5: 0}   # rod -> digit; 2,605 with the ones digit on rod 4
MUST_CLEAR = (3, 5)                       # the zero (tens) and the spare rod right of the ones rod
DIGIT_RODS = (1, 2, 4)
PER_DIGIT = 0.3
FRAME_START = np.array(SOROBAN.pos, float)

_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Failures stay failed for the whole trial: only a reset clears them."""
    _T.clear()
    _T.update(failed=None, step=0, hold=None, worst_past_stop=0.0)


_reset()


def moved(joints: dict, rod: int, bead: str) -> float:
    """How far a bead has moved from its cleared stop toward the beam (m)."""
    q = float(joints[S.joint_name(rod, bead)])
    return -q if bead == "h" else q


def empty_space(joints: dict, rod: int, bead: str) -> float:
    """The empty rod between the bead and the beam, not counting the beads in between (m)."""
    return S.TRAVEL - moved(joints, rod, bead)


def bead_state(joints: dict, rod: int, bead: str) -> str:
    e = empty_space(joints, rod, bead)
    if e <= AT_BEAM:
        return "at"
    return "away" if e >= AWAY else "undecided"


def rod_digit(joints: dict, rod: int):
    """5 for the 5-bead at the beam plus 1 for each 1-bead at the beam; None (unreadable) if any bead is undecided."""
    states = {b: bead_state(joints, rod, b) for b in ("h", *S.EARTH)}
    if "undecided" in states.values():
        return None
    return 5 * (states["h"] == "at") + sum(states[b] == "at" for b in S.EARTH)


def readings(joints: dict) -> dict:
    return {r: rod_digit(joints, r) for r in range(1, 6)}


def _update(objects) -> None:
    """Called every control step; latches the first failure."""
    s = objects["soroban"]
    past = max(max(moved(s.joints, r, b) - S.TRAVEL, -moved(s.joints, r, b)) for r, b in S.bead_names())
    _T["worst_past_stop"] = max(_T["worst_past_stop"], past)
    if _T["failed"]:
        return
    if past > DAMAGE:
        _T["failed"] = f"damage: a bead {past * 1000:.1f} mm past its stop"
    elif np.linalg.norm(np.asarray(s.pos) - FRAME_START) > FRAME_MOVED:
        _T["failed"] = "the frame moved"


def score(objects) -> float:
    """0 for the whole trial after a failure (a bead forced more than 4 mm past its stop; the frame moved more than 1 cm).
    Otherwise 0 unless rods 3 and 5 are cleared (every bead away; an unreadable rod is not cleared), else 0.3 for each of
    rod 1 reading 2, rod 2 reading 6 and rod 4 reading 5."""
    _update(objects)
    if _T["failed"]:
        return 0.0
    read = readings(objects["soroban"].joints)
    if any(read[r] != 0 for r in MUST_CLEAR):
        return 0.0
    return round(PER_DIGIT * sum(read[r] == TARGET[r] for r in DIGIT_RODS), 3)


def _positions(joints: dict) -> np.ndarray:
    return np.array([moved(joints, r, b) for r, b in S.bead_names()])


def success(objects) -> bool:
    """The rods read 2, 6, 0, 5, 0 and no bead moves more than 1 mm while the harness holds this for 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    _T["step"] += 1
    s = score(objects)
    joints = objects["soroban"].joints
    read = readings(joints)
    now = _positions(joints)
    ok = not _T["failed"] and all(read[r] == TARGET[r] for r in TARGET)
    if not ok:
        _T["hold"], holding = None, False
    elif _T["hold"] is None:
        _T["hold"], holding = now, True          # the 1 s window starts here
    elif np.abs(now - _T["hold"]).max() > STILL:
        _T["hold"], holding = now, False         # a bead moved more than 1 mm: the harness's 1 s count starts again
    else:
        holding = True
    _log(objects, s, read, holding)
    return holding


def _tcp() -> dict:
    """Each arm's TCP (gripper link origin moved 13.4 cm along a straight-down approach), for debugging only."""
    env = scenekit.env()
    if env is None:
        return {}
    links = scenekit.robot_links(env)
    return {side: [round(float(v), 4) for v in links[side]["gripper"] - (0.0, 0.0, 0.134)] for side in links}


def _log(objects, s: float, read: dict, holding: bool) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    joints = objects["soroban"].joints
    row = {"step": _T["step"], "score": s, "failed": _T["failed"], "read": read, "holding": holding,
           "past_stop_mm": round(_T["worst_past_stop"] * 1000, 2), "tcp": _tcp(),
           "moved_mm": {f"{r}{b}": round(moved(joints, r, b) * 1000, 2) for r, b in S.bead_names()}}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Closed fingers turned to close along x (yaw -90 left, +90 right): the closed tips are about 13 mm along the rod and
# 26 mm across it over their lowest 16 mm. 1-beads are pushed from the V-shaped groove behind the group to be set; a
# 5-bead lies against the far bar, so the tips press down on it just behind its rim and slide it to the beam.
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
TIP = 0.013                         # fingertips below the TCP
GROOVE_DEPTH = 0.0045               # fingertips this far below the bead tops in a groove
PRESS = 0.005                       # fingertips commanded this far into the top of a 5-bead (the arm is compliant)
FIVE_OFFSET = 0.003                 # ... and this far from its centre toward the far bar, behind its rim
ABOVE = 0.075                       # TCP height for moving between rods: fingertips 9 mm above the frame
PUSH_PAST = 0.004                   # push this far past the stop, so the group surely reaches the beam


def _wp(side, t, x, y, h, label) -> Waypoint:
    return Waypoint(t, (float(x), float(y), TOP + h), FINGER_CLOSED, label, YAW[side])


def _groove_x(rod_beads_up: int) -> float:
    """x of the groove behind the n-th 1-bead (counting from the beam), in the cleared stack."""
    b = S.EARTH[rod_beads_up - 1]
    return S.EARTH_X[b] - S.BEAD_T / 2


def push_ones(side, rod: int, n: int, travel: float = S.TRAVEL + PUSH_PAST) -> list[Waypoint]:
    """Push the n 1-beads nearest the beam up to it, from the groove behind them."""
    y = S.ROD_Y[rod - 1]
    x0 = _groove_x(n)
    h = S.BEAD_TOP - GROOVE_DEPTH + TIP
    return [_wp(side, 2.0, x0, y, ABOVE, f"above rod {rod}, behind 1-bead {n}"),
            _wp(side, 1.5, x0, y, h, "down into the groove"),
            _wp(side, 2.0, x0 + travel, y, h, f"push {n} 1-bead(s) to the beam"),
            _wp(side, 0.6, x0 + travel, y, ABOVE, "up")]


def slide_five(side, rod: int, travel: float = S.TRAVEL + 0.002, back: bool = False) -> list[Waypoint]:
    """Press on the 5-bead's top and slide it toward the beam (or back to the frame with back=True)."""
    y = S.ROD_Y[rod - 1]
    x0 = S.HEAVEN_X - (S.TRAVEL if back else 0.0) + (-FIVE_OFFSET if back else FIVE_OFFSET)
    x1 = x0 + (travel if back else -travel)
    h = S.BEAD_TOP - PRESS + TIP
    return [_wp(side, 2.0, x0, y, ABOVE, f"above rod {rod}'s 5-bead"),
            _wp(side, 1.5, x0, y, h, "press on its top"),
            _wp(side, 2.0, x1, y, h, "slide it " + ("back to the frame" if back else "to the beam")),
            _wp(side, 0.6, x1, y, ABOVE, "up")]


def clear(side, t: float = 1.5) -> list[Waypoint]:
    y = 0.25 if side == "left" else -0.25
    return [_wp(side, t, 0.25, y, 0.16, "clear of the soroban")]


def wait(side, t: float) -> list[Waypoint]:
    y = 0.25 if side == "left" else -0.25
    return [_wp(side, t, 0.25, y, 0.16, "wait")]


def demo():
    """Open loop, so it uses nothing a policy could not see. The left arm sets rod 1 (2: two 1-beads) and rod 2 (6: one
    1-bead, then the 5-bead); the right arm sets rod 4 (5: the 5-bead). TASK_DEMO picks a variant for checking the
    scorer: ones (every bead counted as 1), shifted (one rod to the right), nozero (2, 6, 5 on rods 2-4), halfway (rod
    4's 5-bead left halfway), redo (a 5-bead set on the spare rod and slid back first), nothing, force (a 5-bead
    pressed hard and driven 2 cm past its stop)."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"left": clear("left", 2.0) + wait("left", 8.0)}
    if mode == "force":      # a 5-bead commanded 9 mm into its top and driven 2 cm past its stop at the beam
        right = slide_five("right", 4, travel=S.TRAVEL + 0.02)
        right[1:3] = [Waypoint(w.duration, (w.pos[0], w.pos[1], w.pos[2] - 0.004), w.finger, w.label, w.yaw)
                      for w in right[1:3]]
        return {"right": right + clear("right") + wait("right", 3.0)}
    if mode == "ones":       # every bead worth 1: 3 1-beads for the 5 on rod 4, then 2 on rod 1 and 3 for the 6 on rod 2
        return _right_then_left(push_ones("right", 4, 3), push_ones("left", 1, 2) + push_ones("left", 2, 3))
    if mode == "shifted":    # the rightmost rod taken as the ones place: 2, 6, 0, 5 on rods 2, 3, 4, 5
        return _right_then_left(slide_five("right", 5),
                                push_ones("left", 2, 2) + push_ones("left", 3, 1) + slide_five("left", 3))
    if mode == "nozero":     # the zero skipped: 2, 6, 5 on rods 2, 3, 4
        return _right_then_left(slide_five("right", 4),
                                push_ones("left", 2, 2) + push_ones("left", 3, 1) + slide_five("left", 3))
    if mode == "redo":       # out and back: a 5-bead set on the spare rod 5 and slid back to the frame, then the job
        return _right_then_left(slide_five("right", 5) + slide_five("right", 5, back=True) + slide_five("right", 4),
                                push_ones("left", 1, 2) + push_ones("left", 2, 1) + slide_five("left", 2))
    left = push_ones("left", 1, 2) + push_ones("left", 2, 1) + slide_five("left", 2) + clear("left") + wait("left", 2.0)
    if mode == "halfway":    # rod 4's 5-bead slid only halfway: rod 4 unreadable
        right = slide_five("right", 4, travel=S.TRAVEL / 2) + clear("right")
    else:                    # the demo: the right arm works at the far end of rod 4 while the left works near rod 1
        right = slide_five("right", 4) + clear("right")
    return {"left": left, "right": right}


def _right_then_left(right: list[Waypoint], left: list[Waypoint]) -> dict:
    """The right arm works first and clears; the left arm waits beside the table until then, so the wrists never meet."""
    busy = sum(w.duration for w in right) + 1.5
    return {"right": right + clear("right") + wait("right", 2.0),
            "left": wait("left", busy) + left + clear("left") + wait("left", 2.0)}
