"""On the station table, a printed topographic map (an invented 1:10,000 sheet, north to the far side), laminated and
mounted on hardboard, is taped down between the arms. Its 1 km grid squares are 10 cm on the map: easting lines 31 to 35
run west to east (to the robot's right), northing lines 32 to 35 south to north, and every line carries its two-digit
number on two opposite edges. Beside the board stand two map markers, red on the left and blue on the right.

The robot is told to put the red marker in grid square 3234 and the blue marker on grid reference 346 332.
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
import layout as L  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

a.build()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
MAP_Z = TOP + L.BOARD_T  # the map's surface
START = {"red_marker": (0.25, 0.33), "blue_marker": (0.25, -0.33)}
MARKERS = list(START)

OBJECTS = [
    UsdBody("map_board", usd="assets/map_board.usda", pos=(0.0, 0.0, TOP), mass=None,
            size=(L.BOARD_X[1] - L.BOARD_X[0], L.BOARD_Y[1] - L.BOARD_Y[0], L.BOARD_T), friction=(0.5, 0.4)),
    *[UsdBody(n, usd=f"assets/{n}.usda", pos=(x, y, TOP + a.MARKER_H / 2 + 0.0002), mass=a.MARKER_KG,
              size=(2 * a.MARKER_R, 2 * a.MARKER_R, a.MARKER_H), friction=a.MARKER_FRICTION, solver_iters=32,
              torsional_patch=0.006)
      for n, (x, y) in START.items()],
]

# -- scoring (spec: "How it is scored") ----------------------------------------------------------------------------
TILT_MAX = 5.0  # degrees from vertical
SEAT_TOL = 0.003  # m: the base's centre from the map surface
TCP_CLEAR = 0.06  # m: both TCPs at least this far from the middle of the marker's axis
STILL = 0.01  # m/s
STILL_STEPS = int(round(1.0 * steptime.CONTROL_HZ))  # slower than STILL for the last 1 s


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def base_centre(st) -> np.ndarray:
    """Where the marker's axis meets its base."""
    return np.asarray(st.pos, float) - _rot(st.quat_xyzw)[:, 2] * (a.MARKER_H / 2)


def tilt(st) -> float:
    return math.degrees(math.acos(float(np.clip(abs(_rot(st.quat_xyzw)[2, 2]), 0.0, 1.0))))


def on_board(xy) -> bool:
    return L.BOARD_X[0] <= xy[0] <= L.BOARD_X[1] and L.BOARD_Y[0] <= xy[1] <= L.BOARD_Y[1]


def placed_checks(st, tcps, still_steps: int) -> dict:
    b = base_centre(st)
    up = _rot(st.quat_xyzw)[2, 2] > 0  # standing on its base, not on its top
    return {
        "upright": up and tilt(st) <= TILT_MAX,
        "on_map": abs(b[2] - MAP_Z) <= SEAT_TOL and on_board(b[:2]),
        "released": all(float(np.linalg.norm(np.asarray(t, float) - np.asarray(st.pos, float))) >= TCP_CLEAR for t in tcps),
        "still": still_steps >= STILL_STEPS,
    }


def judge(states: dict, tcps: list, still: dict) -> dict:
    """The spec's tests for both markers. Returns the per-marker verdicts."""
    red, blue = states["red_marker"], states["blue_marker"]
    rc = placed_checks(red, tcps, still["red_marker"])
    bc = placed_checks(blue, tcps, still["blue_marker"])
    r_xy, b_xy = base_centre(red)[:2], base_centre(blue)[:2]
    red_in = L.inside_square(r_xy[0], r_xy[1], *L.RED_SQUARE)
    blue_off = float(np.linalg.norm(b_xy - np.array(L.BLUE_POINT)))
    blue_point = blue_off <= L.POINT_TOL
    blue_square = L.inside_square(b_xy[0], b_xy[1], *L.BLUE_SQUARE)
    red_placed, blue_placed = all(rc.values()), all(bc.values())
    return {
        "red_pass": red_placed and red_in,
        "blue_pass": blue_placed and blue_point,
        "blue_square": blue_placed and blue_square and not blue_point,
        "red_failing": [k for k, v in rc.items() if not v] + ([] if red_in else ["not_in_3234"]),
        "blue_failing": [k for k, v in bc.items() if not v] + ([] if blue_point else [f"off_point_{blue_off * 100:.2f}cm"]),
        "red_xy": r_xy, "blue_xy": b_xy,
    }


def points(verdict: dict) -> float:
    """Run-limit score: 0.5 for red, plus 0.5 for blue on its point or 0.25 for blue only in its square."""
    return 0.5 * verdict["red_pass"] + (0.5 if verdict["blue_pass"] else 0.25 if verdict["blue_square"] else 0.0)


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, still={n: 0 for n in MARKERS}, verdict=None)


_reset_state()


def update(states: dict, tcps: list) -> dict:
    """One control step: count each marker's still steps and judge both markers."""
    _STATE["step"] += 1
    for n in MARKERS:
        slow = float(np.linalg.norm(states[n].lin_vel)) < STILL
        _STATE["still"][n] = _STATE["still"][n] + 1 if slow else 0
    _STATE["verdict"] = judge(states, tcps, _STATE["still"])
    return _STATE["verdict"]


def _trace(record: dict) -> None:
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "trace.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


@on_step
def _monitor(environment) -> None:
    from yam_arena.runtime import tcp_world
    from yam_arena.scene import object_states

    base = environment.unwrapped
    origin = base.scene.env_origins[0]
    origin = getattr(origin, "torch", origin).detach().cpu().numpy()
    tcps = [tcp_world(base.scene[f"{side}_robot"]) - origin for side in ("left", "right")]
    states = object_states(environment, MARKERS)
    v = update(states, tcps)
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2), "score_now": points(v),
           "red": [round(float(c), 4) for c in v["red_xy"]], "blue": [round(float(c), 4) for c in v["blue_xy"]],
           "red_failing": v["red_failing"], "blue_failing": v["blue_failing"],
           "tcp": [[round(float(c), 4) for c in t] for t in tcps]}
    for n in MARKERS:
        st = states[n]
        rec[n] = [round(float(c), 4) for c in base_centre(st)] + [round(tilt(st), 1)]
    _trace(rec)


def success(objects) -> bool:
    """Red in square 3234 and blue within 1 cm of 346 332, both placed; the harness ends the trial at 1.0 after 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    v = _STATE["verdict"]
    return bool(v and v["red_pass"] and v["blue_pass"])


def score(objects) -> float:
    """At the run limit: 0.5 if red passes, plus 0.5 if blue passes or 0.25 if blue is only in square 3433."""
    v = _STATE["verdict"]
    return points(v) if v else 0.0


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
PAD = 0.005  # fingertips this far above the surface the marker stands on
TRAVEL = TOP + 0.075  # TCP while carrying: the held marker's base is then about 5 cm up
HOVER = 0.025  # pauses this far above each grip and set-down height, so the arm settles first
HOLD = -(2 * a.MARKER_R - 0.008) / 2  # pads 8 mm narrower than the marker: about 20 N a finger
LEFT, RIGHT = -math.pi / 2, math.pi / 2  # fingers along x for each arm
PARK = {"left": (0.30, 0.31, TOP + 0.18), "right": (0.30, -0.31, TOP + 0.18)}


def _grip_z(surface: float) -> float:
    return surface + PAD + PAD_TIP_PAST_TCP


def _move(src, dst, yaw: float, release: bool = True, travel: float = TRAVEL) -> list[Waypoint]:
    """Pick the marker standing on the table at src (x, y) and stand it on the map at dst (x, y), 1 mm above it."""
    (sx, sy), (dx, dy) = src, dst
    gz, pz = _grip_z(TOP), _grip_z(MAP_Z + 0.001)
    out = [
        Waypoint(2.0, (sx, sy, travel), FINGER_OPEN, "over marker", yaw),
        Waypoint(1.0, (sx, sy, gz + HOVER), FINGER_OPEN, "near", yaw),
        Waypoint(0.5, (sx, sy, gz + HOVER), FINGER_OPEN, "settle", yaw),
        Waypoint(0.8, (sx, sy, gz), FINGER_OPEN, "down", yaw),
        Waypoint(0.4, (sx, sy, gz), FINGER_OPEN, "settle", yaw),
        Waypoint(0.6, (sx, sy, gz), HOLD, "grip", yaw),
        Waypoint(1.0, (sx, sy, travel), HOLD, "lift", yaw),
        Waypoint(2.2, (dx, dy, travel), HOLD, "carry", yaw),
        Waypoint(1.0, (dx, dy, pz + HOVER), HOLD, "lower", yaw),
        Waypoint(0.6, (dx, dy, pz + HOVER), HOLD, "settle", yaw),
        Waypoint(0.8, (dx, dy, pz), HOLD, "set down", yaw),
        Waypoint(0.4, (dx, dy, pz), HOLD, "settle", yaw),
    ]
    if release:
        out += [Waypoint(0.8, (dx, dy, pz), FINGER_OPEN, "let go", yaw),
                Waypoint(0.8, (dx, dy, travel), FINGER_OPEN, "rise", yaw)]
    else:
        out += [Waypoint(4.0, (dx, dy, pz), HOLD, "keep holding", yaw)]
    return out


def _park(arm: str, yaw: float, s: float = 1.5) -> Waypoint:
    return Waypoint(s, PARK[arm], FINGER_OPEN, "park", yaw)


def _pad(plans: dict) -> dict:
    """Both arms end parked, and wait there for the success hold."""
    for arm, yaw in (("left", LEFT), ("right", RIGHT)):
        plan = plans.setdefault(arm, [])
        if not plan or plan[-1].label != "keep holding":
            plan.append(_park(arm, yaw))
    span = max(sum(w.duration for w in p) for p in plans.values())
    for p in plans.values():
        p.append(Waypoint(span - sum(w.duration for w in p) + 3.0, p[-1].pos, p[-1].finger, "wait", p[-1].yaw))
    return plans


# Targets (x, y) on the map
RED_AT = L.point(32.5, 34.5)  # the middle of square 3234: (0.41, +0.05)
BLUE_AT = L.BLUE_POINT  # (0.285, -0.165)
VARIANTS = {
    # mode: (red target, blue target); see demo()
    "success": (RED_AT, BLUE_AT),
    "crossing": (L.point(32, 34), BLUE_AT),  # red on the crossing of lines 32 and 34
    "southwest": (L.point(31.5, 33.5), BLUE_AT),  # red in square 3133, south-west of the crossing
    "offpoint": (RED_AT, L.point(34.5, 33.5)),  # blue in the middle of square 3433, 2.9 cm off its point
    "held": (RED_AT, BLUE_AT),  # blue still held at the end
}
NORTH_FIRST_RED = L.point(34.5, 32.5)  # 3234 read northing first: square 3432
NORTH_FIRST_BLUE = (0.41, -0.025)  # within 1.5 cm of (0.425, -0.025), where 346 332 lands read northing first


def demo():
    """Both arms at once: the left arm stands the red marker in the middle of square 3234 and the right arm stands the blue
    marker on the centre of the 100 m square 346 332, 25 cm apart. Variants (TASK_DEMO), to check the scorer: crossing,
    southwest, offpoint and held move the markers as VARIANTS says; northfirst reads every reference northing first (the
    right arm sets blue, then the left arm reaches across for red); nothing does nothing."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return _pad({"left": [Waypoint(3.0, PARK["left"], FINGER_OPEN, "hold", LEFT)]})
    if mode == "northfirst":
        right = _move(START["blue_marker"], NORTH_FIRST_BLUE, RIGHT, travel=TOP + 0.05) + [_park("right", RIGHT)]
        wait = sum(w.duration for w in right)
        left = [Waypoint(wait, PARK["left"], FINGER_OPEN, "wait clear", LEFT)]
        left += _move(START["red_marker"], NORTH_FIRST_RED, LEFT, travel=TOP + 0.05)
        return _pad({"left": left, "right": right})
    red_at, blue_at = VARIANTS[mode]
    return _pad({"left": _move(START["red_marker"], red_at, LEFT),
                 "right": _move(START["blue_marker"], blue_at, RIGHT, release=mode != "held")})
