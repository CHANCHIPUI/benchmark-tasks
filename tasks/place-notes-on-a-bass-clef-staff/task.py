"""On the station table, a music teaching board lies flat between the arms: white melamine printed with a five-line staff
(lines along y, pitch rising away from the arms), a bass clef at its left end, and bar lines that make four bars. Beside
each end of the board, a shallow beech tray holds two loose note heads: black-lacquered beech discs, 40 mm across and
18 mm thick. The board and trays are fixed; the note heads are loose.
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
from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

a.build()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
BOARD_Z = TOP + a.BOARD_TOP  # the board's printed surface
FLOOR_Z = TOP + a.TRAY_FLOOR  # the trays' floor

# -- layout (station frame, metres: x forward, y to the robot's left) ---------------------------------------------
START = {
    "head_1": (a.TRAY_XY["left"][0], a.TRAY_SLOTS["left"][0]),
    "head_2": (a.TRAY_XY["left"][0], a.TRAY_SLOTS["left"][1]),
    "head_3": (a.TRAY_XY["right"][0], a.TRAY_SLOTS["right"][0]),
    "head_4": (a.TRAY_XY["right"][0], a.TRAY_SLOTS["right"][1]),
}
HEADS = list(START)

OBJECTS = [
    *[UsdBody(n, usd="assets/note_head.usda", pos=(x, y, FLOOR_Z + a.DISC_H / 2 + 0.0002), mass=a.DISC_MASS,
              size=(2 * a.DISC_R, 2 * a.DISC_R, a.DISC_H), friction=a.DISC_FRICTION, solver_iters=32,
              torsional_patch=0.008)
      for n, (x, y) in START.items()],
    UsdBody("bench", usd="assets/bench.usda", pos=(0.0, 0.0, TOP), mass=None, size=(0.25, 0.63, 0.012),
            friction=(0.5, 0.4)),
]

# -- scoring ------------------------------------------------------------------------------------------------------
# Staff positions 0..8 from line 1 (nearest the arms) up: line n is 2(n - 1), space n is 2n - 1. Bass clef: lines G B D F A,
# spaces A C E G, from the bottom.
POS = {"line 1": 0, "space 1": 1, "line 2": 2, "space 2": 3, "line 3": 4, "space 3": 5, "line 4": 6, "space 4": 7, "line 5": 8}
TARGET = [("D", "line 3"), ("F", "line 4"), ("C", "space 2"), ("E", "space 3")]  # bars 1 to 4, left to right
TARGET_X = [a.POSITION_X[POS[p]] for _, p in TARGET]
PITCH_TOL = 0.0075  # m, front to back, from the target line's or space's centre line
BAR_CLEAR = 0.025  # m, the head's centre from each of its bar's bar lines
TILT_MAX = 10.0  # degrees
FLAT_TOL = 0.003  # m, the head's underside from the board's surface
TCP_CLEAR = 0.04  # m, both grippers' TCPs from the head's centre
STILL = 0.01  # m/s
BOARD_X = (a.BOARD_XY[0] - a.BOARD[0] / 2, a.BOARD_XY[0] + a.BOARD[0] / 2)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _tilt(st) -> float:
    """Degrees between the disc's axis and the vertical. A disc is the same either way up, so face down counts as flat."""
    return math.degrees(math.acos(float(np.clip(abs(_rot(st.quat_xyzw)[2, 2]), 0.0, 1.0))))


def _underside(st) -> float:
    """The lowest point of the disc (a cylinder of radius DISC_R and height DISC_H about its local z)."""
    t = math.radians(_tilt(st))
    return float(st.pos[2]) - (a.DISC_H / 2 * math.cos(t) + a.DISC_R * math.sin(t))


def bar_of(st) -> int | None:
    """The bar (0-based) whose two bar lines the head's centre lies between, over the board, seen from above."""
    x, y = float(st.pos[0]), float(st.pos[1])
    if not BOARD_X[0] < x < BOARD_X[1]:
        return None
    for k in range(4):
        if a.BAR_EDGES[k + 1] < y < a.BAR_EDGES[k]:
            return k
    return None


def bar_checks(k: int, states: dict, tcps: list) -> dict:
    """The five conditions of 'How it is scored' for bar k (0-based), from the heads' states and the TCP positions."""
    inside = [n for n in HEADS if bar_of(states[n]) == k]
    if len(inside) != 1:
        return {"one_head": False}
    st = states[inside[0]]
    x, y = float(st.pos[0]), float(st.pos[1])
    centre = np.asarray(st.pos, float)
    return {
        "one_head": True,
        "clear_of_bar_lines": min(a.BAR_EDGES[k] - y, y - a.BAR_EDGES[k + 1]) >= BAR_CLEAR,
        "right_pitch": abs(x - TARGET_X[k]) <= PITCH_TOL,
        "flat": _tilt(st) < TILT_MAX and abs(_underside(st) - BOARD_Z) <= FLAT_TOL,
        "released_still": all(float(np.linalg.norm(np.asarray(t, float) - centre)) >= TCP_CLEAR for t in tcps)
                          and float(np.linalg.norm(st.lin_vel)) < STILL,
    }


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(passes=[False] * 4, step=0)


_reset_state()


def update(states: dict, tcps: list) -> list:
    """One control step: judge the four bars. Nothing latches: heads may be lifted and placed again."""
    _STATE["step"] += 1
    checks = [bar_checks(k, states, tcps) for k in range(4)]
    _STATE["passes"] = [all(c.values()) for c in checks]
    return checks


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
    states = object_states(environment, HEADS)
    checks = update(states, tcps)
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2),
           "bars": [[c for c, v in chk.items() if not v] for chk in checks],
           "tcp": [[round(float(v), 4) for v in t] for t in tcps]}
    if os.environ.get("TASK_SCORE_DIR"):
        from scenekit import arm_joints
        names = [f"joint{i}" for i in range(1, 7)]
        rec["q"] = [[round(float(v), 4) for v in arm_joints(environment, side, names)] for side in ("left", "right")]
    for n in HEADS:
        st = states[n]
        rec[n] = [round(float(v), 4) for v in st.pos] + [round(_tilt(st), 1), round(float(np.linalg.norm(st.lin_vel)), 4)]
    _trace(rec)


def success(objects) -> bool:
    """All four bars pass; the harness ends the trial at 1.0 after 1 s of this."""
    steptime.tick()  # the harness calls success() once per control step
    return bool(all(_STATE["passes"]))


def score(objects) -> float:
    """0.25 for each bar that passes at the end (see bar_checks)."""
    return 0.25 * sum(_STATE["passes"])


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
TRAVEL = TOP + 0.08  # a held head's bottom is then about 6.2 cm up, clear of the rims and the board
LIFT_PAD = 0.005  # pad tips this far above the surface the head lies on
GRIP_Z = FLOOR_Z + LIFT_PAD + PAD_TIP_PAST_TCP
SET_Z = BOARD_Z + 0.001 + LIFT_PAD + PAD_TIP_PAST_TCP  # the held head's underside 1 mm above the board
OPEN = -0.035  # pads 70 mm apart: 15 mm clear of the 40 mm head on each side
HOLD = -(2 * a.DISC_R - 0.008) / 2  # pads 8 mm narrower than the head: about 20 N a finger
YAW = 0.0  # fingers along y throughout: clear of the tray rims in front of and behind each head, no wrist turn
PARK = {"left": (0.22, 0.44, TOP + 0.14), "right": (0.22, -0.34, TOP + 0.14)}  # outboard, clear of the clef in view
HOVER = 0.025  # pauses this far above each grip and set-down height, so the arm settles before the last centimetres
# Open-loop aim corrections (m), measured once from a logged run: where the arm's TCP lands for a commanded place.
AIM = {}

# Where each bar's head is set: well inside the bar, toward the arm that sets it.
BAR_SET_Y = [0.17, 0.07, -0.07, -0.17]


def _move(arm: str, label: str, src, dst, release: bool = True) -> list[Waypoint]:
    """Pick the head lying at src (x, y) in a tray and set it on the board at dst (x, y)."""
    sx, sy = src
    dx, dy = AIM.get((arm, round(dst[0], 4), round(dst[1], 4)), (0.0, 0.0))
    dx, dy = dst[0] + dx, dst[1] + dy
    yp = yd = YAW
    out = [
        Waypoint(1.8, (sx, sy, TRAVEL), OPEN, f"over {label}", yp),
        Waypoint(0.9, (sx, sy, GRIP_Z + HOVER), OPEN, "near", yp),
        Waypoint(0.5, (sx, sy, GRIP_Z + HOVER), OPEN, "settle", yp),
        Waypoint(0.8, (sx, sy, GRIP_Z), OPEN, "down", yp),
        Waypoint(0.3, (sx, sy, GRIP_Z), OPEN, "settle", yp),
        Waypoint(0.5, (sx, sy, GRIP_Z), HOLD, "grip", yp),
        Waypoint(0.8, (sx, sy, TRAVEL), HOLD, "lift", yp),
        Waypoint(2.2, (dx, dy, TRAVEL), HOLD, "carry", yd),
        Waypoint(0.9, (dx, dy, SET_Z + HOVER), HOLD, "lower", yd),
        Waypoint(0.5, (dx, dy, SET_Z + HOVER), HOLD, "settle", yd),
        Waypoint(0.8, (dx, dy, SET_Z), HOLD, "set down", yd),
        Waypoint(0.3, (dx, dy, SET_Z), HOLD, "settle", yd),
    ]
    if release:
        # Rise until the pad tips are 1 cm above the head's top, then the next waypoint (the next pick, or the park
        # pose) is a straight line up and away, so the arm is clear of the board soon after the head is released.
        out += [Waypoint(0.6, (dx, dy, SET_Z), OPEN, "let go", yd),
                Waypoint(0.5, (dx, dy, SET_Z + a.DISC_H + 0.004), OPEN, "rise", yd)]
    else:
        out += [Waypoint(3.0, (dx, dy, SET_Z), HOLD, "keep holding", yd)]
    return out


def _span(plan) -> float:
    return sum(w.duration for w in plan)


def _schedule(rounds) -> dict:
    """rounds: a list of rounds, each a list of (arm, waypoints) run at the same time. An arm with nothing to do in a round
    waits at its park pose, outboard, so only one arm is ever near the middle bars."""
    plans = {"left": [], "right": []}
    for moves in rounds:
        busy = dict(moves)
        length = max(_span(w) for w in busy.values())
        for arm in plans:
            if arm in busy:
                plans[arm] += busy[arm]
                if _span(busy[arm]) < length:
                    last = plans[arm][-1]
                    plans[arm].append(Waypoint(length - _span(busy[arm]), last.pos, last.finger, "wait", last.yaw))
            else:
                plans[arm].append(Waypoint(length, PARK[arm], OPEN, "wait clear", YAW))
    return plans


def _plan(bar_x: list, bar_y: list = BAR_SET_Y, held: bool = False) -> dict:
    """Heads to bars 1..4 at x bar_x[k] (None leaves the bar empty), y bar_y[k]. The left arm sets bars 1 and 2 from its
    tray, the right arm bars 4 and 3 from its tray: round 1 both arms (bars 1 and 4, 34 cm apart), round 2 the left arm
    (bar 2), round 3 the right arm (bar 3)."""
    jobs = {0: ("left", "head_2"), 1: ("left", "head_1"), 3: ("right", "head_4"), 2: ("right", "head_3")}
    rounds = []
    for group in ((0, 3), (1,), (2,)):
        moves = []
        for k in group:
            if bar_x[k] is None:
                continue
            arm, head = jobs[k]
            moves.append((arm, _move(arm, f"{head} to bar {k + 1}", START[head], (bar_x[k], bar_y[k]),
                                     release=not (held and k == 2))))
        if moves:
            rounds.append(moves)
    plans = _schedule(rounds)
    for arm, plan in plans.items():
        if not (held and arm == "right"):
            plan.append(Waypoint(1.0, PARK[arm], OPEN, "park", YAW))
        plan.append(Waypoint(1.5, plan[-1].pos, plan[-1].finger, "settle", plan[-1].yaw))
    return plans


def _x(position: str) -> float:
    return a.POSITION_X[POS[position]]


def demo():
    """The bass reading: D on line 3 in bar 1, F on line 4 in bar 2, C in space 2 in bar 3, E in space 3 in bar 4.
    Variants (TASK_DEMO), to check the scorer:
      treble    the treble reading: D line 4, F line 5, C space 3, E space 4 (0)
      reverse   the right pitches in reverse order: E, C, F, D (0)
      ascending the four letters in pitch order, as a scale: C, D, E, F (0)
      onetreble right except E on its treble place, space 4 (0.75)
      held      right, but the head in bar 3 is still held at the end (0.75)
      chord     D and F both in bar 1, bar 2 empty (0.5)
      gap       right, except F set 12 mm above line 4, between it and space 4 (0.75)
      barline   right, except D set 2 cm from bar 1's left bar line (0.75)
      nothing   does nothing (0)"""
    mode = os.environ.get("TASK_DEMO", "success")
    right = [_x("line 3"), _x("line 4"), _x("space 2"), _x("space 3")]
    if mode == "nothing":
        return {arm: [Waypoint(4.0, PARK[arm], OPEN, "park", YAW)] for arm in PARK}
    if mode == "treble":
        return _plan([_x("line 4"), _x("line 5"), _x("space 3"), _x("space 4")])
    if mode == "reverse":
        return _plan([_x("space 3"), _x("space 2"), _x("line 4"), _x("line 3")])
    if mode == "ascending":
        return _plan([_x("space 2"), _x("line 3"), _x("space 3"), _x("line 4")])
    if mode == "onetreble":
        return _plan(right[:3] + [_x("space 4")])
    if mode == "held":
        return _plan(right, held=True)
    if mode == "chord":  # F goes into bar 1 with D, 5 cm to its right, so bar 2 stays empty
        ys = list(BAR_SET_Y)
        ys[0], ys[1] = 0.19, 0.14
        return _plan(right, bar_y=ys)
    if mode == "gap":
        return _plan([right[0], right[1] + 0.012] + right[2:])
    if mode == "barline":
        ys = list(BAR_SET_Y)
        ys[0] = a.BAR_EDGES[0] - 0.020
        return _plan(right, bar_y=ys)
    return _plan(right)
