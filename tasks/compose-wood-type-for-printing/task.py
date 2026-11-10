"""On the station table, a letterpress composing bench. Further out, between the arms, a steel imposing surface carries a
cast-iron chase whose inside is divided by three fixed hardwood furniture strips into four bays in a row, each 5.6 cm
wide and 13 cm long. Nearer the arms, on a dark rubber bench mat, lie five loose 18-line wood-type sorts (end-grain
maple, 76.2 mm head to foot and type-high, 23.3 mm), face up, each with its long side front to back: the capitals L, P,
J and K in a row from left to right, and an E further to the right. Each face carries its letter in dried black ink.
The chase, the furniture, the surface and the mat are fixed; the sorts are loose.
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
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

a.build()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
BED_Z = TOP + a.BED  # the imposing surface's top
MAT_TOP = TOP + a.MAT[2]
CX = a.CX

# -- layout (station frame, metres: x forward, y to the robot's left) ---------------------------------------------
# letter: (x, y, heading of the letter's head in degrees; 0 is away from the arms)
START = {"L": (0.20, 0.24, 0.0), "P": (0.20, 0.165, 0.0), "J": (0.20, 0.0, 0.0), "K": (0.20, -0.165, 0.0),
         "E": (0.215, -0.25, 180.0)}
SORTS = [f"sort_{c}" for c in START]


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))


OBJECTS = [
    *[UsdBody(f"sort_{c}", usd=f"assets/sort_{c}.usda", pos=(x, y, MAT_TOP + a.TYPE_HIGH / 2 + 0.0002),
              rot_xyzw=_yaw_quat(h), mass=a.MASS[c], size=(a.BODY, a.SET_WIDTH[c], a.TYPE_HIGH),
              friction=a.SORT_FRICTION, solver_iters=32, torsional_patch=0.008)
      for c, (x, y, h) in START.items()],
    UsdBody("bench", usd="assets/bench.usda", pos=(0.0, 0.0, TOP), mass=None, size=(0.20, 0.60, 0.022),
            friction=(0.5, 0.4)),
]

# -- scoring ------------------------------------------------------------------------------------------------------
FORME = ["P", "L", "E", "K"]  # bays 1 to 4, left to right as seen from the arms: KELP mirror-reversed
SEAT_TOL = 0.002  # m: the sort's bottom from the bed
TILT_MAX = 5.0  # degrees
LINE_TOL = 0.004  # m: the sort's centre from the bays' centre line, front to back
HEAD_TOL = 15.0  # degrees from straight away from the arms
OVER_EPS = 0.001  # m: overlap below this does not count as "over" (numerical contact)
TCP_CLEAR = 0.03  # m: both grippers' TCPs at least this far from the sort
STILL = 0.01  # m/s
DROP_Z = TOP - 0.05  # a sort's centre below this has fallen off the table


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _half(name: str) -> np.ndarray:
    return np.array([a.BODY / 2, a.SET_WIDTH[name[-1]] / 2, a.TYPE_HIGH / 2])


def _corners(name: str, st) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return np.asarray(st.pos, float) + (signs * _half(name)) @ _rot(st.quat_xyzw).T


def _cross2(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    p = sorted(map(tuple, np.round(points, 9)))
    if len(p) < 3:
        return np.array(p)

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and _cross2(np.subtract(out[-1], out[-2]), np.subtract(q, out[-2])) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _bay_rect(k: int) -> tuple[float, float, float, float]:
    y = a.BAY_Y[k]
    return CX - a.BAY_L / 2, CX + a.BAY_L / 2, y - a.BAY_W / 2, y + a.BAY_W / 2


def _over(name: str, st, rect) -> bool:
    """Any part of the sort, seen from above, over the rectangle (by more than OVER_EPS): separating-axis test."""
    poly = _hull(_corners(name, st)[:, :2])
    x0, x1, y0, y1 = rect
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.linalg.norm(e) > 1e-9:
            axes.append(np.array([-e[1], e[0]]) / np.linalg.norm(e))
    for ax in axes:
        p, b = poly @ ax, box @ ax
        if min(p.max(), b.max()) - max(p.min(), b.min()) <= OVER_EPS:
            return False
    return True


def _tcp_distance(name: str, st, tcp) -> float:
    local = _rot(st.quat_xyzw).T @ (np.asarray(tcp, float) - np.asarray(st.pos, float))
    return float(np.linalg.norm(np.maximum(np.abs(local) - _half(name), 0.0)))


def _tilt(st) -> float:
    return math.degrees(math.acos(float(np.clip(_rot(st.quat_xyzw)[2, 2], -1.0, 1.0))))


def _heading(st) -> float:
    """Where the letter's head points, seen from above: degrees from straight away from the arms."""
    r = _rot(st.quat_xyzw)
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def bay_checks(k: int, states: dict, tcps: list) -> dict:
    """The six conditions of 'How it is scored' for bay k (0-based), from the sorts' states and the TCP positions."""
    name = f"sort_{FORME[k]}"
    st = states[name]
    x0, x1, y0, y1 = rect = _bay_rect(k)
    x, y = float(st.pos[0]), float(st.pos[1])
    low = float(_corners(name, st)[:, 2].min())
    return {
        "in_bay": x0 < x < x1 and y0 < y < y1,
        "seated": abs(low - BED_Z) <= SEAT_TOL and _tilt(st) < TILT_MAX,
        "on_line": abs(x - CX) <= LINE_TOL,
        "head": abs(_heading(st)) <= HEAD_TOL,
        "alone": not any(_over(n, states[n], rect) for n in SORTS if n != name),
        "released": all(_tcp_distance(name, st, t) >= TCP_CLEAR for t in tcps)
                    and float(np.linalg.norm(st.lin_vel)) < STILL,
    }


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(failed=False, why=[], passes=[False] * 4, step=0)


_reset_state()


def update(states: dict, tcps: list) -> list:
    """One control step: latch a drop the moment it happens (only a reset clears it) and judge the four bays."""
    _STATE["step"] += 1
    for n in SORTS:
        if float(states[n].pos[2]) < DROP_Z and n not in _STATE["why"]:
            _STATE["failed"] = True
            _STATE["why"].append(n)
    checks = [bay_checks(k, states, tcps) for k in range(4)]
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
    states = object_states(environment, SORTS)
    checks = update(states, tcps)
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2), "failed": _STATE["why"],
           "bays": [[k for k, v in c.items() if not v] for c in checks],
           "tcp": [[round(float(v), 4) for v in t] for t in tcps]}
    if os.environ.get("TASK_SCORE_DIR"):
        from scenekit import arm_joints
        names = [f"joint{i}" for i in range(1, 7)]
        rec["q"] = [[round(float(v), 4) for v in arm_joints(environment, side, names)] for side in ("left", "right")]
    for n in SORTS:
        st = states[n]
        rec[n] = [round(float(v), 4) for v in st.pos] + [round(_tilt(st), 1), round(_heading(st), 1)]
    _trace(rec)


def success(objects) -> bool:
    """All four bays pass and no sort has fallen off the table; the harness ends the trial at 1.0 after 1 s of this."""
    steptime.tick()  # the harness calls success() once per control step
    return bool(not _STATE["failed"] and all(_STATE["passes"]))


def score(objects) -> float:
    """0 if any sort has ever fallen off the table. Otherwise 0.25 for each bay that passes (see bay_checks)."""
    if _STATE["failed"]:
        return 0.0
    return 0.25 * sum(_STATE["passes"])


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
TRAVEL = TOP + 0.085  # a held sort's bottom is then 6.7 cm up, clear of the chase (2.2 cm) and the other sorts
LIFT_PAD = 0.005  # pad tips this far above the surface the sort stands on
GRIP_Z = MAT_TOP + LIFT_PAD + PAD_TIP_PAST_TCP
SET_Z = BED_Z + 0.0005 + LIFT_PAD + PAD_TIP_PAST_TCP  # the held sort's bottom 0.5 mm above the bed
HOLD = -(a.BODY - 0.008) / 2  # pads 8 mm narrower than the sort's body: about 20 N a finger
LEFT, RIGHT = -math.pi / 2, math.pi / 2  # fingers along x for each arm
PARK = {"left": (0.28, 0.30, TOP + 0.17), "right": (0.28, -0.30, TOP + 0.17)}
HOVER = 0.025  # pauses this far above each grip and set-down height, so the arm settles before the last few centimetres
# Spots on the mat, reachable by both arms, where the reading-order variant passes a sort from one arm to the other.
HAND_OVER = {"K": (0.20, 0.07), "P": (0.20, -0.07)}


def _bay(k: int) -> tuple[float, float, float]:
    return CX, a.BAY_Y[k], SET_Z


def _on_mat(xy) -> tuple[float, float, float]:
    return xy[0], xy[1], GRIP_Z


def _move(letter: str, src, dst, yaw_pick: float, yaw_place: float, release: bool = True) -> list[Waypoint]:
    """Pick the sort at src (x, y, grip height) by its head and foot ends and set it down at dst (x, y, set height)."""
    sx, sy, sz = src
    dx, dy, dz = dst
    out = [
        Waypoint(2.0, (sx, sy, TRAVEL), FINGER_OPEN, f"over {letter}", yaw_pick),
        Waypoint(1.0, (sx, sy, sz + HOVER), FINGER_OPEN, "near", yaw_pick),
        Waypoint(0.6, (sx, sy, sz + HOVER), FINGER_OPEN, "settle", yaw_pick),
        Waypoint(1.0, (sx, sy, sz), FINGER_OPEN, "down", yaw_pick),
        Waypoint(0.4, (sx, sy, sz), FINGER_OPEN, "settle", yaw_pick),
        Waypoint(0.6, (sx, sy, sz), HOLD, "grip head and foot", yaw_pick),
        Waypoint(1.0, (sx, sy, TRAVEL), HOLD, "lift", yaw_pick),
        Waypoint(2.4, (dx, dy, TRAVEL), HOLD, "carry", yaw_place),
        Waypoint(1.0, (dx, dy, dz + HOVER), HOLD, "lower", yaw_place),
        Waypoint(0.6, (dx, dy, dz + HOVER), HOLD, "settle", yaw_place),
        Waypoint(1.0, (dx, dy, dz), HOLD, "set down", yaw_place),
        Waypoint(0.4, (dx, dy, dz), HOLD, "settle", yaw_place),
    ]
    if release:
        out += [Waypoint(1.0, (dx, dy, dz), FINGER_OPEN, "let go", yaw_place),
                Waypoint(0.8, (dx, dy, TRAVEL), FINGER_OPEN, "rise", yaw_place)]
    else:
        out += [Waypoint(3.0, (dx, dy, dz), HOLD, "keep holding", yaw_place)]
    return out


def _start(letter: str) -> tuple[float, float, float]:
    return _on_mat(START[letter][:2])


def _span(plan) -> float:
    return sum(w.duration for w in plan)


def _schedule(rounds) -> dict:
    """rounds: a list of rounds, each a list of (arm, waypoints) run at the same time. An arm with nothing to do in a round
    waits at its park pose (with the yaw its next move starts with), so only one arm is ever near the other's work."""
    plans = {"left": [], "right": []}
    yaw0 = {"left": LEFT, "right": RIGHT}
    for k, moves in enumerate(rounds):
        busy = dict(moves)
        length = max(_span(w) for w in busy.values())
        for arm in plans:
            if arm in busy:
                plans[arm] += busy[arm]
                if _span(busy[arm]) < length:
                    last = plans[arm][-1]
                    plans[arm].append(Waypoint(length - _span(busy[arm]), last.pos, last.finger, "wait", last.yaw))
            else:
                upcoming = next((dict(r)[arm][0].yaw for r in rounds[k + 1:] if arm in dict(r)), yaw0[arm])
                plans[arm].append(Waypoint(length, PARK[arm], FINGER_OPEN, "wait clear", upcoming))
    return plans


def demo():
    """Round 1, both arms at once: the left arm sets P in bay 1 and the right arm K in bay 4 (bays 20 cm apart). Round 2:
    the left arm sets L in bay 2 while the right arm waits clear. Round 3: the right arm grips E at yaw -90 degrees, turns
    it in the grasp and sets it in bay 3 at +90 degrees, while the left arm waits clear. Neighbouring bays are never worked
    at once: the wrists, about 5 cm ahead of the fingers, would meet. Variants (TASK_DEMO), to check the scorer: reading
    sets K, E, L, P in bays 1 to 4 as they lie (E still turned), passing K and P between the arms on the mat; eturned sets
    E unturned; jforl sets the J in bay 2; held keeps holding E in bay 3; nothing does nothing."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"left": [Waypoint(4.0, PARK["left"], FINGER_OPEN, "hold", LEFT)]}
    if mode == "reading":
        def parked(arm, yaw):  # each move here crosses the midline, so the arm parks before the other one starts
            return Waypoint(1.5, PARK[arm], FINGER_OPEN, "park", yaw)

        rounds = [
            [("left", _move("P", _start("P"), _on_mat(HAND_OVER["P"]), LEFT, LEFT) + [parked("left", LEFT)])],
            [("right", _move("K", _start("K"), _on_mat(HAND_OVER["K"]), RIGHT, RIGHT) + [parked("right", RIGHT)])],
            [("left", _move("K", _on_mat(HAND_OVER["K"]), _bay(0), LEFT, LEFT) + [parked("left", LEFT)])],
            [("right", _move("P", _on_mat(HAND_OVER["P"]), _bay(3), RIGHT, RIGHT) + [parked("right", RIGHT)])],
            [("left", _move("L", _start("L"), _bay(2), LEFT, LEFT) + [parked("left", LEFT)])],
            [("right", _move("E", _start("E"), _bay(1), RIGHT, RIGHT))],  # E as it lies, still turned
        ]
    else:
        second = "J" if mode == "jforl" else "L"
        e_pick = RIGHT if mode == "eturned" else LEFT  # gripping at -90 and setting at +90 turns E end for end
        rounds = [
            [("left", _move("P", _start("P"), _bay(0), LEFT, LEFT)), ("right", _move("K", _start("K"), _bay(3), RIGHT, RIGHT))],
            [("left", _move(second, _start(second), _bay(1), LEFT, LEFT))],
            [("right", _move("E", _start("E"), _bay(2), e_pick, RIGHT, release=mode != "held"))],
        ]
    plans = _schedule(rounds)
    for arm, plan in plans.items():
        if not (mode == "held" and arm == "right"):
            plan.append(Waypoint(1.2, PARK[arm], FINGER_OPEN, "park", plan[-1].yaw))
        plan.append(Waypoint(1.5, plan[-1].pos, plan[-1].finger, "settle", plan[-1].yaw))
    return plans
