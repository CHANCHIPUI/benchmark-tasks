"""On the station table, a 9x9 Go teaching board lies flat between the arms with a beginner's capture problem in its
near-right corner. Black's bowl (three black stones) and its upturned lid stand to the right of the board; White's bowl
and lid stand on the far side. The board, bowls and lids are fixed; the stones are loose.

Columns A to J run from the robot's left (Go boards skip I), rows 1 to 9 from the robot's edge. X is Black, O White:

     E  F  G  H  J
 5   .  .  X  .  .
 4   .  .  O  .  .
 3   .  X  O  X  X
 2   .  .  X  O  O
 1   .  .  .  X  .

H2-J2 has one liberty, J1. A black stone there captures them; G3-G4 (diagonal to H2, so a separate group) keeps two.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from collections import deque

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
BOARD_Z = TOP + a.BOARD_TOP  # the board's surface

# -- objects ------------------------------------------------------------------------------------------------------
BOARD_STONES = {f"black_{p}": p for p in a.BLACK} | {f"white_{p}": p for p in a.WHITE}
LIVE = [n for n, p in BOARD_STONES.items() if p not in a.CAPTURED]
DEAD = [f"white_{p}" for p in a.CAPTURED]
BOWL = [f"bowl_{k}" for k in range(1, 4)]
STONES = list(BOARD_STONES) + BOWL
BOWL_START = {n: (a.BLACK_BOWL_XY[0] + dx, a.BLACK_BOWL_XY[1]) for n, dx in zip(BOWL, a.BOWL_STONES_X)}


def _stone(name: str, color: str, x: float, y: float, z: float) -> UsdBody:
    return UsdBody(name, usd=f"assets/{color}_stone.usda", pos=(x, y, z + 0.0002), mass=a.STONE_MASS,
                   size=(2 * a.STONE_R, 2 * a.STONE_R, a.STONE_H), friction=a.STONE_FRICTION, solver_iters=32,
                   torsional_patch=0.008)


OBJECTS = [
    *[_stone(n, n.split("_")[0], *a.point(p), BOARD_Z) for n, p in BOARD_STONES.items()],
    *[_stone(n, "black", x, y, TOP + a.BOWL_BASE) for n, (x, y) in BOWL_START.items()],
    UsdBody("go_set", usd="assets/go_set.usda", pos=(0.0, 0.0, TOP), mass=None, size=(a.BOARD[0], a.BOARD[1], a.BOARD[2]),
            friction=(0.5, 0.4)),
]

# -- scoring (see "How it is scored" in the spec) -----------------------------------------------------------------
ON_POINT = 0.0125  # m, a quarter of the 50 mm pitch
ON_TILT = 10.0  # degrees
ON_UNDERSIDE = 0.003  # m, the underside's centre above the board
LEAVE_LIFT = 0.005  # m
LEAVE_TILT = 30.0  # degrees
RELEASE = 0.06  # m, both TCPs from the stone's centre
LID_IN = a.LID_IN_R  # m, from the lid's centre, seen from above
STILL = 0.002  # m in 1 s
STILL_STEPS = 20  # 1 s at 20 Hz


def _tilt(st) -> float:
    """Degrees between the stone's axis and the vertical (0 flat, dome up; 180 upside down)."""
    x, y, _, _ = (float(v) for v in st.quat_xyzw)
    return math.degrees(math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def _centre(st) -> np.ndarray:
    """The stone's centre: halfway up its axis from the underside's centre (the body origin)."""
    x, y, z, w = (float(v) for v in st.quat_xyzw)
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return np.asarray(st.pos, float) + axis * a.STONE_H / 2


def over_board(st) -> bool:
    x, y = float(st.pos[0]), float(st.pos[1])
    return a.BOARD_X_RANGE[0] < x < a.BOARD_X_RANGE[1] and a.BOARD_Y_RANGE[0] < y < a.BOARD_Y_RANGE[1]


def released(st, tcps) -> bool:
    c = _centre(st)
    return all(float(np.linalg.norm(np.asarray(t, float) - c)) >= RELEASE for t in tcps)


def on_point(st, name: str) -> bool:
    px, py = a.point(name)
    return (math.hypot(float(st.pos[0]) - px, float(st.pos[1]) - py) <= ON_POINT and _tilt(st) < ON_TILT
            and abs(float(st.pos[2]) - BOARD_Z) <= ON_UNDERSIDE)


def left_point(st, name: str) -> bool:
    px, py = a.point(name)
    return (math.hypot(float(st.pos[0]) - px, float(st.pos[1]) - py) > ON_POINT or _tilt(st) > LEAVE_TILT
            or float(st.pos[2]) - BOARD_Z > LEAVE_LIFT)


def in_black_lid(st, tcps) -> bool:
    lx, ly = a.BLACK_LID_XY
    return math.hypot(float(st.pos[0]) - lx, float(st.pos[1]) - ly) < LID_IN and released(st, tcps)


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, failed=None, gone=set(), history=deque(maxlen=STILL_STEPS + 1), flags={}, success=False)


_reset_state()


def update(states: dict, tcps: list) -> dict:
    """One control step: latch the failures, then judge the end-state conditions. Returns the flags."""
    _STATE["step"] += 1
    t = round(_STATE["step"] / steptime.CONTROL_HZ, 2)
    played = [n for n in BOWL if over_board(states[n])]
    answer_held = any(on_point(states[n], a.ANSWER) for n in BOWL)
    if _STATE["failed"] is None:
        for n in LIVE:
            if left_point(states[n], BOARD_STONES[n]):
                _STATE["failed"] = f"live stone {BOARD_STONES[n]} left its point at {t} s"
                break
    for n in DEAD:  # judged once, at the moment each captured stone first leaves its point
        if n not in _STATE["gone"] and left_point(states[n], BOARD_STONES[n]):
            _STATE["gone"].add(n)
            if _STATE["failed"] is None and not answer_held:
                _STATE["failed"] = f"{BOARD_STONES[n]} left its point before a stone was on {a.ANSWER}, at {t} s"
    _STATE["history"].append(np.array([np.asarray(states[n].pos, float) for n in STONES]))
    hist = _STATE["history"]
    still = len(hist) > STILL_STEPS and float(np.max(np.linalg.norm(hist[-1] - hist[0], axis=1))) <= STILL
    right_move = (len(played) == 1 and on_point(states[played[0]], a.ANSWER) and released(states[played[0]], tcps))
    flags = {
        "played": len(played),
        "right_move": right_move,
        "dead_in_lid": all(in_black_lid(states[n], tcps) for n in DEAD),
        "dead_off_board": all(not over_board(states[n]) for n in DEAD),
        "still": still,
    }
    _STATE["flags"] = flags
    _STATE["success"] = (_STATE["failed"] is None and right_move and flags["dead_in_lid"] and still)
    return flags


def end_score() -> float:
    """The score at the run limit, when success has not held for 1 s."""
    f = _STATE["flags"]
    if _STATE["failed"] is not None or not f.get("right_move"):
        return 0.0
    return 0.75 if f["dead_off_board"] else 0.5


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
    states = object_states(environment, STONES)
    flags = update(states, tcps)
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2), "failed": _STATE["failed"], **flags,
           "success": _STATE["success"], "tcp": [[round(float(v), 4) for v in t] for t in tcps]}
    if os.environ.get("TASK_SCORE_DIR"):
        from scenekit import arm_joints
        names = [f"joint{i}" for i in range(1, 7)]
        rec["q"] = [[round(float(v), 4) for v in arm_joints(environment, side, names)] for side in ("left", "right")]
    for n in STONES:
        st = states[n]
        rec[n] = [round(float(v), 4) for v in st.pos] + [round(_tilt(st), 1)]
    _trace(rec)


def success(objects) -> bool:
    """The right move, H2 and J2 in Black's lid, every stone still, no failure; the harness ends at 1.0 after 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    return bool(_STATE["success"])


def score(objects) -> float:
    """At the run limit: 0.75 for the right move with both captured stones off the board, 0.5 for the right move with
    either still on it, otherwise 0. Any latched failure scores 0."""
    return end_score()


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
YAW = math.radians(45.0)  # on the board: fingers close along the diagonal from (row+1, toward J) to (row-1, toward A)
BOWL_YAW = 0.0  # in the bowl: fingers close along y, across the row of stones, so they stay inside the bowl's wall
OPEN = -0.026  # pads 52 mm apart: 6 mm clear of a 40 mm stone on each side
HOLD = -(2 * a.STONE_R - 0.008) / 2  # pads 8 mm narrower than the stone: a firm pinch
TIP = 0.0015  # pad tips this far above the surface the stone lies on
GRIP_BOARD = BOARD_Z + TIP + PAD_TIP_PAST_TCP
GRIP_BOWL = TOP + a.BOWL_BASE + TIP + PAD_TIP_PAST_TCP
SET_BOARD = BOARD_Z + 0.001 + TIP + PAD_TIP_PAST_TCP  # the held stone's underside 1 mm above the board
DROP_LID = TOP + a.LID_FLOOR + 0.020 + TIP + PAD_TIP_PAST_TCP  # its underside 2 cm above the lid's floor
TRAVEL = TOP + 0.10
HOVER = 0.025
PARK = (0.20, -0.40, TOP + 0.16)
# Open-loop aim corrections (m), measured once from a logged run: where the TCP lands for a commanded place.
AIM: dict = {}


def _aim(xy) -> tuple[float, float]:
    dx, dy = AIM.get((round(xy[0], 4), round(xy[1], 4)), (0.0, 0.0))
    return xy[0] + dx, xy[1] + dy


def _pick(label: str, xy, grip_z: float, yaw: float = YAW) -> list[Waypoint]:
    x, y = _aim(xy)
    return [
        Waypoint(1.6, (x, y, TRAVEL), OPEN, f"over {label}", yaw),
        Waypoint(0.9, (x, y, grip_z + HOVER), OPEN, "near", yaw),
        Waypoint(0.4, (x, y, grip_z + HOVER), OPEN, "settle", yaw),
        Waypoint(0.8, (x, y, grip_z), OPEN, "down", yaw),
        Waypoint(0.3, (x, y, grip_z), OPEN, "settle", yaw),
        Waypoint(0.5, (x, y, grip_z), HOLD, "grip", yaw),
        Waypoint(0.9, (x, y, TRAVEL), HOLD, "lift", yaw),
    ]


def _place(label: str, xy, set_z: float, rise: float) -> list[Waypoint]:
    x, y = _aim(xy)
    return [
        Waypoint(1.8, (x, y, TRAVEL), HOLD, f"carry to {label}", YAW),
        Waypoint(0.9, (x, y, set_z + HOVER), HOLD, "lower", YAW),
        Waypoint(0.4, (x, y, set_z + HOVER), HOLD, "settle", YAW),
        Waypoint(0.7, (x, y, set_z), HOLD, "set down", YAW),
        Waypoint(0.3, (x, y, set_z), HOLD, "settle", YAW),
        Waypoint(0.5, (x, y, set_z), OPEN, "let go", YAW),
        Waypoint(0.6, (x, y, set_z + rise), OPEN, "rise", YAW),
    ]


def _play(dst: str, src: str = "bowl_2") -> list[Waypoint]:
    """Take a black stone from Black's bowl and set it on the point dst."""
    return _pick(src, BOWL_START[src], GRIP_BOWL, BOWL_YAW) + _place(dst, a.point(dst), SET_BOARD, a.STONE_H + 0.006)


LID_SPOTS = {"J2": (a.BLACK_LID_XY[0] - 0.025, a.BLACK_LID_XY[1]), "H2": (a.BLACK_LID_XY[0] + 0.025, a.BLACK_LID_XY[1]),
             "G3": (a.BLACK_LID_XY[0], a.BLACK_LID_XY[1] + 0.03), "G4": (a.BLACK_LID_XY[0], a.BLACK_LID_XY[1] - 0.03)}


def _remove(p: str, to=None, label: str = "Black's lid") -> list[Waypoint]:
    """Lift the stone on point p and drop it at `to` (by default its own spot in Black's lid, so they lie side by side)."""
    to = LID_SPOTS[p] if to is None else to
    return _pick(p, a.point(p), GRIP_BOARD) + _place(label, to, DROP_LID, 0.03)


def _finish(plan: list[Waypoint]) -> dict:
    plan = plan + [Waypoint(1.5, PARK, OPEN, "park", YAW), Waypoint(3.0, PARK, OPEN, "settle", YAW)]
    return {"right": plan}


def demo():
    """J1 from Black's bowl, then J2 and H2 into Black's lid (1.0). Variants (TASK_DEMO), to check the scorer:
      failure  J1, then all four white stones into the lid (J2, G3, G4, H2): G3 and G4 are live (0)
      g1       the stone on G1, the hole beside H2 (0)
      h4       the stone on H4, a liberty of the live pair G3-G4 (0)
      leave    J1, and the captured stones left on the board (0.5)
      table    J1, then H2 and J2 onto the table beside Black's lid (0.75)
      early    J2 into the lid before the stone on J1, then J1 and H2 (0)
      two      J1, then a second black stone on H4 (0)
      liftback J1, then H1 (live) lifted and set back on its point, then J2 and H2 into the lid (0)
      nothing  does nothing (0)"""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"right": [Waypoint(4.0, PARK, OPEN, "park", YAW)]}
    if mode == "g1":
        return _finish(_play("G1"))
    if mode == "h4":
        return _finish(_play("H4"))
    if mode == "leave":
        return _finish(_play("J1"))
    if mode == "table":
        return _finish(_play("J1") + _remove("J2", (0.30, -0.36), "the table") + _remove("H2", (0.24, -0.36), "the table"))
    if mode == "early":
        return _finish(_remove("J2") + _play("J1") + _remove("H2"))
    if mode == "two":
        return _finish(_play("J1") + _play("H4", "bowl_3"))
    if mode == "liftback":
        back = _pick("H1", a.point("H1"), GRIP_BOARD) + _place("H1", a.point("H1"), SET_BOARD, a.STONE_H + 0.006)
        return _finish(_play("J1") + back + _remove("J2") + _remove("H2"))
    if mode == "failure":
        # G3 and G4 go before H2: once H2 and J2 are both in the lid, the success state holds and ends the trial.
        return _finish(_play("J1") + _remove("J2") + _remove("G3") + _remove("G4") + _remove("H2"))
    return _finish(_play("J1") + _remove("J2") + _remove("H2"))
