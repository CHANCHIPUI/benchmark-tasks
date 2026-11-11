"""On the station table, a braille lesson laid out for a reader sitting on the robot's side. Between the arms, three
half-dozen grey moulded-pulp egg cartons, lids cut off, stand in a row from left to right on a dark grey board, each with
two columns of three cups. They are empty. Nearer each arm, a pine rack holds five red-painted wooden balls (40 mm), one
in each socket. A ball in a cup stands for a raised braille dot. The board, cartons and racks are fixed; the balls are
loose.
"""

from __future__ import annotations

import itertools
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
CUPS = a.cup_centres()  # (carton 0..2, dot 1..6) -> (x, y)
START = a.balls_start()  # 10 (x, y): left rack x 0.18 .. 0.42, then the right rack
BALLS = [f"ball_{i}" for i in range(len(START))]
LEFT_BALLS, RIGHT_BALLS = list(range(5)), list(range(5, 10))

OBJECTS = [
    *[UsdBody(n, usd="assets/ball.usda", pos=(x, y, TOP + a.rack_centre() + 0.0003), mass=a.BALL_MASS,
              size=(2 * a.BALL_R,) * 3, friction=a.BALL_FRICTION, solver_iters=16)
      for n, (x, y) in zip(BALLS, START)],
    UsdBody("bench", usd="assets/bench.usda", pos=(0.0, 0.0, TOP), mass=None, size=(0.22, 0.48, 0.035),
            friction=(0.6, 0.5)),
]

# -- scoring ------------------------------------------------------------------------------------------------------
WORD = "hid"
BRAILLE = {"h": {1, 2, 5}, "i": {2, 4}, "d": {1, 4, 5}}
TARGET = [BRAILLE[c] for c in WORD]  # cartons 1 to 3, left to right as the reader sees them
SEATED_Z = TOP + a.seated_centre()  # a ball's centre seated alone in a cup
CUP_R = 0.015  # m: a filled cup's ball centre is this close to the cup's centre, seen from above
SEAT_TOL = 0.010  # m: and no higher than this above the seated height
TCP_CLEAR = 0.04  # m: both TCPs at least this far from a ball for it to count as released
# Still: slower than 1 cm/s, measured from positions over the last 0.25 s. (The simulator's reported speed of a ball
# resting in a cup can stay at a few cm/s for seconds while the ball does not move at all.)
STILL = 0.01  # m/s
STILL_WINDOW = 5  # control steps


def on_carton(p) -> int | None:
    """The carton (0..2) whose outline, seen from above, holds the point p, or None."""
    for k, cy in enumerate(a.CARTON_Y):
        if abs(p[0] - a.CARTON_X) <= a.OUTER[0] / 2 and abs(p[1] - cy) <= a.OUTER[1] / 2:
            return k
    return None


def judge(pos: dict, vel: dict, tcps: list) -> dict:
    """Every carton's filled dots, stray balls and stillness, from ball positions (name -> xyz), speeds and the TCPs."""
    cells = [{"dots": set(), "stray": [], "moving": []} for _ in a.CARTON_Y]
    for n in BALLS:
        p = np.asarray(pos[n], float)
        k = on_carton(p)
        if k is None:
            continue
        free = all(float(np.linalg.norm(p - np.asarray(t, float))) >= TCP_CLEAR for t in tcps)
        dot = next((d for d in range(1, 7) if math.hypot(p[0] - CUPS[(k, d)][0], p[1] - CUPS[(k, d)][1]) <= CUP_R), None)
        if dot is not None and p[2] <= SEATED_Z + SEAT_TOL and free and dot not in cells[k]["dots"]:
            cells[k]["dots"].add(dot)
        else:
            cells[k]["stray"].append(n)
        if float(vel[n]) >= STILL:
            cells[k]["moving"].append(n)
    for k, c in enumerate(cells):
        c["correct"] = c["dots"] == TARGET[k] and not c["stray"]
    return {"cells": cells, "still": not any(c["moving"] for c in cells)}


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(correct=[False] * 3, still=False, step=0, history=[])


_reset_state()


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
    states = object_states(environment, BALLS)
    pos = {n: np.asarray(states[n].pos, float) for n in BALLS}
    hist = _STATE["history"]
    hist.append(pos)
    del hist[:-(STILL_WINDOW + 1)]
    span = (len(hist) - 1) / steptime.CONTROL_HZ
    # Speed from the positions over the last 0.25 s; before that window has filled (right after a reset), not still.
    vel = {n: (float(np.linalg.norm(pos[n] - hist[0][n])) / span if span > 0 else 1.0) for n in BALLS}
    j = judge(pos, vel, tcps)
    _STATE["step"] += 1
    _STATE["correct"] = [c["correct"] for c in j["cells"]]
    _STATE["still"] = j["still"]
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2),
           "cells": [[sorted(c["dots"]), c["stray"], c["correct"]] for c in j["cells"]], "still": j["still"],
           "tcp": [[round(float(v), 4) for v in t] for t in tcps]}
    for n in BALLS:
        rec[n] = [round(float(v), 4) for v in pos[n]] + [round(vel[n], 3)]
    _trace(rec)


def success(objects) -> bool:
    """All three cells correct and every ball on a carton still; the harness ends the trial at 1.0 after 1 s of this."""
    steptime.tick()  # the harness calls success() once per control step
    return bool(all(_STATE["correct"]) and _STATE["still"])


def score(objects) -> float:
    """Without success: 1/3 for each correct cell (see judge), at most 2/3, since 1.0 needs the cells still for 1 s
    (success). Nothing in simulation latches: the cartons are fixed and rigid."""
    return min(sum(_STATE["correct"]), 2) / 3.0


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
OPEN = -0.030  # pads 60 mm apart: 10 mm clear of the ball on each side
HOLD = -0.017  # pads commanded 34 mm apart on a 40 mm ball
GRIP_Z = TOP + a.rack_centre() + 0.003  # TCP: pad tips 10 mm below the ball's centre, 6.7 mm above the rack
RIM_Z = TOP + a.BASE + a.RIM
RELEASE_Z = RIM_Z + 0.023  # TCP: the held ball's bottom 3 mm above the rim, pad tips 10 mm above it
TRAVEL = TOP + 0.10  # a held ball's bottom 7.7 cm up, 3 cm over seated balls
HOVER = 0.025
YAW_Y = 0.0  # fingers close along y
YAW_X = {"left": -math.pi / 2, "right": math.pi / 2}  # fingers close along x
PARK = {"left": (0.25, 0.30, TOP + 0.15), "right": (0.25, -0.30, TOP + 0.15)}
CLEAR_RIGHT = (0.30, -0.27, TOP + 0.13)  # over the right rack, off the cartons in the top camera
WAIT_OVER_RACK = {"left": (0.30, 0.28, TRAVEL + 0.02), "right": (0.30, -0.28, TRAVEL + 0.02)}

VARIANTS = {  # dots per carton, left to right; each a scorer check from the spec's examples
    "success": [{1, 2, 5}, {2, 4}, {1, 4, 5}],  # h i d: 1.0
    "jef": [{2, 4, 5}, {1, 5}, {1, 2, 4}],  # every cell's columns swapped: 0
    "fej": [{1, 2, 4}, {1, 5}, {2, 4, 5}],  # the whole row mirrored: 0
    "across": [{2, 3, 6}, {3, 5}, {2, 5, 6}],  # built for a reader across the table (turned 180 degrees): 0
    "rows": [{1, 3, 4}, {4, 5}, {1, 3, 5}],  # dots numbered across the rows (m, -, o): 0
    "upside": [{2, 3, 5}, {2, 6}, {3, 5, 6}],  # cells upside down: 0
    "dih": [{1, 4, 5}, {2, 4}, {1, 2, 5}],  # right letters in reverse order: 1/3
    "stray": [{1, 2, 5}, {2, 4}, {1, 4, 5}],  # hid, with a ninth ball laid on carton 1's middle ridge before carton 2: 2/3
    "held": [{1, 2, 5}, {2, 4}, {1, 4, 5}],  # hid, the last ball still held over its cup at the end: 2/3
}


def _neighbours(k: int, dot: int) -> dict:
    """The cups next to (k, dot) along y (other column, same row) and along x (rows above and below)."""
    col, row = (0 if dot <= 3 else 1), (dot - 1) % 3
    other = (1 - col) * 3 + row + 1
    along_x = [col * 3 + r + 1 for r in (row - 1, row + 1) if 0 <= r <= 2]
    return {"y": [other], "x": along_x}


def _plan_cell(k: int, dots: set) -> list:
    """An order and a closing axis for each dot, so that the fingers never open toward a ball already seated."""
    for order in itertools.permutations(sorted(dots)):
        placed, out = set(), []
        for d in order:
            nb = _neighbours(k, d)
            axis = next((ax for ax in ("y", "x") if not placed & set(nb[ax])), None)
            if axis is None:
                break
            out.append((d, axis))
            placed.add(d)
        else:
            return out
    raise ValueError(f"no free axis for carton {k + 1} dots {sorted(dots)}")


def _ball_move(arm: str, src, dst, axis: str, release: bool = True, drop_z: float = RELEASE_Z) -> list:
    """Pick the ball at src (x, y) from its rack socket and let it drop into the cup at dst (x, y)."""
    sx, sy = src
    dx, dy = dst
    yaw = YAW_Y if axis == "y" else YAW_X[arm]
    out = [
        Waypoint(1.4, (sx, sy, TRAVEL), OPEN, "over the ball", YAW_Y),
        Waypoint(0.7, (sx, sy, GRIP_Z + HOVER), OPEN, "near", YAW_Y),
        Waypoint(0.3, (sx, sy, GRIP_Z + HOVER), OPEN, "settle", YAW_Y),
        Waypoint(0.6, (sx, sy, GRIP_Z), OPEN, "down", YAW_Y),
        Waypoint(0.2, (sx, sy, GRIP_Z), OPEN, "settle", YAW_Y),
        Waypoint(0.5, (sx, sy, GRIP_Z), HOLD, "grip", YAW_Y),
        Waypoint(0.8, (sx, sy, TRAVEL), HOLD, "lift", YAW_Y),
        Waypoint(1.4, (dx, dy, TRAVEL), HOLD, "carry", yaw),
        Waypoint(0.7, (dx, dy, drop_z + HOVER), HOLD, "lower", yaw),
        Waypoint(0.3, (dx, dy, drop_z + HOVER), HOLD, "settle", yaw),
        Waypoint(0.5, (dx, dy, drop_z), HOLD, "over the cup", yaw),
        Waypoint(0.3, (dx, dy, drop_z), HOLD, "settle", yaw),
    ]
    if release:
        out += [Waypoint(0.4, (dx, dy, drop_z), OPEN, "let go", yaw),
                Waypoint(0.6, (dx, dy, TRAVEL), OPEN, "rise", yaw)]
    else:
        out += [Waypoint(3.0, (dx, dy, drop_z), HOLD, "keep holding", yaw)]
    return out


def _span(plan) -> float:
    return sum(w.duration for w in plan)


def _pad(plan: list, length: float, label: str = "wait") -> list:
    if _span(plan) < length:
        last = plan[-1]
        plan = plan + [Waypoint(length - _span(plan), last.pos, last.finger, label, last.yaw)]
    return plan


def demo():
    """Open loop. Round 1, both arms at once (wrists at least 20 cm apart): the left arm fills carton 1 and the right
    arm carton 3, from their own racks. Round 2: the left arm fills carton 2's left column while the right arm picks its
    next ball and waits over its rack. Round 3: the right arm fills carton 2's right column while the left arm waits
    parked. In each cell, the dots go in an order that lets the fingers close along an axis with no seated ball next to
    the cup. TASK_DEMO picks a variant from VARIANTS (scorer checks), or `nothing`."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"left": [Waypoint(4.0, PARK["left"], OPEN, "hold", YAW_Y)]}
    target = VARIANTS[mode]
    supply = {"left": [START[i] for i in LEFT_BALLS], "right": [START[i] for i in RIGHT_BALLS]}
    jobs = {"left": [], "right": []}  # (src, dst, axis)
    for arm, k in (("left", 0), ("right", 2)):
        for d, axis in _plan_cell(k, target[k]):
            jobs[arm].append((supply[arm].pop(0), CUPS[(k, d)], axis))
    middle = _plan_cell(1, target[1])
    mid = {"left": [(d, ax) for d, ax in middle if d <= 3], "right": [(d, ax) for d, ax in middle if d > 3]}
    plans = {"left": [], "right": []}
    # Round 1: the outer cartons.
    r1 = {arm: [w for src, dst, ax in jobs[arm] for w in _ball_move(arm, src, dst, ax)] for arm in plans}
    length = max(_span(p) for p in r1.values())
    for arm in plans:
        plans[arm] += _pad(r1[arm], length)
    # Round 2: carton 2's left column by the left arm; the right arm takes its next ball meanwhile.
    held = mode == "held"
    r2_left = []
    if mode == "stray":  # first a ninth ball laid on carton 1's middle ridge, between seated dots 2 and 5
        ridge = (a.CARTON_X, a.CARTON_Y[0])
        r2_left += _ball_move("left", supply["left"].pop(0), ridge, "x", drop_z=TOP + 0.075)
    r2_left += [w for d, ax in mid["left"] for w in _ball_move("left", supply["left"].pop(0), CUPS[(1, d)], ax)]
    r2_left += [Waypoint(1.2, PARK["left"], OPEN, "park", YAW_Y)]
    right_mid = list(mid["right"])
    pre = None
    r2_right = []
    if right_mid:
        pre = supply["right"].pop(0)
        first = _ball_move("right", pre, CUPS[(1, right_mid[0][0])], right_mid[0][1])
        r2_right = first[:7] + [Waypoint(1.2, WAIT_OVER_RACK["right"], HOLD, "wait over the rack", YAW_Y)]
    length = max(_span(r2_left), _span(r2_right) if r2_right else 0.0)
    plans["left"] += _pad(r2_left, length, "wait parked")
    plans["right"] += _pad(r2_right, length) if r2_right else [Waypoint(length, PARK["right"], OPEN, "wait clear", YAW_Y)]
    # Round 3: carton 2's right column by the right arm, the first ball already in hand.
    r3 = []
    for i, (d, ax) in enumerate(right_mid):
        last = held and i == len(right_mid) - 1
        if i == 0:
            r3 += _ball_move("right", pre, CUPS[(1, d)], ax, release=not last)[7:]
        else:
            r3 += _ball_move("right", supply["right"].pop(0), CUPS[(1, d)], ax, release=not last)
    if r3 and not held:
        # The last ball: let it settle with the open fingers still round it (its TCP within 4 cm, so the cell cannot
        # count yet), then rise and leave toward the right arm's own side. Success needs the TCP 4 cm away and then holds
        # 1 s before the trial ends, so the arm is off the cartons in the top camera's last frame.
        let_go = r3[-2]
        r3[-1:] = [Waypoint(0.5, let_go.pos, OPEN, "let the ball settle", let_go.yaw),
                   Waypoint(0.3, (let_go.pos[0], let_go.pos[1], RIM_Z + 0.055), OPEN, "rise", let_go.yaw),
                   Waypoint(0.8, CLEAR_RIGHT, OPEN, "clear the cartons", let_go.yaw)]
    if r3:
        plans["right"] += r3
        plans["left"].append(Waypoint(_span(r3), PARK["left"], OPEN, "wait parked", YAW_Y))
    for arm, plan in plans.items():
        if not (held and arm == "right"):
            plan.append(Waypoint(1.2, PARK[arm], OPEN, "park", plan[-1].yaw))
        plan.append(Waypoint(2.0, plan[-1].pos, plan[-1].finger, "settle", plan[-1].yaw))
    return plans
