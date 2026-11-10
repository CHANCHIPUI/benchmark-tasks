"""On the station table, in front of the arms: an opened battery pack on a dismantling bench, and a grey parts tray.

The pack's dark grey steel housing (16 x 37 cm, walls 2.5 cm) holds four brushed-aluminium modules side by side, each
15 x 7 x 4 cm, with two terminal posts on its lid (a nickel pad and an M6 stud): one 2 cm from the front end, one 2 cm
from the back end, with a red (+) or black (-) mark beside each. The modules are wired in series. At the back, two copper
busbars (11 x 2.5 x 0.5 cm) join the left pair and the right pair; their nuts are off, so each rests on two studs. At
the front, orange cables are bolted to the posts: the middle two run to a black socket in the gap between the middle
modules, and the outer two are the pack's main leads. An orange service disconnect plug (3 x 4.4 x 4.5 cm, with a grey
grip on top) sits 2 cm deep in that socket and closes the string between the two halves. A high-voltage warning
label is on the socket and on the housing's front wall. The grey tray (inside 9 x 26 cm) is in front of the pack.
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

import pack_usd as P  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = P.build()

PACK = UsdBody("battery_pack", usd=USD["pack"], pos=(0.0, 0.0, TOP), size=(0.16, 0.37, 0.061))
TRAY = UsdBody("parts_tray", usd=USD["tray"], pos=(0.0, 0.0, TOP), size=(0.098, 0.258, 0.015))
BARS = {name: UsdBody(name, usd=USD["busbar"], pos=(x, y, TOP + P.BAR_Z), size=P.BAR_SIZE, mass=P.BAR_MASS,
                      friction=(0.9, 0.8), torsional_patch=0.008)
        for name, (x, y) in P.BARS.items()}
PLUG = UsdBody("service_plug", usd=USD["plug"], pos=(*P.SOCKET_CENTRE, TOP + P.PLUG_Z), size=P.PLUG_SIZE,
               mass=P.PLUG_MASS, friction=(0.9, 0.8), torsional_patch=0.01)
OBJECTS = [PACK, TRAY, *BARS.values(), PLUG]

# -- scoring --------------------------------------------------------------------------------------------------------
MOVED = 0.005           # m: a busbar has moved once its centre is this far from its start
CLEAR = 0.03            # m: a busbar must stay this far above the stud tip of any post it did not start on ...
NEAR = 0.02             # m: ... while any part of it, seen from above, is this close to that post's axis
PLUG_XY = 0.015         # m: the plug is in its socket within this of the socket's centre (top view) ...
RIM_Z = TOP + P.RIM     # ... with its lowest point below the socket's rim
IN_TRAY_LOW = 0.015     # m: a busbar in the tray has its lowest point within this of the tray floor
STUD_TIP_Z = TOP + P.STUD_TIP
TRAY_FLOOR_Z = TOP + P.TRAY_FLOOR
POSTS = [(name, np.array((x, y))) for name, x, y, _ in P.posts()]
START = {name: np.array(spec.pos) for name, spec in BARS.items()}
OWN = {name: {p for p, xy in POSTS if min(abs(xy[1] - (START[name][1] + s * P.BAR_HOLE_Y)) for s in (-1, 1)) < 0.005
              and abs(xy[0] - START[name][0]) < 0.005} for name in BARS}
assert all(len(v) == 2 for v in OWN.values()), OWN

_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Failures stay failed for the whole trial: only a reset clears them."""
    _T.clear()
    _T.update(failed=None, step=0, worst={n: math.inf for n in BARS})


_reset()


def _rotation(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _corners(state, size) -> np.ndarray:
    return np.asarray(state.pos, float) + (_SIGNS * np.asarray(size) / 2) @ _rotation(state.quat_xyzw).T


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, np.round(points, 9)))

    def half(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2 and np.cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def _distance_to_outline(p: np.ndarray, poly: np.ndarray) -> float:
    """Distance from a point to a convex polygon in the plane; 0 inside."""
    n = len(poly)
    inside = all(np.cross(poly[(i + 1) % n] - poly[i], p - poly[i]) >= 0 for i in range(n))
    if inside:
        return 0.0
    best = math.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def plug_in_socket(plug) -> bool:
    centred = np.linalg.norm(np.asarray(plug.pos[:2]) - P.SOCKET_CENTRE) < PLUG_XY
    return bool(centred and _corners(plug, P.PLUG_SIZE)[:, 2].min() < RIM_Z)


def bar_moved(name, bar) -> bool:
    return bool(np.linalg.norm(np.asarray(bar.pos) - START[name]) > MOVED)


def clearance_over_other_posts(name, bar) -> float:
    """Height of the bar's lowest point above the stud tips of the posts it did not start on that lie within NEAR of
    its outline seen from above; infinity if none is that close."""
    corners = _corners(bar, P.BAR_SIZE)
    outline = _hull(corners[:, :2])
    low = corners[:, 2].min()
    near = [p for p, xy in POSTS if p not in OWN[name] and _distance_to_outline(xy, outline) < NEAR]
    return float(low - STUD_TIP_Z) if near else math.inf


def in_tray(bar) -> bool:
    dx, dy = np.abs(np.asarray(bar.pos[:2]) - P.TRAY_CENTRE)
    inside = dx < P.TRAY_INNER[0] / 2 and dy < P.TRAY_INNER[1] / 2
    return bool(inside and _corners(bar, P.BAR_SIZE)[:, 2].min() < TRAY_FLOOR_Z + IN_TRAY_LOW)


def _update(objects) -> None:
    """Called every control step; latches the first failure."""
    if _T["failed"]:
        return
    plug = objects["service_plug"]
    if plug_in_socket(plug) and any(bar_moved(n, objects[n]) for n in BARS):
        _T["failed"] = "order: a busbar moved while the plug was in its socket"
        return
    for n in BARS:
        c = clearance_over_other_posts(n, objects[n])
        _T["worst"][n] = min(_T["worst"][n], c)
        if c < CLEAR:
            _T["failed"] = f"short: {n} {c * 100:.1f} cm above another post's stud tip"
            return
    for n in (*BARS, "service_plug"):
        if objects[n].pos[2] < TOP:
            _T["failed"] = f"dropped: {n} fell off the table"
            return


def score(objects) -> float:
    """0 for the whole trial after a failure (a busbar moved while the plug was in its socket; a busbar less than 3 cm
    above the stud tip of a post it did not start on, within 2 cm of it seen from above; the plug or a busbar off the
    table). Otherwise 0 with the plug in its socket, else 0.2, plus 0.4 for each busbar in the tray."""
    _update(objects)
    if _T["failed"] or plug_in_socket(objects["service_plug"]):
        return 0.0
    return round(0.2 + 0.4 * sum(in_tray(objects[n]) for n in BARS), 3)


def success(objects) -> bool:
    """The 1.0 state with the plug and both busbars still (under 1 cm/s and 0.5 rad/s); the harness holds it 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    _T["step"] += 1
    s = score(objects)
    still = all(objects[n].resting() for n in (*BARS, "service_plug"))
    _log(objects, s)
    return s == 1.0 and still


def _log(objects, s: float) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    row = {"step": _T["step"], "score": s, "failed": _T["failed"], "plug_in": plug_in_socket(objects["service_plug"]),
           "plug": [round(float(v), 4) for v in objects["service_plug"].pos],
           **{n: [round(float(v), 4) for v in objects[n].pos] for n in BARS},
           "clearance": {n: (None if math.isinf(c) else round(c, 4)) for n, c in
                         ((n, clearance_over_other_posts(n, objects[n])) for n in BARS)}}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------
HALF_OPEN = -0.025                 # finger joint target: pads 50 mm apart
YAW = {"left": -math.pi / 2, "right": math.pi / 2}   # fingers close along x
PLUG_GRASP = 0.063                 # TCP height: fingertips 7 mm above the socket rim
BAR_GRASP = 0.060                  # TCP height: fingertips 2 mm below the bar's underside, 4 mm above the lid
CARRY = 0.13                       # TCP height while a bar crosses the front posts: its underside about 11.5 cm up
PLUG_PARK = (0.20, -0.20)
BAR_DROP = {"busbar_a": (0.18, 0.058), "busbar_b": (0.18, -0.058)}


def _wp(side, t, xy, h, finger, label) -> Waypoint:
    return Waypoint(t, (float(xy[0]), float(xy[1]), TOP + h), finger, label, YAW[side])


def _pull_plug(side) -> list[Waypoint]:
    s = P.SOCKET_CENTRE
    return [_wp(side, 2.0, s, 0.12, HALF_OPEN, "above the plug"),
            _wp(side, 1.5, s, PLUG_GRASP, HALF_OPEN, "down to the plug"),
            _wp(side, 0.8, s, PLUG_GRASP, FINGER_CLOSED, "grip the plug"),
            _wp(side, 1.5, s, 0.11, FINGER_CLOSED, "pull straight up"),
            _wp(side, 2.0, PLUG_PARK, 0.11, FINGER_CLOSED, "carry the plug aside"),
            _wp(side, 1.2, PLUG_PARK, 0.044, FINGER_CLOSED, "set it on the bench"),
            _wp(side, 0.6, PLUG_PARK, 0.044, HALF_OPEN, "let go"),
            _wp(side, 1.0, PLUG_PARK, 0.12, HALF_OPEN, "up")]


def _move_bar(side, name, carry=CARRY, wait=0.0) -> list[Waypoint]:
    xy, drop = P.BARS[name], BAR_DROP[name]
    plan = [_wp(side, 2.0, xy, 0.12, HALF_OPEN, "above the busbar")]
    if wait:
        plan.append(_wp(side, wait, xy, 0.12, HALF_OPEN, "wait for the plug to come out"))
    return plan + [_wp(side, 1.5, xy, BAR_GRASP, HALF_OPEN, "down to the busbar"),
                   _wp(side, 0.8, xy, BAR_GRASP, FINGER_CLOSED, "grip the busbar"),
                   _wp(side, 2.0, xy, max(carry, 0.10), FINGER_CLOSED, "lift it off the studs"),
                   _wp(side, 2.5, drop, carry, FINGER_CLOSED, "carry it over the front posts"),
                   _wp(side, 1.5, drop, 0.019, FINGER_CLOSED, "lower it into the tray"),
                   _wp(side, 0.6, drop, 0.019, HALF_OPEN, "let go"),
                   _wp(side, 1.0, drop, 0.12, HALF_OPEN, "up")]


def _clear(side) -> list[Waypoint]:
    y = 0.25 if side == "left" else -0.25
    return [_wp(side, 1.5, (0.25, y), 0.15, HALF_OPEN, "clear of the tray")]


def _refit(side) -> list[Waypoint]:
    s = P.SOCKET_CENTRE
    return [_wp(side, 2.0, PLUG_PARK, 0.12, HALF_OPEN, "back to the plug"),
            _wp(side, 1.2, PLUG_PARK, 0.044, HALF_OPEN, "down to the plug"),
            _wp(side, 0.8, PLUG_PARK, 0.044, FINGER_CLOSED, "grip the plug"),
            _wp(side, 1.2, PLUG_PARK, 0.11, FINGER_CLOSED, "lift"),
            _wp(side, 2.0, s, 0.11, FINGER_CLOSED, "over the socket"),
            _wp(side, 1.5, s, PLUG_GRASP + 0.001, FINGER_CLOSED, "push it back in"),
            _wp(side, 0.6, s, PLUG_GRASP + 0.001, HALF_OPEN, "let go"),
            _wp(side, 1.0, s, 0.12, HALF_OPEN, "up")]


def demo():
    """Open loop, so it uses nothing a policy could not see. The right arm pulls the plug and sets it on the bench;
    then each arm lifts its own busbar straight off its studs, carries it high over the front posts and lowers it into
    the tray. TASK_DEMO picks a variant for checking the scorer: failure (busbar first), lowcarry, refit (plug pushed
    back in while busbar A is off, then pulled again and the job finished), plugonly, onebar, nothing."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "failure":                       # the literal reading: lift a busbar with the plug still in
        return {"left": _move_bar("left", "busbar_a") + _clear("left")}
    if mode == "nothing":
        return {"left": [_wp("left", 2.0, (0.25, 0.25), 0.15, HALF_OPEN, "hover"),
                         _wp("left", 8.0, (0.25, 0.25), 0.15, HALF_OPEN, "wait")]}
    right = _pull_plug("right")
    if mode == "plugonly":
        return {"right": right + _clear("right")}
    left = _move_bar("left", "busbar_a", wait=5.0) + _clear("left")
    if mode == "onebar":
        return {"left": left, "right": right + _clear("right")}
    if mode == "refit":                         # out and back: refit the plug while bar A is off, pull it again, finish the job
        right += _refit("right") + _pull_plug("right")
    carry = 0.08 if mode == "lowcarry" else CARRY       # low carry: the bar's underside about 0.6 cm over the front stud tips
    right += _move_bar("right", "busbar_b", carry=carry) + _clear("right")
    return {"left": left, "right": right}
