"""On the station table, in front of the arms: an Avantco F100 countertop electric deep fryer, its front toward the
arms. Its 177FBASKET wire basket of fries is down in the oil on the bottom grate, its coated handle rising toward the
arms. The fryer's own basket hanger is its control head's top front edge.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
         --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Lift the fry basket out of the oil and hang it on the fryer's hanger to drain."

TOP = g.TOP
GOLD = (0.93, 0.76, 0.36)

# -- objects --------------------------------------------------------------------------------------------------------------
FRYER = UsdBody("fryer", usd="assets/fryer.usda", pos=(g.FRONT_X, g.CENTRE_Y, TOP), size=(g.DEPTH, g.WIDTH, g.HEIGHT),
                friction=(0.5, 0.4))
BASKET = UsdBody("basket", usd="assets/basket.usda", pos=g.basket_origin(g.REST_U, g.RACK), size=g.BASKET,
                 mass=g.BASKET_MASS, friction=(0.9, 0.8), torsional_patch=0.01)


def _fries() -> list:
    """Thirty fries lying across the basket's floor in three layers of ten (two columns of five)."""
    L = g.BASKET[0]
    ox, oy, oz = g.basket_origin(g.REST_U, g.RACK)
    out = []
    for layer in range(3):
        for col in (-1, 1):
            for row in range(5):
                x = ox - L / 2 + 0.035 + row * 0.042 + 0.012 * layer
                y = oy + col * 0.042 + 0.008 * (layer - 1)
                z = oz + g.WALL + g.FRY[2] / 2 + 0.0005 + layer * (g.FRY[2] + 0.001)
                out.append(Box(f"fry_{len(out)}", pos=(x, y, z), size=g.FRY, mass=g.FRY_MASS, color=GOLD))
    return out


FRIES = _fries()
OBJECTS = [FRYER, BASKET] + FRIES

# -- monitor ----------------------------------------------------------------------------------------------------------------
_STATE: dict = {}
HOLD_STEPS = 20            # control steps (1 s) hung, drained and with the arms clear before it counts as let go
L, W, H = g.BASKET
_P = g.HANDLE_PITCH
_HANDLE_DIR = np.array([-math.cos(_P), 0.0, math.sin(_P)])
# the basket's parts as boxes in its own frame: (rotation, centre, size)
_ROT_HANDLE = np.array([[math.cos(math.pi + _P), 0, math.sin(math.pi + _P)], [0, 1, 0],
                        [-math.sin(math.pi + _P), 0, math.cos(math.pi + _P)]])
BASKET_PARTS = [
    (np.eye(3), np.array([0.0, 0.0, H / 2]), np.array([L, W, H])),
    (_ROT_HANDLE, np.array([-L / 2, 0.0, H + g.HANDLE_STEM]) + _HANDLE_DIR * g.HANDLE_LEN / 2,
     np.array([g.HANDLE_LEN, g.HANDLE[0], g.HANDLE[1]])),
    (np.eye(3), np.array([-L / 2, 0.0, H + g.HANDLE_STEM / 2]), np.array([g.STEM_T, g.STEM_T, g.HANDLE_STEM])),
    (np.eye(3), np.array([L / 2 + g.HOOK_REACH / 2, 0.0, H - g.HOOK_LIP / 2]),
     np.array([g.HOOK_REACH, g.HOOK_W, g.HOOK_LIP])),
]
_OIL_SIZE = np.array([g.VAT_U[1] - g.VAT_U[0], 2 * g.VAT_HALF, g.OIL - g.VAT_FLOOR])
OIL_BOX = (np.eye(3), np.array([g.FRONT_X + sum(g.VAT_U) / 2, g.CENTRE_Y, TOP + (g.OIL + g.VAT_FLOOR) / 2]), _OIL_SIZE,
           None)   # the oil in the vat; edge points filled in below
# the wire basket's own surfaces, for "touching the hot basket": floor, four walls, the handle's bare riser, the hook's
# tab and lip (the coated grip is the only part the robot may touch)
HOT_PARTS = [(np.eye(3), np.array([0.0, 0.0, g.WALL / 2]), np.array([L, W, g.WALL])),
             (np.eye(3), np.array([-L / 2 + g.WALL / 2, 0.0, H / 2]), np.array([g.WALL, W, H])),
             (np.eye(3), np.array([L / 2 - g.WALL / 2, 0.0, H / 2]), np.array([g.WALL, W, H])),
             (np.eye(3), np.array([0.0, -W / 2 + g.WALL / 2, H / 2]), np.array([L, g.WALL, H])),
             (np.eye(3), np.array([0.0, W / 2 - g.WALL / 2, H / 2]), np.array([L, g.WALL, H])),
             (np.eye(3), np.array([-L / 2, 0.0, H + g.HANDLE_STEM / 2]), np.array([g.STEM_T, g.STEM_T, g.HANDLE_STEM])),
             (np.eye(3), np.array([L / 2 + g.HOOK_REACH / 2, 0.0, H - g.HOOK_T / 2]), np.array([g.HOOK_REACH, g.HOOK_W, g.HOOK_T])),
             (np.eye(3), np.array([L / 2 + g.HOOK_REACH - g.HOOK_T / 2, 0.0, H - g.HOOK_LIP / 2]),
              np.array([g.HOOK_T, g.HOOK_W, g.HOOK_LIP]))]
BASKET_POINTS = np.array([[sx * L / 2, sy * W / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, H)]
                         + [[L / 2 + g.HOOK_REACH, sy * g.HOOK_W / 2, H - g.HOOK_LIP] for sy in (-1, 1)])
FLOOR_CORNERS = np.array([[sx * L / 2, sy * W / 2, 0.0] for sx in (-1, 1) for sy in (-1, 1)])


def _edge_points(size, step: float = 0.005) -> np.ndarray:
    half, pts = np.asarray(size) / 2, []
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        t = np.linspace(-half[a], half[a], max(2, int(np.ceil(size[a] / step)) + 1))
        for sb in (-1.0, 1.0):
            for sc in (-1.0, 1.0):
                p = np.zeros((len(t), 3))
                p[:, a], p[:, b], p[:, c] = t, sb * half[b], sc * half[c]
                pts.append(p)
    return np.concatenate(pts)


PART_EDGES = [_edge_points(size) for _, _, size in BASKET_PARTS]
HOT_EDGES = [_edge_points(size) for _, _, size in HOT_PARTS]
OIL_BOX = OIL_BOX[:3] + (_edge_points(_OIL_SIZE, 0.01),)


def _rotation(xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(dipped=False, hot=False, hot_gap=1.0, out=False, on_rim=False, held=False, grip_gap=1.0, over_vat=True, spilled=0, hung=False, near=False, clearance=1.0,
                  clear=False, clear_steps=0, bottom=g.RACK - g.OIL, tilt=0.0, ever_out=False)


_reset_state()


def _robot_hulls(environment) -> list:
    from yam_arena.analysis import world_hulls

    from scenekit import robot_link_poses

    hulls = []
    for poses in robot_link_poses(environment).values():
        hulls.extend(h for _, h in world_hulls(poses))
    return hulls


def _distance(hulls, boxes) -> float:
    """The smallest gap between convex hulls and boxes (rotation, centre, size, edge points), negative on overlap."""
    from yam_arena.analysis import box_signed_distance

    verts = np.concatenate([h.vertices for h in hulls])
    h_lo = np.array([h.vertices.min(axis=0) for h in hulls])
    h_hi = np.array([h.vertices.max(axis=0) for h in hulls])
    best = 1.0
    for R, centre, size, local_edges in boxes:
        best = min(best, float(box_signed_distance(verts, centre, R, size).min()))
        edges = centre + local_edges @ R.T
        lo, hi = edges.min(axis=0) - 0.02, edges.max(axis=0) + 0.02
        for i in np.flatnonzero(np.all(h_hi >= lo, axis=1) & np.all(h_lo <= hi, axis=1)):
            p = hulls[i].planes
            best = min(best, float((edges @ p[:, :3].T - p[:, 3]).max(axis=1).min()))
    return best


def basket_pose(states) -> tuple[np.ndarray, np.ndarray]:
    s = states["basket"]
    return np.asarray(s.pos, float), _rotation(s.quat_xyzw)


def assess(states) -> dict:
    """What the cameras (or a person at the fryer) can see of the basket and the fries, from their poses."""
    o, R = basket_pose(states)
    floor = o + FLOOR_CORNERS @ R.T
    bottom = float(floor[:, 2].min()) - TOP
    u, v = floor[:, 0] - g.FRONT_X, floor[:, 1] - g.CENTRE_Y
    # (the opening's back edge is the control head's face, which a hung basket leans on: 5 mm allowed there only)
    outside = (u < g.VAT_U[0]) | (u > g.VAT_U[1] + 0.005) | (np.abs(v) > g.VAT_HALF)
    # resting on the fryer (deck, rims or the head's top): a floor corner outside the tank's opening, down at the height
    # of the surface under it
    # every corner of the basket and the hook's lip, not just the floor (unless hung, when the lip rightly sits over the
    # head's top behind the hanger)
    pts = o + BASKET_POINTS @ R.T
    pu, pv = pts[:, 0] - g.FRONT_X, pts[:, 1] - g.CENTRE_Y
    p_out = (pu < g.VAT_U[0]) | (pu > g.VAT_U[1] + 0.005) | (np.abs(pv) > g.VAT_HALF)
    p_out[-2:] |= pu[-2:] > g.HEAD_U                       # the hook's lip: over the head's top counts, with no allowance
    under = np.where(pu > g.HEAD_U, g.HEAD_TOP, g.DECK)
    on_rim = bool(np.any(p_out & (pts[:, 2] - TOP <= under + g.RIM_REST)))
    cu, cv = float(u.mean()), float(v.mean())                  # over the tank: the floor's centre inside the opening
    over_vat = bool(g.VAT_U[0] <= cu <= g.VAT_U[1] and abs(cv) <= g.VAT_HALF)
    inside = 0
    for f in FRIES:
        p = R.T @ (np.asarray(states[f.name].pos, float) - o)
        if abs(p[0]) <= L / 2 + g.FRY_TOL and abs(p[1]) <= W / 2 + g.FRY_TOL and -g.FRY_TOL <= p[2] <= H + 0.010:
            inside += 1
    rod = R.T @ (np.array([g.FRONT_X + g.ROD_U, g.CENTRE_Y, TOP + g.ROD_Z]) - o)
    tab_under = H - g.HOOK_T
    lip_in = L / 2 + g.HOOK_REACH - g.HOOK_T
    # hung: the rod inside the hook's notch, between the back wall and the lip, under the tab and above the lip's end
    hung = (L / 2 - 0.001 <= rod[0] <= lip_in + 0.001
            and H - g.HOOK_LIP - g.ROD_R <= rod[2] <= tab_under - g.ROD_R + g.HOOK_SEAT
            and abs(rod[1]) <= g.ROD_HALF)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
    on_rim = on_rim and not hung
    return dict(bottom=bottom, out=bool(bottom >= g.OIL + g.DRAIN_CLEAR), over_vat=over_vat, on_rim=on_rim,
                spilled=int(g.FRIES_N - inside), hung=bool(hung), tilt=tilt)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    states = object_states(environment, ["basket"] + [f.name for f in FRIES])
    S.update(assess(states))
    S["ever_out"] = S["ever_out"] or S["out"]
    hulls = _robot_hulls(environment)
    if _distance(hulls, [OIL_BOX]) < 0.0:
        S["dipped"] = True                                 # part of the robot went into the oil: latched
    o, R = basket_pose(states)
    boxes = [(R @ Rp, o + R @ c, size, edges) for (Rp, c, size), edges in zip(BASKET_PARTS, PART_EDGES)]
    S["clearance"] = _distance(hulls, boxes)
    S["hot_gap"] = _distance(hulls, [(R @ Rp, o + R @ c, size, edges)            # the wire basket and its hook:
                                     for (Rp, c, size), edges in zip(HOT_PARTS, HOT_EDGES)])   # all but the handle
    if S["hot_gap"] < g.TOUCH:
        S["hot"] = True                                    # touched the hot basket, not its coated handle: latched
    S["near"] = bool(S["clearance"] < g.HANDS_OFF)       # part of an arm within 2 cm of the basket
    S["grip_gap"] = _distance(hulls, [boxes[1]])           # the coated grip
    S["held"] = bool(S["grip_gap"] < g.TOUCH)              # held: a hand on the coated grip
    S["clear"] = not S["near"]
    good = (S["hung"] and S["out"] and S["over_vat"] and not S["on_rim"] and S["spilled"] == 0 and S["clear"]
            and not S["dipped"] and not S["hot"])
    S["clear_steps"] = S["clear_steps"] + 1 if good else 0


def success(objects) -> bool:
    """Hung on the hanger by its hook, its floor at least 2 cm above the oil and inside the vat's opening, every fry still
    in it, no part of the robot ever in the oil or on the basket but its handle, and both arms at least 2 cm from it (yam_arena ends the trial once this has held
    for SUCCESS_HOLD_S)."""
    S = _STATE
    return bool(S["hung"] and S["out"] and S["over_vat"] and not S["on_rim"] and S["spilled"] == 0 and S["clear"]
                and not S["dipped"] and not S["hot"])


def score(objects) -> float:
    """At the end of the trial, the first that applies (success scores 1.0):
    - 0 if any part of the robot went into the oil (its collision hulls into the oil's volume in the vat), or touched
      the basket anywhere but its coated handle (the wire basket and hook are hot in real use), at any time;
    - 1.0: for the last 1 s, hung on the hanger, its floor 2 cm or more above the oil, every fry in it, and the arms
      2 cm or more from the basket;
    - 0 if it rests on the deck or the rims (a floor corner outside the tank's opening, at the deck's height);
    - 0.5: its floor 2 cm or more above the oil, its centre over the tank's opening, every fry in it, and either hung
      on the hanger or held (a hand on its coated grip);
    - 0.25: the same with one to three fries out of it;
    - 0 otherwise (still in the oil, dropped, laid across the vat's rims, carried off, or more fries lost)."""
    S = _STATE
    if S["dipped"] or S["hot"]:
        return 0.0
    if S["clear_steps"] >= HOLD_STEPS:
        return 1.0
    if S["on_rim"]:
        return 0.0                                         # resting on the deck or the rims, not held or hung
    supported = S["hung"] or S["held"]                    # hanging by its hook, or held by its coated grip
    if S["out"] and S["over_vat"] and supported and S["spilled"] == 0:
        return 0.5
    if S["out"] and S["over_vat"] and supported and S["spilled"] <= g.SPILL_SOME:
        return 0.25
    return 0.0
