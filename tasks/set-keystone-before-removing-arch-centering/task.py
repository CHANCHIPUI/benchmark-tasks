"""On the station table, in front of the right arm: a model stone arch, built as masons build one. Eight blue-grey
sandstone voussoirs rest on a timber centering (a solid pine turning piece) between two fixed sandstone abutments; the
arch spans 25 cm along y, opens toward the arms, and rises 12.75 cm. Every stone is laid except the keystone at the
crown, so a wedge-shaped gap shows at the top. The keystone, a slightly larger wedge of the same stone, stands upright on
its narrow end on the table in front of the arch. The centering has a pine board on its front with an upright beech peg
to pull it by.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import arch_usd as A  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 2.0
MODE = os.environ.get("TASK_DEMO", "success")
USD = A.build()

XA, YC = 0.30, -0.20                  # the arch's mid-plane (x) and crown line (y)
KEY_START = (0.15, -0.30)             # the keystone stands here, turned a quarter turn (its depth along y)
VOUSSOIRS = (-4, -3, -2, -1, 1, 2, 3, 4)
STONE_NAMES = {k: ("keystone" if k == 0 else f"voussoir_{k:+d}".replace("+", "p").replace("-", "m")) for k in range(-4, 5)}


def _quat_x(a: float) -> tuple[float, float, float, float]:
    return (math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))


def _stone_world(k: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    c, tilt = A.stone_pose(k)
    return (XA + c[0], YC + c[1], TOP + c[2]), _quat_x(tilt)


KEY_LOCAL_BOTTOM = float(A.stone_corners(0)[:, 2].min())
KEY_LOCAL_TOP = float(A.stone_corners(0)[:, 2].max())
KEY_SEATED = _stone_world(0)
KEY_START_POSE = ((KEY_START[0], KEY_START[1], TOP - KEY_LOCAL_BOTTOM + 0.0003), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
# Physics checks: TASK_DEMO=keyed starts with the keystone already seated.
_key_pos, _key_rot = KEY_SEATED if MODE == "keyed" else KEY_START_POSE

STONES = [UsdBody(STONE_NAMES[k], usd=USD[k], pos=_stone_world(k)[0], rot_xyzw=_stone_world(k)[1], size=(0.05, 0.04, 0.035),
                  mass=A.stone_mass(k), friction=A.STONE_FRICTION, solver_iters=64) for k in VOUSSOIRS]
KEYSTONE = UsdBody("keystone", usd=USD["keystone"], pos=_key_pos, rot_xyzw=_key_rot, size=(0.06, 0.044, 0.046),
                   mass=A.stone_mass(0), friction=A.STONE_FRICTION, solver_iters=64)
ABUTMENTS = [UsdBody(n, usd=USD[n], pos=(XA, YC, TOP), size=(0.05, 0.06, 0.0375), mass=None, friction=A.STONE_FRICTION)
             for n in ("abutment_left", "abutment_right")]
CENTERING = UsdBody("centering", usd=USD["centering"], pos=(XA, YC, TOP), size=(0.134, 0.247, 0.0925),
                    mass=A.centering_mass(), friction=A.PINE_FRICTION, friction_min=True, solver_iters=64)
OBJECTS = [*ABUTMENTS, CENTERING, *STONES, KEYSTONE]

# -- geometry for the scorer ----------------------------------------------------------------------------------------
DISTURB = 0.005                        # m: a voussoir this far from where it was built disturbs the arch
SEAT_TOL = 0.005                       # m
LEVEL_DEG = 5.0
CLEAR = 0.020                          # m: no part of a gripper this near a stone
STILL = 0.005                          # m/s
OUT_X = (XA - A.DEPTH / 2 - 0.005, XA + A.DEPTH / 2 + 0.005)   # the centering must lie wholly outside this x range
HOLD_STEPS = int(round(SUCCESS_HOLD_S * 20))
CENTERING_PTS = A.centering_points()


def _rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _planes(corners: np.ndarray) -> np.ndarray:
    """The outward face planes (n, d) of a convex hexahedron given by its 8 corners: n.p - d > 0 outside."""
    c = corners.mean(axis=0)
    out = []
    n = len(corners)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nrm = np.cross(corners[j] - corners[i], corners[k] - corners[i])
                if np.linalg.norm(nrm) < 1e-9:
                    continue
                nrm /= np.linalg.norm(nrm)
                side = (corners - corners[i]) @ nrm
                if np.all(side <= 1e-7) or np.all(side >= -1e-7):
                    if (c - corners[i]) @ nrm > 0:
                        nrm = -nrm
                    d = float(nrm @ corners[i])
                    if not any(np.allclose(nrm, p[:3], atol=1e-6) for p in out):
                        out.append(np.r_[nrm, d])
    return np.array(out)


STONE_PLANES = {k: _planes(A.stone_corners(k)) for k in range(-4, 5)}


def _gap(points: np.ndarray, state, k: int) -> float:
    """A lower bound on the distance from the points to stone k (its largest face-plane distance; 0 inside)."""
    local = (points - np.asarray(state.pos)) @ _rot(state.quat_xyzw)
    p = STONE_PLANES[k]
    signed = local @ p[:, :3].T - p[:, 3]
    return float(max(0.0, signed.max(axis=1).min()))


def _gripper_points_local() -> dict[str, np.ndarray]:
    """Points on each gripper part's collision meshes (finger tips and palm), in that link's frame."""
    from yam_arena import ASSET_DIR, YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = ET.parse(YAM_URDF).getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in ("tip_left", "tip_right", "gripper"):
            continue
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            path = ASSET_DIR / mesh.get("filename")
            raw = np.fromfile(path, dtype=np.uint8)
            n = int(np.frombuffer(raw[80:84].tobytes(), dtype="<u4")[0])
            tri = np.frombuffer(raw[84:84 + 50 * n].tobytes(), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            v = np.unique(tri["v"].reshape(-1, 3).astype(float), axis=0)
            pts.append(v @ rpy_matrix(rpy).T + xyz)
        if pts:
            v = np.vstack(pts)
            rng = np.random.default_rng(0)
            out[name] = v[rng.choice(len(v), size=min(80, len(v)), replace=False)]
    return out


_GRIPPER_LOCAL: dict = {}


def gripper_points() -> np.ndarray:
    """Points on both grippers (fingers and palm) in the station frame, from the simulator's link poses."""
    from scenekit import _np, env

    environment = env()
    if environment is None:
        return np.zeros((0, 3))
    if not _GRIPPER_LOCAL:
        _GRIPPER_LOCAL.update(_gripper_points_local())
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        poses = _np(robot.data.body_link_pose_w)[0]
        for link, local in _GRIPPER_LOCAL.items():
            i = robot.body_names.index(link)
            out.append(poses[i, :3] - origin + local @ _rot(poses[i, 3:7]).T)
    return np.vstack(out)


def keystone_top(key) -> np.ndarray:
    return np.asarray(key.pos) + _rot(key.quat_xyzw) @ np.array([0.0, 0.0, KEY_LOCAL_TOP])


def tilt_deg(state) -> float:
    return float(np.degrees(np.arccos(np.clip(_rot(state.quat_xyzw)[2, 2], -1.0, 1.0))))


def seated(key) -> bool:
    """Upright within 5 deg, top within 5 mm of its seated height, centre within 5 mm of the crown line and the mid-plane."""
    top = keystone_top(key)
    return bool(tilt_deg(key) <= LEVEL_DEG and abs(top[2] - (TOP + A.KEY_TOP_Z)) <= SEAT_TOL
                and abs(key.pos[1] - YC) <= SEAT_TOL and abs(key.pos[0] - XA) <= SEAT_TOL)


def centering_x(centering) -> tuple[float, float]:
    pts = np.asarray(centering.pos) + CENTERING_PTS @ _rot(centering.quat_xyzw).T
    return float(pts[:, 0].min()), float(pts[:, 0].max())


def centering_out(centering) -> bool:
    lo, hi = centering_x(centering)
    return bool(hi < OUT_X[0] or lo > OUT_X[1])


def gripper_gap(objects) -> float:
    pts = gripper_points()
    if not len(pts):
        return 1.0
    return min(_gap(pts, objects[STONE_NAMES[k]], k) for k in range(-4, 5))


# -- tracker ---------------------------------------------------------------------------------------------------------
_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Failures stay failed for the whole trial: only a reset clears them."""
    _T.clear()
    _T.update(built=None, disturbed=None, step=0, held=0, state={})


_reset()


def _update(objects) -> dict:
    _T["step"] += 1
    now = {k: np.asarray(objects[STONE_NAMES[k]].pos, float) for k in VOUSSOIRS}
    if _T["built"] is None:
        _T["built"] = now
    moved = {k: float(np.linalg.norm(now[k] - _T["built"][k])) for k in VOUSSOIRS}
    worst = max(moved, key=moved.get)
    if not _T["disturbed"] and moved[worst] > DISTURB:
        _T["disturbed"] = f"voussoir {worst:+d} moved {moved[worst] * 1000:.1f} mm (step {_T['step']})"
    key, centering = objects["keystone"], objects["centering"]
    s = {"moved": moved, "seated": seated(key), "out": centering_out(centering),
         "still": all(np.linalg.norm(objects[STONE_NAMES[k]].lin_vel) < STILL for k in range(-4, 5))}
    s["clear"] = gripper_gap(objects) >= CLEAR if (s["seated"] and s["out"]) or os.environ.get("TASK_SCORE_DIR") else True
    s["full"] = bool(not _T["disturbed"] and s["seated"] and s["out"] and s["clear"] and s["still"])
    _T["held"] = _T["held"] + 1 if s["full"] else 0
    _T["state"] = s
    return s


def score(objects) -> float:
    """0 for the whole trial once any voussoir has moved more than 5 mm from where it was built. Otherwise 1.0 once the
    keystone is seated, the centering is out, both grippers are clear and everything is still for 2 s; 0.5 with the
    keystone seated; else 0."""
    if not _T["state"]:
        _update(objects)
    s = _T["state"]
    if _T["disturbed"]:
        return 0.0
    if s["full"] and _T["held"] >= HOLD_STEPS:
        return 1.0
    return 0.5 if s["seated"] else 0.0


def success(objects) -> bool:
    """The 1.0 condition; the harness ends the trial once it has held for 2 s."""
    steptime.tick()  # the harness calls success() once per control step
    s = _update(objects)
    _log(objects, s)
    return s["full"] and MODE != "keyed"      # TASK_DEMO=keyed is a physics check: keep running to the run limit


def _log(objects, s: dict) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    key, cen = objects["keystone"], objects["centering"]
    authored = {k: np.array(_stone_world(k)[0]) for k in VOUSSOIRS}
    from_auth = {k: float(np.linalg.norm(np.asarray(objects[STONE_NAMES[k]].pos) - authored[k])) for k in VOUSSOIRS}
    lo, hi = centering_x(cen)
    row = {"step": _T["step"], "disturbed": _T["disturbed"], "held": _T["held"],
           "max_moved_mm": round(max(s["moved"].values()) * 1000, 3),
           "moved_mm": {k: round(v * 1000, 3) for k, v in s["moved"].items()},
           "max_from_authored_mm": round(max(from_auth.values()) * 1000, 3),
           "seated": s["seated"], "out": s["out"], "clear": s["clear"], "still": s["still"], "full": s["full"],
           "key": [round(float(v), 4) for v in key.pos], "key_top_mm": round((keystone_top(key)[2] - TOP) * 1000, 2),
           "key_tilt": round(tilt_deg(key), 2), "centering_x": [round(lo, 4), round(hi, 4)],
           "centering": [round(float(v), 4) for v in cen.pos], "gripper_gap_mm": round(gripper_gap(objects) * 1000, 1),
           "max_speed": round(max(float(np.linalg.norm(objects[STONE_NAMES[k]].lin_vel)) for k in range(-4, 5)), 4)}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------------
YAW0, YAW90 = 0.0, math.pi / 2
GRIP_ABOVE_BOTTOM = 0.015             # fingertips this far above the keystone's narrow end
PAD_TIP = 0.013                       # fingertips are this far below the TCP
KEY_PICK_H = GRIP_ABOVE_BOTTOM + PAD_TIP                          # TCP above the table when gripping the standing keystone
KEY_BOTTOM_SEATED = A.KEY_TOP_Z - (KEY_LOCAL_TOP - KEY_LOCAL_BOTTOM)
CARRY_H = 0.175                       # the keystone's narrow end about 2 cm above the stones at the crown
SEAT_H = KEY_BOTTOM_SEATED + KEY_PICK_H - 0.003                    # commanded 3 mm past the seat: the arm presses and gives
PEG = (XA - A.CENTER_DEPTH / 2 - A.BOARD[0] + 0.012, YC)          # the pull peg's axis
PEG_GRIP_H = A.BOARD[2] + A.PEG_H - 0.030 + PAD_TIP               # fingertips 3 cm below the peg's top
PULL = 0.065                          # the centering's back face ends 1.3 cm in front of the arch's front face
PEG_OPEN = -0.024                     # finger target: pads about 5 cm apart, clear of the centering's front
KEY_OPEN = -0.0375                    # finger target: pads about 7.5 cm apart, 7.5 mm clear of the keystone's faces
KEY_LET_GO = FINGER_OPEN              # wide open: the loaded grip holds the keystone off-centre, so a partial opening pushes it
RISE_H = 0.19                         # TCP height when moving between the crown and the peg: fingertips 4 cm above the keystone
KEY_PLACE_DX = -0.014                 # the loaded grip carries the keystone 14 mm off the tool point along x (from logged runs)
CLEAR_XY = (0.20, -0.50)             # out to the right: the arm hides none of the arch from the top camera
CLEAR_H = 0.10


def _wp(t, xy, h, finger, label, yaw=YAW0) -> Waypoint:
    return Waypoint(t, (float(xy[0]), float(xy[1]), TOP + h), finger, label, yaw)


def _set_keystone(turn: bool = True) -> list[Waypoint]:
    k = KEY_START
    c = (XA + KEY_PLACE_DX, YC) if turn else (XA, YC - KEY_PLACE_DX)   # the grip offset lies along the closing direction
    yaw = YAW90 if turn else YAW0
    # Unturned, the keystone is 6 cm long across the gap and comes to rest on the crown stones' tops (12.6 cm up).
    seat = SEAT_H if turn else 0.1263 + KEY_PICK_H - 0.003
    return [_wp(2.0, k, 0.10, KEY_OPEN, "above the keystone"),
            _wp(1.5, k, KEY_PICK_H, KEY_OPEN, "down to the keystone"),
            _wp(0.8, k, KEY_PICK_H, FINGER_CLOSED, "grip it across its parallel faces"),
            _wp(1.5, k, CARRY_H, FINGER_CLOSED, "lift it"),
            _wp(2.5, c, CARRY_H, FINGER_CLOSED, "over the crown, turning a quarter turn", yaw),
            _wp(2.0, c, seat + 0.015, FINGER_CLOSED, "lower it into the gap", yaw),
            _wp(1.2, c, seat, FINGER_CLOSED, "press it home", yaw),
            _wp(0.5, c, seat, FINGER_CLOSED, "hold", yaw),
            _wp(1.5, c, seat, KEY_LET_GO, "let go, gently", yaw),
            _wp(1.2, c, RISE_H, KEY_LET_GO, "straight up, clear of the keystone", yaw)]


def _pull_centering() -> list[Waypoint]:
    p, q = PEG, (PEG[0] - PULL, PEG[1])
    return [_wp(2.0, p, RISE_H, PEG_OPEN, "above the peg", YAW90),
            _wp(1.5, p, PEG_GRIP_H, PEG_OPEN, "down to the peg", YAW90),
            _wp(0.8, p, PEG_GRIP_H, FINGER_CLOSED, "grip the peg", YAW90),
            _wp(2.5, q, PEG_GRIP_H, FINGER_CLOSED, "draw the centering out toward the robot", YAW90),
            _wp(0.5, q, PEG_GRIP_H, PEG_OPEN, "let go", YAW90)]


def _wait(t: float, label: str = "wait clear of the arch") -> list[Waypoint]:
    return [_wp(1.5, CLEAR_XY, CLEAR_H, FINGER_OPEN, "move clear, up and out to the right"), _wp(t, CLEAR_XY, CLEAR_H, FINGER_OPEN, label)]


def demo():
    """Open loop, so it uses nothing a policy could not see. The right arm sets the keystone (gripped across its parallel
    faces, turned a quarter turn over the crown, pressed into the gap), then draws the centering out by its peg and
    waits clear. TASK_DEMO picks a variant: failure (draw the centering first), nothing (wait), keyed (physics check: the
    keystone starts seated; draw the centering, then wait), seat_only (set the keystone and leave the centering in),
    unturned (lower the keystone onto the gap without turning it, then draw the centering)."""
    if MODE == "nothing":
        return {"right": [_wp(2.0, CLEAR_XY, 0.18, FINGER_OPEN, "hover"), _wp(56.0, CLEAR_XY, 0.18, FINGER_OPEN, "wait")]}
    if MODE == "failure":
        return {"right": _pull_centering() + _wait(6.0)}
    if MODE == "keyed":
        return {"right": _pull_centering() + _wait(45.0)}
    if MODE == "unturned":             # scorer check: the keystone lowered onto the gap without its quarter turn
        return {"right": _set_keystone(turn=False) + _pull_centering() + _wait(6.0)}
    if MODE == "seat_only":
        return {"right": _set_keystone() + _wait(6.0)}
    return {"right": _set_keystone() + _pull_centering() + _wait(6.0)}
