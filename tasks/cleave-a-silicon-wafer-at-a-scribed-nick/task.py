"""In a black lab tray in front of the arms: a 100 mm silicon test wafer, polished face up, lies in the pocket of a
round PETG stage, its primary flat toward the robot. A black tick on its rim marks where a 30 mm test piece is to be
cleaved off; there the stage's notch leaves the rim overhanging. To the right, outside the tray, a pen-style diamond
scriber stands tip down in a blue PLA stand. To the left, in the tray, 3-point wafer-cleaving pliers lie on edge in a
grey PLA stand, jaws forward.

The robot scribes a short nick at the tick with the scriber, then breaks the wafer from the nick with the pliers,
touching the wafer only with the scriber's tip and the pliers' jaw inserts.

Simplified physics: rigid bodies. Fracture is a rule read from PhysX contact forces (task.json, "scoring"):
when it fires, the wafer is replaced at the same pose by its two pieces, pre-cut on the line through the nick.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import cleave_usd as C  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, before_reset, on_reset, on_step, set_joint_targets  # noqa: E402
from yam_arena.analysis import link_hulls  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
HZ = 20.0

# -- layout (station frame, m) -------------------------------------------------------------------------------------------
WAFER_C = np.array([0.460, -0.020])                    # wafer and stage centre
TICK = np.array([WAFER_C[0] - math.sqrt(C.WAFER_R ** 2 - C.TARGET_Y ** 2), 0.0])   # (0.41417, 0): the rim at the tick
STAGE_TOP = TOP + C.STAGE_H
TRAY_C = np.array([0.460, 0.0])
STAND_C = np.array([0.366, -0.250])
PEN_START = np.array([STAND_C[0], STAND_C[1], TOP + C.STAND[2] - C.STAND_DEPTH])    # the point at the hole's floor
PLIERS_Y = 0.075
PIVOT_START = np.array([0.468, PLIERS_Y, TOP + C.CRADLE_AXIS_Z])
SHELF_C = np.array([0.36, 0.0, 0.40])

USD = C.build()
LINES = C.LINES_MM
PIECES = [f"{side}_{C.sfx(k)}" for k in LINES for side in ("cpn", "main")]
WAFERS = ["wafer"] + PIECES


def _park(i):
    gx, gy = i % 6, i // 6
    return np.array([SHELF_C[0] - 0.39 + 0.15 * gx, SHELF_C[1] - 0.45 + 0.15 * gy, SHELF_C[2] + C.SHELF[2] / 2 + 0.0005])


PARK = {n: _park(i) for i, n in enumerate(PIECES + ["wafer"])}

OBJECTS = [
    UsdBody("stage", usd=USD["stage"], pos=(WAFER_C[0], WAFER_C[1], TOP), size=(0.12, 0.12, 0.12), mass=None,
            friction=(0.5, 0.45)),
    UsdBody("tray", usd=USD["tray"], pos=(TRAY_C[0], TRAY_C[1], TOP), size=(0.308, 0.408, 0.025), mass=None,
            friction=(0.5, 0.45)),
    UsdBody("wafer", usd=USD["wafer"], pos=(WAFER_C[0], WAFER_C[1], STAGE_TOP + 0.00002), size=(0.1, 0.1, 0.0005),
            mass=0.00954, friction=(0.35, 0.3), solver_iters=64),
    *[UsdBody(n, usd=USD[n], pos=tuple(PARK[n]), size=(0.1, 0.1, 0.0005), mass=float(
        C.poly_area(C.clip(C.wafer_outline(), (0, 1 if n.startswith("cpn") else -1),
                           (1 if n.startswith("cpn") else -1) * (C.TARGET_Y + LINES[i // 2] / 1000.0)))
        * C.WAFER_T_REAL * C.SI_DENSITY), friction=(0.35, 0.3), solver_iters=64) for i, n in enumerate(PIECES)],
    UsdBody("stand", usd=USD["stand"], pos=(STAND_C[0], STAND_C[1], TOP), size=C.STAND, mass=None, friction=(0.5, 0.45)),
    UsdBody("scriber", usd=USD["scriber"], pos=tuple(PEN_START), size=(0.009, 0.009, 0.14), mass=C.PEN_MASS,
            friction=(0.8, 0.7), solver_iters=32, torsional_patch=0.004),
    UsdBody("cradle", usd=USD["cradle"], pos=(PIVOT_START[0], PLIERS_Y, TOP), size=(0.1, 0.04, 0.12), mass=None,
            friction=(0.5, 0.45)),
    UsdBody("pliers", usd=USD["pliers"], pos=tuple(PIVOT_START), size=(0.152, 0.016, 0.06), articulation=True,
            joint_pos=(("pivot", 0.0),), drives=(("pivot", C.PIVOT_K, C.PIVOT_D, 5.0),), friction=(1.0, 0.9),
            solver_iters=32, torsional_patch=0.010),
    UsdBody("shelf", usd=USD["shelf"], pos=tuple(SHELF_C), size=C.SHELF, mass=None, friction=(0.8, 0.7)),
]

# -- scoring terms (task.json, "scoring") --------------------------------------------------------------------------------
F_TOUCH = 0.005          # N: contact
F_SCRIBE = 0.3           # N: a scribe stroke from here ...
F_CHIP = 3.0             # ... chips the edge from here ...
F_SHATTER_SCRIBE = 6.0   # ... and shatters the wafer above this
F_CLEAVE = 12.0          # N on the jaw ridge, with a nick within 3 mm of it
F_NO_NICK = 24.0         # N on the jaw ridge with no nick there: shatters
NICK_TO_RIDGE = 0.003
EDGE_BAND = 0.005        # a nick lies wholly within this of the rim; a slide this far in is a scratch
NICK_MIN = 0.001         # stroke length for a nick, and for a scratch
NICK_RIM = 0.001         # a nick ends this close to the rim, or runs off it
NICK_TICK = 0.003        # the 0.25 tier: a nick whose rim end is this close to the tick
STRIKE_V = 0.3           # m/s
OFF_STAGE = 0.005        # a piece more than this below the stage top has left it
CLEAR = 0.03             # fingertips this far from every piece
STILL_V = 0.001          # m/s
HOLD = round(1.0 * HZ)   # the harness's 1 s (SUCCESS_HOLD_S)
TIP_ZONE = 0.003         # the scriber's tip: its point and the first 3 mm of the collet cone
FLAT_TILT = 5.0          # degrees: "flat"
SWAP_QUIET = 10          # control steps after the swap in which new contacts are not judged as strikes
GRIP_HOLD = 0.03         # a TCP this close to the pliers' grip centre holds the pliers
AT_STOP = 0.004          # rad: the pivot this close to its closing stop has the jaws closed
R_GRIP, R_CONTACT = -C.GRIP_X, C.CONTACT_R
TOOLS = ["scriber", "jaw_lower", "jaw_upper", "a_frame", "b_frame"]
FINGERS = ["L:gripper", "L:tip_left", "L:tip_right", "R:gripper", "R:tip_left", "R:tip_right"]
FILTERS = TOOLS + FINGERS
FI = {n: i for i, n in enumerate(FILTERS)}
GRIP_LINKS = ("gripper", "tip_left", "tip_right")


def _grid_points(h, step=0.006) -> np.ndarray:
    lo, hi = h.vertices.min(0), h.vertices.max(0)
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + step, step) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    inside = (g @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1) <= 0
    return np.vstack([h.vertices, g[inside]])


GRIP_POINTS = {link: np.vstack([_grid_points(h) for h in link_hulls()[link]]) for link in GRIP_LINKS}


def _box_points(size, centre, q, step=0.002) -> np.ndarray:
    """Surface sample points of a box (pliers frame)."""
    sx, sy, sz = size
    g = [np.linspace(-s / 2, s / 2, max(2, int(math.ceil(s / step)) + 1)) for s in (sx, sy, sz)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    half = np.array(size) / 2
    on = np.any(np.isclose(np.abs(P), half), axis=1)
    P = P[on]
    w, x, y, z = q
    R = _rot((x, y, z, w))
    return P @ R.T + np.asarray(centre)


RIDGE_PT = C.ridge_point()        # the ridge's top at the contact centre, jaw_lower link frame
PAD_PT = C.pad_point()            # the pads' face there (on the centre line), jaw_upper link frame
OUTLINE = C.wafer_outline(96)


def _rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


_PG = C.pliers_geometry()
BODY_POINTS = {n: np.vstack([_box_points(s, c, q) for _, s, c, q, _ in _PG[n]]) for n in ("a_frame", "b_frame")}


def _piece_poly(name):
    if name == "wafer":
        return OUTLINE
    side, s = name.split("_")
    k = 0 if s == "0" else (int(s[1:]) * (-1 if s[0] == "m" else 1))
    y = C.TARGET_Y + k / 1000.0
    return C.clip(OUTLINE, (0, 1), y + C.GAP) if side == "cpn" else C.clip(OUTLINE, (0, -1), -(y - C.GAP))


POLY = {n: _piece_poly(n) for n in WAFERS}


def rim_depth(p) -> float:
    """How far point p (wafer frame, xy) lies inside the whole wafer's rim (negative outside)."""
    x, y = float(p[0]), float(p[1])
    return min(C.WAFER_R - math.hypot(x, y), x + C.FLAT_D)


def _inside(poly, p, margin=0.0) -> bool:
    """Point inside convex polygon (counter-clockwise or not) by at least margin."""
    P = np.asarray(poly)
    c = P.mean(0)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        d = b - a
        n = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-12)
        if n @ (c - a) > 0:
            n = -n                                      # outward
        if n @ (np.asarray(p) - a) > -margin:
            return False
    return True


# -- per-physics-step contact monitor -----------------------------------------------------------------------------------
# A PhysX rigid-contact view (the mechanism behind Isaac Lab's ContactSensor) on the wafer and every piece, filtered by
# the scriber, the pliers' four links and the six finger links, read after every physics step (240 Hz). The control step
# uses the mean and the peak of the 12 reads.
_MON: dict = {"view": None, "err": None, "sum": None, "peak": None, "n": 0}


def _enable_contact_processing() -> None:
    try:
        import carb

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # pragma: no cover
        pass


def _make_view(mgr) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    sv = mgr.get_physics_sim_view()
    if sv is None:
        return
    stage = get_current_stage()
    sens, filt = {}, {}
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parts = p.split("/")
        name = parts[-1]
        if name in WAFERS and parts[-2] == "env_0":
            sens[name] = p
        elif name == "scriber" and parts[-2] == "env_0":
            filt[name] = p
        elif name in ("jaw_lower", "jaw_upper", "a_frame", "b_frame") and parts[-2] == "pliers":
            filt[name] = p
        elif name in GRIP_LINKS and ("/LeftArm/" in p or "/RightArm/" in p):
            filt[("L:" if "/LeftArm/" in p else "R:") + name] = p
    if len(sens) != len(WAFERS) or len(filt) != len(FILTERS):
        raise RuntimeError(f"cleave monitor: sensors {sorted(sens)}, filters {sorted(filt)}")
    view = sv.create_rigid_contact_view([sens[n] for n in WAFERS], filter_patterns=[[filt[n] for n in FILTERS]] * len(WAFERS),
                                        max_contact_data_count=8 * len(WAFERS) * len(FILTERS))
    _MON.update(view=view)
    _zero()
    print(f"[cleave] contact monitor: {len(WAFERS)} wafer bodies x {len(FILTERS)} filters", flush=True)


def _zero() -> None:
    _MON.update(sum=np.zeros((len(WAFERS), len(FILTERS), 3)), peak=np.zeros((len(WAFERS), len(FILTERS))), n=0)


def _poll(mgr) -> None:
    try:
        if _MON["view"] is None:
            _make_view(mgr)
        if _MON["view"] is None:
            return
        dt = mgr.get_physics_dt()
        f = _np(_MON["view"].get_contact_force_matrix(dt=dt)).reshape(len(WAFERS), len(FILTERS), 3)
    except Exception as e:  # pragma: no cover
        if _MON["err"] is None:
            _MON["err"] = repr(e)
            print(f"[cleave] monitor error: {e!r}", flush=True)
        return
    _MON["sum"] += f
    _MON["peak"] = np.maximum(_MON["peak"], np.linalg.norm(f, axis=2))
    _MON["n"] += 1


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_cleave_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _poll(cls)

    step._cleave_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)
    mgr.step = classmethod(step)


class _MgrFinder(importlib.abc.MetaPathFinder):
    """Patches Isaac Lab's PhysxManager when it is first imported (it needs Kit, so not at task import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _PHYSX_MGR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_module
        return spec


if _PHYSX_MGR in sys.modules:
    _patch_manager(sys.modules[_PHYSX_MGR])
elif not any(isinstance(f, _MgrFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _MgrFinder())


# -- swapping the wafer for its two pieces ---------------------------------------------------------------------------
def _set_visible(name: str, visible: bool) -> None:
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if p.endswith(f"/env_0/{name}/visual"):
                img = UsdGeom.Imageable(prim)
                img.MakeVisible() if visible else img.MakeInvisible()
                return
    except Exception as e:  # pragma: no cover
        print(f"[cleave] visibility: {e!r}", flush=True)


def _write_pose(env, name, pose7, vel6) -> None:
    import torch

    asset = env.unwrapped.scene[name]
    dev = asset.device
    asset.write_root_pose_to_sim_index(root_pose=torch.tensor([pose7], dtype=torch.float32, device=dev))
    asset.write_root_velocity_to_sim_index(root_velocity=torch.tensor([vel6], dtype=torch.float32, device=dev))


def _swap(env, k) -> None:
    """Replace the wafer by the pieces cut on line k, at the wafer's pose and velocity; park the wafer."""
    base = env.unwrapped
    w = base.scene["wafer"]
    pose = _np(w.data.root_link_pose_w)[0].tolist()
    vel = _np(w.data.root_vel_w)[0].tolist() if hasattr(w.data, "root_vel_w") else [0.0] * 6
    origin = _np(base.scene.env_origins[0])
    for n in (f"cpn_{C.sfx(k)}", f"main_{C.sfx(k)}"):
        _write_pose(env, n, pose, vel)
        _set_visible(n, True)
    park = PARK["wafer"] + origin
    _write_pose(env, "wafer", [*park.tolist(), *pose[3:7]], [0.0] * 6)
    _set_visible("wafer", False)


# -- the trial-long tracker -----------------------------------------------------------------------------------------------
_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Only a reset clears the tracker (and with it any latched failure)."""
    _T.clear()
    _T.update(step=0, failed=None, active=["wafer"], line=None, offmark=False, nicks=[], stroke=None,
              cut_len=0.0, prev_touch=np.zeros((len(WAFERS), len(FILTERS)), bool), swap_step=-999, clear_steps=0,
              reading=None, world=None, ridge_max=0.0, scribe_max=0.0, pending_swap=None, gate_now=False)
    _zero()


_reset()


@before_reset
def _pliers_spring(environment) -> None:
    try:
        set_joint_targets(environment, "pliers", {"pivot": C.SPRING_TARGET})
    except Exception as e:  # pragma: no cover
        print(f"[cleave] spring target: {e!r}", flush=True)


def _link_vel(asset, i=None) -> np.ndarray:
    d = asset.data
    if i is None:
        for a in ("root_lin_vel_w", "root_link_lin_vel_w"):
            if hasattr(d, a):
                return _np(getattr(d, a))[0][:3]
        return _np(d.root_vel_w)[0][:3]
    for a in ("body_lin_vel_w", "body_link_lin_vel_w", "body_com_lin_vel_w"):
        if hasattr(d, a):
            return _np(getattr(d, a))[0][i][:3]
    return _np(d.body_vel_w)[0][i][:3]


@on_step
def _read(environment) -> None:
    """Every control step: poses and velocities, and the contact means since the last step."""
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    if _T["pending_swap"] is not None:
        _swap(environment, _T["pending_swap"])
        _T["swap_step"] = _T["step"]
        _T["pending_swap"] = None
    W = {"bodies": {}, "vel": {}, "fingers": {}, "tcp": {}, "links": {}, "lvel": {}}
    for n in _T["active"]:
        a = base.scene[n]
        pose = _np(a.data.root_link_pose_w)[0]
        W["bodies"][n] = (pose[:3] - origin, _rot(pose[3:7]))
        W["vel"][n] = _link_vel(a)
    sc = base.scene["scriber"]
    pose = _np(sc.data.root_link_pose_w)[0]
    W["links"]["scriber"] = (pose[:3] - origin, _rot(pose[3:7]))
    W["lvel"]["scriber"] = _link_vel(sc)
    pl = base.scene["pliers"]
    poses = _np(pl.data.body_link_pose_w)[0]
    names = list(pl.body_names)
    for n in ("jaw_lower", "jaw_upper", "a_frame", "b_frame"):
        i = names.index(n)
        W["links"][n] = (poses[i][:3] - origin, _rot(poses[i][3:7]))
        W["lvel"][n] = _link_vel(pl, i)
    W["pivot"] = float(_np(pl.data.joint_pos)[0][0])
    bp, bR = W["links"]["b_frame"]
    W["grip_c"] = bp + bR @ np.array([C.GRIP_X, 0.0, 0.0])
    try:
        set_joint_targets(environment, "pliers", {"pivot": C.SPRING_TARGET})   # the return spring's preload
    except Exception:  # pragma: no cover
        pass
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        poses = _np(robot.data.body_link_pose_w)[0]
        pts = []
        for link in GRIP_LINKS:
            i = robot.body_names.index(link)
            p = poses[i]
            R = _rot(p[3:7])
            pts.append(GRIP_POINTS[link] @ R.T + (p[:3] - origin))
            W["lvel"][("L:" if side == "left" else "R:") + link] = _link_vel(robot, i)
            if link == "gripper":
                W["tcp"][side] = p[:3] - origin + R @ np.array([0.0, 0.0, 0.134])
        W["fingers"][side] = np.vstack(pts)
        W.setdefault("squeeze", {})[side] = _finger_force(robot)
    k = max(_MON["n"], 1)
    W["f_mean"] = np.linalg.norm(_MON["sum"] / k, axis=2) if _MON["sum"] is not None else np.zeros((len(WAFERS), len(FILTERS)))
    W["f_peak"] = _MON["peak"].copy() if _MON["peak"] is not None else np.zeros((len(WAFERS), len(FILTERS)))
    W["polls"] = _MON["n"]
    _zero()
    _T["world"] = W
    _track(W, environment)


_FINGER_IDS: dict = {}


def _finger_force(robot) -> float:
    """The gripper's squeeze (N per finger): the finger drives' effort, as the actuator model reports it."""
    key = id(robot)
    if key not in _FINGER_IDS:
        _FINGER_IDS[key] = robot.find_joints(["joint7", "joint8"], preserve_order=True)[0]
    ids = _FINGER_IDS[key]
    try:
        eff = _np(robot.data.applied_torque)[0][ids]
    except Exception:  # pragma: no cover
        from yam_arena.constants import GRIPPER

        q = _np(robot.data.joint_pos)[0][ids]
        qt = _np(robot.data.joint_pos_target)[0][ids]
        eff = np.clip(GRIPPER["stiffness"] * (qt - q), -GRIPPER["effort"], GRIPPER["effort"])
    return float(np.mean(np.abs(eff)))


def clamp_force(W) -> tuple:
    """The pliers' clamp on the wafer (N at the ridge) and whether the jaws are closed on it: at their stop, with the
    ridge's mid-point over the wafer and level with its lower face. The force is the holding gripper's squeeze on the
    grips, less the return spring, through the lever (grip centre and contact centre from the pivot)."""
    if "wafer" not in W["bodies"]:
        return 0.0, False
    frame = W["bodies"]["wafer"]
    jp, jR = W["links"]["jaw_lower"]
    rc = _to_wafer(jp + jR @ RIDGE_PT, frame)
    on = (W["pivot"] <= C.CLOSE_LIMIT + AT_STOP and _inside(OUTLINE, rc[:2], 0.0005) and abs(rc[2]) <= 0.0006)
    holders = [s for s in ("left", "right") if np.linalg.norm(W["tcp"][s] - W["grip_c"]) <= GRIP_HOLD]
    if not on or not holders:
        return 0.0, on
    squeeze = max(W["squeeze"][s] for s in holders)
    spring = C.PIVOT_K * (C.SPRING_TARGET - W["pivot"]) / R_GRIP
    return max(0.0, squeeze - spring) * R_GRIP / R_CONTACT, on


def _fail(why) -> None:
    if _T["failed"] is None:
        _T["failed"] = why
        print(f"[cleave] latched 0 at step {_T['step']}: {why}", flush=True)


def _to_wafer(p, frame):
    pos, R = frame
    return (np.asarray(p) - pos) @ R


def _end_stroke() -> None:
    """Judge a finished scribe stroke on the whole wafer: a nick, and whether it chipped the edge."""
    s = _T["stroke"]
    _T["stroke"] = None
    if s is None or s["body"] != "wafer":
        return
    pts = np.array(s["pts"])
    if len(pts) < 2:
        return
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    depths = np.array([rim_depth(p) for p in pts])
    ok = length >= NICK_MIN and depths.max() <= EDGE_BAND and depths[-1] <= NICK_RIM
    if ok:
        end = pts[-1]
        nick = dict(end=end, y=float(end[1] - C.TARGET_Y), chip=s["fmax"] >= F_CHIP, fmax=s["fmax"], length=length,
                    step=_T["step"])
        _T["nicks"].append(nick)
        print(f"[cleave] nick at step {_T['step']}: rim end y {1000 * nick['y']:+.2f} mm from the target, "
              f"{1000 * length:.1f} mm long, peak {s['fmax']:.2f} N{' (chip)' if nick['chip'] else ''}", flush=True)
    else:
        print(f"[cleave] stroke without a nick at step {_T['step']}: {1000 * length:.1f} mm, deepest "
              f"{1000 * depths.max():.1f} mm in, ends {1000 * depths[-1]:.1f} mm in, peak {s['fmax']:.2f} N", flush=True)


def _track(W, env) -> None:
    T = _T
    T["step"] += 1
    step = T["step"]
    rows = {n: WAFERS.index(n) for n in T["active"]}
    fm, fp = W["f_mean"], W["f_peak"]
    # 1. fingers on the wafer or a piece
    for n, r in rows.items():
        for f in FINGERS:
            if fp[r, FI[f]] > F_TOUCH:
                _fail(f"finger {f} touched {n}")
    # 2. new contacts striking at speed (the scriber, the pliers, a finger)
    touch = fp > F_TOUCH
    if step - T["swap_step"] > SWAP_QUIET:
        for n, r in rows.items():
            for f in FILTERS:
                j = FI[f]
                if touch[r, j] and not T["prev_touch"][r, j]:
                    v = float(np.linalg.norm(W["lvel"][f] - W["vel"][n]))
                    if v > STRIKE_V:
                        _fail(f"shattered: {f} struck {n} at {v:.2f} m/s")
    T["prev_touch"] = touch
    # 3. anything but the scriber's tip or the jaw inserts on a top face
    sp, sR = W["links"]["scriber"]
    axis = sR[:, 2]
    for n, r in rows.items():
        frame = W["bodies"][n]
        top = C.WAFER_T
        if fp[r, FI["scriber"]] > F_TOUCH:
            for z in np.linspace(TIP_ZONE, C.PEN_L, 28):
                rad = C.PEN_R if z >= C.PEN_CONE_L else 0.0012 + (0.0035 - 0.0012) * z / C.PEN_CONE_L
                q = _to_wafer(sp + axis * z, frame)
                low = q[2] - rad * math.sqrt(max(0.0, 1 - (frame[1][:, 2] @ axis) ** 2))
                if low - top < 0.001 and _inside(POLY[n], q[:2], 0.0005):
                    _fail(f"scriber barrel on the top face of {n}")
                    break
        for b in ("a_frame", "b_frame"):
            if fp[r, FI[b]] > F_TOUCH:
                lp, lR = W["links"][b]
                q = _to_wafer(BODY_POINTS[b] @ lR.T + lp, frame)
                hit = (q[:, 2] > top - 0.0003) & (q[:, 2] < top + 0.001)
                if hit.any() and any(_inside(POLY[n], p[:2], 0.0005) for p in q[hit][:200]):
                    _fail(f"pliers body ({b}) on the top face of {n}")
    # 4. scribe strokes: nick, chip, scratch, too hard
    tip_force, tip_body = 0.0, None
    for n, r in rows.items():
        if fm[r, FI["scriber"]] > tip_force:
            tip_force, tip_body = float(fm[r, FI["scriber"]]), n
    T["scribe_max"] = max(T["scribe_max"], tip_force)
    if tip_force > F_SHATTER_SCRIBE:
        _fail(f"shattered: scribe force {tip_force:.1f} N")
    if tip_body is not None and tip_force >= F_SCRIBE:
        q = _to_wafer(sp, W["bodies"][tip_body])
        s = T["stroke"]
        if s is not None and s["body"] != tip_body:
            _end_stroke()
            s = None
        if s is None:
            s = T["stroke"] = dict(body=tip_body, pts=[], fmax=0.0, deep=0.0)
        if s["pts"] and rim_depth(q[:2]) > EDGE_BAND:
            seg = float(np.linalg.norm(q[:2] - s["pts"][-1]))
            if rim_depth(s["pts"][-1]) > EDGE_BAND:
                s["deep"] += seg
                if s["deep"] >= NICK_MIN:
                    _fail(f"scratch: the tip slid {1000 * s['deep']:.1f} mm more than 5 mm in from the rim")
        s["pts"].append(q[:2].copy())
        s["fmax"] = max(s["fmax"], tip_force)
    elif T["stroke"] is not None:
        _end_stroke()
    # 5. the cleave rule
    if "wafer" in rows and T["line"] is None and not T["offmark"]:
        f_ridge, _ = clamp_force(W)
        T["ridge_max"] = max(T["ridge_max"], f_ridge)
        if f_ridge >= F_CLEAVE:
            frame = W["bodies"]["wafer"]
            jp, jR = W["links"]["jaw_lower"]
            rc = _to_wafer(jp + jR @ RIDGE_PT, frame)[:2]
            rd = (frame[1].T @ jR[:, 0])[:2]
            rd = rd / max(np.linalg.norm(rd), 1e-9)
            near = []
            for nk in T["nicks"]:
                d = nk["end"] - rc
                dist = abs(d[0] * rd[1] - d[1] * rd[0])
                if dist <= NICK_TO_RIDGE:
                    near.append((dist, nk))
            if near:
                nk = min(near, key=lambda t: t[0])[1]
                k = int(round(1000 * nk["y"]))
                if abs(1000 * nk["y"]) > 6.5:
                    T["offmark"] = True
                    print(f"[cleave] off-mark break at step {step}: nick {1000 * nk['y']:+.1f} mm from the target", flush=True)
                else:
                    T["line"] = dict(k=k, chip=nk["chip"], step=step, ridge=f_ridge)
                    T["pending_swap"] = k
                    T["active"] = [f"cpn_{C.sfx(k)}", f"main_{C.sfx(k)}"]
                    print(f"[cleave] cleaved at step {step}: ridge {f_ridge:.1f} N, line {k:+d} mm, nick "
                          f"{1000 * nk['y']:+.2f} mm", flush=True)
            elif f_ridge >= F_NO_NICK:
                _fail(f"shattered: {f_ridge:.1f} N on the jaw ridge with no nick within 3 mm")
    # 6. pieces on the stage, face up
    for n in T["active"]:
        if n not in W["bodies"]:
            continue
        pos, R = W["bodies"][n]
        P = np.c_[POLY[n], np.zeros(len(POLY[n]))] @ R.T + pos
        if P[:, 2].min() + TOP < STAGE_TOP - OFF_STAGE:
            _fail(f"{n} left the stage")
        if R[2, 2] < 0:
            _fail(f"{n} turned face down")
    # 7. the gate
    tools_off = all(fp[r, FI[t]] <= F_TOUCH for r in rows.values() for t in TOOLS)
    gaps = []
    for n in T["active"]:
        if n not in W["bodies"]:
            continue
        for side, pts in W["fingers"].items():
            gaps.append(_slab_dist(_to_wafer(pts, W["bodies"][n]), POLY[n]))
    T["gap"] = min(gaps) if gaps else 0.0
    still = all(np.linalg.norm(W["vel"][n]) < STILL_V for n in T["active"] if n in W["vel"])
    fresh = T["pending_swap"] is None and (step - T["swap_step"]) > 2
    T["gate_now"] = bool(tools_off and T["gap"] >= CLEAR and still and fresh)
    T["clear_steps"] = T["clear_steps"] + 1 if T["gate_now"] else 0
    if T["clear_steps"] >= HOLD:
        T["reading"] = _tier(W)
    if step % 20 == 0 or step < 3:
        _log(W, fm)


def _slab_dist(q, poly) -> float:
    """Least distance from points q [n,3] (piece frame) to the piece as a solid: its outline swept through its thickness."""
    P = np.asarray(poly)
    c = P.mean(0)
    xy = q[:, :2]
    out = np.zeros(len(q))
    seg = np.full(len(q), np.inf)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        d = b - a
        nrm = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-12)
        if nrm @ (c - a) > 0:
            nrm = -nrm
        out = np.maximum(out, (xy - a) @ nrm)                       # > 0: outside this edge
        t = np.clip(((xy - a) @ d) / max(d @ d, 1e-18), 0, 1)
        seg = np.minimum(seg, np.linalg.norm(xy - (a + t[:, None] * d), axis=1))
    horiz = np.where(out > 0, seg, 0.0)
    vert = np.maximum(np.maximum(-q[:, 2], q[:, 2] - C.WAFER_T), 0.0)
    return float(np.hypot(horiz, vert).min())


def _tier(W) -> float:
    """The end-state table (task.json, "scoring"): the score the state earns, the latches aside."""
    T = _T
    if T["offmark"]:
        return 0.0
    if T["line"] is None:
        for nk in T["nicks"]:
            if np.linalg.norm(nk["end"] - np.array([-math.sqrt(C.WAFER_R ** 2 - C.TARGET_Y ** 2), C.TARGET_Y])) <= NICK_TICK:
                return 0.25
        return 0.0
    k = T["line"]["k"]
    width = 30.0 - k                        # mm: from the line (world y k mm) to the coupon's farthest rim point (+30 mm)
    flat = True
    for n in T["active"]:
        if n not in W["bodies"]:
            return 0.0
        _, R = W["bodies"][n]
        if math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))) > FLAT_TILT:
            flat = False
    if 29.0 <= width <= 31.0 and not T["line"]["chip"] and flat:
        return 1.0
    if 28.0 <= width <= 32.0:
        return 0.75
    if 26.0 <= width <= 34.0:
        return 0.5
    return 0.0


def _score() -> float:
    T = _T
    if T["failed"] is not None or T["world"] is None:
        return 0.0
    return float(T["reading"]) if T["reading"] is not None else 0.0


def score(objects) -> float:
    """task.json, "scoring": the last reading taken while the gate held for 1 s; 0 once a failure has latched."""
    return _score()


def success(objects) -> bool:
    """The 1.0 state with the gate's conditions now; the harness holds it 1 s (SUCCESS_HOLD_S), the gate's 1 s."""
    steptime.tick()
    T = _T
    if T["failed"] is not None or T["world"] is None or not T["gate_now"]:
        return False
    return _tier(T["world"]) == 1.0


def _log(W, fm) -> None:
    T = _T
    parts = [f"[cleave] step {T['step']} polls {W['polls']}"]
    for n in T["active"]:
        if n in W["bodies"]:
            pos, R = W["bodies"][n]
            parts.append(f"{n} {np.round(1000 * (pos - [WAFER_C[0], WAFER_C[1], STAGE_TOP]), 2).tolist()} "
                         f"tilt {math.degrees(math.acos(max(-1, min(1, R[2, 2])))):.2f} v {1000 * np.linalg.norm(W['vel'][n]):.1f}mm/s")
    r = WAFERS.index(T["active"][0])
    parts.append("f " + " ".join(f"{f}:{fm[WAFERS.index(n), FI[f]]:.2f}" for n in T["active"] for f in TOOLS
                                 if fm[WAFERS.index(n), FI[f]] > 0.001))
    sp, sR = W["links"]["scriber"]
    if "wafer" in W["bodies"]:
        q = _to_wafer(sp, W["bodies"]["wafer"])
        parts.append(f"tip(w) {np.round(1000 * q, 2).tolist()} rimdepth {1000 * rim_depth(q[:2]):.1f}")
    jl, jR = W["links"]["jaw_lower"]
    ju, uR = W["links"]["jaw_upper"]
    ridge = jl + jR @ RIDGE_PT
    pad = ju + uR @ PAD_PT
    parts.append(f"ridge {np.round(1000 * (ridge - [0, 0, TOP]), 2).tolist()} pad {np.round(1000 * (pad - [0, 0, TOP]), 2).tolist()}"
                 f" pivot {math.degrees(W['pivot']):.1f} squeezeL {W['squeeze']['left']:.1f} clamp {clamp_force(W)[0]:.1f}"
                 f" on {int(clamp_force(W)[1])}")
    tcp = W["tcp"]
    parts.append(f"tcpL {np.round(1000 * (tcp['left'] - [0, 0, TOP]), 1).tolist()} tcpR {np.round(1000 * (tcp['right'] - [0, 0, TOP]), 1).tolist()}"
                 f" gap {1000 * T.get('gap', 0):.0f}mm gate {int(T['gate_now'])} reading {T['reading']} nicks {len(T['nicks'])}"
                 f" line {T['line']['k'] if T['line'] else None} fail {T['failed']}")
    del r
    print(" | ".join(parts), flush=True)


# -- scripted demo ----------------------------------------------------------------------------------------------------------
import yam_arena.demo as _demo  # noqa: E402


class Rot(float):
    """A waypoint 'yaw' that carries a full TCP rotation (columns: pinch axis, gy, approach)."""

    def __new__(cls, R):
        g = float.__new__(cls, 0.0)
        g.R = np.asarray(R, float)
        return g


_straight_down = getattr(_demo.grip_rotation, "straight_down", _demo.grip_rotation)


def _grip_rotation(yaw: float = 0.0) -> np.ndarray:
    R = getattr(yaw, "R", None)
    return R if R is not None else _straight_down(float(yaw))


_grip_rotation.straight_down = _straight_down
_demo.grip_rotation = _grip_rotation


def horizontal(pinch) -> Rot:
    """Approach along +x; the fingers close along `pinch` (a unit vector perpendicular to x)."""
    gz = np.array([1.0, 0.0, 0.0])
    gx = np.asarray(pinch, float)
    return Rot(np.c_[gx, np.cross(gz, gx), gz])


PLIERS_GRIP = horizontal((0, 0, 1))      # left hand: fingers above and below the handle ends
PEN_GRIP = horizontal((0, -1, 0))        # right hand: fingers either side of the upright pen
PEN_AHEAD = float(os.environ.get("PEN_AHEAD", 0.006))   # the pen's axis this far ahead of the TCP, between the pads
PEN_HOLD = 0.050                         # TCP this far above the pen's point
PEN_DY = float(os.environ.get("PEN_DY", -0.001))        # runs 3-5: the hand settles ~0.9 mm toward +y over the tick
PRESS = float(os.environ.get("PRESS", 0.010))           # the point commanded this far below the wafer's face: the
                                                         # arm gives about 0.14 N/mm (run 3: 3 mm gave 0.3 N)
JAW_DZ = float(os.environ.get("JAW_DZ", 0.0093))        # measured in run 2: held level, the jaws ride 9.1 mm below
JAW_DY = float(os.environ.get("JAW_DY", -0.0003))       # the commanded TCP height (arm sag and the pliers' pitch in
                                                         # the grip), and the hand 1.8 mm toward +y
ENGAGE = 0.006                           # the jaws' tips this far past the rim at the tick


def _wp(t, p, finger, label, yaw) -> Waypoint:
    return Waypoint(t, (float(p[0]), float(p[1]), float(p[2])), finger, label, yaw)


class Plan:
    def __init__(self, side, yaw):
        self.side, self.w, self.t, self.yaw = side, [], 0.0, yaw
        self.p, self.finger = None, FINGER_OPEN

    def go(self, t, p=None, finger=None, label="", yaw=None):
        self.p = self.p if p is None else np.asarray(p, float)
        self.finger = self.finger if finger is None else finger
        self.yaw = self.yaw if yaw is None else yaw
        self.w.append(_wp(t, self.p, self.finger, label, self.yaw))
        self.t += t
        return self

    def until(self, t, label="wait"):
        if t > self.t + 1e-9:
            self.go(t - self.t, label=label)
        return self


def sep(mm: float) -> float:
    """Finger target for pads `mm` apart."""
    return -mm / 2000.0


def demo():
    """Open loop. Right hand: scriber out of its stand, nick at the tick, back in the stand. Left hand: pliers from their
    stand, jaws onto the rim at the nick, squeeze, open, draw the pliers back 4.5 cm clear of the pieces and hold still."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {}
    R = Plan("right", PEN_GRIP)
    L = Plan("left", PLIERS_GRIP)
    # -- right: the scriber
    pen_grasp = PEN_START + [-PEN_AHEAD, 0.0, PEN_HOLD]
    R.go(2.0, pen_grasp + [-0.035, 0.0, 0.03], sep(94), "to the stand, hand level")
    R.go(1.5, pen_grasp + [-0.035, 0.0, 0.0], label="level with the pen")
    R.go(1.5, pen_grasp, label="fingers round the pen")
    R.go(1.0, finger=sep(0), label="close on the pen")
    lift = pen_grasp + [0.0, 0.0, 0.040]
    R.go(1.5, lift, label="pen out of the stand")
    face = STAGE_TOP + C.WAFER_T
    start = np.array([TICK[0] + 0.003, TICK[1] + PEN_DY])      # 3 mm in from the rim at the tick
    end = np.array([TICK[0] - 0.001, TICK[1] + PEN_DY])        # 1 mm past the rim
    above = np.array([start[0] - PEN_AHEAD, start[1], face + PEN_HOLD + 0.012])
    R.go(3.0, above, label="over the tick")
    R.go(2.0, [above[0], above[1], face + PEN_HOLD + 0.002], label="near the face")
    R.go(1.5, [above[0], above[1], face + PEN_HOLD - PRESS], label="set the point down")
    R.go(0.5, label="settle")
    R.go(2.0, [end[0] - PEN_AHEAD, end[1], face + PEN_HOLD - PRESS], label="draw off the edge")
    R.go(1.0, [end[0] - PEN_AHEAD - 0.002, end[1], face + PEN_HOLD + 0.012], label="lift")
    T_LEFT = R.t
    R.go(3.0, lift, label="back over the stand")
    R.go(2.0, pen_grasp + [0.0, 0.0, 0.002], label="pen into the stand")
    R.go(0.8, finger=sep(30), label="release")
    R.go(1.5, pen_grasp + [-0.035, 0.0, 0.0], label="back out")
    R.go(2.0, pen_grasp + [-0.02, 0.05, 0.06], sep(60), "clear")
    # -- left: the pliers, once the right hand has left the tick
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.station import BASES

    p0, r0 = ik.tcp_pose(np.array(READY, float), np.array(BASES["left"]))
    L.go(T_LEFT, p0, sep(94), "wait at ready", yaw=Rot(r0))
    grip = np.array([PIVOT_START[0] + C.GRIP_X, PLIERS_Y, PIVOT_START[2]])
    L.go(2.0, grip + [-0.08, 0.0, 0.03], sep(94), "behind the handles, hand level", yaw=PLIERS_GRIP)
    L.go(1.5, grip + [-0.06, 0.0, 0.0], label="level with the handles")
    L.go(1.5, grip, label="fingers round the handle ends")
    L.go(1.2, finger=sep(57), label="hold at rest")
    L.go(1.0, finger=sep(24), label="squeeze shut to carry")
    L.go(1.5, grip + [0.0, 0.0, 0.02], label="lift off the stand")
    tip_from_tcp = C.JAW_X[1] - C.GRIP_X
    wz = STAGE_TOP + C.WAFER_T / 2 + JAW_DZ
    back = np.array([TICK[0] - 0.025 - tip_from_tcp, JAW_DY, wz])
    L.go(2.5, [back[0], PLIERS_Y, grip[2] + 0.02], label="back out")
    L.go(2.0, [back[0], JAW_DY, wz + 0.02], label="to the tick's line")
    L.go(1.5, back, label="down to the wafer's plane")
    L.go(1.2, finger=sep(57), label="let the jaws open")
    L.go(0.8, label="settle")
    engaged = np.array([TICK[0] + ENGAGE - tip_from_tcp, JAW_DY, wz])
    L.go(3.0, engaged, label="jaws onto the rim")
    L.go(0.5, label="settle")
    L.go(4.0, finger=sep(33), label="squeeze")
    L.go(1.0, label="hold")
    L.go(0.3, finger=sep(57), label="open the jaws")
    L.go(0.9, engaged + [-0.045, 0.0, 0.0], label="draw the pliers back clear of the pieces")
    end_t = max(L.t, R.t) + 2.5
    L.until(end_t, "still")
    R.until(end_t, "still")
    return {"left": L.w, "right": R.w}
