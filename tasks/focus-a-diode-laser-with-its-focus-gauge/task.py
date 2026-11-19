"""On the station table: an open-frame 20 W diode laser engraver (red aluminium frame, gold gantry beam), unplugged,
with its honeycomb working panel inside the frame and a 3 mm basswood plywood sheet lying loose on the panel. The laser
module hangs from its carriage near the front of the frame, 12 mm too high: its focal length setting bar is stowed up
on its magnet (+y side), its thumb-screw handle tight (-y side).

The robot flips the bar down, holds the module while the other hand loosens the handle, lowers the module until the
bar rests on the sheet, tightens the handle, flips the bar back up onto its magnet and lets go.

Simplified physics: rigid bodies. The thumb screw's clamp, the bar's magnet and the handle's thread friction are rules
in this file (task.json, "fidelity").
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

import focus_usd as F  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, before_reset, on_reset, on_step  # noqa: E402
from yam_arena.analysis import box_signed_distance, link_hulls  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
HZ = 20.0

# -- layout (station frame, m) -------------------------------------------------------------------------------------------
MOD_XY = F.MODULE_XY                                   # the module's axis (0.30, 0.05)
HONEY_C = np.array([(F.HONEY[0] + F.HONEY[1]) / 2, (F.HONEY[2] + F.HONEY[3]) / 2])
SHEET_C = np.array([0.35, 0.05])                       # sheet x 0.25..0.45, y -0.10..+0.20
SHEET_TOP0 = F.HONEY[4] + F.SHEET[2]                   # 2.5 cm above the table
G = F.bar_reach()                                      # the hanging bar's lowest point below the shield edge (8.1 mm)
E0 = 0.012                                             # start: the hanging bar's tip would be 12 mm above the sheet
START_Z = SHEET_TOP0 + E0 + G                          # shield edge above the table at the start (4.51 cm)

USD = F.build()

# -- the rules' constants (task.json, "fidelity") -----------------------------------------------------------------------
K_LOCK, D_LOCK = 3.0e5, 1000.0        # the clamp: a stiff drive holds the slide where it was when the handle locked
D_FREE = 5.0                          # N s/m: the free slide's damping (estimate)
LOCK_DEG = 10.0                       # the handle within this of its tight stop: locked
H_FREE, H_TIGHT = 0.05, 0.05          # N m to turn the handle (estimate); the same over its whole travel
K_HANDLE, D_HANDLE = 5.0, 0.002       # the handle's friction hold (target follows it; torque capped as above)
MAG_DEG, MAG_TORQUE = 5.0, 0.02       # the magnet holds the bar within 5 deg of stowed with up to 0.02 N m (estimate)
K_MAG, D_BAR = 1.0, 1.0e-4

OBJECTS = [
    UsdBody("engraver", usd=USD["machine"], pos=(MOD_XY[0], MOD_XY[1], TOP), size=(0.59, 0.60, 0.11), mass=None,
            friction=(0.5, 0.45)),
    UsdBody("honeycomb", usd=USD["honeycomb"], pos=(HONEY_C[0], HONEY_C[1], TOP), size=(0.5, 0.5, F.HONEY[4]),
            mass=None, friction=(0.5, 0.45)),
    UsdBody("sheet", usd=USD["sheet"], pos=(SHEET_C[0], SHEET_C[1], TOP + F.HONEY[4]), size=F.SHEET,
            mass=F.SHEET_MASS, friction=(0.5, 0.45), solver_iters=32),
    UsdBody("laser", usd=USD["laser"], pos=(MOD_XY[0], MOD_XY[1], TOP + START_Z), size=(0.06, 0.06, 0.128),
            articulation=True, fix_root=True, joint_pos=(("slide", 0.0), ("bar", 0.0), ("handle", 0.0)),
            drives=(("slide", K_LOCK, D_LOCK, 2000.0), ("bar", K_MAG, D_BAR, MAG_TORQUE),
                    ("handle", K_HANDLE, D_HANDLE, H_TIGHT)),
            friction=(0.8, 0.7), solver_iters=32, torsional_patch=0.006),
]

# -- scoring terms (task.json, "scoring") ---------------------------------------------------------------------------------
IN_FOCUS = 0.0005
NEAR_FOCUS = 0.002
STOW_TOL = 10.0                       # degrees from the stowed angle
DROP = 0.002                          # m of descent while no gripper touches the module's body
DROP_SPEED = 0.03                     # m/s: the module descending faster than this, touched or not, has dropped
                                      # (read as more than 1.5 mm of descent in one 50 ms control step)
F_TOUCH = 0.05                        # N: contact
SHEET_MOVED = 0.003                   # m, any corner
CLEAR = 0.03                          # fingertips this far from the module, its bar and handle, and the sheet
STILL_STEP = 0.01 / HZ                # 1 cm/s, as displacement over one 50 ms control step
HOLD = round(1.0 * HZ)                # the harness's 1 s (SUCCESS_HOLD_S)
GRIP_LINKS = ("gripper", "tip_left", "tip_right")
FINGERS = [f"{s}:{n}" for s in ("L", "R") for n in GRIP_LINKS]


def _grid_points(h, step=0.006) -> np.ndarray:
    lo, hi = h.vertices.min(0), h.vertices.max(0)
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + step, step) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    inside = (g @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1) <= 0
    return np.vstack([h.vertices, g[inside]])


GRIP_POINTS = {link: np.vstack([_grid_points(h) for h in link_hulls()[link]]) for link in GRIP_LINKS}


def _rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _ry(deg) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


# Boxes for the fingertip clearance, each in its link's authored frame (centre, rotation, size).
_T_BAR = F.BAR_TOP + F.BAR_L
BOXES = {
    "module": [(np.array([0, 0, F.BODY_H / 2]), np.eye(3), np.array([F.BODY, F.BODY, F.BODY_H])),
               (np.array([F.BAR_PIVOT[0], (F.BODY / 2 + F.BAR_PIVOT[1]) / 2, F.BAR_PIVOT[2]]), np.eye(3),
                np.array([0.012, F.BAR_PIVOT[1] - F.BODY / 2, 0.012])),
               (np.array([F.HANDLE_AXIS[0], (F.HANDLE_AXIS[1] - F.BODY / 2) / 2, F.HANDLE_AXIS[2]]), np.eye(3),
                np.array([0.011, -F.HANDLE_AXIS[1] - F.BODY / 2, 0.011]))],
    "bar": [(F.BAR_PIVOT + (_T_BAR / 2 - F.BAR_TOP) * F.bar_dir(F.STOW_DEG), _ry(F.STOW_DEG),
             np.array([F.BAR_W, F.BAR_COL_T, _T_BAR]))],
    "handle": [(F.HANDLE_AXIS + np.array([-(F.HANDLE_L + 0.006) / 2 + 0.006, 0, 0]), np.eye(3),
                np.array([F.HANDLE_L + 0.006, F.HANDLE_W + 0.003, F.HANDLE_T + 0.002]))],
}
SHEET_CORNERS = np.array([[sx * F.SHEET[0] / 2, sy * F.SHEET[1] / 2, F.SHEET[2]] for sx in (-1, 1) for sy in (-1, 1)])
TIP_XY = MOD_XY + (F.BAR_PIVOT + F.BAR_L * F.bar_dir(F.HANG_DEG))[:2]     # the hanging bar's tip (station xy)


# -- per-physics-step monitor: contact of the fingers with the module's body, and the drop rule ---------------------------
_MON: dict = {"view": None, "err": None, "asset": None, "slide": None, "touch_steps": 0, "polls": 0, "peak": 0.0}


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
    sens, filt, bar_p, sheet_p = None, {}, None, None
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parts = p.split("/")
        name = parts[-1]
        if name == "module" and parts[-2] == "laser":
            sens = p
        elif name == "bar" and parts[-2] == "laser":
            bar_p = p
        elif name == "sheet" and parts[-2] == "env_0":
            sheet_p = p
        elif name in GRIP_LINKS and ("/LeftArm/" in p or "/RightArm/" in p):
            filt[("L:" if "/LeftArm/" in p else "R:") + name] = p
    if sens is None or len(filt) != len(FINGERS):
        raise RuntimeError(f"focus monitor: module {sens}, fingers {sorted(filt)}")
    view = sv.create_rigid_contact_view([sens], filter_patterns=[[filt[n] for n in FINGERS]],
                                        max_contact_data_count=16 * len(FINGERS))
    _MON.update(view=view)
    if bar_p is not None and sheet_p is not None:
        _MON["bar_view"] = sv.create_rigid_contact_view([bar_p], filter_patterns=[[filt[n] for n in FINGERS] + [sheet_p]],
                                                         max_contact_data_count=16 * (len(FINGERS) + 1))
    print(f"[focus] contact monitor: module body x {len(FINGERS)} finger links", flush=True)


def _poll(mgr) -> None:
    """After every physics step (240 Hz): is any gripper touching the module's body, and how far has the module
    descended since it was last touched?"""
    try:
        if _MON["view"] is None:
            _make_view(mgr)
        if _MON["view"] is None:
            return
        f = _np(_MON["view"].get_contact_force_matrix(dt=mgr.get_physics_dt())).reshape(len(FINGERS), 3)
    except Exception as e:  # pragma: no cover
        if _MON["err"] is None:
            _MON["err"] = repr(e)
            print(f"[focus] monitor error: {e!r}", flush=True)
        return
    if _MON.get("bar_view") is not None:
        try:
            fb = _np(_MON["bar_view"].get_contact_force_matrix(dt=mgr.get_physics_dt())).reshape(len(FINGERS) + 1, 3)
            _MON["bar_sum"] = _MON.get("bar_sum", 0.0) + fb
        except Exception:  # pragma: no cover
            pass
    peak = float(np.linalg.norm(f, axis=1).max())
    touch = peak > F_TOUCH
    _MON["polls"] += 1
    _MON["peak"] = max(_MON["peak"], peak)
    _MON["touch_steps"] += int(touch)
    asset = _MON["asset"]
    if asset is None or not _T:
        return
    q = float(_np(asset.data.joint_pos)[0][_MON["slide"]])
    T = _T
    if touch or T["z_hi"] is None:
        T["z_hi"] = q
    else:
        T["z_hi"] = max(T["z_hi"], q)
        T["drop_max"] = max(T["drop_max"], T["z_hi"] - q)
        if T["z_hi"] - q >= DROP:
            _fail(f"dropped: the module descended {1000 * (T['z_hi'] - q):.1f} mm with no gripper on its body")


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_focus_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _poll(cls)

    step._focus_hooked = True
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


# -- the rules: thumb-screw clamp, handle friction, bar magnet -----------------------------------------------------------
_IDS: dict = {}


def _joint_ids(asset) -> dict:
    if not _IDS:
        names = list(asset.joint_names)
        _IDS.update({n: names.index(n) for n in ("slide", "bar", "handle")})
        bnames = list(asset.body_names)
        _IDS.update({f"body:{n}": bnames.index(n) for n in ("module", "bar", "handle")})
    return _IDS


def _write(asset, what, joint, value) -> None:
    ids = [_IDS[joint]]
    if what == "stiffness":
        asset.write_joint_stiffness_to_sim_index(stiffness=float(value), joint_ids=ids)
    elif what == "damping":
        asset.write_joint_damping_to_sim_index(damping=float(value), joint_ids=ids)
    elif what == "effort":
        asset.write_joint_effort_limit_to_sim_index(limits=float(value), joint_ids=ids)


def _targets(asset, values: dict) -> None:
    import torch

    names = list(values)
    ids = [_IDS[n] for n in names]
    v = torch.tensor([[float(values[n]) for n in names]], dtype=torch.float32, device=asset.device)
    asset.actuators.target_command.set_position_index(value=v, joint_ids=ids)


def _apply_rules(asset, q) -> None:
    """Every control step: the clamp (locked within 10 deg of the handle's stop), the handle's friction, the magnet."""
    T = _T
    psi = math.degrees(q["handle"])
    locked = psi >= -LOCK_DEG
    if locked and not T["locked"]:
        T["q_lock"] = q["slide"]
        _write(asset, "stiffness", "slide", K_LOCK)
        _write(asset, "damping", "slide", D_LOCK)
        print(f"[focus] step {T['step']}: locked at slide {1000 * q['slide']:+.2f} mm (handle {psi:.1f} deg)", flush=True)
    elif not locked and T["locked"]:
        _write(asset, "stiffness", "slide", 0.0)
        _write(asset, "damping", "slide", D_FREE)
        print(f"[focus] step {T['step']}: unlocked at slide {1000 * q['slide']:+.2f} mm (handle {psi:.1f} deg)", flush=True)
    T["locked"] = locked
    tight = psi > -LOCK_DEG
    if tight != T["tight_zone"]:
        _write(asset, "effort", "handle", H_TIGHT if tight else H_FREE)
        T["tight_zone"] = tight
    mag = abs(math.degrees(q["bar"])) <= MAG_DEG
    if mag != T["magnet"]:
        _write(asset, "stiffness", "bar", K_MAG if mag else 0.0)
        T["magnet"] = mag
    # the handle's thread friction: the drive's target stays put (the handle sticks) until a push saturates the drive,
    # then trails the handle by the torque cap (it slips)
    cap = H_TIGHT if tight else H_FREE
    err = T["h_target"] - q["handle"]
    if abs(K_HANDLE * err) > cap:
        T["h_target"] = q["handle"] + math.copysign(cap / K_HANDLE, err)
    # the clamp carries the module's weight (like the real thumb screw's friction): its target sits mg/K above the
    # position where it locked, so the bar is not left holding the module up
    hold = F.MODULE_MASS * 9.81 / K_LOCK
    _targets(asset, {"slide": T["q_lock"] + hold, "bar": 0.0, "handle": T["h_target"]})


# -- the trial-long tracker ----------------------------------------------------------------------------------------------
_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Only a reset clears the tracker (and with it any latched failure)."""
    _T.clear()
    _T.update(step=0, failed=None, z_hi=None, drop_max=0.0, locked=True, q_lock=0.0, tight_zone=True, magnet=True,
              world=None, prev=None, hold=0, state=None, sheet0=None, sheet_max=0.0, e_min=9.0, h_target=0.0, v_max=0.0)
    if _env is not None:
        try:
            asset = _env.unwrapped.scene["laser"]
            _joint_ids(asset)
            _write(asset, "stiffness", "slide", K_LOCK)
            _write(asset, "damping", "slide", D_LOCK)
            _write(asset, "effort", "handle", H_TIGHT)
            _write(asset, "stiffness", "bar", K_MAG)
            _targets(asset, {"slide": 0.0, "bar": 0.0, "handle": 0.0})
        except Exception as e:  # pragma: no cover
            print(f"[focus] reset rules: {e!r}", flush=True)


_reset()


@before_reset
def _before(environment) -> None:
    try:
        asset = environment.unwrapped.scene["laser"]
        _joint_ids(asset)
        _write(asset, "stiffness", "slide", K_LOCK)
        _write(asset, "damping", "slide", D_LOCK)
        _targets(asset, {"slide": 0.0, "bar": 0.0, "handle": 0.0})
    except Exception as e:  # pragma: no cover
        print(f"[focus] before reset: {e!r}", flush=True)


def _fail(why) -> None:
    if _T.get("failed") is None and _T:
        _T["failed"] = why
        print(f"[focus] latched 0 at step {_T['step']}: {why}", flush=True)


@on_step
def _read(environment) -> None:
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    asset = base.scene["laser"]
    ids = _joint_ids(asset)
    if _MON["asset"] is None:
        _MON["asset"], _MON["slide"] = asset, ids["slide"]
    qa = _np(asset.data.joint_pos)[0]
    q = {n: float(qa[ids[n]]) for n in ("slide", "bar", "handle")}
    poses = _np(asset.data.body_link_pose_w)[0]
    links = {n: (poses[ids[f"body:{n}"]][:3] - origin, _rot(poses[ids[f"body:{n}"]][3:7])) for n in ("module", "bar", "handle")}
    sh = base.scene["sheet"]
    sp = _np(sh.data.root_link_pose_w)[0]
    sheet = (sp[:3] - origin, _rot(sp[3:7]))
    W = {"q": q, "links": links, "sheet": sheet, "fingers": {}, "tcp": {}}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        rp = _np(robot.data.body_link_pose_w)[0]
        pts = []
        for link in GRIP_LINKS:
            i = robot.body_names.index(link)
            R = _rot(rp[i][3:7])
            pts.append(GRIP_POINTS[link] @ R.T + (rp[i][:3] - origin))
            if link == "gripper":
                W["tcp"][side] = rp[i][:3] - origin + R @ np.array([0.0, 0.0, 0.134])
        W["fingers"][side] = np.vstack(pts)
    W["touch_steps"], W["polls"], W["peak"] = _MON["touch_steps"], _MON["polls"], _MON["peak"]
    bs = _MON.get("bar_sum", 0.0)
    W["bar_f"] = (np.asarray(bs) / max(_MON["polls"], 1)) if not np.isscalar(bs) else None
    _MON.update(touch_steps=0, polls=0, peak=0.0, bar_sum=0.0)
    _T["step"] += 1
    _apply_rules(asset, q)
    _track(W)


def focus_error(W) -> float:
    """e: the hanging bar's lowest point (computed with the bar on its stop) above the sheet's top surface there."""
    mpos, _ = W["links"]["module"]
    z_tip = mpos[2] - G
    p, R = W["sheet"]
    top = p + R @ np.array([0, 0, F.SHEET[2]])
    n = R[:, 2]
    z_sheet = top[2] - (n[0] * (TIP_XY[0] - top[0]) + n[1] * (TIP_XY[1] - top[1])) / n[2]
    return float(z_tip - z_sheet)


def _sheet_corners(W) -> np.ndarray:
    p, R = W["sheet"]
    return SHEET_CORNERS @ R.T + p


def finger_gap(W) -> float:
    """Least distance from any gripper's finger links to the module (body, bracket, boss), bar, handle or sheet."""
    best = 9.0
    pts = np.vstack(list(W["fingers"].values()))
    for link, boxes in BOXES.items():
        lp, lR = W["links"][link]
        for c, r, s in boxes:
            d = box_signed_distance(pts, lp + lR @ c, lR @ r, s)
            best = min(best, float(d.min()))
    p, R = W["sheet"]
    d = box_signed_distance(pts, p + R @ np.array([0, 0, F.SHEET[2] / 2]), R, np.array(F.SHEET))
    return min(best, float(d.min()))


def _track(W) -> None:
    T = _T
    q = W["q"]
    corners = _sheet_corners(W)
    if T["sheet0"] is None:
        T["sheet0"] = corners
    moved = float(np.linalg.norm(corners - T["sheet0"], axis=1).max())
    T["sheet_max"] = max(T["sheet_max"], moved)
    if moved > SHEET_MOVED:
        _fail(f"sheet moved: a corner is {1000 * moved:.1f} mm from where it lay")
    e = focus_error(W)
    T["e_min"] = min(T["e_min"], e)
    gap = finger_gap(W)
    still = True
    if T["prev"] is not None:
        pq, pc = T["prev"]
        still = (abs(q["slide"] - pq["slide"]) < STILL_STEP and abs(q["bar"] - pq["bar"]) * F.BAR_L < STILL_STEP
                 and abs(q["handle"] - pq["handle"]) * F.HANDLE_L < STILL_STEP
                 and float(np.linalg.norm(corners - pc, axis=1).max()) < STILL_STEP)
    else:
        still = False
    if T["prev"] is not None:
        dz = T["prev"][0]["slide"] - q["slide"]                     # descent over this 50 ms step
        T["v_max"] = max(T["v_max"], dz * HZ)
        if dz * HZ > DROP_SPEED:
            _fail(f"dropped: the module descended {1000 * dz:.2f} mm in one 50 ms step ({100 * dz * HZ:.1f} cm/s)")
    T["prev"] = (dict(q), corners)
    state = dict(e=e, in_focus=abs(e) <= IN_FOCUS, near=abs(e) <= NEAR_FOCUS,
                 locked=math.degrees(q["handle"]) >= -LOCK_DEG, stowed=abs(math.degrees(q["bar"])) <= STOW_TOL,
                 released=gap >= CLEAR, still=still, gap=gap)
    state["full"] = state["in_focus"] and state["locked"] and state["stowed"] and state["released"] and still
    T["hold"] = T["hold"] + 1 if (state["full"] and T["failed"] is None) else 0
    T["state"] = state
    T["world"] = W
    if (T["step"] % 10 == 0 or T["step"] < 3 or (W.get("bar_f") is not None and np.abs(W["bar_f"][:6]).sum() > 0.01)
            or (230 <= T["step"] <= 520 and T["step"] % 3 == 0)):
        _log(W, state)


def _tier() -> float:
    T = _T
    s = T["state"]
    if T["failed"] is not None or s is None:
        return 0.0
    if T["hold"] >= HOLD:
        return 1.0
    rs = s["released"] and s["still"]
    if s["in_focus"] and s["locked"] and rs:
        return 0.75
    if s["near"] and not s["in_focus"] and s["locked"] and rs:
        return 0.5
    if s["in_focus"] and not s["locked"] and rs:
        return 0.25
    return 0.0


def score(objects) -> float:
    """task.json, "scoring": 0 once a failure has latched; otherwise the end state's tier (1.0 after 1 s of it)."""
    return _tier()


def success(objects) -> bool:
    """The 1.0 state now; the harness holds it 1 s (SUCCESS_HOLD_S), the same 20 steps score() counts."""
    steptime.tick()
    T = _T
    return bool(T and T["failed"] is None and T["state"] is not None and T["state"]["full"])


def _log(W, s) -> None:
    T = _T
    q = W["q"]
    tl, tr = W["tcp"]["left"], W["tcp"]["right"]
    print(f"[focus] step {T['step']} slide {1000 * q['slide']:+.2f}mm e {1000 * s['e']:+.2f}mm handle "
          f"{math.degrees(q['handle']):+.1f} bar {math.degrees(q['bar']):+.1f} locked {int(s['locked'])} "
          f"touch {W['touch_steps']}/{W['polls']} peak {W['peak']:.1f}N drop {1000 * T['drop_max']:.2f}mm vmax {1000 * T['v_max']:.1f}mm/s "
          f"sheet {1000 * T['sheet_max']:.2f}mm gap {1000 * s['gap']:.0f}mm still {int(s['still'])} hold {T['hold']} "
          f"tcpL {np.round(1000 * (tl - [0, 0, TOP]), 1).tolist()} tcpR {np.round(1000 * (tr - [0, 0, TOP]), 1).tolist()} "
          f"tier {_tier()} fail {T['failed']}"
          + (f" barF fingers {np.round(W['bar_f'][:6].sum(0), 2).tolist()} sheet {np.round(W['bar_f'][6], 2).tolist()}"
             if W.get("bar_f") is not None else ""), flush=True)


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


def tilted(tilt_deg: float, toward_y: float) -> Rot:
    """Fingers closing along x; the approach tilted tilt_deg from straight down, toward +y (toward_y=1) or -y."""
    t = math.radians(tilt_deg)
    gz = np.array([0.0, toward_y * math.sin(t), -math.cos(t)])
    gx = np.array([1.0, 0.0, 0.0])
    return Rot(np.c_[gx, np.cross(gz, gx), gz])


PAD = 0.013                                         # fingertips past the TCP along the approach
LEFT_TILT = tilted(float(os.environ.get("LEFT_TILT", 45.0)), -1.0)     # left hand from +y, pointing toward the module
RIGHT_TILT = tilted(float(os.environ.get("RIGHT_TILT", 40.0)), +1.0)   # right hand from -y
_fd = math.radians(float(os.environ.get("FORK_DOWN", 8.0)))
FORK = Rot(np.c_[[0.0, math.sin(_fd), math.cos(_fd)], np.cross([0.0, math.cos(_fd), -math.sin(_fd)], [0.0, math.sin(_fd), math.cos(_fd)]),
                 [0.0, math.cos(_fd), -math.sin(_fd)]])   # right hand pointing +y, 8 deg down; fingers open above and below
LIFT_PRE = float(os.environ.get("LIFT_PRE", 0.019))   # the holding hand pushes up this much before the handle frees
OVER = float(os.environ.get("OVER", 0.002))           # lowered this far past the bar's contact
LOWER_S = float(os.environ.get("LOWER_S", 6.0))       # seconds for the lowering (a slow, steady descent)
GRIP_DZ = float(os.environ.get("GRIP_DZ", 0.006))     # TCP this far below the module's top
R_SWEEP = float(os.environ.get("R_SWEEP", 0.036))     # the fingertip's circle round the bar's axis
SWEEP_DY = float(os.environ.get("SWEEP_DY", -0.004))  # the fingertip's centre this far toward the module from the bar's plane


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

    def tip(self, t, tip, finger=None, label="", yaw=None):
        """Go with the fingertips' centre at `tip` (TCP = tip - PAD along the approach)."""
        rot = (yaw if yaw is not None else self.yaw)
        R = getattr(rot, "R", None)
        a = R[:, 2] if R is not None else np.array([0.0, 0.0, -1.0])
        return self.go(t, np.asarray(tip, float) - PAD * a, finger, label, yaw)

    def until(self, t, label="wait"):
        if t > self.t + 1e-9:
            self.go(t - self.t, label=label)
        return self


def sep(mm: float) -> float:
    """Finger target for pads `mm` apart."""
    return -mm / 2000.0


def _w(p) -> np.ndarray:
    """Station point from (x, y, height above the table)."""
    return np.array([p[0], p[1], TOP + p[2]], float)


def demo():
    """Open loop. Left: flip the bar down, grip the module's fan housing, carry its weight, lower it onto the bar;
    after the right hand has locked the handle, let go and sweep the bar back up onto its magnet. Right: press the
    handle down 45 deg to loosen it, then lift it back to its stop."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {}
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.station import BASES

    L = Plan("left", LEFT_TILT)
    R = Plan("right", RIGHT_TILT)
    bx, by = MOD_XY + F.BAR_PIVOT[:2]
    piv0 = np.array([bx, by, START_Z + F.BAR_PIVOT[2]])            # the bar's axis at the start (height above table)
    piv1 = piv0 - [0, 0, E0]                                         # ... once focused

    def circ(piv, deg, r=R_SWEEP):
        return piv + r * F.bar_dir(deg)

    pL, rL = ik.tcp_pose(np.array(READY, float), np.array(BASES["left"]))
    pR, rR = ik.tcp_pose(np.array(READY, float), np.array(BASES["right"]))
    # -- left: flip the bar down (sweep the stowed bar's upper part forward)
    s0 = circ(piv0, 188.0)
    L.tip(2.5, _w(s0 + [0, 0.015, 0.035]), sep(0), "over the stowed bar")
    L.tip(1.0, _w(s0), label="behind the bar's top")
    for a in (178.0, 166.0, 154.0, 145.0):
        L.tip(0.3, _w(circ(piv0, a)), label="flip the bar down")
    away = circ(piv0, 145.0) + [0, 0.025, 0.035]
    L.tip(0.8, _w(away), label="away from the falling bar")
    L.tip(1.0, _w(away + [0, 0.03, 0.06]), label="up")
    # -- left: grip the fan housing, straight down, fingers across y
    top = START_Z + F.BODY_H
    grip = np.array([MOD_XY[0], MOD_XY[1], top - GRIP_DZ])
    HOVER = top + 0.017                                          # open fingertips 4 mm over the module's top
    L.go(1.5, _w([MOD_XY[0], MOD_XY[1] + 0.07, HOVER]), sep(92), "beside the module, open", yaw=0.0)
    L.go(1.0, _w([MOD_XY[0], MOD_XY[1], HOVER]), label="over the module")
    L.go(1.2, _w(grip), label="fingers round the fan housing")
    L.go(1.0, finger=sep(30), label="grip")
    L.go(0.8, _w(grip + [0, 0, LIFT_PRE]), label="take the module's weight")
    T_GRIPPED = L.t
    # -- right: a fork round the handle's lever. The hand points along +y (8 deg down), its fingers open 44 mm, one
    # above the lever and one below, 20 mm from its axis: the upper finger presses it down, the lower one lifts it.
    hx, hy, hz = MOD_XY[0] + F.HANDLE_AXIS[0], MOD_XY[1] + F.HANDLE_AXIS[1], START_Z + F.HANDLE_AXIS[2]
    FORK_GAP = float(os.environ.get("FORK_GAP", 0.044))
    half = FORK_GAP / 2                                              # fork centre to each finger's inner face
    RF = float(os.environ.get("RF", 0.020))                         # where the fork holds the lever, from its axis
    FY = float(os.environ.get("FY", 0.012))                         # fingertips' end this far past the lever's centre (+y)
    PRESS = float(os.environ.get("PRESS", 0.002))                   # pressed this far past the lever (arm compliance)
    ht = (F.HANDLE_T + 0.002) / 2                                    # the lever collider's half thickness

    def lever(deg, r=None, lowered=False):
        a = math.radians(deg)
        r = RF if r is None else r
        return np.array([hx - r * math.cos(a), hy + FY, hz - (E0 if lowered else 0.0) - r * math.sin(a)])

    out = np.array([0.0, -0.045, 0.0])
    XP = float(os.environ.get("XP", 0.036))     # press: the fork's centre this far in front of the axis; its fingers'
    XL = float(os.environ.get("XL", 0.032))
    P_DEG = float(os.environ.get("P_DEG", 44.0))  # the press aims the lever this far down     # inner edges then reach about 22 (press) and 18 mm (lift) from the axis

    def at(x_off, deg_z, dz, lowered=False):
        """Fork tip at x = axis - x_off, at the lever's height (angle deg_z, 20 mm out) plus dz."""
        z = lever(deg_z, lowered=lowered)[2] + dz
        return np.array([hx - x_off, hy + FY, z])

    R.go(T_GRIPPED - 4.0, pR, FINGER_OPEN, "wait at ready", yaw=Rot(rR))
    R.tip(3.0, _w(at(XP, 0, 0) + out), sep(FORK_GAP * 1000), "beside the handle, fork open", yaw=FORK)
    R.tip(1.0, _w(at(XP, 0, 0)), label="fork round the lever")
    R.go(0.3, label="settle")
    R.tip(1.5, _w(at(XP, P_DEG, -(half + ht - 0.003 + PRESS))), label="press the handle down")
    R.go(0.3, label="hold it down")
    R.tip(0.5, _w(at(XP, P_DEG, 0)), label="fork centred on the lever")
    R.tip(0.8, _w(at(XP, P_DEG, 0) + out), label="fork off the lever")
    T_LOOSE = R.t
    # -- left: lower onto the bar's contact
    L.until(T_LOOSE, "hold the module")
    if mode == "slip":                                           # a careless hand: the grip eases and the module slides
        L.go(0.4, finger=sep(64), label="grip loosens")
    L.go(LOWER_S, _w(grip - [0, 0, E0 + OVER]), label="lower onto the bar")
    L.go(0.6, label="settle on the bar")
    T_DOWN = L.t
    # -- right: lift the handle back to its stop with the fork's lower finger
    R.tip(1.0, _w(at(XL, P_DEG, 0, True) + out), label="beside the lever, lowered")
    R.until(T_DOWN, "wait for the module")
    R.tip(1.0, _w(at(XL, P_DEG, 0, True)), label="fork round the lever")
    R.go(0.3, label="settle")
    for a in (30, 10):
        R.tip(0.7, _w(at(XL, a, half + ht - 0.003, True)), label="lift the handle")
    R.tip(0.8, _w(at(XL, -8, half + ht + 0.020, True)), label="push it tight")
    R.go(0.4, label="hold it tight")
    T_LOCKED = R.t
    R.tip(0.5, _w(at(XL, 0, half - ht - 0.008, True)), label="ease off, lower finger below the lever")
    R.tip(0.8, _w(at(XL, 0, half - ht - 0.008, True) + out), label="fork off the lever")
    R.tip(0.8, _w(at(XL, 0, 0, True) + [-0.01, -0.06, 0.03]), label="away from the handle")
    R.go(3.5, pR, label="back to ready", yaw=Rot(rR))
    # -- left: let go, then sweep the bar back up onto its magnet
    L.until(T_LOCKED, "hold the module")
    L.go(0.6, finger=sep(92), label="let go")
    L.go(1.0, _w([MOD_XY[0], MOD_XY[1], HOVER]), label="rise")
    L.go(1.0, _w([MOD_XY[0], MOD_XY[1] + 0.07, HOVER]), label="out to the side")
    s1 = circ(piv1, -28.0)
    L.tip(1.2, _w(np.array([s1[0], 0.125, top + 0.03])), sep(0), "beside the module, high", yaw=LEFT_TILT)
    L.tip(1.2, _w(s1 + [0, 0.02, 0.07]), label="over the hanging bar")
    L.tip(1.0, _w(s1 + [0, 0, 0.03]), label="behind the bar")
    L.tip(0.8, _w(s1 + [0, SWEEP_DY, 0]), label="behind the bar, low")
    for a in np.arange(-12.0, 157.0, 12.0):
        lift = 0.003 if 40.0 <= a <= 140.0 else 0.0            # over the frame's front rail
        L.tip(0.3, _w(circ(piv1, float(a)) + [0, SWEEP_DY, lift]), label="sweep the bar up")
    L.tip(0.3, _w(circ(piv1, 160.0) + [0, SWEEP_DY, 0]), label="onto the magnet")
    L.tip(0.8, _w(circ(piv1, 150.0) + [0, 0.03, 0.01]), label="out of the bar's plane")
    L.tip(1.2, _w(circ(piv1, 150.0) + [-0.02, 0.06, 0.06]), label="clear")
    end_t = max(L.t, R.t) + 2.5
    L.until(end_t, "still")
    R.until(end_t, "still")
    del pL, rL
    return {"left": L.w, "right": R.w}
