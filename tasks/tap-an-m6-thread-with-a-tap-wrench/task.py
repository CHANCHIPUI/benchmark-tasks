"""In front of the right arm: a drill-press vise holds a round aluminium tool adapter plate for a robot wrist (80 mm
across, 6 mm thick) on two steel parallels, tool face up and flush with the jaw tops. One of its four pre-drilled,
countersunk 5 mm tool holes is ringed in blue marker. To the right of the vise, a T-handle tap wrench holding an M6 plug
tap stands upright in a grey PLA stand, its T-bar pointing away from the robot. A red band is painted round the tap's
thread 13.5 mm above its tip.

The robot taps an M6 thread through the marked hole: it starts the tap square, backs it off at least a quarter turn
after every turn in, stops at the red band, backs the tap out and stands the wrench back in its stand.

Simplified physics: rigid bodies. Cutting is a rule (task.json, "scoring"): when the tap bites, drives on the wrench's
joints hold it on the thread's helix (1 mm per turn) and resist its turning with the cutting torques.
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

import steptime  # noqa: E402
import tap_usd as G  # noqa: E402
from scenekit import UsdBody, before_reset, on_reset, on_step, set_joint_targets, set_joint_velocity_targets  # noqa: E402
from yam_arena.analysis import link_hulls  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
HZ = 20.0

# -- layout (station frame, m) -------------------------------------------------------------------------------------------
HOLE = np.array([0.280, -0.180])                 # the marked hole's axis
PLATE_TOP = TOP + G.JAW_TOP                       # 7.3 cm up
STAND_C = np.array([0.180, -0.300])
TIP0 = np.array([STAND_C[0], STAND_C[1], TOP + 0.0025])    # the tap's tip as the wrench stands in its stand
STAND_TOP = TOP + G.STAND[2]
SEAT_POS = np.array([HOLE[0], HOLE[1], PLATE_TOP])
SEAT_PARK = np.array([0.36, 0.0, 0.30])          # under the table: where the seat goes when the tap bites

USD = G.build()
JN = G.JOINT_NAMES

OBJECTS = [
    UsdBody("vise", usd=USD["vise"], pos=(HOLE[0], HOLE[1], TOP), size=(0.26, 0.15, 0.073), mass=None, friction=(0.6, 0.5)),
    UsdBody("plate", usd=USD["plate"], pos=tuple(SEAT_POS), size=(0.08, 0.08, 0.006), mass=None, friction=(0.5, 0.45)),
    UsdBody("seat", usd=USD["seat"], pos=tuple(SEAT_POS), size=(0.008, 0.008, 0.006), mass=None, friction=(0.3, 0.25)),
    UsdBody("stand", usd=USD["stand"], pos=(STAND_C[0], STAND_C[1], TOP), size=G.STAND, mass=None, friction=(0.5, 0.45)),
    UsdBody("tapwrench", usd=USD["tapwrench"], pos=tuple(TIP0), size=(0.1, 0.016, 0.12), articulation=True,
            fix_root=True, joint_pos=tuple((n, 0.0) for n in JN), joint_damping=0.0,
            drives=tuple((n, 0.0, 0.0, 1.0e4) for n in JN), friction=(1.3, 1.2), solver_iters=32, torsional_patch=0.004),
]

# -- the thread rule and the scoring terms (task.json, "scoring") -------------------------------------------------------
PITCH = 0.001                # m per turn
BITE_R = 0.0005              # the tip within this of the hole's axis
BITE_F = 3.0                 # N: the push on the countersink
BITE_TILT = 10.0             # degrees
SEAT_DEPTH = 0.004           # the tip seated in the countersink: between these depths below the plate's top face
SEAT_MIN = 0.0015
SQUARE = 2.0                 # degrees
THROUGH = 0.010              # m of greatest advance
STARTED = 0.006
TAP_OUT = 0.002              # the tip this far above the plate's top face (or this far from the hole: OUT_R)
OUT_R = 0.006
SNAP_BEND = 1.5              # N m at the plate's top face
SNAP_TWIST = 5.0             # N m
THREAD_LEN = 0.024           # forcing past this much advance snaps the tap
T_CLEAN, T_CHIP, T_RERUN, T_REVERSE = 0.8, 0.8, 0.2, 0.3   # N m
BACKOFF = math.pi / 2        # rad
CHIP_MAX = 2 * math.pi       # rad of new thread since the last back-off
IN_STAND_TILT = 15.0         # degrees from vertical
CLEAR = 0.03                 # m: fingertips from the wrench and tap
STILL_V = 0.01               # m/s
HOLD = round(1.0 * HZ)       # the harness's 1 s (SUCCESS_HOLD_S)
# drives while the tap is in its thread
K_LAT, D_LAT = 5.0e4, 100.0          # N/m, N s/m: the tip on the thread's axis
K_AX, D_AX = 5.0e4, 100.0            # along it (the helix)
K_TILT, D_TILT = 200.0, 0.5          # N m/rad: the tap's axis on the thread's axis
D_SPIN = 30.0                        # N m s/rad: the turning resistance's damper, capped at the cutting torque
M_STICK = 0.05                       # N m: during the first turn the chamfer holds the tap at the tilt it is left at,
                                     # against moments up to this (it yields to the hand, but does not fall over)
RAMP = 20                            # control steps: once the axis is fixed, the tilt drives' targets move onto it
GRIP_LINKS = ("gripper", "tip_left", "tip_right")


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)


def _rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _grid_points(h, step=0.006) -> np.ndarray:
    lo, hi = h.vertices.min(0), h.vertices.max(0)
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + step, step) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    inside = (g @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1) <= 0
    return np.vstack([h.vertices, g[inside]])


GRIP_POINTS = {link: np.vstack([_grid_points(h) for h in link_hulls()[link]]) for link in GRIP_LINKS}
# the wrench and tap as capsules in the wrench frame: (end a, end b, radius)
CAPSULES = [((0, 0, 0), (0, 0, G.TAP_OUT), G.SHANK_R), ((0, 0, G.TAP_OUT), (0, 0, G.TAP_OUT + G.BODY_L), G.BODY_R),
            ((-G.BAR_L / 2, 0, G.BAR_Z), (G.BAR_L / 2, 0, G.BAR_Z), G.BAR_R)]


def _seg_dist(p, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = np.clip(((p - a) @ d) / (d @ d), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


def _tilt_deg(u) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u[2])))))


# -- per-physics-step contact monitor: the tap's push on the countersink -------------------------------------------------
_MON: dict = {"view": None, "err": None, "sum": 0.0, "peak": 0.0, "n": 0}


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
    seat = wrench = None
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/env_0/" not in p or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parts = p.split("/")
        if parts[-1] == "seat" and parts[-2] == "env_0":
            seat = p
        elif parts[-1] == "wrench" and parts[-2] == "tapwrench":
            wrench = p
    if seat is None or wrench is None:
        raise RuntimeError(f"tap monitor: seat {seat}, wrench {wrench}")
    _MON["view"] = sv.create_rigid_contact_view([seat], filter_patterns=[[wrench]], max_contact_data_count=64)
    print(f"[tap] contact monitor: {seat} x {wrench}", flush=True)


def _poll(mgr) -> None:
    try:
        if _MON["view"] is None:
            _make_view(mgr)
        if _MON["view"] is None:
            return
        f = _np(_MON["view"].get_contact_force_matrix(dt=mgr.get_physics_dt())).reshape(-1, 3)[0]
    except Exception as e:  # pragma: no cover
        if _MON["err"] is None:
            _MON["err"] = repr(e)
            print(f"[tap] monitor error: {e!r}", flush=True)
        return
    m = float(np.linalg.norm(f))
    _MON["sum"] += m
    _MON["peak"] = max(_MON["peak"], m)
    _MON["n"] += 1


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_tap_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _poll(cls)

    step._tap_hooked = True
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


# -- drives ------------------------------------------------------------------------------------------------------------
_IDS: dict = {}


def _ids(asset) -> list:
    if "j" not in _IDS:
        _IDS["j"] = asset.find_joints(JN, preserve_order=True)[0]
        _IDS["names"] = list(asset.joint_names)
    return _IDS["j"]


def _write_gains(env, stiff, damp, effort) -> None:
    """Stiffness, damping and effort limit of the six joints (order JN), written only when they change."""
    import torch

    key = (tuple(stiff), tuple(damp), tuple(effort))
    if _T.get("gains") == key:
        return
    asset = env.unwrapped.scene["tapwrench"]
    ids = _ids(asset)
    dev = asset.device
    asset.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([stiff], dtype=torch.float32, device=dev), joint_ids=ids)
    asset.write_joint_damping_to_sim_index(damping=torch.tensor([damp], dtype=torch.float32, device=dev), joint_ids=ids)
    asset.write_joint_effort_limit_to_sim_index(limits=torch.tensor([effort], dtype=torch.float32, device=dev), joint_ids=ids)
    _T["gains"] = key


FREE_GAINS = ([0.0] * 6, [0.0] * 6, [1.0e4] * 6)


def _move_seat(env, pos) -> None:
    import torch

    asset = env.unwrapped.scene["seat"]
    origin = _np(env.unwrapped.scene.env_origins[0])
    pose = [*(np.asarray(pos) + origin).tolist(), 0.0, 0.0, 0.0, 1.0]
    asset.write_root_pose_to_sim_index(root_pose=torch.tensor([pose], dtype=torch.float32, device=asset.device))


# -- the trial-long tracker -----------------------------------------------------------------------------------------------
_T: dict = {}


@on_reset
def _reset(env=None) -> None:
    """Only a reset clears the tracker (and with it any latched failure)."""
    gains = _T.get("gains")
    _T.clear()
    _T.update(step=0, failed=None, phase="free", bitten=False, world=None, qs_prev=None, spin=0.0, spin_bite=0.0,
              th=0.0, th_max=0.0, th_far=0.0, front_bo=0.0, backoffs=0, chip_max=0.0, axis=None, axis_sum=np.zeros(3),
              p_bite=None, square=None, tau=T_CLEAN, dir=0, bend=0.0, bend_max=0.0, twist=0.0, push=0.0, push_max=0.0,
              gate=0, gate_now=False, tilt_start=[], gains=gains, seat_moved=False, snap_at=None)
    _MON.update(sum=0.0, peak=0.0, n=0)
    if env is not None:
        _write_gains(env, *FREE_GAINS)
        _move_seat(env, SEAT_POS)


_reset()


@before_reset
def _free_joints(env) -> None:
    try:
        _T["gains"] = None
        _write_gains(env, *FREE_GAINS)
        set_joint_targets(env, "tapwrench", {n: 0.0 for n in JN})
        set_joint_velocity_targets(env, "tapwrench", {n: 0.0 for n in JN})
    except Exception as e:  # pragma: no cover
        print(f"[tap] before reset: {e!r}", flush=True)


def _fail(why) -> None:
    if _T["failed"] is None:
        _T["failed"] = why
        print(f"[tap] latched 0 at step {_T['step']}: {why}", flush=True)


def _axis_angles(u):
    """Joint angles (j_a about x, then j_b about the turned y) that point the tap's axis along u."""
    b = math.asin(max(-1.0, min(1.0, float(u[0]))))
    a = math.atan2(-float(u[1]), float(u[2]))
    return a, b


@on_step
def _read(env) -> None:
    base = env.unwrapped
    origin = _np(base.scene.env_origins[0])
    asset = base.scene["tapwrench"]
    ids = _ids(asset)
    q = _np(asset.data.joint_pos)[0][ids]
    dq = _np(asset.data.joint_vel)[0][ids]
    names = list(asset.body_names)
    wi = names.index("wrench")
    pose = _np(asset.data.body_link_pose_w)[0][wi]
    tip = pose[:3] - origin
    R = _rot(pose[3:7])
    for a in ("body_link_lin_vel_w", "body_lin_vel_w", "body_com_lin_vel_w"):
        if hasattr(asset.data, a):
            vel = _np(getattr(asset.data, a))[0][wi][:3]
            break
    # "still" uses how far the tip moved over the control step (50 ms), not the solver's instantaneous velocity,
    # which jitters by more than 1 cm/s while the wrench sits on the stand's chamfer
    prev = _T.get("tip_prev")
    _T["tip_prev"] = tip.copy()
    if prev is not None:
        vel = (tip - prev) * HZ
    W = {"q": q, "dq": dq, "tip": tip, "R": R, "u": R[:, 2], "vel": vel, "fingers": {}}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        poses = _np(robot.data.body_link_pose_w)[0]
        pts = []
        for link in GRIP_LINKS:
            p = poses[robot.body_names.index(link)]
            pts.append(GRIP_POINTS[link] @ _rot(p[3:7]).T + (p[:3] - origin))
            if link == "gripper":
                W.setdefault("tcp", {})[side] = p[:3] - origin + _rot(p[3:7]) @ np.array([0.0, 0.0, 0.134])
                W.setdefault("grot", {})[side] = _rot(p[3:7])
        W["fingers"][side] = np.vstack(pts)
    k = max(_MON["n"], 1)
    W["push"] = _MON["sum"] / k
    _MON.update(sum=0.0, peak=0.0, n=0)
    _T["world"] = W
    _track(W, env)


def _unwrap(T, qs) -> float:
    if T["qs_prev"] is None:
        T["qs_prev"] = qs
    d = qs - T["qs_prev"]
    d = (d + math.pi) % (2 * math.pi) - math.pi
    T["qs_prev"] = qs
    T["spin"] += d
    return d


def _track(W, env) -> None:
    T = _T
    T["step"] += 1
    step = T["step"]
    q, dq, tip, u = W["q"], W["dq"], W["tip"], W["u"]
    _unwrap(T, float(q[5]))
    T["push"] = W["push"]
    lateral = float(np.linalg.norm(tip[:2] - HOLE))
    tilt = _tilt_deg(u)
    turning_in = float(dq[5]) < -0.05            # clockwise from above: the spin joint's angle falls
    # -- the bite (or running back into a thread already started)
    if T["phase"] == "free" and T["failed"] is None:
        if not T["bitten"]:
            T["push_max"] = max(T["push_max"], W["push"]) if lateral <= 0.002 else T["push_max"]
            if (lateral <= BITE_R and PLATE_TOP - SEAT_DEPTH <= tip[2] <= PLATE_TOP - SEAT_MIN and W["push"] >= BITE_F
                    and tilt <= BITE_TILT and turning_in):
                T.update(phase="start", bitten=True, spin_bite=T["spin"], th=0.0, th_max=0.0, th_far=0.0, front_bo=0.0,
                         p_bite=np.array([HOLE[0], HOLE[1], tip[2]]), axis_sum=np.zeros(3), tilt_start=[],
                         stick=[float(q[3]), float(q[4])])
                _move_seat(env, SEAT_PARK)
                print(f"[tap] bite at step {step}: tip {np.round(1000 * (tip - [HOLE[0], HOLE[1], PLATE_TOP]), 2).tolist()} mm, "
                      f"push {W['push']:.2f} N, tilt {tilt:.2f} deg", flush=True)
        else:
            th_here = T["spin_bite"] - T["spin"]
            depth = float(T["p_bite"][2] - tip[2])
            if (lateral <= BITE_R and abs(depth - PITCH * th_here / (2 * math.pi)) <= 0.0005 and th_here > 0.0
                    and turning_in):
                T["phase"] = "start" if T["axis"] is None else "thread"
                T["stick"] = [float(q[3]), float(q[4])]
                print(f"[tap] back into the thread at step {step}", flush=True)
    if T["phase"] in ("start", "thread"):
        th = T["spin_bite"] - T["spin"]                 # rad turned in since the bite
        prev = T["th"]
        T["th"] = th
        # back-offs and the chip run
        if th > T["th_max"]:
            if T["phase"] == "start":
                T["axis_sum"] += u * (th - max(T["th_max"], 0.0))
                T["tilt_start"].append(tilt)
            T["th_max"] = th
        T["th_far"] = max(T["th_far"], th)
        if th <= T["th_far"] - BACKOFF:
            T["backoffs"] += 1
            T["th_far"] = th
            T["front_bo"] = T["th_max"]
            print(f"[tap] back-off {T['backoffs']} at step {step}: {math.degrees(th):.0f} deg turned in "
                  f"(deepest {math.degrees(T['th_max']):.0f})", flush=True)
        chip = T["th_max"] - T["front_bo"]
        T["chip_max"] = max(T["chip_max"], chip)
        if chip > CHIP_MAX + 1e-6:
            _fail(f"chip not broken: {math.degrees(chip):.0f} deg of new thread since the last back-off")
        # the thread's axis: the mean over the first turn after the bite
        if T["phase"] == "start" and T["th_max"] >= 2 * math.pi:
            ax = T["axis_sum"] / np.linalg.norm(T["axis_sum"])
            T["axis"] = ax
            T["square"] = _tilt_deg(ax) <= SQUARE
            T["phase"] = "thread"
            T["ramp"] = (step, float(q[3]), float(q[4]))
            print(f"[tap] thread axis fixed at step {step}: {_tilt_deg(ax):.2f} deg from the hole's axis "
                  f"({'square' if T['square'] else 'crooked'}); tilt over the first turn "
                  f"{min(T['tilt_start']):.2f}-{max(T['tilt_start']):.2f} deg", flush=True)
        adv = PITCH * th / (2 * math.pi)
        # the turning resistance
        d = th - prev
        if d > 1e-4:
            T["dir"] = 1
        elif d < -1e-4:
            T["dir"] = -1
        if T["dir"] > 0:
            T["tau"] = (T_CLEAN + T_CHIP * chip / (2 * math.pi)) if th >= T["th_max"] - 0.02 else T_RERUN
        elif T["dir"] < 0:
            T["tau"] = T_REVERSE if T["th_max"] - th < BACKOFF else T_RERUN
        if adv > THREAD_LEN:
            T["tau"] = SNAP_TWIST
        omega = abs(float(dq[5]))
        T["twist"] = min(D_SPIN * omega, T["tau"])
        # the drives' targets: the tip on the helix along the thread's axis (the hole's axis during the first turn)
        ax = T["axis"] if T["axis"] is not None else np.array([0.0, 0.0, 1.0])
        tgt = T["p_bite"] - adv * ax
        jt = tgt - TIP0
        targets = {"j_x": float(jt[0]), "j_y": float(jt[1]), "j_z": float(jt[2])}
        stiff = [K_LAT, K_LAT, K_AX, 0.0, 0.0, 0.0]
        damp = [D_LAT, D_LAT, D_AX, 0.0, 0.0, D_SPIN]
        effort = [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, round(T["tau"], 3)]
        stiff[3] = stiff[4] = K_TILT
        damp[3] = damp[4] = D_TILT
        if T["axis"] is not None:
            a, b = _axis_angles(ax)
            s0, a0, b0 = T.get("ramp", (step - RAMP, a, b))
            f = min(1.0, (step - s0) / RAMP)
            targets.update(j_a=a0 + f * (a - a0), j_b=b0 + f * (b - b0))
        else:
            lim = M_STICK / K_TILT                     # stick: the target is dragged along only beyond the hold
            for i in (0, 1):
                err = float(q[3 + i]) - T["stick"][i]
                if abs(err) > lim:
                    T["stick"][i] = float(q[3 + i]) - math.copysign(lim, err)
            targets.update(j_a=T["stick"][0], j_b=T["stick"][1])
            effort[3] = effort[4] = M_STICK
        # the bending moment at the plate's top face: the springs' restoring force and moment on the tap (at its tip),
        # taken about the axis's point on the face (the dampers only steady the solver and are left out)
        F = np.array([K_LAT * (targets["j_x"] - q[0]), K_LAT * (targets["j_y"] - q[1]), K_AX * (targets["j_z"] - q[2])])
        M = np.zeros(3)
        if True:
            ta = K_TILT * (targets["j_a"] - q[3])
            tb = K_TILT * (targets["j_b"] - q[4])
            ca, sa = math.cos(q[3]), math.sin(q[3])
            M = ta * np.array([1.0, 0.0, 0.0]) + tb * np.array([0.0, ca, sa])
        h = float(T["p_bite"][2] - adv * ax[2]) - PLATE_TOP          # tip height relative to the face (< 0: below)
        lever = -h * ax                                                # from the tip up the axis to the face
        Mq = M - np.cross(lever, F)
        bend = float(np.linalg.norm(Mq - (Mq @ ax) * ax))
        T["bend"] = bend
        if T["phase"] == "thread":
            T["bend_max"] = max(T["bend_max"], bend)
            if bend >= SNAP_BEND:
                _snap(env, f"tap snapped: bending {bend:.2f} N m at the plate's face")
            elif T["twist"] >= SNAP_TWIST - 1e-6 and turning_in:
                _snap(env, f"tap snapped: twist {T['twist']:.2f} N m")
        # out of the thread: turned back past the bite
        if T["phase"] != "snapped" and th < 0.0:
            T["phase"] = "free"
            _write_gains(env, *FREE_GAINS)
            print(f"[tap] out of the thread at step {step}", flush=True)
        elif T["phase"] in ("start", "thread"):
            set_joint_targets(env, "tapwrench", targets)
            set_joint_velocity_targets(env, "tapwrench", {n: 0.0 for n in JN})
            _write_gains(env, stiff, damp, effort)
    # a tap turned back out but still in the hole's mouth rests there: its tilt sticks as in the first turn
    if T["phase"] == "free" and T["bitten"] and T["failed"] is None:
        if lateral <= OUT_R and tip[2] < PLATE_TOP + TAP_OUT:
            if T.get("rest") is None:
                T["rest"] = [float(q[3]), float(q[4]), float(q[2])]
            lim = M_STICK / K_TILT
            for i in (0, 1):
                err = float(q[3 + i]) - T["rest"][i]
                if abs(err) > lim:
                    T["rest"][i] = float(q[3 + i]) - math.copysign(lim, err)
            # it sits on the thread's first turn: it cannot drop deeper without being screwed in, but lifts freely
            kz = K_AX if float(q[2]) < T["rest"][2] - 0.0002 else 0.0
            set_joint_targets(env, "tapwrench", {"j_z": T["rest"][2], "j_a": T["rest"][0], "j_b": T["rest"][1]})
            _write_gains(env, [0.0, 0.0, kz, K_TILT, K_TILT, 0.0], [0.0, 0.0, D_AX if kz else 0.0, D_TILT, D_TILT, 0.0],
                         [1.0e4, 1.0e4, 1.0e4, M_STICK, M_STICK, 1.0e4])
        elif T.get("rest") is not None:
            T["rest"] = None
            _write_gains(env, *FREE_GAINS)
    # -- the end state and the gate
    out = tip[2] >= PLATE_TOP + TAP_OUT or float(np.linalg.norm(tip[:2] - HOLE)) > OUT_R
    T["out"] = bool(out and T["phase"] in ("free",))
    in_stand = (float(np.linalg.norm(tip[:2] - STAND_C)) <= G.STAND_HOLE_R and tip[2] < STAND_TOP
                and tilt <= IN_STAND_TILT)
    gaps = []
    for pts in W["fingers"].values():
        local = (pts - tip) @ W["R"]
        gaps.append(min(float(_seg_dist(local, a, b).min()) - r for a, b, r in CAPSULES))
    T["gap"] = min(gaps)
    still = float(np.linalg.norm(W["vel"])) < STILL_V
    T["in_stand"] = in_stand
    T["gate_now"] = bool(in_stand and T["gap"] >= CLEAR and still and T["out"])
    T["gate"] = T["gate"] + 1 if (T["gate_now"] and _tier() == 1.0) else 0
    if step % 20 == 0 or step < 3 or (T["phase"] != "free" and step % 5 == 0) or (140 <= step <= 260 and step % 5 == 0):
        _log(W, lateral, tilt)


def _snap(env, why) -> None:
    _fail(why)
    _T["phase"] = "snapped"
    _T["snap_at"] = _T["step"]
    _write_gains(env, *FREE_GAINS)
    try:
        _set_visible_tap(False)
    except Exception:  # pragma: no cover
        pass


def _set_visible_tap(visible: bool) -> None:
    """A snapped tap: its lower part stays in the hole (drawn by hiding the wrench's tap and showing the stub)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/env_0/tapwrench/wrench/visual/" in p and p.rsplit("/", 1)[-1] in ("tap", "band", "flute0", "flute1",
                                                                               "flute2", "flute3"):
            img = UsdGeom.Imageable(prim)
            img.MakeVisible() if visible else img.MakeInvisible()


def _tier() -> float:
    """The end-state table (task.json, "scoring") without the 1 s gate: the score the state earns now."""
    T = _T
    if T["failed"] is not None or T["axis"] is None or not T["square"]:
        return 0.0
    adv_max = PITCH * T["th_max"] / (2 * math.pi)
    if adv_max >= THROUGH:
        if not T["out"]:
            return 0.5
        if T["gate_now"]:
            return 1.0
        return 0.75
    if adv_max >= STARTED:
        return 0.25
    return 0.0


def _score() -> float:
    T = _T
    if T["world"] is None:
        return 0.0
    t = _tier()
    if t == 1.0 and T["gate"] < HOLD:
        return 0.75
    return t


def score(objects) -> float:
    """task.json, "scoring": the end-state table read now; 1.0 only once its state has held 1 s; 0 once latched."""
    return _score()


def success(objects) -> bool:
    """The 1.0 state now; the harness holds it 1 s (SUCCESS_HOLD_S), the same 1 s as score()'s."""
    steptime.tick()
    return _T["world"] is not None and _tier() == 1.0


def _log(W, lateral, tilt) -> None:
    T = _T
    tip = W["tip"]
    adv = PITCH * T["th"] / (2 * math.pi) if T["bitten"] else 0.0
    tcp = W.get("tcp", {}).get("right")
    gR = W.get("grot", {}).get("right")
    if gR is not None:
        ga = -gR[:, 2]                                   # the gripper's approach axis, flipped to point up
        bar = W["R"][:, 0]
        bar_in_grip = gR.T @ bar                         # the T-bar's direction in the gripper frame
        rel = tcp - (tip + W["R"] @ np.array([0, 0, G.BAR_Z]))   # TCP from the bar's centre point
        grip_txt = (f"grip tilt {_tilt_deg(ga):.2f} dir {np.round(ga[:2] * 1000, 1).tolist()} wrench dir "
                    f"{np.round(W['u'][:2] * 1000, 1).tolist()} bar-in-grip {np.round(bar_in_grip, 3).tolist()} "
                    f"tcp-from-bar {np.round(1000 * rel, 1).tolist()} ")
    else:
        grip_txt = ""
    print(f"[tap] step {T['step']} {T['phase']} tip {np.round(1000 * (tip - [HOLE[0], HOLE[1], PLATE_TOP]), 2).tolist()} "
          f"lat {1000 * lateral:.2f}mm tilt {tilt:.2f} push {W['push']:.2f} turned {math.degrees(T['th']):.0f} "
          f"max {math.degrees(T['th_max']):.0f} adv {1000 * adv:.2f} bo {T['backoffs']} chipmax {math.degrees(T['chip_max']):.0f} "
          f"tau {T['tau']:.2f} bend {T['bend']:.3f}/{T['bend_max']:.3f} twist {T['twist']:.2f} "
          f"axis {None if T['axis'] is None else round(_tilt_deg(T['axis']), 2)} out {int(T.get('out', False))} "
          f"stand {int(T.get('in_stand', False))} gap {1000 * T.get('gap', 0):.0f} tier {_tier()} fail {T['failed']} "
          f"tcpR {None if tcp is None else np.round(1000 * (tcp - [0, 0, TOP]), 1).tolist()} "
          f"{grip_txt}q {np.round(W['q'], 4).tolist()}", flush=True)


# -- scripted demo ----------------------------------------------------------------------------------------------------------
import yam_arena.demo as _demo  # noqa: E402
from yam_arena import ik as _ik  # noqa: E402


class Arc(float):
    """A waypoint 'yaw' (its value: the yaw at the end) for a turn of the hand about a vertical axis through (cx, cy):
    the TCP circles the axis by dpsi while the grip yaws by the same angle, and its height moves linearly."""

    def __new__(cls, yaw_end, cx, cy, dpsi):
        g = float.__new__(cls, yaw_end)
        g.arc = (cx, cy, dpsi)
        return g


def _target(self, t: float):
    prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
    for w in self.waypoints:
        goal, rot = np.asarray(w.pos, dtype=float), _demo.grip_rotation(w.yaw)
        if t < w.duration:
            s = _ik.smoothstep(t / w.duration)
            arc = getattr(w.yaw, "arc", None)
            finger = prev_finger + s * (w.finger - prev_finger)
            if arc is not None and arc[0] is None:          # a turn of the hand about its own axis
                return (prev_pos + s * (goal - prev_pos), _demo.grip_rotation(float(w.yaw) - arc[2] + s * arc[2]),
                        finger, w.label)
            if arc is not None:
                cx, cy, dpsi = arc
                ang = s * dpsi
                c, sn = math.cos(ang), math.sin(ang)
                dx, dy = prev_pos[0] - cx, prev_pos[1] - cy
                pos = np.array([cx + c * dx - sn * dy, cy + sn * dx + c * dy, prev_pos[2] + s * (goal[2] - prev_pos[2])])
                return pos, _demo.grip_rotation(float(w.yaw) - dpsi + ang), finger, w.label
            return prev_pos + s * (goal - prev_pos), _ik.slerp_matrix(prev_rot, rot, s), finger, w.label
        t -= w.duration
        prev_pos, prev_rot, prev_finger = goal, rot, w.finger
    return prev_pos, prev_rot, prev_finger, "done"


_demo.ArmPlan.target = _target

CLAMP = 0.008                     # the TCP this far below the T-bar's axis: the pads clamp the knurled body, the bar
                                  # runs between the fingers and drives against their sides once the body slips
TIP_SEATED = PLATE_TOP - 0.00236  # the tip resting in the countersink (measured from the colliders)
PUSH = float(os.environ.get("TAP_PUSH", 0.005))     # the TCP commanded this far below the seated grip: about 3.8 N
                                                    # (run 3: 0.6 N/mm plus the wrench's 0.8 N)
YAW_IN = math.radians(115.0)      # each grasp turning in starts at this yaw and turns 180 deg clockwise
SLIP = math.radians(float(os.environ.get("TAP_SLIP", 8.0)))   # the wrist's lag at the cutting torque (K 10 N m/rad)
BACKOFF_CMD = math.radians(150.0)
RISE = 0.032                      # fingertips clear the T-bar's top
# the tip lands this far (m, world x, y) from the TCP's commanded point over the hole (arm's offset there)
LAND = (float(os.environ.get("TAP_LAND_X", 0.99)) / 1000, float(os.environ.get("TAP_LAND_Y", 1.49)) / 1000)   # run 8
N_IN = int(os.environ.get("TAP_N_IN", 22))           # half turns in: about 11 mm, the band at the plate
N_OUT = N_IN + 2                                      # the last ones turn the freed tap in the hole
N_OUT_EXTRA = math.radians(20.0)
# the base joint's twist under the tap's torque moves the hand about the base; offset per N m (m, world x, y, z)
_BH = np.array([HOLE[0], HOLE[1] + 0.305])
TWIST_COMP = float(os.environ.get("TAP_TWIST", 1.0)) * (np.linalg.norm(_BH) / 80.0) * np.array(
    [_BH[1] / np.linalg.norm(_BH), -_BH[0] / np.linalg.norm(_BH), 0.0])
OPEN = -0.020                     # finger targets: pads 40 mm apart, clear of the 20 mm body and of the T-bar within
                                  # about 45 deg of the fingers' gap (measured on the finger hulls)
SHUT = 0.0                        # a firm clamp (50 N a finger): the hand holds the tap's tilt for the first turn
LIGHT = -0.0095                   # a light clamp (about 2.5 N): centres the body; the T-bar drives against the fingers'
                                  # sides (26 deg of play each way) without the hand holding the tap's tilt
OUT_CLAMP = -0.007                # 15 N: enough friction to turn the tap out (0.2 N m)
N_FIRM = 2                        # grasps with the firm clamp (about the first turn)
BACKOFF_PLAY = math.radians(175.0)   # with the light clamp the hand first turns through the fingers' play on the
                                     # T-bar (about 70 deg, run 10), then backs the tap off about 105 deg


class Plan:
    def __init__(self, yaw):
        self.w, self.t, self.yaw = [], 0.0, yaw
        self.p, self.finger = None, FINGER_OPEN

    def go(self, t, p=None, finger=None, label="", yaw=None):
        self.p = self.p if p is None else np.asarray(p, float)
        self.finger = self.finger if finger is None else finger
        y = float(self.yaw) if yaw is None else yaw        # an Arc applies to its own waypoint only
        self.yaw = float(y)
        self.w.append(Waypoint(t, tuple(float(v) for v in self.p), self.finger, label, y))
        self.t += t
        return self

    def turn(self, t, dpsi, z, tau=0.0, label="turn", finger=None):
        """Yaw the hand by dpsi (rad, + counter-clockwise from above) about its own vertical axis, ending at height z.
        tau: the torque the tap resists with (N m, + while turning in); the hand is shifted to offset the twist of the
        arm's base joint under it (joint 1: 80 N m/rad, 0.307 m from the hole), so the tap is not pushed sideways."""
        end = np.array([HOLE[0] + LAND[0], HOLE[1] + LAND[1], z]) + tau * TWIST_COMP
        return self.go(t, end, finger, label, Arc(float(self.yaw) + dpsi, None, None, dpsi))


def demo():
    """Open loop, right arm only; the left arm stays at its ready pose."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {}
    n_in = 6 if mode == "start" else N_IN
    P = Plan(0.0)
    bar0 = TIP0[2] + G.BAR_Z
    g0 = np.array([STAND_C[0], STAND_C[1], bar0 - CLAMP])
    P.go(2.0, g0 + [0, 0, 0.06], OPEN, "over the wrench", yaw=0.0)
    P.go(1.5, g0, label="fingers round the body, the T-bar between them")
    P.go(0.5, label="settle")
    P.go(0.6, finger=SHUT, label="clamp the body")
    carry_z = PLATE_TOP + 0.010 + G.BAR_Z - CLAMP
    P.go(1.5, [g0[0], g0[1], carry_z], label="lift out of the stand")
    hole = np.array([HOLE[0] + LAND[0], HOLE[1] + LAND[1], 0.0])
    P.go(2.5, hole + [0, 0, carry_z], label="carry over the hole", yaw=YAW_IN)
    seat_z = TIP_SEATED + G.BAR_Z - CLAMP
    P.go(1.5, hole + [0, 0, seat_z + 0.003], label="tip over the countersink")
    P.go(0.8, hole + [0, 0, seat_z + 0.0005], label="tip into the countersink")
    P.go(0.8, hole + [0, 0, seat_z - PUSH], label="push")
    P.go(0.5, label="settle")
    half = PITCH / 2
    z = seat_z                            # the clamp's nominal height for the current advance
    zp = seat_z - PUSH
    for k in range(n_in):
        zz = zp if k == 0 else z - 0.001
        bo = BACKOFF_CMD if k < N_FIRM else BACKOFF_PLAY
        P.turn(2.2 if k == 0 else 2.0, -math.pi - SLIP, zz - half, tau=1.0, label=f"turn in {k + 1}")
        db = PITCH * bo / (2 * math.pi)
        P.turn(1.2, bo, zz - half + db, tau=-0.3, label="back off")
        P.turn(1.2, -bo, zz - half, tau=0.2, label="forward again")
        z -= half
        P.go(0.3, finger=OPEN, label="open")
        P.go(0.4, hole + [0, 0, z + RISE], label="rise")
        if k < n_in - 1:
            P.turn(1.0, math.pi + SLIP, z + RISE, label="back round")
            P.go(0.4, hole + [0, 0, z - 0.001], label="down round the body")
            P.go(0.3, finger=SHUT if k + 1 < N_FIRM else LIGHT, label="clamp")
    if mode == "start":
        P.go(2.0, label="hold")
        return {"right": P.w}
    # turning out: grasps from YAW_IN - 180 deg, each 180 deg counter-clockwise (no slip: 0.2 N m)
    P.go(0.6, hole + [0, 0, z + RISE], label="over the body", yaw=YAW_IN - math.pi)
    for k in range(N_OUT):
        last = k == N_OUT - 1
        P.go(0.4, hole + [0, 0, z - 0.001], label="down round the body")
        P.go(0.3, finger=SHUT if last else OUT_CLAMP, label="clamp")
        P.turn(1.5, math.pi + (N_OUT_EXTRA if last else 0.0), z - 0.001 + half, tau=-0.2, label=f"turn out {k + 1}")
        z += half
        if last:
            break
        P.go(0.3, finger=OPEN, label="open")
        P.go(0.4, hole + [0, 0, z + RISE], label="rise")
        P.turn(1.0, -math.pi, z + RISE, label="back round")
    # lift the free wrench out and stand it in its stand
    P.go(1.0, P.p + [0, 0, 0.03], label="lift out of the hole")
    P.go(2.5, [g0[0], g0[1], carry_z + 0.01], label="carry to the stand", yaw=0.0)
    P.go(1.0, g0 + [0, 0, 0.065], label="over the stand")          # the point 7 mm above the stand's top
    P.go(0.8, label="settle")                                       # run 13: the hand still 6 mm short here
    P.go(2.0, g0 + [0, 0, 0.012], label="tap into the stand's hole")
    P.go(1.0, g0 + [0, 0, 0.002], label="down onto the chamfer")
    P.go(0.4, finger=OPEN, label="release")
    P.go(1.0, g0 + [0, 0, 0.07], label="rise clear")
    P.go(1.0, g0 + [0.02, 0.06, 0.07], label="back away")
    P.go(2.5, label="still")
    return {"right": P.w}
