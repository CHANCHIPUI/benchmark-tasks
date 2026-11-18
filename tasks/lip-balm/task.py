"""On the station table, in front of the arms: a care manikin's head (a life-size vinyl practice head) lies on a towel on
a board on a kitchen scale, turned toward the robot, its face 45 degrees above horizontal. Beside it, on the overbed
table's white tray, lies the resident's lip balm stick, capped.

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

from scenekit import Shape, UsdBody, after_spawn, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Uncap the resident's lip balm, apply it along the upper and lower lips from corner to corner, only on "
               "the lips, then cap it and put it back on the tray.")

TOP = g.TOP
WHITE = (0.93, 0.93, 0.91)
WOOD = (0.72, 0.58, 0.40)
GREY = (0.55, 0.56, 0.58)
DARK = (0.10, 0.11, 0.12)

# -- objects -----------------------------------------------------------------------------------------------------------
_ALONG_Y = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))     # the stick's +z (base to tip) along +y
_BASE = np.array([g.STICK_C[0], g.STICK_C[1] - g.CAPPED_L / 2, TOP + g.TRAY_T + g.TUBE_R])
HEAD = UsdBody("head", usd="assets/head.usda", pos=tuple(np.r_[g.MOUTH[:2], TOP + g.MOUTH[2]]), size=(0.2, 0.25, 0.2),
               friction=(0.6, 0.5))
STICK = UsdBody("stick", usd="assets/stick.usda", pos=tuple(_BASE), rot_xyzw=_ALONG_Y, size=(0.016, 0.069, 0.016),
                articulation=True, joint_pos=(("slide", 0.0),), drives=(("slide", g.SLIDE_K, 2.0),),
                friction=(1.2, 1.0), solver_iters=32, torsional_patch=0.006)
CAP_SEAT_GAP = 0.0003       # the cap's rim rests 0.3 mm off the tube's shoulder when seated
CAP = UsdBody("cap", usd="assets/cap.usda", pos=tuple(_BASE + [0, g.TUBE_L + CAP_SEAT_GAP, 0]), rot_xyzw=_ALONG_Y,
              size=(0.016, 0.02, 0.016), mass=g.CAP_MASS, friction=(1.2, 1.0), solver_iters=32,
              torsional_patch=0.006)


def _box(name, x, y, z, color, collide=True):
    return Shape(name, pos=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, TOP + (z[0] + z[1]) / 2),
                 size=(x[1] - x[0], y[1] - y[0], z[1] - z[0]), color=color, collide=collide)


_sx, _sy = g.SCALE_C
_SX = (_sx - g.SCALE_SIZE[0] / 2, _sx + g.SCALE_SIZE[0] / 2)
_SY = (_sy - g.SCALE_SIZE[1] / 2, _sy + g.SCALE_SIZE[1] / 2)
_B0 = g.SCALE_SIZE[2]
_T0 = _B0 + g.BOARD_T
SHAPES = [
    _box("scale", _SX, _SY, (0.0, _B0), GREY),
    _box("scale_display", (_SX[0] - 0.002, _SX[0]), (_sy - 0.03, _sy + 0.03), (0.004, 0.016), DARK, collide=False),
    _box("board", g.BOARD_X, g.BOARD_Y, (_B0, _T0), WOOD),
    _box("towel", g.TOWEL_X, g.TOWEL_Y, (_T0, g.BED), WHITE),
    Shape("towel_roll", kind="capsule", radius=g.ROLL_R, height=g.ROLL_Y[1] - g.ROLL_Y[0] - 2 * g.ROLL_R, axis="Y",
          pos=(g.ROLL_X, sum(g.ROLL_Y) / 2, TOP + g.BED + g.ROLL_R), color=WHITE),
    _box("tray_floor", g.TRAY_X, g.TRAY_Y, (0.0, g.TRAY_T), WHITE),
    _box("tray_rim_n", (g.TRAY_X[0], g.TRAY_X[0] + 0.003), g.TRAY_Y, (0.0, g.TRAY_RIM), WHITE),
    _box("tray_rim_f", (g.TRAY_X[1] - 0.003, g.TRAY_X[1]), g.TRAY_Y, (0.0, g.TRAY_RIM), WHITE),
    _box("tray_rim_r", g.TRAY_X, (g.TRAY_Y[0], g.TRAY_Y[0] + 0.003), (0.0, g.TRAY_RIM), WHITE),
    _box("tray_rim_l", g.TRAY_X, (g.TRAY_Y[1] - 0.003, g.TRAY_Y[1]), (0.0, g.TRAY_RIM), WHITE),
]
OBJECTS = [HEAD, STICK, CAP] + SHAPES


@after_spawn("cap")
def _cap_filtered(prim) -> None:
    """The cap is one solid collider (no thin wall for the fingers to squeeze into). It passes through the tube and the
    bullet (filtered pairs): its friction fit is held in code (see _cap_fit), not by contact or a PhysX joint."""
    from pxr import UsdPhysics

    env = prim.GetPath().GetParentPath()
    fp = UsdPhysics.FilteredPairsAPI.Apply(prim)
    fp.CreateFilteredPairsRel().SetTargets([env.AppendChild("stick").AppendChild("tube"),
                                            env.AppendChild("stick").AppendChild("bullet")])


# -- scoring samples (station frame, relative to the head's origin) ----------------------------------------------------
_L = np.load(pathlib.Path(__file__).resolve().parent / "assets" / "lips.npz")
HEAD_POS = np.r_[g.MOUTH[:2], TOP + g.MOUTH[2]]
LIP = _L["lip_pts"].astype(float) + HEAD_POS
LIP_REGION = _L["lip_region"].astype(int)
SKIN = _L["skin_pts"].astype(float) + HEAD_POS
BAND = _L["band_pts"].astype(float) + HEAD_POS
ALL = np.vstack([LIP, BAND, SKIN])
N_LIP, N_BAND = len(LIP), len(BAND)
REGION_SIZE = np.bincount(LIP_REGION[LIP_REGION >= 0], minlength=2 * g.N_REGIONS)
HOLD_STEPS = 20              # 1 s at 20 Hz
DOME_C = g.tip_from_base() - g.BULLET_R     # the dome's centre along the stick, in the bullet's frame
FAR = g.BULLET_R + 0.004     # a dome centre this far from every sample is touching the head somewhere else: skin

_STATE: dict = {}
_VIEWS: dict = {}
SHOW_BALM = __import__("os").environ.get("LB_SHOW_BALM") == "1"   # draw the balm marks (display only; demo videos)
_LIP_NRM = _L["lip_nrm"].astype(float)
_VIS: dict = {}
_DEBUG = bool(__import__("os").environ.get("LB_DEBUG"))
SEAT = g.TUBE_L + CAP_SEAT_GAP  # the seated cap's rim along the tube from its base end
JAM_OFF, JAM_ANG = 0.002, 10.0  # a cap this far off line while over the neck is jammed: it must come off it to seat
HOLD_GAP = 0.016            # a hand takes hold of a part when its pads close below the part's 16 mm diameter
HOLD_GAP_KEEP = 0.0165      # and keeps hold while they stay within 0.5 mm of it (pad gap read off the finger joints)
HOLD_F = 0.3                # N, with each of its two pads pressing on that part
PUSH_ON = 1.0               # N: the axial push that seats a cap lined up in its last 1 mm
SETTLE_STEPS = 8            # just pulled off, the cap stays heavy and weightless 0.4 s more, settling in the hand
                            # (dropped to 2 g at once under the squeeze and the released pull, it can shoot out)
LOCK_MASS = 5.0             # kg: the cap's mass while the fit holds it on the tube (gravity off), so a hand pulling or
                            # pushing on it meets the fit's resistance and the force can be read off its motion


def _t(x):
    return getattr(x, "torch", x)


def _qrot_z(q):
    """The z column of the rotation of an xyzw quaternion (torch)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    import torch
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _sync_cap(base) -> None:
    """The friction fit, while the cap is on (or lined up in its last 1 mm): the part not being held follows the held
    one rigidly (the cap the tube, unless only the cap is held). Run before every physics substep. When the cap is the
    follower it is heavy and weightless (LOCK_MASS): what the fingers did to it over the last substep, the change of its
    velocity times its mass, is the force they put on the fit."""
    import torch

    S = _STATE
    if not (S.get("attached") or S.get("engaged")):
        _FIT["v_set"] = None
        return
    cap, stick = base.scene["cap"], base.scene["stick"]
    i = _FIT["i_tube"]
    if S.get("leader") == "cap":
        _FIT["v_set"] = None
        pose = _t(cap.data.root_link_pose_w)[0].clone()
        vel = _t(cap.data.root_link_vel_w)[0].clone()
        r = -_qrot_z(pose[3:7]) * S["lock_s"]           # from the cap's rim to the tube's base end
        out = pose.clone()
        out[:3] = pose[:3] + r
        v = vel.clone()
        v[:3] = vel[:3] + torch.linalg.cross(vel[3:6], r)
        stick.write_root_link_pose_to_sim_index(root_pose=out.reshape(1, 7))
        stick.write_root_link_velocity_to_sim_index(root_velocity=v.reshape(1, 6))
        return
    if _FIT.get("v_set") is not None and _FIT.get("heavy") in ("cap", "both"):
        dv = _t(cap.data.root_link_vel_w)[0, :3] - _FIT["v_set"]
        _FIT["f_sum"] = _FIT["f_sum"] + dv * (LOCK_MASS / _FIT["dt"])
        _FIT["n"] += 1
    pose = _t(stick.data.body_link_pose_w)[0, i].clone()
    vel = _t(stick.data.body_link_vel_w)[0, i].clone()
    r = _qrot_z(pose[3:7]) * S["lock_s"]                # from the tube's base end to the cap's rim
    out = pose.clone()
    out[:3] = pose[:3] + r
    v = vel.clone()
    v[:3] = vel[:3] + torch.linalg.cross(vel[3:6], r)
    cap.write_root_link_pose_to_sim_index(root_pose=out.reshape(1, 7))
    cap.write_root_link_velocity_to_sim_index(root_velocity=v.reshape(1, 6))
    _FIT["v_set"] = v[:3].clone()


def _set_heavy(base, which: str) -> None:
    """Which parts are heavy (LOCK_MASS) and weightless: "both" while the fit is on and a hand holds the cap, "cap" (the
    follower) while it is on and no hand holds the cap, "" while it is off. A heavy follower meets fingers closing on it
    or pulling on it with resistance, as the stick as a whole would; a light leader under a firm squeeze jitters, and
    the follower would carry that jitter into the other hand. Weightless for at most one control step after a hand
    lets go of the cap."""
    import torch
    import warp as wp

    if _FIT.get("heavy", "unset") == which:
        return
    cap, stick = base.scene["cap"], base.scene["stick"]
    if "m0" not in _FIT:
        _FIT["m0"] = _t(cap.data.body_mass).clone()
        _FIT["I0"] = _t(cap.data.body_inertia).clone()
        _FIT["sm0"] = _t(stick.data.body_mass).clone()
        _FIT["sI0"] = _t(stick.data.body_inertia).clone()
    hc, ht = which in ("cap", "both"), which == "both"
    k = LOCK_MASS / float(_FIT["m0"].reshape(-1)[0]) if hc else 1.0
    cap.set_masses_index(masses=(_FIT["m0"] * k).reshape(1, 1).to(cap.device))
    cap.set_inertias_index(inertias=(_FIT["I0"] * k).reshape(1, 1, 9).to(cap.device))
    cap.root_view.set_disable_gravities(wp.from_numpy(np.array([[1 if hc else 0]], np.uint8),
                                                      dtype=wp.uint8, device="cpu"),
                                        wp.from_numpy(np.array([0], np.uint32), dtype=wp.uint32, device="cpu"))
    i = _FIT["i_tube"]
    m, I = _FIT["sm0"].clone().reshape(1, -1), _FIT["sI0"].clone().reshape(1, -1, 9)
    if ht:
        k = LOCK_MASS / float(m[0, i])
        m[0, i] *= k
        I[0, i] *= k
    stick.set_masses_index(masses=m.to(stick.device))
    stick.set_inertias_index(inertias=I.to(stick.device))
    n = m.shape[1]
    stick.root_view.set_disable_gravities(wp.from_numpy(np.full((1, n), 1 if ht else 0, np.uint8),
                                                        dtype=wp.uint8, device="cpu"),
                                          wp.from_numpy(np.array([0], np.uint32), dtype=wp.uint32, device="cpu"))
    _FIT["heavy"], _FIT["v_set"] = which, None
    _FIT["f_sum"], _FIT["n"] = torch.zeros(3, device=cap.device), 0


_FIT: dict = {}


def _install_fit(environment) -> None:
    """Hook the scene's per-substep write so the fit is enforced before every physics substep."""
    base = environment.unwrapped
    scene = base.scene
    _FIT["i_tube"] = scene["stick"].body_names.index("tube")
    from yam_arena import constants as C
    _FIT["dt"] = float(C.PHYSICS_DT)
    _FIT.pop("heavy", None)
    _set_heavy(base, "cap")                              # it starts on, following the tube
    if getattr(scene, "_cap_fit", False):
        return
    write = scene.write_data_to_sim

    def write_and_fit(*a, **k):
        out = write(*a, **k)
        _sync_cap(base)
        return out

    scene.write_data_to_sim = write_and_fit
    scene._cap_fit = True


def _holders(environment) -> dict:
    """Which part each hand holds: its pads closed below the part's 16 mm diameter, each pad pressing on that part
    (PhysX contact forces) with HOLD_F or more. Also each hand's tool point and pad gap, for the log."""
    from yam_arena.constants import FINGER_JOINTS

    from scenekit import arm_joints

    V = _views(environment)
    dt = V["dt"]
    fc = np.linalg.norm(_np(V["cap_f"].get_contact_force_matrix(dt))[0], axis=-1)    # (4 pads)
    ft = np.linalg.norm(_np(V["tube_f"].get_contact_force_matrix(dt))[0], axis=-1)
    from yam_arena.constants import TCP_OFFSET

    from scenekit import robot_link_poses

    out, gaps = {}, {}
    tcps = {}
    for side, poses in robot_link_poses(environment).items():
        T = poses["gripper"]
        tcps[side] = T[:3, :3] @ np.asarray(TCP_OFFSET) + T[:3, 3]
    _FIT["tcps"] = tcps
    for k, side in enumerate(("left", "right")):
        gap = -float(np.sum(arm_joints(environment, side, FINGER_JOINTS)))
        gaps[side] = gap
        keep = _STATE.get("holders", {}).get(side) is not None
        if gap >= (HOLD_GAP_KEEP if keep else HOLD_GAP):
            continue
        c, t = fc[2 * k:2 * k + 2], ft[2 * k:2 * k + 2]
        if c.min() >= HOLD_F and c.sum() >= t.sum():
            out[side] = "cap"
        elif t.min() >= HOLD_F:
            out[side] = "tube"
    return out, gaps, fc, ft


def _quat_wxyz(R) -> tuple:
    t = np.trace(R)
    if t > 0:
        s_ = 0.5 / math.sqrt(t + 1.0)
        return 0.25 / s_, (R[2, 1] - R[1, 2]) * s_, (R[0, 2] - R[2, 0]) * s_, (R[1, 0] - R[0, 1]) * s_
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s_ = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s_
    q[j] = (R[j, i] + R[i, j]) / s_
    q[k] = (R[k, i] + R[i, k]) / s_
    return (R[k, j] - R[j, k]) / s_, q[0], q[1], q[2]


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(attached=True, engaged=False, settle=0, lock_s=SEAT, leader="tube", pull_steps=0, f_axial=0.0, guided=True,
                  holders={}, head_other=0.0, contact=None, marked=np.zeros(N_LIP, bool), skin_marked=np.zeros(len(SKIN), bool), coated=np.zeros(2 * g.N_REGIONS, bool), why=[], zero=False,
                  scale=[], scale_avg=0.0, tip_force=0.0, peak_tip=0.0, peak_scale=0.0, cap_off=False,
                  seated=True, in_tray=True, hands_off=False, still=True, ok_steps=0, returned=False)
    _VIEWS.clear()
    if environment is not None:
        _install_fit(environment)


_reset_state()


def _fail(why: str) -> None:
    if why not in _STATE["why"]:
        _STATE["why"].append(why)
    _STATE["zero"] = True


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), np.float64)       # a warp array
    return np.asarray(x, np.float64)


def _views(environment) -> dict:
    """PhysX contact views: the head against the stick's two links, and the bullet against the head."""
    if not _VIEWS:
        base = environment.unwrapped
        psv = base.sim.physics_sim_view
        ns = "/World/envs/env_*"
        _VIEWS["head"] = psv.create_rigid_contact_view(f"{ns}/head", filter_patterns=[f"{ns}/stick/bullet",
                                                                                         f"{ns}/stick/tube"])
        _VIEWS["bullet"] = psv.create_rigid_contact_view(f"{ns}/stick/bullet", filter_patterns=[f"{ns}/head"])
        from yam_arena.robot import asset_link_paths
        lp = asset_link_paths()
        pads = [f"{ns}/{arm}{lp[tip]}" for arm in ("LeftArm", "RightArm") for tip in ("tip_left", "tip_right")]
        _VIEWS["cap_f"] = psv.create_rigid_contact_view(f"{ns}/cap", filter_patterns=pads)
        _VIEWS["tube_f"] = psv.create_rigid_contact_view(f"{ns}/stick/tube", filter_patterns=pads)
        from yam_arena import constants as C
        _VIEWS["dt"] = float(C.PHYSICS_DT)
    return _VIEWS


def _rotation(xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def stick_pose(environment):
    """The tube's base-end centre, its axis (base to tip), the bullet's slide position (m, 0 = fully out, negative =
    pushed in), and the tube's velocity."""
    base = environment.unwrapped
    a = base.scene["stick"]
    origin = _np(base.scene.env_origins[0])
    pose = _np(a.data.body_link_pose_w)[0]
    i_tube, i_bullet = a.body_names.index("tube"), a.body_names.index("bullet")
    p, R = pose[i_tube, :3] - origin, _rotation(pose[i_tube, 3:7])
    q = float(_np(a.data.joint_pos)[0][a.joint_names.index("slide")])
    vel = _np(a.data.body_link_vel_w)[0][i_tube, :3]
    bullet_c = pose[i_bullet, :3] - origin + _rotation(pose[i_bullet, 3:7]) @ [0, 0, DOME_C]
    return p, R[:, 2], q, vel, bullet_c


def _tube_rotation(environment) -> np.ndarray:
    base = environment.unwrapped
    a = base.scene["stick"]
    pose = _np(a.data.body_link_pose_w)[0][a.body_names.index("tube")]
    return _rotation(pose[3:7])


def cap_pose(environment):
    base = environment.unwrapped
    c = base.scene["cap"]
    origin = _np(base.scene.env_origins[0])
    pose = _np(c.data.root_link_pose_w)[0]
    return pose[:3] - origin, _rotation(pose[3:7])[:, 2], _np(c.data.root_lin_vel_w)[0]


def cap_relation(p, a, cp, ca):
    """The cap's rim along the tube's axis from its base end (m), its centre's distance off that axis, and the angle
    between the two axes (deg)."""
    d = cp - p
    s = float(d @ a)
    off = float(np.linalg.norm(d - s * a))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ ca)))))
    return s, off, ang


def _low_point(p0, p1, r) -> tuple[float, np.ndarray]:
    """A horizontal-ish cylinder's lowest point (height, and the end it is at)."""
    a = (p1 - p0) / max(1e-9, np.linalg.norm(p1 - p0))
    drop = r * math.sqrt(max(0.0, 1 - a[2] ** 2))
    end = p0 if p0[2] < p1[2] else p1
    return float(end[2] - drop), end


def _in_tray(pt) -> bool:
    return g.TRAY_X[0] <= pt[0] <= g.TRAY_X[1] and g.TRAY_Y[0] <= pt[1] <= g.TRAY_Y[1]


def _hull_gap(environment, pts: np.ndarray) -> float:
    """The smallest distance from any part of either arm (its collision hulls) to these points."""
    from yam_arena.analysis import world_hulls

    from scenekit import robot_link_poses

    best = 1.0
    for poses in robot_link_poses(environment).values():
        for _, h in world_hulls(poses):
            if np.any(np.all(h.vertices.max(0) + 0.05 >= pts.min(0), 0) & np.all(h.vertices.min(0) - 0.05 <= pts.max(0), 0)):
                p = h.planes
                best = min(best, float((pts @ p[:, :3].T - p[:, 3]).max(axis=1).min()))
    return best


def _fit_step(environment) -> None:
    """The cap's friction fit, once per control step:
    - on: the part not held follows the held one (_sync_cap). It comes off only when one hand holds the cap and the
      other the tube (pads closed on them, pressing) and the axial pull on the cap is CAP_PULL (4 N) or more, with the
      hands drawing apart along the stick, for two control steps (0.1 s) (a hand still pushing a seated cap reads a
      spurious outward force when it twists it). Squeezing (no axial component), weight (not on the fit; the follower is weightless) or a
      knock (needs both hands holding, and 0.1 s) can't take it off; nor can one hand, or hands not gripping;
    - off: once a held cap comes into line (within 1 mm and 5 degrees) into its last 1 mm, having come onto the neck
      without jamming (never 2 mm or 10 degrees off line while over it), it is engaged: held there in line, and it seats
      when pushed on with PUSH_ON (1 N) or more; drawn back (1 N for 0.1 s) or let go of, it is off again. After
      any change to off it stays heavy 0.4 s (SETTLE_STEPS) so it doesn't shoot out of a squeezing hand."""
    S = _STATE
    base = environment.unwrapped
    if "i_tube" not in _FIT:
        _install_fit(environment)
    p, a, _, _, _ = stick_pose(environment)
    cp, ca, _ = cap_pose(environment)
    holders, gaps, fc, ft = _holders(environment)
    S["holders"] = holders
    held = set(holders.values())
    f = 0.0                                              # the axial force the fingers put on the cap (+ = pulling off)
    if _FIT.get("n"):
        f = float((_FIT["f_sum"] / _FIT["n"]).cpu().numpy() @ a)
    import torch
    _FIT["f_sum"], _FIT["n"] = torch.zeros(3, device=_FIT["f_sum"].device), 0
    S["f_axial"] = f
    on_cap = [h for h, part in holders.items() if part == "cap"]
    on_tube = [h for h, part in holders.items() if part == "tube"]
    if S["attached"]:
        drawing = False                                  # the hands drawing apart along the stick
        if on_cap and on_tube:
            sep = float((_FIT["tcps"][on_cap[0]] - _FIT["tcps"][on_tube[0]]) @ a)
            prev = S.get("sep_prev")
            drawing = prev is not None and sep > prev
            S["sep_prev"] = sep
        else:
            S["sep_prev"] = None
        if on_cap and on_tube and f >= g.CAP_PULL and drawing:
            S["pull_steps"] += 1
        else:
            S["pull_steps"] = 0
        if S["pull_steps"] >= 2:                         # pulled off
            S["attached"], S["guided"], S["pull_steps"] = False, False, 0
            S["settle"] = SETTLE_STEPS
    elif S["engaged"]:
        S["back_steps"] = S.get("back_steps", 0) + 1 if f >= PUSH_ON else 0
        if not on_cap or S["back_steps"] >= 2:
            S["engaged"], S["back_steps"] = False, 0     # let go of, or drawn back (1 N, 0.1 s): off again
            S["settle"] = SETTLE_STEPS
        elif f <= -PUSH_ON:
            S["engaged"], S["attached"], S["lock_s"] = False, True, SEAT     # pushed home: seated
    else:
        s, off, ang = cap_relation(p, a, cp, ca)
        if not (s < SEAT + g.NECK_L and s + g.CAP_L > 0 and off < g.CAP_R + g.TUBE_R):
            S["guided"] = True                           # clear of the neck: a fresh start
        elif off > JAM_OFF or ang > JAM_ANG:
            S["guided"] = False                          # on the neck well out of line: jammed (it can't seat from here)
        if on_cap and S["guided"] and off <= 0.001 and ang <= 5.0 and SEAT - 0.005 <= s <= SEAT + g.CAP_SEATED:
            S["engaged"], S["lock_s"] = True, float(min(max(s, SEAT), SEAT + g.CAP_SEATED))
    locked = S["attached"] or S["engaged"]
    if locked:
        S["leader"] = "cap" if (held == {"cap"} and S["attached"]) else "tube"
    else:
        S["leader"] = "tube"
    S["settle"] = max(0, S.get("settle", 0) - 1) if not locked else 0
    _set_heavy(base, ("both" if on_cap else "cap") if locked else ("cap" if S["settle"] > 0 else ""))
    key = (S["attached"], S["engaged"], S["leader"], tuple(sorted(holders.items())))
    S["_fit_n"] = S.get("_fit_n", 0) + 1
    if _DEBUG and (S.get("_last_fit") != key or abs(f) > 0.5 or S["_fit_n"] % 10 == 0):
        S["_last_fit"] = key
        s, off, ang = cap_relation(p, a, cp, ca)
        print(f"[fit] on {S['attached']} engaged {S['engaged']} leader {S['leader']} holders {holders} gaps "
              f"{ {k: round(v * 1000, 1) for k, v in gaps.items()} } pads cap {np.round(fc, 1).tolist()} tube "
              f"{np.round(ft, 1).tolist()} axial {f:5.2f} N rim {s * 1000:.1f} mm off {off * 1000:.1f} ang {ang:.1f}",
              flush=True)


@on_step
def _monitor(environment) -> None:
    S = _STATE
    V = _views(environment)
    dt = V["dt"]
    head_net = _np(V["head"].get_net_contact_forces(dt))[0]
    head_m = _np(V["head"].get_contact_force_matrix(dt))[0]          # (2 filters, 3)
    bullet_net = _np(V["bullet"].get_net_contact_forces(dt))[0]
    bullet_head = _np(V["bullet"].get_contact_force_matrix(dt))[0][0]

    if _DEBUG and (np.linalg.norm(head_net) > 0.01 or np.linalg.norm(bullet_net) > 0.01):
        print(f"[contact] head_net {np.round(head_net, 3)} head_m {np.round(head_m, 3).tolist()} bullet_net "
              f"{np.round(bullet_net, 3)} bullet_head {np.round(bullet_head, 3)}", flush=True)

    # 1. the scale: the vertical force pressing the head down, averaged over 0.25 s
    S["head_other"] = float(np.linalg.norm(head_net - head_m[0]))   # on the head from anything but the bullet
    S["scale"] = (S["scale"] + [max(0.0, -float(head_net[2]))])[-g.SCALE_AVG_STEPS:]
    S["scale_avg"] = float(np.mean(S["scale"]))
    S["peak_scale"] = max(S["peak_scale"], S["scale_avg"])
    if len(S["scale"]) == g.SCALE_AVG_STEPS and S["scale_avg"] >= g.SCALE_LIMIT_N:
        _fail("pressed on the head: the scale rose 100 g")

    # 4. a gripper or the cap touching the head: any force on it but the stick's
    other = head_net - head_m.sum(0)
    if np.linalg.norm(other) > 0.05:
        _fail("a gripper or the cap touched the head")

    _fit_step(environment)
    p, a, q, vel, dome = stick_pose(environment)
    cp, ca, cvel = cap_pose(environment)
    if (S["attached"] or S["engaged"]) and S["leader"] == "tube":   # the cap rides on the tube (as written)
        cp, ca, cvel = p + a * S["lock_s"], a, vel
    s, off, ang = cap_relation(p, a, cp, ca)
    # the bullet inside the cap's bore (its contacts there are with the cap's inside)
    dc = dome - cp
    along = float(dc @ ca)
    radial = float(np.linalg.norm(dc - along * ca))
    inside = bool(radial < g.CAP_BORE + 0.001 and -0.002 < along < g.CAP_L)
    S["cap_off"] = S["cap_off"] or not S["attached"]
    if not S["attached"]:
        # the cap passes through the stick (filtered), so the dome against its rim or outside is found geometrically:
        # the dome's sphere reaching into the cap's body, unless it is past the rim inside the bore (the inside)
        dr, da = max(0.0, radial - g.CAP_R), max(0.0, -along, along - g.CAP_L)
        touching = math.hypot(dr, da) < g.BULLET_R - 0.0003
        in_bore = radial + g.BULLET_R <= g.CAP_BORE + 0.0003 or (radial < g.CAP_BORE and along >= g.BULLET_R)
        if touching and not in_bore:
            _fail("the balm touched something other than the lips")   # the dome against the cap's rim or outside

    # 2. the balm: contact with the head marks the lips (0.3 N or more) and the skin (0.1 N or more) near the contact
    f = float(np.linalg.norm(bullet_head))
    S["tip_force"] = f
    S["peak_tip"] = max(S["peak_tip"], f)
    S["contact"] = None
    if f >= g.SKIN_FORCE:
        d = np.linalg.norm(ALL - dome, axis=1)
        k = int(np.argmin(d))
        if d[k] > FAR:
            _fail("balm on the skin")                      # touching the head away from the mouth
        else:
            c = ALL[k]
            S["contact"] = c
            if _DEBUG:
                rel = c - HEAD_POS
                kind = "lip" if k < N_LIP else ("band" if k < N_LIP + N_BAND else "skin")
                ds = float(np.linalg.norm(SKIN - c, axis=1).min())
                print(f"[touch] f {f:4.2f} u {rel @ g.LIP_LINE * 1000:6.1f} v {rel[1] * 1000:6.1f} {kind:4s}"
                      f"{' r%d' % LIP_REGION[k] if k < N_LIP else ''} skin {ds * 1000:4.1f} mm dome-gap "
                      f"{(d[k] - g.BULLET_R) * 1000:5.2f}", flush=True)
            if f >= g.LIP_FORCE:
                S["marked"] |= np.linalg.norm(LIP - c, axis=1) <= g.MARK_R
            near_skin = np.linalg.norm(SKIN - c, axis=1) <= g.MARK_R
            if np.any(near_skin):
                _fail("balm on the skin")
                S["skin_marked"] |= near_skin            # for the display only

    # 3. the bullet touching anything but the head or the inside of its cap
    if np.linalg.norm(bullet_net - bullet_head) > 0.05 and not inside:
        _fail("the balm touched something other than the lips")

    # 5. the stick or cap on the table outside the tray, or off the table
    tip_end = p + a * g.TUBE_L
    for p0, p1, r in ((p, tip_end, g.TUBE_R), (cp, cp + ca * g.CAP_L, g.CAP_R)):
        z, end = _low_point(p0, p1, r)
        if z < TOP - 0.02:
            _fail("the stick or cap fell off the table")
        elif z < TOP + 0.0012 and not _in_tray(end):
            _fail("the stick or cap touched the table")

    counts = np.bincount(LIP_REGION[S["marked"] & (LIP_REGION >= 0)], minlength=2 * g.N_REGIONS)
    S["coated"] = counts >= g.COATED * REGION_SIZE

    # the return: capped, in the tray, let go of, still
    ends = np.array([p, tip_end, cp + ca * g.CAP_L])
    S["seated"] = bool(S["attached"] and off < 0.001 and ang < 5.0 and s <= SEAT + g.CAP_SEATED)
    S["in_tray"] = bool(all(_in_tray(e) for e in ends) and ends[:, 2].max() < TOP + g.TRAY_RIM + 0.01)
    axis_pts = np.array([p + a * t for t in np.linspace(0, g.CAPPED_L, 8)])
    S["hands_off"] = bool(_hull_gap(environment, axis_pts) - g.TUBE_R >= g.HANDS_OFF)
    S["still"] = bool(np.linalg.norm(vel) < g.STILL and np.linalg.norm(cvel) < g.STILL)
    S["returned"] = bool(S["seated"] and S["in_tray"] and S["hands_off"] and S["still"])
    S["ok_steps"] = S["ok_steps"] + 1 if (S["returned"] and S["coated"].any()) else 0
    if SHOW_BALM:
        _draw_balm(environment)


def _draw_balm(environment) -> None:
    """Display only: the marked lip points (and any marked skin points) as small balm-red dots, a USD Points prim with
    no collider, no rigid body and no mass, so physics, contacts, the scale reading and the score are untouched."""
    S = _STATE
    n = int(S["marked"].sum() + S["skin_marked"].sum())
    if _VIS.get("n") == n:
        return
    import omni.usd
    from pxr import Gf, UsdGeom, Vt

    stage = omni.usd.get_context().get_stage()
    if "prim" not in _VIS or not _VIS["prim"].GetPrim().IsValid():
        pts = UsdGeom.Points.Define(stage, "/World/BalmMarks")
        pts.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.08, 0.14)]))
        _VIS["prim"] = pts
    origin = _np(environment.unwrapped.scene.env_origins[0])
    P = np.vstack([LIP[S["marked"]] + _LIP_NRM[S["marked"]] * 0.0004, SKIN[S["skin_marked"]]]) + origin
    _VIS["prim"].GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in P]))
    _VIS["prim"].GetWidthsAttr().Set(Vt.FloatArray([0.0016] * len(P)))
    _VIS["n"] = n


def success(objects) -> bool:
    """All 12 lip regions coated and the stick returned (capped, in the tray, both arms 3 cm or more away, still),
    with no zero condition; yam_arena ends the trial once this has held for SUCCESS_HOLD_S (1 s)."""
    S = _STATE
    return bool(not S["zero"] and S["coated"].all() and S["returned"])


def score(objects) -> float:
    """At the end of the trial:
    - 0 if, at any time, the scale (the vertical force on the head, averaged over 0.25 s) rose 100 g; balm marked the
      skin outside the 2 mm band round the lips; the bullet touched anything but the head or the inside of its cap; a
      gripper or the cap touched the head; or the stick or cap touched the table outside the tray or fell off it;
    - otherwise 0.4 x (coated upper regions / 6) + 0.4 x (coated lower regions / 6), a region coated when 50% of its
      points are marked;
    - plus 0.2 when, with at least one region coated, the stick has been capped (within 1 mm), in the tray, let go of
      (both arms 3 cm or more away) and still (under 2 mm/s) for the last 1 s.
    Success (1.0) is all 12 regions coated and the stick returned, held 1 s."""
    S = _STATE
    if S["zero"]:
        return 0.0
    c = S["coated"]
    out = 0.4 * c[:g.N_REGIONS].sum() / g.N_REGIONS + 0.4 * c[g.N_REGIONS:].sum() / g.N_REGIONS
    if S["ok_steps"] >= HOLD_STEPS:
        out += 0.2
    return float(round(out, 4))
