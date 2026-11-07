"""On the station table, in front of the arms: a training thigh (a firm foam cylinder, 12.7 cm across) lying across the
table, hip end on the robot's left. A blue windlass tourniquet is strapped round it, snug, its black windlass rod on
top, 30 degrees off lying along the thigh. A clip on the thigh beside the band, toward the knee, has one hook for the
rod. Beyond the thigh, an aneroid gauge on the band's bladder reads 60 mmHg; its dial is green from 250 to 300.

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

from scenekit import Shape, UsdBody, before_reset, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Tighten the tourniquet on the training leg until its gauge reads green, then lock the rod in its clip."

TOP = g.TOP
GREY = (0.20, 0.22, 0.25)

# -- objects --------------------------------------------------------------------------------------------------------
TOURNIQUET = UsdBody("tourniquet", usd="assets/tourniquet.usda", pos=(g.LEG_X, g.BAND_Y, TOP),
                     size=(2 * g.LEG_R, g.LEG_L, 2 * g.LEG_R), articulation=True,
                     joint_pos=(("twist", 0.0), ("press", 0.0), ("needle", -g.needle_angle(g.P_SNUG))),
                     drives=(("twist", 0.0, g.TWIST_DAMPING), ("press", g.TILT_K, g.TILT_D), ("needle", 0.05, 0.002)),
                     friction=(0.8, 0.7))

# the clip's hook: two prongs either side of where the rod lies along the thigh, on the clip's plate (its floor)
_PRONG_X = g.ROD_R + g.CLIP_GAP + g.PRONG_T / 2
_PRONG_H = g.PRONG_TOP - g.HOOK_FLOOR
HOOK = [Shape(f"clip_prong_{'ab'[s > 0]}", pos=(g.LEG_X + s * _PRONG_X, g.BAND_Y + g.CLIP_Y,
                                                 TOP + g.HOOK_FLOOR + _PRONG_H / 2),
              size=(g.PRONG_T, g.PRONG_L, _PRONG_H), color=GREY) for s in (-1, 1)]
HOOK.append(Shape("clip_plate", pos=(g.LEG_X, g.BAND_Y + g.CLIP_Y, TOP + g.HOOK_FLOOR - g.PLATE[2] / 2),
                  size=g.PLATE, color=GREY))
OBJECTS = [TOURNIQUET] + HOOK

# -- monitor: the band's pull on the windlass, the gauge, the hook, and the score's state ----------------------------
_STATE: dict = {}
HOLD_STEPS = 20           # control steps (1 s) locked, green and with the arms clear before it counts as let go
ROD_SIZE = np.array([g.ROD_L, 2 * g.ROD_R, 2 * g.ROD_R])


def _edge_points(size, step: float = 0.005) -> np.ndarray:
    """Points every 5 mm or less along a box's 12 edges, in its frame (origin at its centre)."""
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


ROD_EDGES = _edge_points(ROD_SIZE)
FIXED_BOXES = [(np.eye(3), np.asarray(s.pos, float) - [0, 0, TOP], np.asarray(s.size, float)) for s in HOOK]
FIXED_BOXES = [(R, c, size, _edge_points(size)) for R, c, size in FIXED_BOXES]


def _asset(environment):
    return environment.unwrapped.scene["tourniquet"]


def _torque(environment, tau: float) -> None:
    """The band's pull on the windlass: a torque about the vertical on the twisted strap, applied every physics step."""
    import torch

    asset = _asset(environment)
    ids = asset.find_bodies(["windlass"])[0]
    value = torch.tensor([[[0.0, 0.0, tau]]], dtype=torch.float32, device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(torques=value, body_ids=ids, is_global=True)


def _set_needle(environment, p: float) -> None:
    from scenekit import set_joint_targets

    set_joint_targets(environment, "tourniquet", {"needle": -g.needle_angle(p)})


def _set_press_target(environment, tilt: float) -> None:
    from scenekit import set_joint_targets

    set_joint_targets(environment, "tourniquet", {"press": tilt})


def hook_dip(theta: float, tilt: float) -> float:
    """How far down the rod's centre line is at the hook (the rod's end on the knee side, -y)."""
    end = 1.0 if math.sin(theta) < 0 else -1.0             # the rod's +x end points toward the knee
    return end * abs(g.CLIP_Y) * math.sin(tilt)


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(twist=0.0, last_q=0.0, tilt=0.0, dip=0.0, locked=False, pressure=g.P_SNUG, aligned=False, theta=g.THETA0,
                  clearance=1.0, clear=True, clear_steps=0, held=False)
    if environment is not None:
        _restart(environment)


@before_reset
def _restart(environment) -> None:
    """Untwist the windlass, lift the rod out of the hook and level it, and zero the gauge before a reset (the reset
    moves the tourniquet only by its root, not its joints)."""
    from scenekit import write_joint_positions

    _torque(environment, 0.0)
    _set_press_target(environment, 0.0)
    _set_needle(environment, g.P_SNUG)
    write_joint_positions(environment, "tourniquet", {"twist": 0.0, "press": 0.0,
                                                      "needle": -g.needle_angle(g.P_SNUG)})


_reset_state()


def _distance(hulls, boxes) -> float:
    """The smallest gap between convex hulls and boxes (rotation, centre, size, edge points): from every hull vertex
    to each box, and from points along each box's edges to each nearby hull (by the hull's face planes, which never
    overestimate it). Negative when they overlap."""
    from yam_arena.analysis import box_signed_distance

    if not boxes or not hulls:
        return 1.0
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


def rod_box(theta: float, tilt: float):
    """The rod as a box: (rotation, centre in the station frame above the table, size)."""
    c, s = math.cos(theta), math.sin(theta)
    ct, st = math.cos(tilt), math.sin(tilt)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return Rz @ Ry, np.array([g.LEG_X, g.BAND_Y, g.ROD_Z]), ROD_SIZE


def _clearance(environment, theta: float, tilt: float) -> float:
    """How far either arm is from the rod and the hook, using the URDF's collision hulls of every link, the gripper
    body and both fingers at their simulated poses."""
    from yam_arena.analysis import world_hulls

    from scenekit import robot_link_poses

    hulls = []
    for poses in robot_link_poses(environment).values():
        for _, h in world_hulls(poses):
            h.vertices[:, 2] -= TOP
            h.planes[:, 3] -= h.planes[:, 2] * TOP
            hulls.append(h)
    R, centre, size = rod_box(theta, tilt)
    return _distance(hulls, [(R, centre, size, ROD_EDGES)] + FIXED_BOXES)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    joints = object_states(environment, ["tourniquet"])["tourniquet"].joints
    q = float(joints["twist"])
    S["twist"] += math.degrees(math.remainder(q - S["last_q"], 2 * math.pi))   # the joint comes back wrapped
    S["last_q"] = q
    S["tilt"] = tilt = float(joints["press"])
    S["theta"] = theta = g.THETA0 + math.radians(S["twist"])
    S["aligned"] = abs(math.remainder(theta - math.pi / 2, math.pi)) <= g.LOCK_ALIGN    # along the thigh
    S["dip"] = dip = hook_dip(theta, tilt)
    if not S["locked"] and S["aligned"] and dip >= g.LOCKED_DEPTH:
        S["locked"] = True                                  # down in the hook: it snaps in and stays down
        end = 1.0 if math.sin(theta) < 0 else -1.0
        _set_press_target(environment, end * g.TILT_MAX)
    elif S["locked"] and dip < g.UNLOCK_DEPTH:
        S["locked"] = False                                 # lifted back out of the hook
        _set_press_target(environment, 0.0)
    S["pressure"] = p = g.pressure(S["twist"])
    _set_needle(environment, p)
    _torque(environment, -g.K_TWIST * math.radians(S["twist"]))
    S["clearance"] = _clearance(environment, theta, tilt)
    S["clear"] = S["clearance"] >= g.CLEAR
    S["clear_steps"] = S["clear_steps"] + 1 if S["locked"] and p >= g.GREEN and S["clear"] else 0


def success(objects) -> bool:
    """The gauge in the green (250 mmHg or more) with the rod locked in the clip's hook and no part of either arm
    within 1 cm of the rod or the hook (yam_arena ends the trial once this has held for SUCCESS_HOLD_S)."""
    return bool(_STATE["locked"] and _STATE["pressure"] >= g.GREEN and _STATE["clear"])


def score(objects) -> float:
    """At the end of the trial (success scores 1.0): 0.5 with the gauge in the green but the rod still held (not locked,
    or locked with a hand still on it in the last 1 s); 0.25 with the rod locked below the green; 0 otherwise (not
    tightened, or let go unlocked so it unwound)."""
    S = _STATE
    green = S["pressure"] >= g.GREEN
    if S["locked"] and green:
        return 1.0 if S["clear_steps"] >= HOLD_STEPS else 0.5
    if green and not S["clear"]:
        return 0.5
    if S["locked"]:
        return 0.25
    return 0.0
