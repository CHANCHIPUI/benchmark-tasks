"""Scripted full-score demo. The left arm takes the basket's coated handle from the side (the gripper level, coming
in square across it from the left, its fingers above and below the flat handle and its finger plates along it), lifts the
basket straight up out of the oil keeping it level, pushes it back until its hook is over the hanger rod, lowers it
until the hook takes its weight, lets go and backs off. Every move follows the basket's measured pose: the hand's
command is corrected each step by the basket's position and tilt errors, since the wrist gives under the load.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record] [--hold] [--keep-hold]

``--hold`` lifts the basket out and holds it over the vat without hanging it (0.5). ``--keep-hold`` hangs it but keeps
hold of the handle (0.5).
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as g  # noqa: E402
from demokit import run, turn, wait  # noqa: E402

L, W, H = g.BASKET
APPROACH_AZ = math.radians(-90)     # the left hand comes in level from the left, square across the handle
APPROACH = np.array([math.cos(APPROACH_AZ), math.sin(APPROACH_AZ), 0.0])
HANG_LIFT_BOTTOM = g.ROD_Z + g.ROD_R + 0.030 - (H - g.HOOK_LIP)   # the hook's lip 3 cm above the hanger's top
SEAT_BOTTOM = g.ROD_Z + g.ROD_R - (H - g.HOOK_T) - 0.008          # lowered past seating: the rod takes the weight
GRIP_OPEN = 0.45                    # the fingers part-open (about 42 mm) to come in round the 8 mm handle
DEBUG = bool(__import__("os").environ.get("FRY_DEBUG"))


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_err(R_goal, R) -> np.ndarray:
    """The small rotation (axis * angle, world frame) taking R to R_goal."""
    E = R_goal @ R.T
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / 2
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    return v / s * math.asin(min(1.0, s))


class Basket:
    def __init__(self, ctx):
        self.ctx = ctx

    def pose(self):
        s = self.ctx.states()["basket"]
        return np.asarray(s.pos, float), quat_to_R(s.quat_xyzw)


def handle_grip(o, R):
    """The hand's grasp on the handle for the basket at (o, R): the tool point at the handle's middle (GRIP_S along it),
    approach level and square across it, fingers closing across the flat bar, finger plates along it."""
    p = g.HANDLE_PITCH
    d_local = np.array([-math.cos(p), 0.0, math.sin(p)])
    point = o + R @ (np.array([-L / 2, 0.0, H + g.HANDLE_STEM]) + d_local * g.GRIP_S)
    z = APPROACH
    d = R @ d_local
    d = d - (d @ z) * z
    y = -d / np.linalg.norm(d)
    x = np.cross(y, z)
    return point, np.column_stack([x, y, z])


def tcp_actual(arm):
    from scenekit import arm_joints, env
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    return ik.tcp_pose(arm_joints(env(), arm.side, ARM_JOINTS), arm.base)


def arrive(arm, tol=0.004, timeout=4.0):
    """Hold the command until the arm's measured tool point is within ``tol`` of it (the arm lags big moves)."""
    for _ in range(round(timeout * 20)):
        tcp, _ = tcp_actual(arm)
        if np.linalg.norm(tcp - arm.pos) < tol:
            return
        yield
    if DEBUG:
        print(f"arrive: still {np.linalg.norm(tcp_actual(arm)[0] - arm.pos) * 1000:.0f} mm off", flush=True)


def to_rotvec_R(v) -> np.ndarray:
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-12 else turn(v / a, a)


def _clip(v, m):
    n = np.linalg.norm(v)
    return v if n <= m else v * (m / n)


CORR = {"p": np.zeros(3), "r": np.zeros(3)}   # the hand's sag correction, kept from one move to the next


def carry(basket, arm, T, goal_o, goal_R, seconds, settle=3.0, gain=1.0, tol=0.003, track=None):
    """Move the basket's origin to ``goal_o`` (rotation ``goal_R``) over ``seconds``, the hand rigidly offset from it by
    the grasp ``T`` (hand pose in the basket's frame), each step correcting the command by the basket's measured
    position and rotation errors (integrated), then hold until it is within ``tol`` of the goal (up to ``settle`` s)."""
    o0, R0 = basket.pose()
    corr_p, corr_r = CORR["p"], CORR["r"]
    n = max(1, round(seconds * 20))
    Tp, TR = T
    for k in range(1, n + round(settle * 20) + 1):
        s = min(1.0, k / n)
        s = s * s * (3 - 2 * s)
        o_des = o0 + s * (goal_o - o0)
        R_des = to_rotvec_R(s * rot_err(goal_R, R0)) @ R0
        o, R = basket.pose()
        e_p, e_r = o_des - o, rot_err(R_des, R)
        if k > n and np.linalg.norm(goal_o - o) < tol and np.linalg.norm(rot_err(goal_R, R)) < math.radians(2):
            break
        corr_p = _clip(corr_p + gain * 0.3 * e_p, 0.18)
        corr_r = _clip(corr_r + gain * 0.6 * e_r, math.radians(35))
        CORR["p"], CORR["r"] = corr_p, corr_r
        Rc = to_rotvec_R(corr_r) @ R_des
        arm.aim(o_des + corr_p + Rc @ Tp, Rc @ TR)
        if track is not None:
            track(k)
        yield
    if DEBUG:
        from scenekit import arm_joints, env
        from yam_arena import ik
        from yam_arena.constants import ARM_JOINTS
        q = arm_joints(env(), arm.side, ARM_JOINTS)
        tcp, _ = ik.tcp_pose(q, arm.base)
        print(f"carry: hand cmd {np.round(arm.pos, 3)} actual {np.round(tcp, 3)} ({np.linalg.norm(tcp - arm.pos) * 1000:.0f} mm off); "
              f"basket rel hand {np.round(basket.pose()[0] - tcp, 3)}", flush=True)
        o, R = basket.pose()
        print(f"carry: goal {np.round(goal_o, 3)} got {np.round(o, 3)} err {np.linalg.norm(goal_o - o) * 1000:.1f} mm "
              f"tilt err {math.degrees(np.linalg.norm(rot_err(goal_R, R))):.1f} deg  corr {np.round(corr_p, 3)}",
              flush=True)


def script(ctx, mode="full"):
    R_arm = ctx.left                    # the left arm works the basket: its wrist sits mid-range for this grasp
    basket = Basket(ctx)
    o, Rb = basket.pose()
    # grasp the handle: come in level along the approach, fingers open, and close on it
    p, Rh = handle_grip(o, Rb)
    stage = p - APPROACH * 0.09 + np.array([-0.06, 0.0, 0.0])     # in front of the fryer, clear of it
    yield from R_arm.move_joints(stage, Rh, 2.2, grip=GRIP_OPEN)
    yield from arrive(R_arm)
    yield from R_arm.move(p - APPROACH * 0.09, Rh, 0.8)
    yield from R_arm.move(p, Rh, 1.2)
    yield from arrive(R_arm)
    if DEBUG:
        from scenekit import arm_joints, env
        from yam_arena import ik
        from yam_arena.constants import ARM_JOINTS
        q = arm_joints(env(), R_arm.side, ARM_JOINTS)
        tcp, rot = ik.tcp_pose(q, R_arm.base)
        o2, R2 = basket.pose()
        print(f"grasp: target {np.round(p, 4)} cmd {np.round(R_arm.pos, 4)} actual {np.round(tcp, 4)}; basket origin "
              f"then {np.round(o, 4)} now {np.round(o2, 4)} tilt {math.degrees(math.acos(min(1, R2[2, 2]))):.1f}; "
              f"approach {np.round(rot[:, 2], 2)} closing {np.round(rot[:, 0], 2)} arm gap {ctx.task._STATE['clearance'] * 100:.1f} cm",
              flush=True)
    yield from R_arm.set_grip(0.0, 0.6)
    yield from wait(0.3)
    if DEBUG:
        from scenekit import arm_joints, env
        from yam_arena.constants import FINGER_JOINTS
        print(f"fingers after closing: {np.round(arm_joints(env(), R_arm.side, FINGER_JOINTS), 4)} (0 = shut, "
              f"-0.047 = open; about -0.004 each on the 8 mm handle)", flush=True)
    o, Rb = basket.pose()
    T = (Rb.T @ (R_arm.pos - o), Rb.T @ R_arm.rot)          # the hand in the basket's frame
    level = np.eye(3)
    rest_o = o.copy()
    up = rest_o.copy()
    up[2] = g.TOP + HANG_LIFT_BOTTOM
    yield from carry(basket, R_arm, T, up, level, 3.5)
    if mode == "hold":
        yield from wait(3.0)
        return
    back = up + [g.HUNG_U - g.REST_U - 0.0005, 0.0, 0.0]         # back against the hanger's plate
    yield from carry(basket, R_arm, T, back, level, 1.5, tol=0.0015)
    seat = back.copy()
    seat[2] = g.TOP + SEAT_BOTTOM
    yield from carry(basket, R_arm, T, seat, level, 1.5, settle=0.5, gain=0.0)   # no correction: let the rod take it
    if mode == "keep":
        yield from wait(4.0)
        return
    # ease the command to where the hand really is, so it doesn't spring up when the load comes off; open part-way
    # (fully open, the lower finger swings onto the basket); drop a little below the handle and back off level
    tcp, rot = tcp_actual(R_arm)
    yield from R_arm.move(tcp, rot, 0.8)
    yield from R_arm.set_grip(GRIP_OPEN, 0.6)
    yield from R_arm.move(R_arm.pos + [0.0, 0.0, -0.008], None, 0.4)
    yield from R_arm.move(R_arm.pos - APPROACH * 0.09, None, 1.0)
    yield from R_arm.go_home(1.8)
    yield from wait(2.0)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    return (f"bottom {S['bottom'] * 100:5.1f} cm  out {S['out']}  over {S['over_vat']}  tilt {S['tilt']:5.1f}  "
            f"hung {S['hung']}  spilled {S['spilled']}  near {S['near']}  dipped {S['dipped']}  "
            f"arm {S['clearance'] * 100:5.1f} cm  hot {S['hot']} {S['hot_gap'] * 1000:5.1f} mm  grip {S['grip_gap'] * 1000:5.1f} mm  "
            f"clear {S['clear_steps']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    for flag in ("hold", "keep-hold"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    mode = "hold" if a.hold else "keep" if a.keep_hold else "full"
    run(a.task, a.out, lambda ctx: script(ctx, mode), max_steps=a.max_steps, record=not a.no_record, log=log,
        log_every=20)
