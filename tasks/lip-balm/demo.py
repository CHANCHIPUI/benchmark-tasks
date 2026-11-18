"""Scripted full-score demo.

1. The left hand lifts the capped stick from the tray by its cap and holds it level in the air. The right hand takes the
   tube's base end in a pen grip (fingers along the stick, as a person holds a lip balm); the left pulls the cap off
   along the stick's axis (the fit, held in task.py, lets go once the two-hand pull has been 4 N or more for 0.1 s with
   the hands drawing apart) and holds it clear.
2. The right hand turns the stick tip-down over the mouth.
3. It traces the balm along the upper lip, then the lower lip, corner to corner on each lip's centre line, then goes
   back along the lower lip nearer its outer edge and makes a short second pass over the upper lip's first third. The
   stick stays within 10 degrees of the lip's local inward normal, so it turns through about 70 degrees as the line wraps
   round the mouth (assets/stroke_plan.npz, from plan_strokes.py, holds the planned axes and hand turns). The tip's
   contact force is held near 0.4 N from the measured force, the press going in along the local normal.
4. It turns the stick level clear of the face and brings it back where the cap came off; the left hand lines the cap up
   and pushes it back on until the fit seats (1 N or more in its last 1 mm), lets go and withdraws.
5. The right hand lays the capped stick in the tray, lets go and withdraws.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record] [--variant NAME]

``--variant`` (development checks, not part of the submission) changes the script: ``light`` (a 0.15 N graze held
under 0.2 N), ``hard`` (presses 3 N), ``sloppy`` (the upper stroke runs 4 mm above the lip's centre line), ``held``
(never recaps), ``upper`` (the upper lip only, then returns the stick), ``nothing``; ``pull1`` (one hand grips the cap
and pulls it along, the tube lying in the tray) and ``apart`` (both hands round cap and tube, not gripping, drawn
apart).
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
from demokit import run, together, turn, wait  # noqa: E402

TOP = g.TOP
GRIP_OPEN = 0.40          # fingers part-open (about 38 mm) to come down round the 16 mm tube
R_GRIP_S = 0.022          # the right hand's tool point along the tube from its base end (pads over 9-35 mm)
CAP_TAKE = 0.016          # the left tool point beyond the cap's rim, along it (fingertips 3 mm short of the rim)
L_HOVER = np.array([0.18, 0.02, TOP + 0.15])  # where the left hand holds the stick by its cap for the handover
L_GRIP_S = g.TUBE_L + 0.015     # the left hand's tool point on the cap, from the tube's base end
CAP_GRIP = 0.10                 # grip command on the cap: a firm hold (pads closed below its 16 mm, pressing), enough
                                # friction for the fit's 4 N pull
CAP_EASE = 0.215                # after the pull: eased to just at the cap's surface, so the fingers don't reach in past
                                # the tube's radius (they would catch on its shoulder as the cap goes back on)
RIGHT_PUSHES = False            # recap by pushing the tube into the held cap, and lay the stick down by the cap
CAP_HOLD = 0.18                 # grip command that just closes the fingers on the cap (about 8.5 mm each)
REGRIP = 0.006                  # before the last push the left hand moves its hold this far out along the cap, so its
                                # fingers clear the tube's shoulder
PEN_OPEN = 0.45                 # fingers part-open to come along the stick round the tube's end
DROP = 0.022                    # the capped stick is let go this far over the tray's floor (the fingers clear it)
ABOVE = 0.006             # tool point above the stick's axis (fingertips clear of the tray's floor)
TILT = math.radians(20)   # the stick, tip down, leans 20 degrees from vertical toward the head (+x)
SPIN = math.radians(-30)  # and the hand is turned 30 degrees about it (it comes in from the front right)
F_TARGET = 0.40           # N at the tip while stroking
TOWARD_JOIN = 0.0008      # aim this far from the lip's centre line toward the other lip (the skin is on the far side)
T_IN = 0.75               # strokes start and end 1.5 mm further in from the scored span's ends (planned points are 2 mm apart)
STROKE_SPEED = 0.008      # m/s along the lip
JOINT_SPEED = math.radians(15)   # rad/s: no joint turns faster than this while stroking
EXTRA_UPPER = True        # a short second pass over the upper lip's first third
EXTRA_UPPER_END = False   # and a short pass back over its last region
SKIN_GUARD = 0.0045       # a contact this near the skin moves the aim 0.8 mm further in, across the lip (the centre
                          # line is 5.3 mm or more from it)
VARIANT = {"name": "full"}
DEBUG = bool(__import__("os").environ.get("LB_DEBUG"))


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def grip_frame(axis, side: str = "right") -> np.ndarray:
    """Tool frame for a hand coming straight down on a level stick along ``axis``: fingers closing across it, pad width
    along it."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(a, z), a, z])


PEN_S = 0.003             # the pen grip's tool point along the tube from its base end (pads over its first 16 mm)
PEN_SPIN = math.pi        # the pen grip's turn about the stick (from dev/handover_scan.py)


def pen_frame(axis, spin=None) -> np.ndarray:
    """Tool frame for a pen grip: fingers pointing along ``axis`` (base to tip), closing across it; turned ``spin``
    about it from fingers closing horizontally."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    hint = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(hint, a)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(a, x), a])
    return turn(a, PEN_SPIN if spin is None else spin) @ R


def end_frame(axis) -> np.ndarray:
    """Tool frame for the left hand coming along a level stick's axis from beyond its cap: fingers pointing back
    along the axis, closing across it horizontally."""
    z = -np.asarray(axis, float) / np.linalg.norm(axis)
    x = -np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def apply_frame(tilt=None, spin=None) -> np.ndarray:
    """The stick's frame (y = its axis, base to tip) while applying: tip down, leaning TILT toward +x, spun SPIN about
    its axis."""
    tilt = TILT if tilt is None else tilt
    spin = SPIN if spin is None else spin
    a = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    side = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
    R = np.column_stack([np.cross(a, side), a, side])
    return turn(a, spin) @ R


def tcp_actual(arm):
    from scenekit import arm_joints, env
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    return ik.tcp_pose(arm_joints(env(), arm.side, ARM_JOINTS), arm.base)


def arrive(arm, tol=0.003, timeout=3.0):
    for _ in range(round(timeout * 20)):
        tcp, _ = tcp_actual(arm)
        if np.linalg.norm(tcp - arm.pos) < tol:
            return
        yield


def stick(ctx):
    """Measured: the tube's base-end centre, its axis, and the cap's rim centre and axis."""
    import task
    from scenekit import env

    p, a, q, vel, dome = task.stick_pose(env())
    cp, ca, _ = task.cap_pose(env())
    return p, a, cp, ca


def lip_lines():
    """Each lip's centre line on the surface, from the resident's right corner to the left, with surface normals:
    (points, normals) in the station frame, every 1 mm across the lip, 1 mm short of the scored span's ends."""
    import task

    L = np.load(pathlib.Path(__file__).resolve().parent / "assets" / "lips.npz")
    P, N, R = task.LIP, L["lip_nrm"].astype(float), task.LIP_REGION
    rel = P - task.HEAD_POS
    u, v = rel @ g.LIP_LINE, rel[:, 1]
    half = g.CORNER_X_MH * g.HEAD_SCALE / 1000 - g.CORNER_IN
    out = {}
    for name, sel in (("upper", (R >= 0) & (R < 6)), ("lower", R >= 6)):
        pts, nrm = [], []
        for u0 in np.arange(-half + 0.001, half - 0.001 + 1e-9, 0.001):
            m = sel & (np.abs(u - u0) <= 0.0008)
            if m.sum() < 2:
                continue
            vm = (v[m].min() + v[m].max()) / 2
            k = np.flatnonzero(m)[np.argmin(np.abs(v[m] - vm))]
            pts.append(P[k])
            nrm.append(N[k] / np.linalg.norm(N[k]))
        out[name] = (np.array(pts), np.array(nrm))
    return out


def stick_frame(axis, x_hint) -> np.ndarray:
    """A frame for the stick: y along its axis (base to tip), x as near ``x_hint`` as square to it allows."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.asarray(x_hint, float) - (np.asarray(x_hint, float) @ a) * a
    x /= np.linalg.norm(x)
    return np.column_stack([x, a, np.cross(x, a)])


class Held:
    """The stick's pose in the right hand's tool frame (its base end and its frame), measured once it is held, or
    nominal (from the planned grasp) for an offline check."""

    def __init__(self, tcp, Rt, p, a):
        S = stick_frame(a, Rt[:, 0])
        self.p_in = Rt.T @ (p - tcp)
        self.S_in = Rt.T @ S

    @classmethod
    def measured(cls, ctx):
        tcp, Rt = tcp_actual(ctx.right)
        p, a, _, _ = stick(ctx)
        return cls(tcp, Rt, p, a)

    def tcp_for(self, base, S_target):
        """Tool pose putting the tube's base end at ``base`` with the stick's frame at ``S_target``."""
        Rt = S_target @ self.S_in.T
        return base - Rt @ self.p_in, Rt


def _align(u, v) -> np.ndarray:
    """The rotation taking unit vector u to v (smallest)."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(u @ v)
    w = np.cross(u, v)
    s = np.linalg.norm(w)
    if s < 1e-9:
        return np.eye(3)
    return turn(w / s, math.atan2(s, c))


def script(ctx):
    import task

    name = VARIANT["name"]
    if name == "nothing":
        yield from wait(5.0)
        return
    S = task._STATE
    Rr, Lr = ctx.right, ctx.left
    p, a, cp, ca = stick(ctx)
    up = np.array([0.0, 0.0, 1.0])
    R_cap = grip_frame(a)

    l_at = p + a * L_GRIP_S + up * ABOVE
    if name in ("pull1", "apart"):                       # checks: these must not uncap
        yield from Lr.move_joints(l_at + up * 0.06, R_cap, 2.0, grip=GRIP_OPEN)
        yield from Lr.move(l_at, None, 1.0)
        if name == "pull1":                              # one hand grips the cap and pulls it along, the tube in the tray
            yield from Lr.set_grip(CAP_GRIP, 0.8)
            yield from wait(0.3)
            for k in range(3):
                yield from Lr.move(Lr.pos + a * 0.02, None, 1.0)
                _dbg(ctx, f"one-hand pull {k}")
        else:                                            # both hands round cap and tube, not gripping, drawn apart
            r_at = p + a * R_GRIP_S + up * ABOVE
            yield from Rr.move_joints(np.array([0.13, -0.20, TOP + 0.22]), grip_frame(a), 1.5, grip=GRIP_OPEN)
            yield from Rr.move(r_at + up * 0.06, None, 1.5)
            yield from Rr.move(r_at, None, 1.0)
            yield from together(Lr.set_grip(0.25, 0.8), Rr.set_grip(0.25, 0.8))
            for k in range(3):
                yield from together(Lr.move(Lr.pos + a * 0.015, None, 1.0), Rr.move(Rr.pos - a * 0.015, None, 1.0))
                _dbg(ctx, f"hands apart {k}")
        yield from together(Lr.set_grip(1.0, 0.6), Rr.set_grip(1.0, 0.6))
        yield from together(Lr.move(Lr.pos + up * 0.06, None, 1.0), Rr.move(Rr.pos + up * 0.06, None, 1.0))
        _dbg(ctx, "check end")
        print(f"[check] {name}: cap on {S['attached']} engaged {S['engaged']} cap_off {S['cap_off']}", flush=True)
        return

    # 1. the left hand lifts the stick by its cap; the right takes the tube's end in a pen grip; the cap comes off ------
    yield from Lr.move_joints(l_at + up * 0.06, R_cap, 2.0, grip=GRIP_OPEN)
    yield from Lr.move(l_at, None, 1.0)
    yield from arrive(Lr)
    yield from Lr.set_grip(CAP_GRIP, 0.8)
    yield from wait(0.3)
    yield from Lr.move(Lr.pos + up * 0.04, None, 0.8)
    yield from Lr.move_joints(L_HOVER, R_cap, 2.0)
    yield from arrive(Lr, tol=0.002)
    p, a, cp, ca = stick(ctx)
    R_pen = pen_frame(a)
    r_at = p + a * PEN_S
    # the right hand goes round in front of the head (its neck points at the right arm), not over it
    yield from Rr.move_joints(np.array([0.13, -0.20, TOP + 0.22]), R_pen, 1.5, grip=PEN_OPEN)
    yield from Rr.move_joints(r_at - a * 0.05, R_pen, 1.5)
    yield from Rr.move(r_at, None, 1.2)
    yield from arrive(Rr, tol=0.002)
    yield from Rr.set_grip(0.0, 0.8)
    yield from wait(0.3)
    _dbg(ctx, "taken")
    pull_from = Lr.pos.copy()
    for k in range(24):                                  # the pull, 1.25 mm a step: the fit gives at 4 N
        if not S["attached"]:
            break
        yield from Lr.move(Lr.pos + a * 0.00125, None, 0.05)
    tl, _ = tcp_actual(Lr)
    yield from Lr.move(tl, None, 0.2)                    # off: take up the arm's lag where the hand is, so it doesn't
    yield from wait(0.3)                                 # spring forward with the cap
    yield from Lr.move(pull_from + a * 0.030, None, 0.8)                 # draw it clear of the neck
    yield from Lr.move(pull_from + a * 0.050 + up * 0.03, None, 0.8)     # the cap held clear, open end toward the stick
    _dbg(ctx, "after the pull")
    hand_tcp, hand_R = tcp_actual(Rr)

    # 2. over the mouth, tip down ----------------------------------------------------------------------------------------
    held = Held.measured(ctx)
    plan = np.load(pathlib.Path(__file__).resolve().parent / "assets" / "stroke_plan.npz")
    r_b = g.BULLET_R
    f_target = {"light": 0.15, "hard": 3.0}.get(name, F_TARGET)   # light: a graze under the 0.3 N that leaves balm

    def frame_at(lip, t):
        """The planned stick frame, dome-centre point and inward normal at fractional index t along a lip."""
        P, Nn, A, SP = (plan[f"{lip}_{k}"] for k in ("pts", "nrm", "axis", "spin"))
        k = int(min(max(math.floor(t), 0), len(P) - 2))
        w = float(np.clip(t - k, 0.0, 1.0))
        pt = P[k] * (1 - w) + P[k + 1] * w
        n = Nn[k] * (1 - w) + Nn[k + 1] * w
        n /= np.linalg.norm(n)
        a = A[k] * (1 - w) + A[k + 1] * w
        a /= np.linalg.norm(a)
        d = (SP[k + 1] - SP[k] + math.pi) % (2 * math.pi) - math.pi
        spin = SP[k] + w * d
        hint = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        S_ = turn(a, spin) @ stick_frame(a, np.cross(hint, a))
        if name == "sloppy" and lip == "upper":
            pt = pt + np.array([0.0, 0.004, 0.0])
        return S_, pt, n

    side = {"upper": np.zeros(3), "lower": np.zeros(3)}  # the measured contact's offset across the lip, fed back
    from scipy.spatial import cKDTree
    skin_tree = cKDTree(task.SKIN)                       # the skin outside the band (balm within 3 mm of it voids the run)
    all_tree = cKDTree(task.ALL)

    def across(lip, t):
        """The unit direction across the lip at t, toward the other lip, square to the line and the normal."""
        P = plan[f"{lip}_pts"]
        k = int(min(max(math.floor(t), 0), len(P) - 2))
        _, _, n = frame_at(lip, t)
        d = np.cross(n, P[k + 1] - P[k])
        d /= np.linalg.norm(d)
        toward = -1.0 if lip == "upper" else 1.0       # the upper lip's join is below it (-y), the lower's above
        return d if d[1] * toward > 0 else -d

    offset = {"v": TOWARD_JOIN}                          # the aim's offset toward the join (negative: toward the border)

    def half_height(lip, t):
        """Half the lip's height across it at t, from the lip samples (m)."""
        P = plan[f"{lip}_pts"]
        k = int(min(max(round(t), 0), len(P) - 1))
        sel = (task.LIP_REGION >= 6) if lip == "lower" else ((task.LIP_REGION >= 0) & (task.LIP_REGION < 6))
        u = (task.LIP - P[k]) @ g.LIP_LINE
        m = sel & (np.abs(u) < 0.001)
        if m.sum() < 2:
            return 0.002
        d = across(lip, t)
        w = (task.LIP[m] - P[k]) @ d
        return float((w.max() - w.min()) / 2)

    def aim_offset(lip, t):
        off = offset["v"]
        if off < 0:                                      # toward the border: keep 1.2 mm of lip outside the aim
            off = -min(-off, max(0.0, half_height(lip, t) - 0.0012))
        return off

    def pose(lip, t, depth):
        S_, pt, n = frame_at(lip, t)
        pt = pt + across(lip, t) * aim_offset(lip, t) - side[lip]
        c = pt + n * (r_b - depth)
        return held.tcp_for(c - S_[:, 1] * task.DOME_C, S_)

    def q_plan(lip, t):
        """The plan's joint angles at fractional index t (interpolated between planned points)."""
        Q = plan[f"{lip}_q"]
        k = int(min(max(math.floor(t), 0), len(Q) - 2))
        w = float(np.clip(t - k, 0.0, 1.0))
        return Q[k] * (1 - w) + Q[k + 1] * w

    corr = np.zeros(3)                                   # the arm's measured tracking error, fed back

    def aim_seeded(lip, t, depth, feedback=True):
        """Track a pose this step, solved from the plan's own joint angles there (so it never jumps branches), with
        the tool point's measured offset from its last target fed back."""
        from yam_arena import ik
        nonlocal corr
        tcp, Rt = pose(lip, t, depth)
        if feedback:
            tcp_a, _ = tcp_actual(Rr)
            corr = np.clip(corr + 0.3 * (tcp_a - Rr.pos), -0.012, 0.012)
            tcp = tcp - corr
        else:
            corr = np.zeros(3)
        seed = q_plan(lip, t)
        if np.abs(Rr.q - seed).max() < 0.3:
            seed = Rr.q
        q, e, r = ik.solve(tcp, ik.matrix_to_quat_xyzw(Rt), q_init=seed, base_pos=Rr.base, iters=60)
        if e > 0.002:
            q, e, r = ik.solve(tcp, ik.matrix_to_quat_xyzw(Rt), q_init=q_plan(lip, t), base_pos=Rr.base, iters=150)
        Rr.q, Rr.rot = q, Rt
        Rr.pos = tcp + corr                                  # the target itself, for the next step's error
        Rr.worst_mm = max(Rr.worst_mm, e * 1000)

    def go_planned(lip, t, depth, seconds):
        """Joint-space move to a planned pose, solved from the plan's own joint angles (keeps its arm posture)."""
        from yam_arena import ik
        tcp, Rt = pose(lip, t, depth)
        q, e, r = ik.solve(tcp, ik.matrix_to_quat_xyzw(Rt), q_init=q_plan(lip, t), base_pos=Rr.base, iters=200)
        q0, g0 = Rr.q.copy(), Rr.grip
        n = max(1, round(seconds * 20))
        from yam_arena.ik import smoothstep
        for k in range(1, n + 1):
            s_ = smoothstep(k / n)
            Rr.q = q0 + s_ * (q - q0)
            yield
        Rr.pos, Rr.rot = tcp, Rt

    yield from Rr.move(Rr.pos + up * 0.10, None, 1.2)                    # up, well clear of the face
    yield from go_planned("upper", 0.0, -0.040, 3.0)
    yield from arrive(Rr, timeout=2.0)
    _dbg(ctx, "over the mouth")

    # 3. stroke each lip --------------------------------------------------------------------------------------------------
    def approach(lip, t, back):
        """From ``back`` m off the lip along its normal, come in until the tip bears on it; returns the depth."""
        depth = -back
        for _ in range(int(back / 0.00025) + 60):
            if S["tip_force"] >= 0.6 * f_target:
                break
            depth += 0.00025 if S["head_other"] < 0.05 else -0.001
            aim_seeded(lip, t, depth)
            yield
        return depth

    def stroke(lip, from_air=True, reverse=False, upto=None):
        n_pts = len(plan[f"{lip}_pts"])
        t0, t1 = T_IN, n_pts - 1 - T_IN
        if reverse:
            t0, t1 = t1, t0
        if upto is not None:                             # a short pass: from the start to this share of the line
            t1 = t0 + upto * (t1 - t0)
        if from_air:
            yield from go_planned(lip, t0, -0.030, 2.0)
        yield from arrive(Rr, tol=0.002, timeout=2.0)
        for k in range(20):                              # in along the normal to 8 mm off the lip
            aim_seeded(lip, t0, -0.030 + 0.022 * (k + 1) / 20)
            yield
        depth = yield from approach(lip, t0, 0.008)
        # the timing along the lip: no faster than STROKE_SPEED, and no joint faster than JOINT_SPEED
        Q = plan[f"{lip}_q"]
        seg = [max(0.002 / STROKE_SPEED, float(np.abs(Q[i + 1] - Q[i]).max()) / JOINT_SPEED) for i in range(len(Q) - 1)]
        times = np.concatenate([[0.0], np.cumsum(seg)])
        T0, T1 = np.interp(t0, np.arange(len(times)), times), np.interp(t1, np.arange(len(times)), times)
        # (reversed, T1 < T0: the same timing, run backward)
        steps = max(2, round(abs(T1 - T0) * 20))
        for k in range(steps + 1):                      # along the lip, holding the force
            t = float(np.interp(T0 + k / steps * (T1 - T0), times, np.arange(len(times))))
            if S["contact"] is not None:              # keep the contact on the planned line, across the lip
                _, pt, n = frame_at(lip, t)
                e = S["contact"] - (pt + across(lip, t) * aim_offset(lip, t))
                d = across(lip, t)
                side[lip] = np.clip(side[lip] + 0.3 * (e @ d) * d, -0.003, 0.003)
            near = S["contact"]
            if near is None:                             # not touching: where the dome would touch, the sample nearest it
                from scenekit import env
                near = task.ALL[all_tree.query(task.stick_pose(env())[4])[1]]
            ds = skin_tree.query(near)[0]
            if ds < SKIN_GUARD:
                side[lip] = side[lip] - across(lip, t) * (0.0008 if ds >= 0.0039 else 0.0015)   # nearing the border:
                if ds < 0.0039:                                                                # aim further in
                    depth -= 0.0003                                                            # (and ease off)
            if S["tip_force"] > 1.3:
                depth -= 0.0015 + 0.0015 * (S["tip_force"] - 1.3)     # a sudden hard press: back off fast
            elif name == "light" and S["tip_force"] > 0.20:
                depth -= 0.0006                                       # light: never reach the 0.3 N that leaves balm
            elif S["head_other"] > 0.05 or S["scale_avg"] > 0.40 or S["tip_force"] > 1.0:
                depth -= 0.0005 + 0.0004 * max(0.0, S["tip_force"] - 1.0)   # too hard, or not the balm: back off
            else:
                depth = float(np.clip(depth + 0.0003 * (f_target - S["tip_force"]), -0.004, 0.003))
            aim_seeded(lip, t, depth)
            if DEBUG and k % 10 == 0:
                tcp_a, R_a = tcp_actual(Rr)
                p_, a_, _, _ = stick(ctx)
                slip = math.degrees(math.acos(np.clip(a_ @ (R_a @ held.S_in[:, 1]), -1, 1)))
                shift = (R_a.T @ (p_ - tcp_a) - held.p_in) * 1000
                print(f"[stroke {lip} t {t:5.2f}] f {S['tip_force']:4.2f} depth {depth * 1000:5.1f} mm  stick turned "
                      f"{slip:4.1f} deg in the hand, base shifted {np.round(shift, 1)} mm  tcp err "
                      f"{np.linalg.norm(tcp_a - Rr.pos) * 1000:4.1f} mm", flush=True)
            yield
        for k in range(16):                              # off the lip, straight out along its normal
            aim_seeded(lip, t1, depth - 0.033 * (k + 1) / 16)
            yield

    def hop(lip, t_at=None):
        """From one lip's end to the next lip's start (or index t_at) through the air: up and back off the face, across,
        and down."""
        from yam_arena import ik
        from yam_arena.ik import smoothstep
        lift = g.FACE_N * 0.03 + up * 0.04
        yield from Rr.move(Rr.pos + lift, None, 1.0)
        n_pts = len(plan[f"{lip}_pts"])
        t_at = T_IN if t_at is None else t_at
        tcp, Rt = pose(lip, t_at, -0.030)
        q, e, r = ik.solve(tcp + lift, ik.matrix_to_quat_xyzw(Rt), q_init=q_plan(lip, t_at), base_pos=Rr.base, iters=200)
        q0 = Rr.q.copy()
        for k in range(1, 41):
            Rr.q = q0 + smoothstep(k / 40) * (q - q0)
            yield
        Rr.pos, Rr.rot = tcp + lift, Rt
        yield from Rr.move(tcp, Rt, 1.0)

    yield from stroke("upper")
    _dbg(ctx, "upper done")
    if name == "upper":
        yield from hop("lower")                          # off to the other side through the air, as after the lower lip
    if name != "upper":
        offset["v"] = 0.0                                # the lower lip's centre line (keep off the mouth's slit)
        yield from hop("lower")
        yield from stroke("lower", from_air=False)
        _dbg(ctx, "lower done")
        # a second pass back along the lower lip, nearer its outer edge (it is 10 mm tall; one pass covers about 60%)
        offset["v"] = 0.0 if name == "light" else -0.0015
        side["lower"] = np.zeros(3)
        yield from stroke("lower", from_air=False, reverse=True)
        offset["v"] = TOWARD_JOIN
        _dbg(ctx, "lower back")
        if EXTRA_UPPER and name == "full":
            # a short second pass over the upper lip's first third (its thinnest regions)
            offset["v"] = TOWARD_JOIN
            side["upper"] = np.zeros(3)
            yield from hop("upper")
            yield from stroke("upper", from_air=False, upto=0.40)
            _dbg(ctx, "upper again")
            if EXTRA_UPPER_END:                          # and over its last region, from the other corner
                side["upper"] = np.zeros(3)
                yield from hop("upper", len(plan["upper_pts"]) - 1 - T_IN)
                yield from stroke("upper", from_air=False, reverse=True, upto=0.25)
                _dbg(ctx, "upper end again")
            offset["v"] = TOWARD_JOIN

    # 4. recap ---------------------------------------------------------------------------------------------------------
    yield from Rr.move(Rr.pos + g.FACE_N * 0.03 + up * 0.06, None, 1.2)     # off the face
    yield from Rr.move(np.array([0.20, -0.20, TOP + 0.26]), None, 1.8)       # to the right, in front of the head
    if name == "held":
        yield from wait(4.0)
        return
    # the stick back level where the cap came off; the left hand brings the cap back along its axis
    # (the stick is turned level first out at the front right, clear of the face, then brought in)
    yield from Rr.move_joints(np.array([0.10, -0.22, TOP + 0.26]), hand_R, 2.0)
    yield from Rr.move_joints(hand_tcp + up * 0.03 - np.array([0.0, 0.03, 0.0]), hand_R, 2.0)
    yield from Rr.move(hand_tcp, hand_R, 1.0)
    yield from arrive(Rr, tol=0.002)
    def rim_error(gap):
        p, a, cp, ca = stick(ctx)
        return p + a * (g.TUBE_L + gap) - cp, ca, a

    for _ in range(6):                                   # line the cap up on the stick's axis, rim 16 mm short
        err, ca, a = rim_error(0.016)
        ang = math.degrees(math.acos(np.clip(ca @ a, -1, 1)))
        if np.linalg.norm(err) < 0.0005 and ang < 1.5:
            break
        yield from Lr.move(Lr.pos, _align(ca, a) @ Lr.rot, 0.4)           # turn first, about the tool point
        yield from wait(0.2)
        err, ca, a = rim_error(0.016)
        yield from Lr.move(Lr.pos + err, None, 0.5)                       # then slide onto the axis
        yield from wait(0.2)
    for gap in (0.010, 0.006, 0.003, 0.0, -0.001, -0.002, -0.003):   # push the cap home along the axis, in line
        if gap == "regrip":                              # the cap is on the neck now: take a new hold further out
            p, a, cp, ca = stick(ctx)
            yield from Lr.set_grip(GRIP_OPEN, 0.4)
            yield from Lr.move(Lr.pos + a * REGRIP, None, 0.5)
            yield from Lr.set_grip(CAP_HOLD, 0.5)              # just closed on the cap: a light hold to push with
            yield from wait(0.2)
            continue
        p, a, cp, ca = stick(ctx)
        if S["attached"]:
            break                                        # pushed home in line: the fit has taken
        if RIGHT_PUSHES:                                 # the left holds the cap still; the right pushes the tube in
            err, ca, a = rim_error(gap)
            yield from Rr.move(Rr.pos - err, None, 0.6)
            yield from wait(0.2)
        else:
            yield from Lr.move(Lr.pos, _align(ca, a) @ Lr.rot, 0.3)
            yield from wait(0.15)
            err, ca, a = rim_error(gap)
            start = Lr.pos.copy()
            for k in range(1, 13):                       # in 12 steps, stopping the moment the fit seats
                if S["attached"]:
                    break
                yield from Lr.move(start + err * k / 12, None, 0.05)
            if S["attached"]:
                break
            yield from wait(0.2)
        if DEBUG:
            p, a, cp, ca = stick(ctx)
            tl, Rl_ = tcp_actual(Lr)
            print(f"[push gap {gap * 1000:4.1f}] rim {((cp - p) @ a) * 1000:5.1f} mm  off {task.cap_relation(p, a, cp, ca)[1] * 1000:4.1f} mm  tilt "
                  f"{math.degrees(math.acos(np.clip(ca @ a, -1, 1))):4.1f}  left err {np.linalg.norm(tl - Lr.pos) * 1000:4.1f} mm"
                  f"  cap in hand {np.round(Rl_.T @ (cp - tl) * 1000, 1)}", flush=True)
    for _ in range(8):                                   # not taken yet: put it in line at its last mm
        if S["attached"]:
            break
        err, ca, a = rim_error(-0.0005)
        yield from Lr.move(Lr.pos, _align(ca, a) @ Lr.rot, 0.3)
        err, ca, a = rim_error(-0.0005)
        yield from Lr.move(Lr.pos + err, None, 0.4)
        yield from wait(0.15)
        if DEBUG:
            p, a, cp, ca = stick(ctx)
            print(f"[line up] {np.round(task.cap_relation(p, a, cp, ca), 4)} attached {S['attached']}", flush=True)
    yield from wait(0.3)
    _dbg(ctx, "recapped")
    # the left hand lets go of the cap and withdraws beyond it
    if RIGHT_PUSHES:
        # the right lets go and withdraws; the left lays the capped stick on the tray's floor by its cap, lets go
        tr, _ = tcp_actual(Rr)
        yield from Rr.move(tr, None, 0.3)
        yield from Rr.set_grip(PEN_OPEN, 0.5)
        p, a, cp, ca = stick(ctx)
        yield from Rr.move(Rr.pos - a * 0.05, None, 0.8)
        yield from Rr.move_joints(np.array([0.13, -0.20, TOP + 0.22]), Rr.rot, 1.5)
        yield from Rr.go_home(2.0)
        p, a, cp, ca = stick(ctx)
        centre = p + a * (g.CAPPED_L / 2)
        goal = np.array([g.STICK_C[0], g.STICK_C[1], TOP + g.TRAY_T + g.TUBE_R + 0.0005])
        yield from Lr.move(Lr.pos + (goal - centre) * np.array([1, 1, 0]) + up * 0.01, None, 1.5)
        p, a, cp, ca = stick(ctx)
        yield from Lr.move(Lr.pos + np.array([0, 0, goal[2] - (p[2] + cp[2]) / 2]), None, 1.2)
        yield from wait(0.3)
        yield from Lr.set_grip(1.0, 0.6)
        yield from wait(0.4)
        yield from Lr.move(Lr.pos + up * 0.06, None, 1.0)
        yield from Lr.go_home(2.0)
        yield from wait(2.0)
        _dbg(ctx, "end")
        return
    tl, _ = tcp_actual(Lr)
    yield from Lr.move(tl, None, 0.4)                    # stop pushing (the arm was held back by the seated cap)
    p, a, cp, ca = stick(ctx)
    def left_dbg(where):
        if DEBUG:
            from scenekit import arm_joints, env
            from yam_arena.constants import FINGER_JOINTS
            p_, a_, cp_, ca_ = stick(ctx)
            tl, Rl_ = tcp_actual(Lr)
            print(f"[{where}] left fingers {np.round(arm_joints(env(), 'left', FINGER_JOINTS) * 1000, 1)} mm  cap rim in "
                  f"hand {np.round(Rl_.T @ (cp_ - tl) * 1000, 1)}  rim on tube {((cp_ - p_) @ a_) * 1000:5.1f} mm", flush=True)

    left_dbg("before release")
    yield from Lr.set_grip(1.0, 0.6)                     # fully open, so the cap is let go of cleanly
    yield from wait(0.5)
    left_dbg("opened")
    yield from Lr.move(Lr.pos + up * 0.04, None, 0.8)
    left_dbg("lifted")
    yield from Lr.move(Lr.pos + a * 0.04, None, 0.6)
    yield from Lr.go_home(2.0)

    # 5. the right hand lays the capped stick in the tray: level, 2 cm over its floor, and lets go -------------------
    p, a, cp, ca = stick(ctx)
    centre = p + a * (g.CAPPED_L / 2)
    goal = np.array([g.STICK_C[0], g.STICK_C[1], TOP + g.TRAY_T + g.TUBE_R + DROP])
    yield from Rr.move(Rr.pos + (goal - centre) * np.array([1, 1, 0]), None, 1.5)
    yield from Rr.move(Rr.pos + (goal - centre) * np.array([0, 0, 1]), None, 1.2)
    yield from wait(0.3)
    yield from Rr.set_grip(PEN_OPEN, 0.4)
    yield from wait(0.4)
    yield from Rr.move(Rr.pos - a * 0.05, None, 0.8)
    yield from Rr.move(Rr.pos + up * 0.06, None, 0.8)
    _dbg(ctx, "dropped")
    yield from Rr.move_joints(np.array([0.13, -0.20, TOP + 0.22]), Rr.rot, 1.5)
    yield from Rr.go_home(2.0)
    yield from wait(2.0)
    _dbg(ctx, "end")


def _dbg(ctx, where: str) -> None:
    import task
    from scenekit import env

    p, a, q, vel, dome = task.stick_pose(env())
    cp, ca, _ = task.cap_pose(env())
    s, off, ang = task.cap_relation(p, a, cp, ca)
    tcp, _ = tcp_actual(ctx.right)
    cov = np.bincount(task.LIP_REGION[task._STATE["marked"] & (task.LIP_REGION >= 0)], minlength=12) / task.REGION_SIZE
    print(f"[{where}] coverage % upper {np.round(cov[:6] * 100).astype(int).tolist()} lower "
          f"{np.round(cov[6:] * 100).astype(int).tolist()}", flush=True)
    print(f"[{where}] stick base {np.round(p, 3)} axis {np.round(a, 2)}  cap rim at {s * 1000:.1f} mm, off-axis "
          f"{off * 1000:.1f} mm, {ang:.1f} deg  right tcp {np.round(tcp, 3)}  cmd {np.round(ctx.right.pos, 3)}  "
          f"why {task._STATE['why']}", flush=True)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    return (f"scale {S['scale_avg'] * 102:5.0f} g (peak {S['peak_scale'] * 102:4.0f})  tip {S['tip_force']:4.2f} N "
            f"(peak {S['peak_tip']:4.2f})  coated {int(S['coated'][:6].sum())}+{int(S['coated'][6:].sum())}  "
            f"marked {int(S['marked'].sum())}  cap_off {S['cap_off']}  seated {S['seated']}  tray {S['in_tray']}  "
            f"off {S['hands_off']}  still {S['still']}  ok {S['ok_steps']}  why {S['why']}")


def check():
    """Offline (no simulator): IK for every planned pose with the nominal grasp, for a few tilts and spins."""
    import types
    from yam_arena import analysis as A, ik, station
    from yam_arena import constants as C
    sys.modules.setdefault("task", None)
    import importlib.util
    spec = importlib.util.spec_from_file_location("task", pathlib.Path(__file__).resolve().parent / "task.py")
    task = importlib.util.module_from_spec(spec)
    sys.modules["task"] = task
    spec.loader.exec_module(task)
    base_r = np.array(station.BASES["right"], float)
    p = np.array(task._BASE)
    a = np.array([0.0, 1.0, 0.0])
    Rg = grip_frame(a)
    tcp0 = p + a * R_GRIP_S + np.array([0, 0, ABOVE])
    held = Held(tcp0, Rg, p, a)
    lines = lip_lines()
    import trimesh
    h = np.load(pathlib.Path(__file__).resolve().parent / "build" / "head.npz", allow_pickle=True)
    pts = trimesh.Trimesh(h["V"], h["F"], process=False).sample(20000) + task.HEAD_POS
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    from demokit import Arm
    seeds = [np.array(C.READY, float)] + [np.array(q, float) for q in Arm.SEEDS]
    seeds += [q + np.array([0, 0, 0, 0, 0, d]) for q in list(seeds) for d in (-1.5, 1.5)]
    seeds += [q + np.array([0, 0, 0, 0, d, 0]) for q in list(seeds[:6]) for d in (-1.0, 1.0)]
    for tilt in (0, 10, 20, 30, 40):
        for spin in range(-180, 180, 30):
            S = apply_frame(math.radians(tilt), math.radians(spin))
            targets = []
            for name in ("upper", "lower"):
                for c, n in zip(*lines[name]):
                    targets.append(held.tcp_for(c + n * g.BULLET_R - S[:, 1] * task.DOME_C, S))
            best = None
            for q0 in seeds:
                q, e, r = ik.solve(targets[0][0], ik.matrix_to_quat_xyzw(targets[0][1]), q_init=q0, base_pos=base_r,
                                   iters=200)
                if best is None or e + r < best[1]:
                    best = (q, e + r)
            q = best[0]
            worst, clear = 0.0, 1.0
            for tcp, Rt in targets:
                q, e, r = ik.solve(tcp, ik.matrix_to_quat_xyzw(Rt), q_init=q, base_pos=base_r, iters=100)
                worst = max(worst, e * 1000 + math.degrees(r))
            if worst < 2:
                for tcp, Rt in targets[::6]:
                    q, e, r = ik.solve(tcp, ik.matrix_to_quat_xyzw(Rt), q_init=q, base_pos=base_r, iters=100)
                    qd = dict(zip(C.ARM_JOINTS, q)) | {k: -0.0075 for k in C.FINGER_JOINTS}
                    for _, hh in A.world_hulls(A.fk_link_poses_world(qd, base_r)):
                        clear = min(clear, float(tree.query(hh.vertices)[0].min()))
                print(f"tilt {tilt:2d} spin {spin:4d}: worst IK mm+deg {worst:6.2f}  arm to head {clear * 1000:5.1f} mm",
                      flush=True)
        print("done tilt", tilt, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="full")
    a = ap.parse_args()
    VARIANT["name"] = a.variant
    if not a.no_record:
        __import__("os").environ.setdefault("LB_SHOW_BALM", "1")   # recorded runs draw the balm marks (display only)
    run(a.task, a.out, script, max_steps=a.max_steps, record=not a.no_record, log=log, log_every=20)
