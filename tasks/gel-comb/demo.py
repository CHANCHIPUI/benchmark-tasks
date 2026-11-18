"""Scripted full-score demo. The right hand presses its two open fingertips on the tray's side walls past the middle
and holds it on the bench; the left settles its pads over the comb's bar (measured tool point within a few tenths of a
millimetre, pads turned to the bar), closes on it and draws it straight up at 3 mm/s until it gives, then slowly
(4 mm/s, lead capped) until the teeth are 2 mm clear of the gel; then lifts it away and lays it on the tissue. The
right lets go, takes the tray by its two open ends (pads on the end faces, a light pinch), lifts it 2 cm, carries it
level over the tank while turning it 90 degrees clockwise (the wells end from the robot's side to the black end),
lowers it onto the gel stage with its fingers going down beside the tray's ends, opens 5 mm, rises clear and goes home.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record] [--variant NAME]

Checks of the scorer (the same script with one change): ``nohold`` leaves the tray unheld during the pull;
``sideways`` drags the comb across its thickness during the pull (the hand commanded 2 mm sideways over 1 s);
``red`` has the left arm carry the tray, turning it 90 degrees counter-clockwise (wells at the red end); ``fast`` and
``yank`` pull at 60 and 150 mm/s; ``nothing`` does nothing.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as g  # noqa: E402
from demokit import run, together, turn, wait  # noqa: E402

DEBUG = bool(os.environ.get("GEL_DEBUG"))
T = g.TOP
HOLD_X, HOLD_PITCH = 0.010, math.radians(40)       # hold-down fingertips on the wall tops (dev/gate5.py)
G_PRESS = 0.072 / 0.094             # pads 72 mm apart: the fingertips over the 4 mm walls' tops
PRESS_DZ = float(os.environ.get("GEL_PRESS", "0.003"))          # commanded this far below the wall tops
COMB_DZ = 0.014                     # tool point over the bar's bottom edge: fingertips 3 mm over the gel
END_DZ = 0.014                      # tool point over the tray's bottom in the end grip: tips 1 mm above its bottom
SQUEEZE = float(os.environ.get("GEL_SQUEEZE", "0.0006"))         # m per pad past the end faces: ~3 N at 5000 N/m
PULL_V = 0.003                      # m/s
PULL_TO = 0.010                     # m: teeth 6 mm clear of the gel
PULL_OUT = 0.006                    # m: once released, drawn at PULL_V_OUT until the teeth are 2 mm clear, then lifted
PULL_V_OUT = float(os.environ.get("GEL_VOUT", "0.004"))
LEAD_OUT = float(os.environ.get("GEL_LEAD_OUT", "0.0016"))       # m, the command's lead once the comb has let go
WIGGLE_OUT = float(os.environ.get("GEL_WIGGLE_OUT", "0.00015"))  # m, the wiggle then (3.8 mm/s at its fastest)
X_BIAS = float(os.environ.get("GEL_XBIAS", "0.0001"))           # m: the hand lags ~0.2 mm in -x while rising
LEAD_MAX = float(os.environ.get("GEL_LEAD", "0.004"))            # the command leads the stuck comb by at most this
                                    # (the arm needs ~2.4 mm of lead to pull 1 N: its joints are soft)
WIGGLE = float(os.environ.get("GEL_WIGGLE", "0.0005"))            # m, vertical wiggle amplitude during the pull
WIGGLE_HZ = 4.0
CARRY_DZ = g.BASE_H + 0.012         # the tray's bottom 12 mm over the tank's rim while carried


def grip_rot(yaw_deg: float) -> np.ndarray:
    from yam_arena.demo import grip_rotation

    return grip_rotation(math.radians(yaw_deg))


def tcp_actual(arm):
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    from scenekit import arm_joints, env

    return ik.tcp_pose(arm_joints(env(), arm.side, ARM_JOINTS), arm.base)


def arrive(arm, tol=0.002, timeout=2.0):
    for _ in range(round(timeout * 20)):
        if np.linalg.norm(tcp_actual(arm)[0] - arm.pos) < tol:
            return
        yield
    if DEBUG:
        print(f"arrive: {arm.side} still {np.linalg.norm(tcp_actual(arm)[0] - arm.pos) * 1000:.1f} mm off", flush=True)


def rot_err(want, Ra):
    """The rotation from ``want`` to ``Ra`` as a rotation vector (rad), in the world frame."""
    E = Ra @ want.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(E) - 1) / 2)))
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2 * math.sin(ang))
    return axis * ang


def settle(arm, goal, want, rounds=8):
    """Settle the measured tool point on ``goal`` (x/y to about 0.1 mm) and the hand on ``want`` (to about 0.15 deg).
    The arm's joints stick (static joint friction: about a millimetre and a third of a degree at the hand), short of
    the command on the side they come from. Each round backs the command off (2 mm, 0.6 deg about the vertical) and
    brings it back slowly, so the joints always come to rest from the same side, then corrects the command by the
    measured error."""
    cmd, Rc = arm.pos.copy(), arm.rot.copy()
    for k in range(rounds):
        yield from arm.move(cmd - [0.002, 0.002, 0.0], turn([0, 0, 1], -0.01) @ Rc, 0.4)
        yield from arm.move(cmd, Rc, 0.8)
        yield from wait(0.4)
        pa, Ra = tcp_actual(arm)
        err, rv = goal - pa, rot_err(want, Ra)
        if DEBUG:
            print(f"  settle {k}: err {np.round(err * 1000, 2).tolist()} mm, rot {np.round(np.degrees(rv), 2).tolist()} deg",
                  flush=True)
        if abs(err[0]) < 0.00012 and abs(err[1]) < 0.0004 and abs(rv[2]) < math.radians(0.12):
            break
        cmd = cmd + np.array([err[0], err[1], 0.0])
        Rc = turn(rv / max(np.linalg.norm(rv), 1e-12), -np.linalg.norm(rv)) @ Rc
    arm.aim(cmd, Rc)


def pose(ctx, name):
    s = ctx.states()[name]
    return np.asarray(s.pos, float), ctx.task._rotation(s.quat_xyzw)


def comb_in_gel(ctx):
    """The comb's bar-centre offset across its thickness from the wells' line, and its two tilts, in the gel frame."""
    go, gR = pose(ctx, "gel")
    co, cR = pose(ctx, "comb")
    p = gR.T @ (co - go)
    rel = gR.T @ cR
    return p[0] - g.WELL_X, math.atan2(rel[0, 2], rel[2, 2]), math.atan2(rel[1, 2], rel[2, 2]), p[2]


# -- the hold-down ---------------------------------------------------------------------------------------------------------
def hold_down(ctx, arm):
    """Both open fingertips press down on the side walls' top edges, 10 mm past the tray's middle, the hand pitched
    40 degrees toward +x (dev/gate5.py: 10 mm between the arms). Pressing there (25 mm in from the far end) keeps the
    near end down against the comb's pull; a pinch near the far end lets the tray pivot about it."""
    to, _ = pose(ctx, "tray")
    ph = HOLD_PITCH
    R = turn([0, 1, 0], ph) @ grip_rot(0)
    tip = np.array([to[0] + HOLD_X, to[1], T + g.WALL_H])
    tcp = tip + g.PAD_TIP * np.array([math.sin(ph), 0.0, math.cos(ph)])
    yield from arm.move_joints(tcp + [0, 0, 0.05], R, 2.5, grip=G_PRESS)
    yield from arm.move(tcp + [0, 0, 0.008], R, 1.0)
    yield from arm.move(tcp - [0, 0, PRESS_DZ], R, 1.0)
    yield from wait(0.5)
    ctx.flags["held"] = True


def let_go_hold(ctx, arm):
    while not ctx.flags.get("comb_out"):
        yield
    yield from arm.move(arm.pos + [0, 0, 0.05], arm.rot, 1.0, grip=1.0)


# -- the comb ----------------------------------------------------------------------------------------------------------------
def pull_comb(ctx, arm, variant):
    co, _ = pose(ctx, "comb")
    R = grip_rot(-90)                                       # pads closing along x, across the bar's thickness
    p = np.array([co[0] + X_BIAS, co[1], co[2] + COMB_DZ])
    yield from arm.move_joints(p + [0, 0, 0.06], R, 2.5, grip=0.3)
    yield from arm.move(p + [0, 0, 0.015], R, 1.0)
    yield from arrive(arm, 0.0015)
    # set the measured tool point over the bar (the arm sags a little): correct the command by the error
    # and turn the hand so the measured pads lie along the bar (a yawed comb pushes its end teeth into their walls)
    co, cR = pose(ctx, "comb")
    want = turn([0, 0, 1], math.atan2(cR[1, 0], cR[0, 0])) @ R
    yield from settle(arm, p + [0, 0, 0.015], want)
    if DEBUG:
        Ra = tcp_actual(arm)[1]
        print(f"comb hand: pos err {np.round((p + [0, 0, 0.015] - tcp_actual(arm)[0]) * 1000, 2).tolist()} mm, "
              f"rot err {math.degrees(np.arccos(np.clip((np.trace(want.T @ Ra) - 1) / 2, -1, 1))):.2f} deg", flush=True)
    yield from arm.move(arm.pos + [0, 0, -0.015], arm.rot, 1.0)
    yield from settle(arm, p, want)
    R = arm.rot.copy()
    while not ctx.flags.get("held") and variant != "nohold":
        yield
    yield from arm.set_grip(0.0, 0.6)
    yield from wait(0.4)
    # draw it straight up, holding the teeth square in their wells (closed loop on offset and tilt)
    fast = variant in ("fast", "yank")
    v = {"fast": 0.060, "yank": 0.150}.get(variant, PULL_V)
    off0, _, _, z0 = comb_in_gel(ctx)
    cmd = arm.pos.copy()
    x_start = cmd[0]
    rot = R.copy()
    n, best, extra = 0, 0.0, 0.0
    while True:
        off, tilt_t, tilt_e, z = comb_in_gel(ctx)
        rise = z - z0
        if rise >= PULL_TO or (not fast and rise >= PULL_OUT):
            break
        n += 1
        hold = ctx.task._STATE["hold"]
        if fast:
            cmd[2] += v / 20
        elif hold == "on":
            cmd[2] = min(cmd[2] + v / 20, tcp_actual(arm)[0][2] + LEAD_MAX)
        elif hold == "releasing":
            pass                    # the comb is giving: hold the command; the arm's own lead (~2.4 mm at 1 N) lifts it
                                    # as the wells' hold fades
        else:
            # let go: draw the teeth out slowly, the command at most LEAD_OUT ahead of the hand; if the arm's joints
            # stick (no rise), let the lead grow by 0.01 mm a step until they slide again, then fall back
            if rise > best + 0.00005:
                best, extra = rise, max(0.0, extra - 0.0001)
            else:
                extra += 0.00001
            cmd[2] = min(cmd[2] + PULL_V_OUT / 20, tcp_actual(arm)[0][2] + LEAD_OUT + extra)
        if variant == "sideways":
            cmd[0] = x_start + min(0.002, 0.002 * n / 20)       # drags the bar across its thickness (the command
                                                                # 2 mm in 1 s; the sticking arm follows ~1 mm)
        # a gentle vertical wiggle while the comb is held keeps the arm's joints sliding: from rest they stick and
        # break loose one after another, and the hand jumps a few tenths of a millimetre sideways when they do;
        # cut to WIGGLE_OUT once the comb has let go, as the comb then follows the hand
        amp = 0.0 if fast else (WIGGLE * min(1.0, n / 10) if hold == "on" else WIGGLE_OUT if hold == "off" else 0.0)
        wig = amp * math.sin(2 * math.pi * WIGGLE_HZ * n / 20)
        arm.aim(cmd + [0, 0, wig], rot)                         # straight up: x, y and the hand's rotation held
        if DEBUG and n % 2 == 0:
            S = ctx.task._STATE
            pa = tcp_actual(arm)[0]
            print(f"   pull {n}: off {off * 1000:+.3f} z {rise * 1000:.2f} f {S['hold_f']:.2f} {S['hold']} cmd-act "
                  f"{np.round((cmd - pa) * 1000, 2).tolist()} max_off {S['max_off'] * 1000:.2f}", flush=True)
        yield
        if n > 400:
            break
    if DEBUG:
        S = ctx.task._STATE
        print(f"pulled: {n} steps, hold {S['hold']} released at {S['rel_force']} N, peak {S['hold_peak']:.2f} N, "
              f"max offset {S['max_off'] * 1000:.2f} mm, tilts {math.degrees(S['max_tilt_t']):.2f}/"
              f"{math.degrees(S['max_tilt_e']):.2f} deg, max up {S['max_up'] * 1000:.1f} mm/s, torn {S['torn_why']}",
              flush=True)
    ctx.flags["comb_out"] = True
    yield from arm.move(arm.pos + [0, 0, 0.05], rot, 1.0)          # 50 mm/s once clear
    yield from arm.move(arm.pos + [0, 0, 0.05], rot, 0.8)


def lay_comb(ctx, arm):
    R = turn([0, 1, 0], math.radians(-50)) @ grip_rot(-90)
    above = np.array([g.TISSUE_X, g.TISSUE_Y, T + 0.08])
    yield from arm.move_joints(above, R, 2.5)
    yield from arm.move(above + [0, 0, -0.045], R, 1.2)
    yield from wait(0.2)
    yield from arm.set_grip(0.4, 0.5)
    yield from wait(0.5)
    yield from arm.move(arm.pos + [0, 0, 0.06], R, 1.0)
    yield from arm.go_home(2.5)


# -- the carry ---------------------------------------------------------------------------------------------------------------
def carry(ctx, arm, variant):
    left = arm.side == "left"
    y0 = -90 if left else 90                                # pads closing along x, on the tray's open ends
    to, _ = pose(ctx, "tray")
    R0 = grip_rot(y0)
    g_end = (g.TRAY_L - 2 * SQUEEZE) / 0.094
    g_open = (g.TRAY_L + 0.016) / 0.094
    grasp = np.array([to[0], to[1], T + END_DZ])
    yield from arm.move_joints(grasp + [0, 0, 0.06], R0, 2.0, grip=g_open)
    yield from arm.move(grasp, R0, 1.5)
    yield from arrive(arm, 0.0015)
    err = grasp - tcp_actual(arm)[0]
    yield from arm.move(arm.pos + [err[0], err[1], 0.0], R0, 0.4)
    yield from arm.set_grip(g_end, 1.0)
    yield from wait(0.5)
    if DEBUG:
        S = ctx.task._STATE
        print(f"end grip: held {S['held']} end force max {S['max_end']:.2f} N", flush=True)
    # lift 20 mm slowly, then up over the rim, then across, turning 90 degrees (clockwise for the right hand)
    yield from arm.move(arm.pos + [0, 0, 0.020], R0, 2.0)
    yield from arm.move([arm.pos[0], arm.pos[1], T + CARRY_DZ + END_DZ], R0, 1.2)
    seat = np.array([g.TANK_X, g.TANK_Y, T + g.STAGE_TOP + END_DZ])
    R1 = grip_rot(0)
    yield from arm.move(seat + [0, 0, CARRY_DZ - g.STAGE_TOP], R1, 3.5)
    yield from arm.move(seat + [0, 0, 0.015], R1, 1.5)
    yield from arrive(arm, 0.0015)
    # correct the tray's measured offset and yaw over the stage
    want = np.array([0.0, 1.0 if left else -1.0])          # the tray's +x axis (away from the wells)
    for _ in range(2):
        to, tR = pose(ctx, "tray")
        err = np.r_[seat[:2] - to[:2], 0.0]
        ang = math.atan2(tR[1, 0], tR[0, 0]) - math.atan2(want[1], want[0])
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        R1 = turn([0, 0, 1], -ang) @ R1
        yield from arm.move(arm.pos + err, R1, 0.6)
        yield from arrive(arm, 0.001)
    # down the last 15 mm at 10 mm/s, onto the stage
    yield from arm.move([arm.pos[0], arm.pos[1], seat[2] - 0.0005], R1, 1.5)
    yield from wait(0.5)
    yield from arm.set_grip(g_end + 0.005 / 0.094, 0.5)
    yield from wait(0.3)
    yield from arm.move(arm.pos + [0, 0, 0.065], R1, 1.5)
    yield from arm.move(arm.pos + [-0.06, 0, 0.02], R1, 1.0)
    yield from arm.go_home(2.5)


def script(ctx, variant: str):
    if variant == "nothing":
        yield from wait(5.0)
        return
    L, R = ctx.left, ctx.right
    if variant == "nohold":
        ctx.flags["held"] = True
        yield from pull_comb(ctx, L, variant)
        yield from wait(2.0)
        return
    yield from together(hold_down(ctx, R), pull_comb(ctx, L, variant))
    yield from together(let_go_hold(ctx, R), lay_comb(ctx, L))
    if variant == "red":
        yield from R.go_home(2.0)
        yield from carry(ctx, L, variant)
    else:
        yield from R.move_joints(R.pos + [0, 0, 0.03], R.rot, 1.0)
        yield from carry(ctx, R, variant)
    yield from wait(2.0)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    off, tt, te, z = comb_in_gel(ctx)
    to, _ = pose(ctx, "tray")
    return (f"hold {S['hold']} f {S['hold_f']:.2f} peak {S['hold_peak']:.2f} rel {S['rel_force']} | comb off "
            f"{off * 1000:.2f} mm tilt {math.degrees(tt):.2f}/{math.degrees(te):.2f} z {z * 1000:.1f} in_gel {S['in_gel']} "
            f"max_off {S['max_off'] * 1000:.2f} max_up {S['max_up'] * 1000:.1f} lift {S['max_lift'] * 1000:.2f} | "
            f"torn {sum(S['torn'])} {[w for w in S['torn_why'] if w][:2]} | tray {np.round(to, 3).tolist()} held {S['held']} "
            f"press {np.round(S['press'], 1).tolist()} Ract {np.round((tcp_actual(ctx.right)[0] - ctx.right.pos) * 1000, 1).tolist()} "
            f"slide {S['slide'] * 1000:.1f} top {S['max_top']:.2f} end {S['max_end']:.2f} | out {S['comb_out']} "
            f"corners {np.round(S['corners'] * 1000, 1).tolist()} black {S['black']} gap {S['hand_gap'] * 1000:.0f} "
            f"rung {S['rung']} why {S['why']} | L {np.round(ctx.left.pos, 3).tolist()} R {np.round(ctx.right.pos, 3).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="full", choices=["full", "nohold", "sideways", "fast", "yank", "red", "nothing"])
    a = ap.parse_args()
    run(a.task, a.out, lambda ctx: script(ctx, a.variant), max_steps=a.max_steps, record=not a.no_record, log=log,
        log_every=int(os.environ.get("GEL_LOG_EVERY", "20")))
