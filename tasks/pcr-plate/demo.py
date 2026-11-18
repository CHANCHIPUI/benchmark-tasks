"""Scripted full-score demo, all with the left arm (the cycler stands in front of it). The hand takes the plate from its
support base by its two long edges (pads on the semi-skirt, fingers closing front to back), lifts it out, carries it
low across the cycler's surround to over the block, lowers it until the wells are in the block's wells, lets go 1.5 mm
above seated, and taps both ends down with closed fingertips. Then it takes the cover's handle bar (hand pitched 45
degrees forward-down), pulls the cover forward to its stop, and pulls the lever down: the fingers close fully and
squeeze the bar; through the first half the hand turns with the lever, for the rest it stays level while the bar turns
in the pads down to its stop (the lever's spring pulls it back toward up until it latches there). It lets go and goes
home.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record] [--variant NAME]

Variants (for checking the scorer): ``ridges`` lets go of the plate half a pitch off (diagonally), on the ridges, then
closes the cover; ``turned`` puts the plate in turned 180 degrees; ``short`` leaves the cover 3 cm short of its stop;
``lever_up`` slides the cover fully forward but leaves the lever up; ``nothing`` does nothing; ``nudge_ridges`` first
pulls the cover 3 mm forward and lets go, then lets go of the plate on the ridges, then closes the cover.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json.
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
from demokit import run, turn, wait  # noqa: E402

DEBUG = bool(os.environ.get("PCR_DEBUG"))
T = g.TOP
YAW = math.radians(90)              # the hand turned so its fingers close along x (across the plate's width)
G_OPEN = 1.0                        # pads 94 mm apart
G_HOLD = 0.85                       # pads commanded to 80 mm on the 85.5 mm plate (a firm squeeze)
PAST = float(os.environ.get("PCR_PAST", "12"))          # degrees: the arc's command runs this far past the stop
LEAD = float(os.environ.get("PCR_LEAD", "0.03"))       # m: and leads it downward this far by its end
G_LEVER = float(os.environ.get("PCR_G_LEVER", "0.0"))     # the grip while pulling the lever against its spring
TIP = 0.013                         # the pad tips past the tool point
GRASP_DZ = 0.006                    # tool point over the deck's top when holding the plate: pad tips 7 mm below it
OVER = 0.017                        # carried with the deck 17 mm above seated: the tips 4.7 mm over the block's top
RELEASE = 0.0015                    # let go with the deck 1.5 mm above seated
BAR_Y = g.CENTRE_Y + 0.05           # where the hand takes the handle bar (5 cm left of its middle)


def grip_rot(yaw=YAW) -> np.ndarray:
    from yam_arena.demo import grip_rotation

    return grip_rotation(yaw)


def tcp_actual(arm):
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    from scenekit import arm_joints, env

    return ik.tcp_pose(arm_joints(env(), arm.side, ARM_JOINTS), arm.base)


def arrive(arm, tol=0.003, timeout=3.0):
    """Hold the command until the measured tool point is within ``tol`` of it."""
    for _ in range(round(timeout * 20)):
        if np.linalg.norm(tcp_actual(arm)[0] - arm.pos) < tol:
            return
        yield
    if DEBUG:
        print(f"arrive: {arm.side} still {np.linalg.norm(tcp_actual(arm)[0] - arm.pos) * 1000:.1f} mm off", flush=True)


def plate_pose(ctx):
    s = ctx.states()["plate"]
    return np.asarray(s.pos, float), ctx.task._rotation(s.quat_xyzw)


def block_xy():
    return np.array([g.FRONT_X + g.BLOCK_U, g.CENTRE_Y])


def bar_point(slide: float, lever: float) -> np.ndarray:
    u, h = g.lever_bar_uh(slide, lever)
    return np.array([g.FRONT_X + u, BAR_Y, T + h])


def load_plate(ctx, arm, variant):
    """Support base -> block."""
    o, R = plate_pose(ctx)
    down = grip_rot()
    grasp = np.array([o[0], o[1], o[2] + GRASP_DZ])
    yield from arm.move_joints(grasp + [0, 0, 0.06], down, 2.5, grip=G_OPEN)
    yield from arm.move(grasp, down, 1.5)
    yield from arrive(arm)
    yield from arm.set_grip(G_HOLD, 0.8)
    yield from wait(0.3)
    # out of the base's wells (12.3 mm deep) straight up, then up to the carrying height
    carry_z = T + g.SEATED_DECK_H + OVER + GRASP_DZ
    yield from arm.move(grasp + [0, 0, 0.025], down, 1.0)
    # back beside the cycler, low (the plate's right edge 1.6 cm left of the cycler's side), then up to carrying height
    target = block_xy().copy()
    yield from arm.move([target[0], grasp[1], grasp[2] + 0.025], down, 1.5)
    yield from arm.move([target[0], grasp[1], carry_z], down, 1.2)
    rot = down
    if variant == "ridges":
        target += [g.PITCH / 2, g.PITCH / 2]
    if variant == "turned":
        rot = grip_rot(YAW - math.pi)
    yield from arm.move([target[0], target[1], carry_z], rot, 3.0)
    yield from arrive(arm, 0.002)
    # correct for the plate's offset in the hand, measured
    for _ in range(2):
        o, R = plate_pose(ctx)
        err = np.r_[target - o[:2], 0.0]
        yield from arm.move(arm.pos + err, rot, 0.6)
        yield from arrive(arm, 0.0015)
    # lower: the wells' tips enter the block's wells 13 mm down; stop 1.5 mm over seated (or on the ridges)
    seat_z = T + g.SEATED_DECK_H + (g.HOLE_DEPTH + 0.0004 if variant == "ridges" else RELEASE) + GRASP_DZ
    yield from arm.move([arm.pos[0], arm.pos[1], seat_z], rot, 2.5)
    yield from arrive(arm, 0.0015)
    yield from wait(0.3)
    yield from arm.set_grip(G_OPEN, 0.8)
    yield from wait(0.5)
    yield from arm.move(arm.pos + [0, 0, 0.03], rot, 1.0)
    if variant != "ridges":
        # tap both ends down with closed fingertips on the caps (1 mm in)
        yield from arm.set_grip(0.0, 0.6)
        cap_z = T + g.SEATED_DECK_H + g.CAP_H + TIP - 0.001
        o, R = plate_pose(ctx)
        for sy in (-1, 1):                      # right end first: the hand ends on the arm's side
            p = o + R @ [0.0, sy * 0.045, 0.0]
            yield from arm.move([p[0], p[1], cap_z + 0.02], rot, 0.8)
            yield from arm.move([p[0], p[1], cap_z], rot, 0.6)
            yield from wait(0.3)
            yield from arm.move([p[0], p[1], cap_z + 0.02], rot, 0.5)
    # clear of the plate: up 12 mm, then toward the arm's side (straight-down reach above this height is poor)
    yield from arm.move(arm.pos + [0, 0, 0.012], rot, 0.5)
    yield from arm.move(arm.pos + [0, 0.12, 0], rot, 1.0)


def close_cover(ctx, arm, variant):
    """Slide the cover forward by its handle bar, then pull the lever down."""
    R_bar = turn([0, 1, 0], math.radians(-45)) @ grip_rot(math.radians(90))    # forward-down, fingers across the bar
    ap = R_bar[:, 2]
    bar = bar_point(ctx.task._STATE["slide"], 0.0)
    yield from arm.move_joints(bar - 0.07 * ap, R_bar, 2.5, grip=G_OPEN)
    yield from arm.move(bar - 0.02 * ap, R_bar, 1.0)
    yield from arm.move(bar, R_bar, 0.8)
    yield from arrive(arm, 0.002)
    yield from arm.set_grip(0.0, 0.7)
    yield from wait(0.3)
    end = -g.COVER_TRAVEL + (0.03 if variant == "short" else -0.003)     # 3 mm past the stop: press it home
    if variant == "nudge":
        end = ctx.task._STATE["slide"] - 0.003
    yield from arm.move(bar_point(end, 0.0), R_bar, 4.0 if variant != "nudge" else 1.0)
    yield from wait(0.5)
    if variant == "nudge":
        yield from release_bar(arm, R_bar, back=0.05, up=0.0, home=True)
        return
    if variant in ("short", "lever_up"):
        yield from release_bar(arm, R_bar, back=0.03, up=0.0, home=False)
        return
    # the lever's first half: the hand turns with it about the pivot; then the hand keeps its orientation (level) and
    # the bar turns between the pads (squeezed by the closed grip) down past its stop
    pivot = bar_point(-g.COVER_TRAVEL, 0.0) - [0, 0, g.LEVER_R]
    yield from arm.set_grip(G_LEVER, 0.4)
    n = 40
    for k in range(1, n + 1):
        q = -math.pi / 4 * k / n
        Ry = turn([0, 1, 0], q)
        arm.aim(pivot + Ry @ [0, 0, g.LEVER_R], Ry @ R_bar)
        yield
    # the second half against the lever's spring (up to 20 N at the bar): the arc runs on past the stop and the command
    # leads it downward (the arm gives under the load: up to ~5 cm near the stop)
    R_hook = turn([0, 1, 0], -math.pi / 4) @ R_bar
    n = 50
    for k in range(1, n + 1):
        q = -math.pi / 4 - (math.pi / 4 + math.radians(PAST)) * k / n
        arm.aim(pivot + turn([0, 1, 0], q) @ [0, 0, g.LEVER_R] - [0, 0, LEAD * k / n], R_hook)
        yield
    yield from wait(0.5)
    yield from release_bar(arm, R_hook)


def release_bar(arm, R, back=0.08, up=0.02, home=True):
    """Let go of the bar, back the open fingers off it along the hand's axis and go home. After the lever (the arm at
    its wrist's limit) the long back-off is followed only roughly but clears the bar; the cover is locked by then. With
    the lever still up (variants short, lever_up) nothing holds the cover where it is and the way home passes over the
    bar, so the hand only backs off 3 cm (reachable) and stays."""
    yield from arm.set_grip(G_OPEN, 0.6)
    yield from arm.move(arm.pos - back * R[:, 2] + [0, 0, up], R, 1.2)
    if home:
        yield from arm.go_home(2.5)
    else:
        yield from wait(2.0)


def _watch_ik(arm):
    """With PCR_DEBUG, print each IK solve worse than 10 mm (target and error)."""
    aim = arm.aim

    def aim_and_check(pos, rot):
        before = arm.worst_mm
        arm.worst_mm = 0.0
        aim(pos, rot)
        if arm.worst_mm > 10:
            print(f"ik {arm.side}: {arm.worst_mm:.0f} mm at {np.round(pos, 3).tolist()}", flush=True)
        arm.worst_mm = max(before, arm.worst_mm)

    arm.aim = aim_and_check


def phase(arm, name, gen):
    """Run a phase; with PCR_DEBUG, report its worst IK error."""
    arm.worst_mm = 0.0
    yield from gen
    if DEBUG:
        print(f"phase {name}: worst IK {arm.worst_mm:.1f} mm", flush=True)


def script(ctx, variant: str):
    if variant == "nothing":
        yield from wait(5.0)
        return
    arm = ctx.left
    if DEBUG:
        _watch_ik(arm)
    worst = 0.0
    if variant == "nudge_ridges":
        yield from phase(arm, "nudge", close_cover(ctx, arm, "nudge"))
        variant = "ridges"
    if variant != "cover_only":
        yield from phase(arm, "load", load_plate(ctx, arm, variant))
        worst = arm.worst_mm
    yield from phase(arm, "cover", close_cover(ctx, arm, variant))
    arm.worst_mm = max(worst, arm.worst_mm)
    yield from wait(1.5)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    c = S["corners"] * 1000
    tcp = tcp_actual(ctx.left)[0]
    po = ctx.states()["plate"].pos
    return (f"L cmd {np.round(ctx.left.pos, 3).tolist()} meas-cmd {np.round((tcp - ctx.left.pos) * 1000, 1).tolist()} "
            f"grip {ctx.left.grip:.2f} plate {np.round(po, 3).tolist()} slide {S['slide'] * 1000:6.1f} lever {math.degrees(S['lever']):6.1f} platen {S['platen'] * 1000:5.2f} "
            f"corners {c[0]:5.1f} {c[1]:5.1f} {c[2]:5.1f} {c[3]:5.1f} block {S['on_block']} eng {S['engaged']} "
            f"or {S['oriented']} turned {S['turned']} held {S['held']} pads {np.round(S['pads'], 1).tolist()} "
            f"rung {S['rung']} gap {S['hand_gap'] * 1000:.0f} why {S['why']} free {S['lever_free']} lock {S['slide_locked']} "
            f"latch {S['latched']} closing_max {S['moving_max']} lim {S.get('limits')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="full", choices=["full", "ridges", "turned", "short", "lever_up", "nothing", "cover_only", "nudge_ridges"])
    a = ap.parse_args()
    run(a.task, a.out, lambda ctx: script(ctx, a.variant), max_steps=a.max_steps, record=not a.no_record, log=log,
        log_every=int(os.environ.get("PCR_LOG_EVERY", "20")))
