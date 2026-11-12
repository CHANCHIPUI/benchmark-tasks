"""Scripted full-score demo. The right arm takes the windlass rod from above by one end, its grip leaning 15 degrees
toward the robot, and turns it anticlockwise (seen from above): past the first time it lies along the thigh (30
degrees in, gauge about 90), to 100 degrees, where its wrist is near its limit. The left arm takes the other end and
holds it still while the right lets go and takes the same end again, its wrist turned the other way round; the left
lets go. The right turns the rod on to 210 degrees, where it lies along the thigh again with the gauge in the green,
now holding its hip end: it lifts that end so the knee end goes down into the clip's hook, lets go and backs off.
Every move follows the rod's measured angle; the script reads the gauge's needle, not the band's internal state.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record] [--early-lock] [--hold] [--slip] [--keep-hold]

``--early-lock`` locks the rod the first time it lies along the thigh, with the gauge at about 90 (0.25): the left
hand lifts the hip end there. ``--hold`` turns it into the green but keeps holding it instead of locking it (0.5).
``--slip`` lets go for the fresh grip with nothing holding the rod, so it unwinds (0). ``--keep-hold`` locks it in
the green but keeps the right hand on the rod (0.5).
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

HUB = np.array([g.LEG_X, g.BAND_Y, g.TOP + g.ROD_Z])
GRIP_R = 0.050            # take the rod 5 cm from its middle (the fingers are 6.8 cm wide along it)
ABOVE = 0.005             # the tool point this far above the rod's axis: the fingertips stay clear of the band
STEP = math.radians(2.0)  # how far the hand's angle advances each move
LEAD_MAX = math.radians(10.0)  # and how far it may run ahead of the rod's measured angle
FIRST_ALIGN = math.radians(-90.0)  # the rod first lies along the thigh 30 degrees in (gauge about 90)
REGRASP_AT = math.radians(-20.0)   # the right hand's wrist is near its limit here: it takes a fresh grip
LOCK_AT = math.radians(89.0)       # along the thigh again, the gauge in the green (about 269)
LEAN = 15.0              # the twisting hand leans this far toward the robot
HOLD_LEAN = 30.0         # the holding hand leans further, which keeps the two arms apart


class Rod:
    """The rod as the cameras see it: its direction (followed through every turn) and its tilt, plus the gauge's
    reading from its needle."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.theta = g.THETA0
        self.last = g.THETA0

    def read(self):
        j = self.ctx.states()["tourniquet"].joints
        a = g.THETA0 + j["twist"]
        self.theta += math.remainder(a - self.last, 2 * math.pi)
        self.last = a
        return self.theta, j["press"]

    def gauge(self) -> float:
        needle = -self.ctx.states()["tourniquet"].joints["needle"]
        return (needle + g.NEEDLE_SWEEP / 2) / g.NEEDLE_SWEEP * g.GAUGE_MAX


def end(theta: float, s: int) -> np.ndarray:
    return HUB + s * GRIP_R * np.array([math.cos(theta), math.sin(theta), 0.0]) + [0.0, 0.0, ABOVE]


def grip(theta: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Straight down with the fingers closing across the rod, leaned ``tilt_deg`` toward the robot."""
    from yam_arena.demo import grip_rotation

    return turn([0, 1, 0], -math.radians(tilt_deg)) @ grip_rotation(theta)


def grip_along_y(theta: float, lean_deg: float) -> np.ndarray:
    """For the rod lying along the thigh (y): straight down with the fingers across it, leaned ``lean_deg`` about their
    closing line so the hand's body is outboard (+y) and the fingers stay level either side of the rod. (Leaning
    about y instead would tip the closing line, putting one finger on top of the rod.)"""
    from yam_arena.demo import grip_rotation

    return turn([1, 0, 0], -math.radians(lean_deg)) @ grip_rotation(theta)


def take(arm, p, rot):
    """Come down on the rod along the grip's approach, and close."""
    a = rot[:, 2]
    yield from arm.move(p - a * 0.06, rot, 1.6, grip=1.0)
    yield from arm.move(p, rot, 0.8)
    yield from arm.set_grip(0.0, 0.5)


def sweep(rod, arm, s, goal, tilt, branch=0.0):
    """Turn the rod to ``goal`` (either way). The hand's angle advances STEP a move and runs up to LEAD_MAX ahead of
    the rod's measured angle: the wrist gives under the band's pull, so the rod trails the hand. It stops once the rod
    is within half a degree of the goal. ``branch`` (0 or pi) picks which way round the fingers straddle the rod."""
    theta, _ = rod.read()
    d = 1.0 if goal > theta else -1.0
    cmd = theta
    for _ in range(800):
        if (goal - theta) * d <= math.radians(0.5):
            return
        cmd += d * STEP
        if (cmd - theta) * d > LEAD_MAX:
            cmd = theta + d * LEAD_MAX
        if (cmd - goal) * d > LEAD_MAX:
            cmd = goal + d * LEAD_MAX
        yield from arm.move(end(cmd, s), grip(cmd + branch, tilt), 0.1)
        theta, _ = rod.read()


def release(arm, up=0.05):
    """Open and lift clear, staying near (for a fresh grasp)."""
    yield from arm.set_grip(1.0, 0.4)
    yield from arm.move(arm.pos + [0.0, 0.0, up], None, 0.6)


def let_go(arm, up=0.06):
    yield from release(arm, up)
    yield from arm.go_home(1.6)


def lock(rod, arm):
    """Lift the held (hip) end until the rod's measured tilt puts its knee end past the clip's lips, onto its plate (the
    arm gives under the strap's spring, so it lifts more than that); the lips then hold it down. The clip's lead-in
    guides the rod's end into its channel on the way down."""
    for _ in range(30):
        theta, tilt = rod.read()
        knee = 1.0 if math.sin(theta) < 0 else -1.0             # which way the rod's +x end points: knee or hip
        if knee * g.LOCK_R * math.sin(tilt) >= g.LOCKED_DEPTH + 0.0003:   # how far down its knee end is at the clip
            break
        yield from arm.move(arm.pos + [0.0, 0.0, 0.003], None, 0.25)
        yield from wait(0.15)
    yield from wait(0.3)


def script(ctx, mode="full"):
    R, L = ctx.right, ctx.left
    rod = Rod(ctx)
    theta, _ = rod.read()
    # grasp 1: the right hand on the rod's +x end, fingers straddling it the "far" way round (branch pi)
    yield from take(R, end(theta, 1), grip(theta + math.pi, LEAN))
    if mode == "early":                                     # the rod lies along the thigh 30 degrees in: lock there
        yield from sweep(rod, R, 1, FIRST_ALIGN, LEAN, math.pi)
        theta, _ = rod.read()
        yield from take(L, end(theta, -1), grip_along_y(theta, 20.0))
        yield from release(R)
        yield from lock(rod, L)
        yield from let_go(L)
        yield from R.go_home(1.6)
        yield from wait(3.0)
        return
    yield from sweep(rod, R, 1, REGRASP_AT, LEAN, math.pi)
    if mode == "slip":                                      # lets go for a fresh grip with nothing holding the rod
        yield from let_go(R)
        yield from wait(4.0)
        return
    # the left hand holds the other end still while the right takes a fresh grip, the other way round
    theta, _ = rod.read()
    yield from take(L, end(theta, -1), grip(theta, HOLD_LEAN))
    yield from release(R)
    theta, _ = rod.read()
    yield from take(R, end(theta, 1), grip(theta, LEAN))
    yield from let_go(L)
    # grasp 2: on until the rod lies along the thigh again, the right hand now on its hip end
    yield from sweep(rod, R, 1, LOCK_AT, LEAN, 0.0)
    yield from wait(0.3)
    if rod.gauge() < g.GREEN:
        print(f"gauge {rod.gauge():.0f} mmHg at the lock angle: not green", flush=True)
    if mode == "hold":
        yield from wait(4.0)
        return
    yield from lock(rod, R)
    if mode == "keep":
        yield from wait(4.0)
        return
    yield from let_go(R)
    yield from wait(2.0)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    return (f"twist {S['twist']:7.1f}  {S['pressure']:5.1f} mmHg  aligned {S['aligned']}  dip {S['dip'] * 1000:5.1f} mm  "
            f"locked {S['locked']}  arms {S['clearance'] * 100:5.1f} cm off  clear steps {S['clear_steps']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--early-lock", action="store_true", help="lock at the first chance, below the green")
    ap.add_argument("--hold", action="store_true", help="turn into the green but keep holding instead of locking")
    ap.add_argument("--slip", action="store_true", help="let go before the other hand has the rod")
    ap.add_argument("--keep-hold", action="store_true", help="lock in the green but keep a hand on the rod")
    a = ap.parse_args()
    mode = ("early" if a.early_lock else "hold" if a.hold else "slip" if a.slip else "keep" if a.keep_hold
            else "full")
    run(a.task, a.out, lambda ctx: script(ctx, mode), max_steps=a.max_steps, record=not a.no_record, log=log,
        log_every=20)
