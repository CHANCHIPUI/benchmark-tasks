"""Scripted full-score demo (reads object poses from the simulator).

1. Right arm, the pen (while the left arm sets the time): it comes down along the pen as it leans in the mug (top
   toward the robot, ~36 degrees), fingers closing across the pen in its lean plane 135 mm above the ball, and lifts
   it straight up out of the mug.
2. Left arm, the time: fingers pointing down astride the dial, the pads on its ribbed side 2 mm above the housing (no
   segment touched); it closes and rolls the wrist clockwise, by feedback on the dial's angle, until the pointer shows
   2:30 PM; opens, lifts and parks with its wrist camera 15 cm from the dial; after the last press it moves the camera
   to 10 cm, 45 degrees above the dial, for the closing close-up (the pen waits 2 s over the dial, then lifts).
3. Right arm, the ten segments (6:00 PM to 10:30 PM), the pen tipped PRESS_TILT with its top toward the robot's right,
   alternating ends (45, 36, 44, 37, ...): see press(). Then it lifts the pen clear and parks it.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record] [--variant NAME]

Variants (scorer checks), each printing its mode:
- ``neighbour``: the 6:00 PM press lands on the gap toward 5:30 PM (1.5 mm off centre) and pushes on until both are
  down (a careless press that catches the 5:30 PM segment as well);
- ``neighbour2``: after the ten, a deliberate extra pen press on the 5:30 PM segment;
- ``wrongtime``: the dial is turned to 3:30 PM;
- ``fingertip``: the segments are pressed with the closed fingertips instead of the pen;
- ``missing``: the 8:00 and 8:30 PM segments are skipped;
- ``nothing``: the arms stay still.
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
from demokit import run, wait  # noqa: E402

DEBUG = bool(os.environ.get("PT_DEBUG"))
GAP_MM = 94.0                       # pads' gap at grip 1.0
PEN_L_GRIP = 0.135                  # TCP this far up the pen from the ball's centre
PEN_SQUEEZE = float(os.environ.get("PT_SQUEEZE", "0.0"))   # grip command on the pen (0: fully closed, 8 mm inside)
PRESS_TILT = math.radians(float(os.environ.get("PT_TILT", "24")))
PRESS_AZ = math.radians(float(os.environ.get("PT_AZ", "270")))    # the pen's top toward the robot's right
RLAND = float(os.environ.get("PT_RLAND", "0.0230"))   # the ball's landing radius on a segment
PRESS_FLIP = float(os.environ.get("PT_FLIP", "1"))
HOVER = 0.0025                      # ball over the segment top while settling (clear of the arm's sway)
APPROACH = 0.004                    # the ball comes down from this far above the segment
MOVE_S = float(os.environ.get("PT_MOVE_S", "0.7"))  # s from one segment to the next
GAIN = 0.12                         # xy feedback per step on the measured ball, above the segment
SETTLE_N = int(os.environ.get("PT_SETTLE_N", "6"))  # steps averaged per settling round over the aim point
SETTLE_TOL = 0.0002                 # settled: the mean error under this
END_BIAS = float(os.environ.get("PT_BIAS", "0.00045"))   # the two end segments: aim this far toward the run's inside
DANGER = 0.00055                    # an end segment's landing this far toward the forbidden neighbour is redone
LEAD = float(os.environ.get("PT_LEAD", "0.0"))   # aim this far against the ball's creep under load
PRESS_V2 = float(os.environ.get("PT_V2", "0.012"))   # m/s down once the ball is on the segment
HOLD_GAIN = float(os.environ.get("PT_HOLD", "0.2"))  # xy feedback per step holding the ball where it landed
FF = float(os.environ.get("PT_FF", "0.45"))           # hand moved back this much per metre down (toward the pen's top)
PRESS_V = float(os.environ.get("PT_V", "0.006"))   # m/s down
OVER_MAX = float(os.environ.get("PT_OVER", "0.035"))   # command at most this far below where the ball landed
DIAL_SQUEEZE = 0.004                # m commanded inside the dial's ribbed side
FREEZE = os.environ.get("PT_FREEZE", "1") == "1"   # hold the hand's xy once the ball is about to land
TRACE = bool(os.environ.get("PT_TRACE"))
LOOK = os.environ.get("PT_LOOK", "1") == "1"         # park the left arm as a close camera on the dial


def grip_for(width_m: float, squeeze: float = 0.0) -> float:
    return max(0.0, (width_m - squeeze) * 1000 / GAP_MM)


def R_from(a, c) -> np.ndarray:
    """Gripper rotation with approach ``a`` (its z) and closing axis ``c`` (its x)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    c = np.asarray(c, float)
    c = c - a * (a @ c)
    c /= np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


def rotz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def tcp_actual(arm):
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    from scenekit import arm_joints, env

    return ik.tcp_pose(arm_joints(env(), arm.side, ARM_JOINTS), arm.base)


def pen_state(ctx):
    """(ball centre, axis tip->top) of the pen, world."""
    return ctx.task.pen_tip(ctx.states())


def dial_angle(ctx) -> float:
    return float(ctx.task._STATE["dial"])


def pen_grip_rot(tilt: float, az: float, flip: float = -1.0) -> np.ndarray:
    """The right hand's rotation holding the pen tipped ``tilt`` from vertical with its top toward azimuth ``az``: the
    approach along the pen toward its point, the closing axis in the tilt plane (``flip`` picks which of the two
    fingers is on the pen's upper side: the wrist's reachable roll)."""
    u = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
    c = flip * np.array([math.cos(tilt) * math.cos(az), math.cos(tilt) * math.sin(az), -math.sin(tilt)])
    return R_from(-u, c)


def settle_to(arm, pos, rot, tol=0.0005, rounds=3, seconds=0.6):
    """Move to ``pos`` and correct the command by the measured tool-point error."""
    target = np.asarray(pos, float)
    cmd = target.copy()
    yield from arm.move(cmd, rot, seconds)
    for _ in range(rounds):
        yield from wait(0.25)
        err = target - tcp_actual(arm)[0]
        if np.linalg.norm(err) < tol:
            break
        cmd = cmd + err
        yield from arm.move(cmd, rot, 0.35)


# -- right arm: the pen ------------------------------------------------------------------------------------------------
def pick_pen(ctx, arm):
    tip, u = pen_state(ctx)
    tilt = math.acos(max(-1.0, min(1.0, u[2])))
    az = math.atan2(u[1], u[0])
    R = pen_grip_rot(tilt, az)
    tcp = tip + PEN_L_GRIP * u
    a = R[:, 2]
    if DEBUG:
        print(f"pen tip {np.round((tip - [0, 0, g.TOP]) * 1000, 1)} tilt {math.degrees(tilt):.1f} az {math.degrees(az):.1f}",
              flush=True)
    yield from arm.move_joints(tcp - 0.07 * a, R, 3.0, grip=grip_for(0.026))
    yield from arm.move(tcp - 0.02 * a, R, 1.0)
    yield from settle_to(arm, tcp, R)
    yield from arm.set_grip(PEN_SQUEEZE, 0.8)
    yield from wait(0.3)
    # straight up out of the mug, then tip it to the pressing angle
    yield from arm.move(arm.pos + np.array([0.0, 0.0, 0.115]), R, 2.0)
    ctx.flags["pen_in_hand"] = pen_in_hand(ctx, arm)
    if DEBUG:
        print("pen in hand (gripper frame, mm):", np.round(ctx.flags["pen_in_hand"][0] * 1000, 2),
              np.round(ctx.flags["pen_in_hand"][1], 3), flush=True)


def pen_in_hand(ctx, arm):
    """The pen's ball centre and axis in the hand's frame (measured)."""
    p, R = tcp_actual(arm)
    tip, u = pen_state(ctx)
    return R.T @ (tip - p), R.T @ u


def tcp_for_tip(ctx, tip_world, R) -> np.ndarray:
    """The TCP that puts the pen's ball at ``tip_world`` with the hand at R (the pen as last measured in the hand)."""
    t_in, _ = ctx.flags["pen_in_hand"]
    return np.asarray(tip_world, float) - R @ t_in


def seg_target(ctx, i: int, r: float = RLAND, h: float = g.SEG_TOP_H, off: float = 0.0) -> np.ndarray:
    """A point on segment i's centre line at radius r (``off`` m along the clockwise tangent), at the dial's measured
    angle and the timer's measured pose."""
    st = ctx.states()["timer"]
    R0 = ctx.task._rotation(st.quat_xyzw)
    a = g.seg_angle_world(i, float(st.joints["dial"]))
    local = np.array([r * math.cos(a) + off * math.sin(a), r * math.sin(a) - off * math.cos(a), h])
    return np.asarray(st.pos, float) + R0 @ local


def press(ctx, arm, R, i: int, off: float = 0.0):
    """Press segment i with the pen's ball, reading the pen's pose (privileged). The aim point is on the segment's
    centre line at RLAND from the axis (``off`` m along the clockwise tangent; the two end segments END_BIAS toward the
    inside of the run). The hand moves smoothly to put the ball HOVER over it and waits for the arm's sway to die
    down; the ball's measured position is averaged over SETTLE_N steps and the hand corrected by the mean error with a
    smooth move, until the mean is within SETTLE_TOL. Then straight down with the hand's xy held. On the segment the
    hand holds the ball where it landed by feedback and leans back toward the pen's top as it goes down (the arm gives
    about 3 mm per newton, and sideways under the tilted pen); it stops once the segment latches (at most OVER_MAX below
    the landing). An end segment's landing more than DANGER toward its forbidden neighbour is lifted off at once (a
    light touch moves no segment far) and redone. The lift starts from the hand's measured pose."""
    S = ctx.task._STATE
    lead = -LEAD * np.array([math.cos(PRESS_AZ + math.pi), math.sin(PRESS_AZ + math.pi), 0.0])
    t_in, u_in = pen_in_hand(ctx, arm)                  # re-read the pen in the hand (it may have shifted a little)
    t0, u0 = ctx.flags["pen_in_hand"]
    if np.linalg.norm(t_in - t0) > 0.0002:
        ctx.flags.setdefault("pen_shift_mm", []).append((i, np.round((t_in - t0) * 1000, 2).tolist()))
    ctx.flags["pen_in_hand"] = (t_in, u_in)

    bias = END_BIAS if i == g.ON_SEG else (-END_BIAS if i == g.OFF_SEG - 1 else 0.0)

    def aim():
        return seg_target(ctx, i, off=off - bias) + lead

    # straight to HOVER over the aim point, smoothly; let the arm's sway die down
    cmd = tcp_for_tip(ctx, aim() + np.array([0.0, 0.0, HOVER]), R)
    yield from arm.move(cmd, R, MOVE_S)
    yield from wait(0.4)
    # settle: average the measured ball over SETTLE_N steps and correct the hand by the mean error with a smooth move
    # (a step would set the arm swaying), until the mean is within SETTLE_TOL (at most 3 corrections)
    for rnd in range(4):
        errs = []
        for _ in range(SETTLE_N):
            yield
            tip = pen_state(ctx)[0]
            errs.append(np.r_[aim()[:2] - tip[:2], aim()[2] + g.PEN_BALL_R + HOVER - tip[2]])
            if TRACE:
                print("HOVER", i, rnd, np.round(errs[-1] * 1000, 2).tolist(), flush=True)
        e = np.mean(errs, axis=0)
        if np.linalg.norm(e[:2]) < SETTLE_TOL or rnd == 3:
            break
        cmd = cmd + np.r_[e[:2], 0.0]
        yield from arm.move(cmd, R, 0.3)
        yield from wait(0.3)
    top = aim()
    n0 = len(S["down"])
    trace = []
    first = None
    back = np.array([math.cos(PRESS_AZ), math.sin(PRESS_AZ), 0.0])     # toward the pen's top
    z_stop = None
    for k in range(600):
        if i in S["down"] and (off == 0.0 or (i - 1) in S["down"]):
            break
        tip, _ = pen_state(ctx)
        above = tip[2] - g.PEN_BALL_R - top[2]
        a = aim()
        rad = a[:2] - seg_target(ctx, i, r=0.0)[:2]
        rad = rad / np.linalg.norm(rad)
        tan = np.array([rad[1], -rad[0]])                  # the clockwise tangent
        err = a[:2] - tip[:2]
        if first is None and above > 0.0003:
            dz = PRESS_V * 0.05                                    # straight down, the hand's xy held
        else:
            if first is None:
                first = tip.copy()
                z_stop = cmd[2] - OVER_MAX
                # an end segment: a landing toward the forbidden neighbour beyond DANGER is lifted off and redone
                # (a light touch moves no segment far)
                t_off = float((tip[:2] - seg_target(ctx, i)[:2]) @ tan)
                if off == 0.0 and ((i == g.ON_SEG and t_off > DANGER) or (i == g.OFF_SEG - 1 and t_off < -DANGER)):
                    ctx.flags.setdefault("relanded", []).append((i, round(t_off * 1000, 2)))
                    break
            # on the segment: hold the ball on the centre line (along the tangent) and where it landed (along the
            # radius) by feedback, and lean the hand back toward the pen's top as it goes down (feedforward against
            # the arm's sideways give)
            dz = PRESS_V2 * 0.05
            hold = first[:2] - tip[:2]
            cmd[:2] = cmd[:2] + HOLD_GAIN * hold + FF * dz * back[:2]
        # the ball has gone off the segment (into a pressed neighbour's slot, or over its end) while the segment has
        # hardly moved: stop at once and lift (pushing on would bend the pen in the fingers)
        if off == 0.0 and first is not None and above < -0.0012 and ctx.states()["timer"].joints[f"seg_{i:02d}"] > -0.0006:
            ctx.flags.setdefault("slipped", []).append(i)
            break
        cmd[2] = cmd[2] - dz if z_stop is None else max(cmd[2] - dz, z_stop)
        arm.aim(cmd, R)
        yield
        if TRACE:
            st = ctx.states()["timer"]
            q = st.joints[f"seg_{i:02d}"]
            trace.append((k, round(float((tip[:2] - seg_target(ctx, i)[:2]) @ tan) * 1000, 2),
                          round(float((tip[:2] - seg_target(ctx, i)[:2]) @ rad) * 1000, 2),
                          round(above * 1000, 2), round(q * 1000, 2), round(math.degrees(S["dial"]), 3)))
        if z_stop is not None and cmd[2] <= z_stop + 1e-6:
            break
    if TRACE:
        for t in trace:
            print("TRACE", i, *t, flush=True)
    log = {"seg": i, "latched": i in S["down"], "new": len(S["down"]) - n0, "steps": len(trace),
           "landed_mm": None if first is None else np.round((first - top) * 1000, 2).tolist(),
           "creep_mm": None if first is None else np.round((pen_state(ctx)[0] - first)[:2] * 1000, 2).tolist()}
    ctx.flags.setdefault("press_log", []).append(log)
    # lift: start the command from where the hand actually is (the arm is bent by the press; from the deep command it
    # would spring sideways and drag the pen across the dial), then straight up
    p_act, _ = tcp_actual(arm)
    arm.pos = p_act
    yield from arm.move(p_act + np.array([0.0, 0.0, 0.008]), R, 0.4)


def right_script(ctx, variant):
    arm = ctx.right
    if variant == "fingertip":
        while not ctx.flags.get("time_done"):
            yield
        yield from fingertip_presses(ctx, arm)
        ctx.flags["presses_over"] = ctx.flags["presses_done"] = True
        return
    yield from pick_pen(ctx, arm)
    while not ctx.flags.get("time_done"):
        yield
    R = pen_grip_rot(PRESS_TILT, PRESS_AZ, PRESS_FLIP)
    # over the dial at a safe height, tipped to the pressing angle
    first = seg_target(ctx, g.ON_SEG) + np.array([0.0, 0.0, 0.03])
    yield from arm.move_joints(tcp_for_tip(ctx, first, R), R, 2.5)
    ctx.flags["pen_in_hand"] = pen_in_hand(ctx, arm)
    segs = list(g.TARGETS)
    segs = [45, 36, 44, 37, 43, 38, 42, 39, 41, 40]      # alternate ends: the drags on the dial cancel
    if os.environ.get("PT_SEGS"):
        segs = [int(v) for v in os.environ["PT_SEGS"].split(",")]
    if variant == "missing":
        segs = [i for i in segs if i not in (40, 41)]
    for i in segs:
        # neighbour variant: the 6:00 PM press lands on the gap toward 5:30 PM (net 1.5 mm off centre) and pushes on
        # until both are down
        off = 0.0015 + END_BIAS if (variant == "neighbour" and i == g.ON_SEG) else 0.0
        for attempt in range(3):
            yield from press(ctx, arm, R, i, off)
            if i in ctx.task._STATE["down"]:
                break
    if variant == "neighbour2":
        # check: after the ten, a deliberate extra press on the 5:30 PM segment (the run then starts half an hour early)
        for attempt in range(3):
            yield from press(ctx, arm, R, g.ON_SEG - 1)
            if g.ON_SEG - 1 in ctx.task._STATE["down"]:
                break
    # hold the pen just over the dial until the left arm's camera is in its close-up pose and 2 s more (the video's
    # closing view), then lift it well clear and park it
    ctx.flags["presses_over"] = True
    if LOOK:
        while not ctx.flags.get("close_ready"):
            yield
        yield from wait(2.0)
    yield from arm.move(arm.pos + np.array([0.0, 0.0, 0.06]), R, 1.0)
    ctx.flags["pen_lifted"] = True
    yield from arm.move(np.array([0.20, -0.18, g.TOP + 0.25]), R, 1.5)
    ctx.flags["presses_done"] = True


def fingertip_presses(ctx, arm):
    """Variant: the closed fingertips (pointing down, tipped as the pen was) press each of the ten segments."""
    S = ctx.task._STATE
    R = pen_grip_rot(PRESS_TILT, PRESS_AZ, PRESS_FLIP)
    tip_in = np.array([0.0, 0.0, 0.134 + 0.0128 - 0.134])      # the fingertips' end, on the approach axis
    ctx.flags["pen_in_hand"] = (tip_in, np.array([0.0, 0.0, -1.0]))
    first = seg_target(ctx, g.ON_SEG) + np.array([0.0, 0.0, 0.03])
    yield from arm.move_joints(tcp_for_tip(ctx, first, R), R, 3.0, grip=0.0)
    for i in g.TARGETS:
        want = seg_target(ctx, i) + np.array([0.0, 0.0, HOVER + 0.002])
        yield from arm.move(tcp_for_tip(ctx, want, R), R, 0.6)
        n = 0
        while i not in S["down"] and n < 60:
            want = want - np.array([0.0, 0.0, PRESS_V * 0.05])
            arm.aim(tcp_for_tip(ctx, want, R), R)
            yield
            n += 1
        yield from arm.move(tcp_for_tip(ctx, seg_target(ctx, i) + np.array([0.0, 0.0, 0.008]), R), R, 0.4)
    yield from arm.move(arm.pos + np.array([0.0, 0.0, 0.06]), R, 1.0)
    yield from arm.move(np.array([0.20, -0.18, g.TOP + 0.25]), R, 1.5)


# -- left arm: the time ---------------------------------------------------------------------------------------------
DIAL_TCP_H = g.FACE_H + 0.002 + 0.013 - 0.0003     # fingertips 2 mm above the housing (pads 13 mm past the TCP)
DIAL_YAW0 = math.radians(270.0)                       # the closing axis (along y) when it takes hold; q6 -45 -> 38 deg


def dial_rot(yaw: float) -> np.ndarray:
    return rotz(yaw) @ R_from([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])


def set_time(ctx, arm, target: float):
    S = ctx.task._STATE
    st = ctx.states()["timer"]
    centre = np.asarray(st.pos, float) + np.array([0.0, 0.0, 0.0])
    p = np.array([centre[0], centre[1], g.TOP + DIAL_TCP_H])
    yaw = DIAL_YAW0
    R = dial_rot(yaw)
    open_w = grip_for(2 * g.RIB_R + 0.012)
    yield from arm.move_joints(p + np.array([0.0, 0.0, 0.06]), R, 3.0, grip=open_w)
    yield from settle_to(arm, p + np.array([0.0, 0.0, 0.015]), R)
    yield from settle_to(arm, p, R)
    corr = arm.pos - p
    yield from arm.set_grip(grip_for(2 * g.RIB_R, DIAL_SQUEEZE), 0.8)
    yield from wait(0.3)
    # roll clockwise (yaw decreasing), led by the measured dial angle
    start = dial_angle(ctx)
    for k in range(400):
        d = dial_angle(ctx)
        err = target - d
        if abs(err) < math.radians(0.3):
            break
        step = max(-math.radians(1.2), min(math.radians(1.2), err))
        lead = math.radians(2.0) * (1 if step > 0 else -1)
        yaw_cmd = DIAL_YAW0 + (d - start) + step + lead * min(1.0, abs(err) / math.radians(3.0))
        arm.aim(p + corr, dial_rot(yaw_cmd))
        yield
    ctx.flags["dial_err_deg"] = math.degrees(dial_angle(ctx) - target)
    yield from wait(0.2)
    yield from arm.set_grip(open_w, 0.6)
    yield from arm.move(arm.pos + np.array([0.0, 0.0, 0.06]), arm.rot, 1.0)
    yield from arm.move(np.array([0.22, 0.22, g.TOP + 0.25]), arm.rot, 1.5)


CLOSE_Q = [-0.592, 2.263, 2.096, -1.614, 0.406, -0.108]
LOOK_Q = [-0.57, 1.55, 1.16, -0.68, 0.38, -0.24]     # left wrist camera 15 cm from the dial, looking down at its
                                                     # left half from the far left (dev/look.py)


def move_q(arm, q1, seconds: float = 2.0, grip: float | None = None):
    """Generator: a smooth joint-space move to the joint angles q1."""
    from yam_arena import ik

    q0, q1, g0 = arm.q.copy(), np.asarray(q1, float), arm.grip
    g1 = g0 if grip is None else grip
    n = max(1, round(seconds * 20))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        arm.grip = g0 + s * (g1 - g0)
        arm.q = q0 + s * (q1 - q0)
        yield
    arm.pos, arm.rot = ik.tcp_pose(arm.q, arm.base)


def left_script(ctx, variant):
    arm = ctx.left
    target = g.TARGET_ANGLE - (math.radians(15.0) if variant == "wrongtime" else 0.0)
    yield from set_time(ctx, arm, target)
    ctx.flags["time_done"] = True
    if LOOK:
        yield from move_q(arm, LOOK_Q, 2.0)
        # after the last press: the wrist camera 10 cm from the dial, 45 degrees above it, from beyond its far side, for
        # the video's closing close-up (the hand stays 37 mm from the timer; dev/look_close.py)
        while not ctx.flags.get("presses_over"):
            yield
        yield from move_q(arm, CLOSE_Q, 1.5)
        ctx.flags["close_ready"] = True


def script(ctx, variant: str):
    from demokit import together

    print(f"MODE variant={variant}", flush=True)
    yield from together(right_script(ctx, variant), left_script(ctx, variant))
    print("PRESS_LOG", ctx.flags.get("press_log"), "dial_err_deg", ctx.flags.get("dial_err_deg"),
          "relanded", ctx.flags.get("relanded"), "slipped", ctx.flags.get("slipped"), "pen_shift", ctx.flags.get("pen_shift_mm"),
          flush=True)
    yield from wait(1.5)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    tip, u = pen_state(ctx)
    return (f"dial {math.degrees(S['dial']):.2f} time {g.fmt_time(S['time'])} down {S['down']} touched {S['touched']} pushed {S['pushed']} "
            f"{S['touch_who']} gap {S['gap'] * 1000:.0f} pen_gap {S['pen_gap'] * 1000:.0f} tier {S['tier']} rung {S['rung']} "
            f"tip {np.round((tip - [0, 0, g.TOP]) * 1000, 1).tolist()} nb_max {S['nb_max'] * 1000:.2f} {S['nb_who']} "
            f"strip {S['strip_moved'] * 1000:.1f}mm {S['strip_turn']:.2f}deg why {S['why']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="full",
                    choices=["full", "neighbour", "neighbour2", "wrongtime", "fingertip", "missing", "nothing", "pick"])
    a = ap.parse_args()
    if a.variant == "nothing":
        def scr(ctx):
            print("MODE variant=nothing", flush=True)
            yield from wait(5.0)
    elif a.variant == "pick":
        def scr(ctx):
            print("MODE variant=pick", flush=True)
            yield from pick_pen(ctx, ctx.right)
            for k in range(40):
                yield
                tin, uin = pen_in_hand(ctx, ctx.right)
                if k % 2 == 0:
                    print("HOLD", k, np.round(tin * 1000, 2).tolist(), np.round(uin, 4).tolist(), flush=True)
            R = pen_grip_rot(PRESS_TILT, PRESS_AZ, PRESS_FLIP)
            first = seg_target(ctx, g.ON_SEG) + np.array([0.0, 0.0, 0.03])
            yield from ctx.right.move_joints(tcp_for_tip(ctx, first, R), R, 2.5)
            ctx.flags["pen_in_hand"] = pen_in_hand(ctx, ctx.right)
            for i in [int(v) for v in os.environ.get("PT_SEGS", "36,41,45").split(",")]:
                yield from press(ctx, ctx.right, R, i)
            yield from wait(2.0)
    else:
        def scr(ctx):
            yield from script(ctx, a.variant)
    res = run(a.task, a.out, scr, max_steps=a.max_steps, record=not a.no_record, log=log,
              log_every=int(os.environ.get("PT_LOG_EVERY", "20")))
    import json

    from demokit import Ctx  # noqa: F401
    S = sys.modules["task"]._STATE if "task" in sys.modules else {}
    extra = {"variant": a.variant, "down": S.get("down"), "time": g.fmt_time(S.get("time", 0)),
             "touched": S.get("touched"), "pushed": S.get("pushed"), "push_who": S.get("push_who"), "presses": S.get("presses"), "nb_max_mm": round(S.get("nb_max", 0) * 1000, 3),
             "strip_moved_mm": round(S.get("strip_moved", 0) * 1000, 2), "strip_turn_deg": round(S.get("strip_turn", 0), 2),
             "lamp": S.get("lamp"), "why": S.get("why")}
    print("EXTRA " + json.dumps(extra), flush=True)
    (pathlib.Path(a.out) / "extra.json").write_text(json.dumps(extra, indent=2))
