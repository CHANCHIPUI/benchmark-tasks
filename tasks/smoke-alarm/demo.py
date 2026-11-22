"""Scripted full-score demo (reads object poses from the simulator: the alarm's pose, its compartment's hinge angle,
the latch tab's travel and the batteries' poses). README.md describes each step.

1. Right arm, open: closed fingertips press the latch tab down until the compartment is freed and springs out past
   its catch, then push the tab's inner face outward along the compartment's arc to its stop (closed loop on the
   hinge angle).
2. Left arm, the pin: fingers closed, tipped 45 degrees, pressing on the front cover beside the compartment's free
   end; it holds the alarm down on its box until the compartment is latched again.
3. Right arm, the old battery: a light pinch across its width on its exposed snap end; the hand rises along the arc
   about the battery's bottom end (under the front cover) without turning, until the snap end is off its contacts and
   clear of the contact block, then draws the battery out along its tipped axis and lays it on the table.
4. Right arm, the new battery: a pinch on its card near its snap end, the hand pitched so the battery hangs tipped;
   turned so its snaps point at the contacts, + to the outer side; slid in bottom end first under the front cover,
   its snap end lowered onto the contacts (the hand goes down along the arc about the bottom end), slid onto them and
   pressed until it snaps.
5. Right arm, latch: the closed fingertips push the tab's outer face back along the arc until the compartment is shut
   past its catch; the left hand lets go.
6. Right arm, turn-over: a pinch across the rim where it overhangs the box, a lift, half a turn of the wrist, the
   alarm laid face down on the box.
7. Both arms, mount: side grips on the alarm's band (each hand along the band's tangent, below the board), lift, carry
   it under the ring, raise it onto the ring's face, turn it clockwise seen from below until it locks, let go.

    python demo.py task.py demo_out [--max-steps 8000] [--no-record] [--variant NAME] [--close-view]

Variants (scorer checks), each printing its mode:
- ``pull_leave``: the old battery taken out and laid on the table, the compartment left open;
- ``no_latch``: the full swap, the compartment left open;
- ``open_only``: the compartment opened, nothing else;
- ``reversed``: the new battery starts on its card turned over (label down); the same moves put it in turned over,
  then the compartment is pushed shut;
- ``latch_empty``: the old battery taken out, the empty compartment pushed shut;
- ``relatch``: the old battery taken out, the empty compartment pushed shut, opened again, the new battery snapped in,
  the compartment latched and the alarm mounted;
- ``no_mount``: the full swap and latch, the alarm left face up on its box;
- ``turned_over``: the full swap and latch and the turn-over, the alarm left face down on its box;
- ``mount_only``: the turn-over and the mount alone (the old battery stays in);
- ``lift_straight``: the old battery pinched and lifted straight up (the front cover stops its bottom end);
- ``nothing``: the arms stay still.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as g  # noqa: E402
from demokit import run, together, turn, wait  # noqa: E402

DEBUG = bool(os.environ.get("SA_DEBUG"))
GAP_MM = 94.0                       # pads' gap at grip 1.0
TIP = 0.013                         # fingertips past the tool point
TAB_C = ((g.TAB_X[0] + g.TAB_X[1]) / 2, (g.TAB_Y[0] + g.TAB_Y[1]) / 2)
FT_HALF = 0.0064                    # half the closed fingertips' thickness along the closing axis
PIN_XY = (g.ALARM_X - 0.030, g.ALARM_Y - 0.020)   # on the front cover's flat middle, beside the compartment's free end
PIN_TILT = math.radians(45.0)
PIN_DEPTH = float(os.environ.get("SA_PIN", "0.040"))      # commanded below the front cover (~13 N)
PRESS_DEPTH = float(os.environ.get("SA_PRESS", "0.026"))  # commanded below where the battery met the floor (press peaks ~20 N)
LEAD = math.radians(float(os.environ.get("SA_LEAD", "6")))
PUSH_X = (g.TAB_X[0] + g.TAB_X[1]) / 2   # the pushing fingertips' centre along the compartment: on the tab, their
                                    # 26 mm width inside the front cover's finger recess
OPEN_Z = g.FLOOR_TOP + g.BATT_T + 0.0025   # the opening push: fingertip bottoms 2.5 mm above the battery
SHUT_Z = 0.020                      # the shutting push, outside the rim
PINCH_X = -0.048                    # the battery's pinch, its centre along the compartment (tray x): the fingers stay
                                    # outside the shroud with the compartment open
PINCH_Z = g.BLOCK_TOP + 0.001       # fingertip bottoms over the floor... 1 mm above the contact block
TILT = math.radians(float(os.environ.get("SA_TILT", "12.5")))  # the battery's snap end tipped up this far to clear the block
PINCH_W_IN = float(os.environ.get("SA_PINCH_W_IN", "0.0235"))  # the new one's pinch: firmer, so it keeps its tilt
PINCH_W = float(os.environ.get("SA_PINCH_W", "0.0245"))  # the pinch's finger opening: a light pinch (the battery 26.5
                                    # wide), firm enough to hold it, loose enough that it turns in the fingers
SLIDE = 0.032                       # slid this far along its tipped axis to come out from under the shroud
SLIDE_IN = float(os.environ.get("SA_SLIDE_IN", "0.030"))   # the new one comes in from this far out along its
                                    # tipped axis: lowered there, its bottom end is clear of the front cover
ENTRY_LIFT = 0.0025                 # ... and this far up (1 mm at the end of the slide)
ENTRY_YAW = math.radians(3.0)       # ... its snap end turned this far toward the alarm's middle while the hand is
                                    # near the latch tab (straightened over the last 16 to 8 mm of the slide)
TRACE = bool(os.environ.get("SA_TRACE"))
PICK_UP = PINCH_Z - g.FLOOR_TOP     # the new battery's pinch: fingertips this far above its bottom
LATCH_OVER = math.radians(20.0)     # the last push aims this far beyond shut (the arm gives about 3 mm per newton)
PICK_YAW0 = float(os.environ.get("SA_PICK_YAW", "-45"))  # the pick's yaw is taken within 90 degrees of this: of
                                     # the two pinches across the battery, the one the wrist can turn to the seat


def grip_for(width_m: float) -> float:
    return max(0.0, min(1.0, width_m * 1000 / GAP_MM))


def R_from(a, c) -> np.ndarray:
    """Gripper rotation with approach ``a`` (its z) and closing axis ``c`` (its x)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    c = np.asarray(c, float)
    c = c - a * (a @ c)
    c /= np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


def tilted(yaw: float, phi: float, toward: float) -> np.ndarray:
    """Fingers pointing down, tipped ``phi`` toward the horizontal direction ``toward``, closing axis horizontal at
    ``yaw``."""
    a = np.array([math.cos(toward) * math.sin(phi), math.sin(toward) * math.sin(phi), -math.cos(phi)])
    return R_from(a, [math.cos(yaw), math.sin(yaw), 0.0])


def down(yaw: float) -> np.ndarray:
    return R_from([0.0, 0.0, -1.0], [math.cos(yaw), math.sin(yaw), 0.0])


def tcp_actual(arm):
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    from scenekit import arm_joints, env

    return ik.tcp_pose(arm_joints(env(), arm.side, ARM_JOINTS), arm.base)


class Scene:
    """The measured scene."""

    def __init__(self, ctx):
        st = ctx.states()
        self.task = ctx.task
        self.alarm = st["smoke_alarm"]
        self.old, self.new = st["old_battery"], st["new_battery"]
        self.hinge = float(self.alarm.joints["hinge"])
        self.tab = float(self.alarm.joints["tab"])
        self.p0 = np.asarray(self.alarm.pos, float)
        self.R0 = ctx.task._rotation(self.alarm.quat_xyzw)
        self.yaw0 = math.atan2(self.R0[1, 0], self.R0[0, 0])
        # the frame the bracket uses (origin at its front's middle, z toward its back): once it is turned face down,
        # pf is on the table and pf + Rf[:, 2] * ALARM_H is the middle of its back
        self.pf, self.Rf = g.front_frame(self.p0, self.R0)

    def tray_point(self, local, z, hinge=None) -> np.ndarray:
        """A point in the compartment's frame (x, y) at height z above the alarm's back (it lies face up), world."""
        th = self.hinge if hinge is None else hinge
        p = g.tray_to_alarm([local[0], local[1], 0.0], th)
        w = self.p0 + self.R0 @ p
        return np.array([w[0], w[1], w[2] + z])

    def tray_rot(self, hinge=None) -> np.ndarray:
        th = self.hinge if hinge is None else hinge
        return self.R0 @ g.rot_z(th)

    def batt(self, name):
        b = getattr(self, "old" if name == "old_battery" else "new")
        return np.asarray(b.pos, float), self.task._rotation(b.quat_xyzw)


def settle_to(arm, pos, rot, tol=0.0006, rounds=3, seconds=0.6):
    """Move to ``pos`` and correct the command by the measured tool-point error."""
    target = np.asarray(pos, float)
    cmd = target.copy()
    yield from arm.move(cmd, rot, seconds)
    for _ in range(rounds):
        yield from wait(0.2)
        err = target - tcp_actual(arm)[0]
        if np.linalg.norm(err) < tol:
            break
        cmd = cmd + err
        yield from arm.move(cmd, rot, 0.3)


# -- left arm: the pin ---------------------------------------------------------------------------------------------------
def pin_pose():
    """Fingers tipped PIN_TILT toward the compartment (the left forearm stays clear of the right arm's), closing axis
    along x; the lowest corner of the closed fingertips on the back, beside the compartment."""
    R = tilted(math.pi, PIN_TILT, -math.pi / 2)
    low = TIP * math.cos(PIN_TILT) + 0.0132 * math.sin(PIN_TILT)
    p = np.array([PIN_XY[0], PIN_XY[1] + TIP * math.sin(PIN_TILT), g.TOP + g.BOX_H + g.ALARM_H + low])
    return p, R


def pin_script(ctx, arm):
    """Holds the alarm down while ctx.flags["pin_want"] is set, waits above it otherwise."""
    p, R = pin_pose()
    wait_p = p + np.array([0.0, 0.03, 0.06])
    yield from arm.move_joints(wait_p, R, 2.0, grip=0.0)
    state = False
    ctx.flags["pin_state"] = state
    while not ctx.flags.get("release_pin"):
        want = bool(ctx.flags.get("pin_want"))
        if want == state:
            yield
            continue
        if want:
            yield from arm.move(p + [0, 0, 0.005], R, 0.8)
            yield from arm.move(p - [0, 0, PIN_DEPTH], R, 1.0)
        else:
            yield from arm.move(wait_p, R, 0.8)
        state = want
        ctx.flags["pin_state"] = state
    yield from arm.move(p + [0, 0, 0.05], R, 0.8)
    ctx.flags["left_free"] = True
    ikmark(ctx, "pin done")


def set_pin(ctx, want: bool):
    ctx.flags["pin_want"] = want
    while ctx.flags.get("pin_state") != want:
        yield


# -- right arm ------------------------------------------------------------------------------------------------------------
def press_tab(ctx, arm):
    sc = Scene(ctx)
    R = down(math.radians(-90))
    top = sc.tray_point(TAB_C, g.TAB_Z[1] + 0.0004 + TIP)
    yield from arm.move_joints(top + [0, 0, 0.04], R, 2.0, grip=0.0)
    yield from settle_to(arm, top + [0, 0, 0.004], R)
    cmd = arm.pos.copy()
    for k in range(200):                                       # down 6 mm/s until the tab has gone 1.6 mm
        if -Scene(ctx).tab >= 0.00155 or top[2] - cmd[2] > 0.012:
            break
        cmd[2] -= 0.0003
        arm.aim(cmd, R)
        yield
    ctx.flags["tab_cmd_mm"] = round((top[2] - cmd[2]) * 1000, 1)
    p_act, _ = tcp_actual(arm)
    arm.pos = p_act
    yield from arm.move(p_act + [0, 0, 0.012], R, 0.4)


def push_arc(ctx, arm, to: float, side: float, z: float, tol: float):
    """Push the tab along the compartment's arc to hinge angle ``to``: the closed fingertips on the tab's inner face
    (``side`` +1, pushing it open) or outer face (-1, pushing it shut), their bottoms ``z`` above the table, led LEAD
    ahead of the measured angle."""
    sc = Scene(ctx)
    yoff = (g.TAB_Y[1] + FT_HALF + 0.0012) if side > 0 else (g.TAB_Y[0] - FT_HALF - 0.0012)
    th = sc.hinge
    R = down(sc.yaw0 + th - math.pi / 2)
    xc = PUSH_X
    p = sc.tray_point((xc, yoff), z + TIP, th)
    yield from arm.move_joints(p + [0, 0, 0.03], R, 1.5, grip=0.0)          # fingers closed: one 12.8 mm block
    yield from settle_to(arm, p, R, rounds=2, seconds=0.5)
    sgn = 1.0 if to > th else -1.0
    stall = 0
    best = th
    for k in range(400):
        sc = Scene(ctx)
        th = sc.hinge
        if (to - th) * sgn <= tol:
            break
        cmd_th = th + sgn * min(LEAD, abs(to - th) + math.radians(1.0))
        y_in = yoff - side * 0.0008                           # press 0.8 mm into the face
        p = sc.tray_point((xc, y_in), z + TIP, cmd_th)
        arm.aim(p, down(sc.yaw0 + cmd_th - math.pi / 2))
        yield
        if TRACE and k % 5 == 0:
            pa, _ = tcp_actual(arm)
            print(f"PUSH k {k} th {math.degrees(th):.2f} cmd {math.degrees(cmd_th):.2f} tcp_err_mm "
                  f"{np.round((p - pa) * 1000, 1).tolist()}", flush=True)
        if (th - best) * sgn > math.radians(0.05):
            best, stall = th, 0
        else:
            stall += 1
        if stall > 60:
            break
    ctx.flags.setdefault("push_log", []).append((round(math.degrees(to), 1), round(math.degrees(Scene(ctx).hinge), 2), k))


def open_door(ctx, arm):
    yield from press_tab(ctx, arm)
    yield from push_arc(ctx, arm, g.SWING, +1, OPEN_Z, math.radians(1.0))
    p_act, _ = tcp_actual(arm)
    arm.pos = p_act
    yield from arm.move(p_act + [0, 0, 0.03], arm.rot, 0.6)


def shut_door(ctx, arm):
    # coming from elsewhere (not from the pocket), first go to above the pocket, where the full demo's hand is
    sc = Scene(ctx)
    stage = sc.tray_point((g.SEAT[0], g.SEAT[1]), 0.10, sc.hinge)
    if np.linalg.norm(tcp_actual(arm)[0] - stage) > 0.06:
        yield from arm.move_joints(stage, down(sc.yaw0 + sc.hinge - math.pi / 2), 1.5)
        ctx.flags["shut_staged"] = True
    yield from push_arc(ctx, arm, math.radians(2.0), -1, SHUT_Z, math.radians(0.5))
    # past the catch: push on toward LATCH_OVER beyond shut until the compartment has stayed shut 0.3 s
    n_shut, over = 0, 0.0
    for k in range(160):
        sc = Scene(ctx)
        n_shut = n_shut + 1 if sc.hinge < g.LATCH_AT - math.radians(0.15) else 0
        if n_shut >= 6:
            break
        over = min(LATCH_OVER, over + math.radians(0.5))
        p = sc.tray_point((PUSH_X, g.TAB_Y[0] - FT_HALF - 0.0004), SHUT_Z + TIP, -over)
        arm.aim(p, down(sc.yaw0 - over - math.pi / 2))
        yield
    ctx.flags["latch_push_deg"] = round(math.degrees(over), 1)
    yield from wait(0.2)
    p_act, _ = tcp_actual(arm)
    arm.pos = p_act
    sc = Scene(ctx)
    away = sc.R0 @ g.rot_z(sc.hinge) @ np.array([0.0, -1.0, 0.0])
    yield from arm.move(p_act + 0.012 * away, arm.rot, 0.4)
    yield from arm.move(arm.pos + [0, 0, 0.05], arm.rot, 0.6)
    ctx.flags["shut_deg"] = round(math.degrees(Scene(ctx).hinge), 3)


def hand_contacts(ctx):
    """Debugging aid: contact forces (N) between the alarm's parts and the right hand's links, and between the old
    battery and the alarm's body."""
    try:
        V = ctx.flags.get("_cview")
        if V is None:
            from yam_arena.robot import asset_link_paths
            from scenekit import env
            import yam_arena.constants as C
            psv = env().unwrapped.sim.physics_sim_view
            ns = "/World/envs/env_*"
            lp = asset_link_paths()
            hands = [f"{ns}/RightArm{lp[link]}" for link in ("gripper", "tip_left", "tip_right")]
            lefts = [f"{ns}/LeftArm{lp[link]}" for link in ("gripper", "tip_left", "tip_right")]
            V = ({**{part: psv.create_rigid_contact_view(f"{ns}/smoke_alarm/{part}", filter_patterns=hands)
                     for part in ("body", "tray", "tab")},
                  "pin": psv.create_rigid_contact_view(f"{ns}/smoke_alarm/body", filter_patterns=lefts)},
                 psv.create_rigid_contact_view(f"{ns}/old_battery", filter_patterns=[f"{ns}/smoke_alarm/body", f"{ns}/smoke_alarm/tray"]),
                 float(C.PHYSICS_DT))
            ctx.flags["_cview"] = V
        out = {}
        for part, v in V[0].items():
            m = v.get_contact_force_matrix(V[2])
            m = np.asarray(m.numpy() if hasattr(m, "numpy") else m).reshape(-1, 3)
            out[part] = [round(float(np.linalg.norm(r)), 2) for r in m]
        m = V[1].get_contact_force_matrix(V[2])
        m = np.asarray(m.numpy() if hasattr(m, "numpy") else m).reshape(-1, 3)
        out["batt_vs_body_tray"] = [np.round(r, 2).tolist() for r in m]
        return out
    except Exception as e:                                  # debugging aid only
        return {"err": repr(e)[:100]}


def lift_straight(ctx, arm):
    """Check: pinch the old battery where the demo does, firmly, and lift the hand straight up 30 mm (its bottom end is
    under the front cover); record how far each end of the battery rose and whether the alarm rose, then let go."""
    sc = Scene(ctx)
    th = sc.hinge
    R = down(sc.yaw0 + th - math.pi / 2)
    p = sc.tray_point((PINCH_X, g.POCKET_YC), PINCH_Z + TIP, th)
    yield from arm.move_joints(p + [0, 0, 0.04], R, 1.5, grip=grip_for(0.042))
    yield from settle_to(arm, p + [0, 0, 0.01], R)
    yield from settle_to(arm, p, R)
    yield from arm.set_grip(grip_for(PINCH_W_IN), 0.6)
    yield from wait(0.3)
    z0 = Scene(ctx).p0[2]
    top, rise = [-1.0, -1.0], 0.0
    for k in range(90):                                          # 30 mm up at 10 mm/s, then held 1.5 s
        arm.aim(p + [0.0, 0.0, 0.0005 * min(k + 1, 60)], R)
        yield
        bb = ctx.task._STATE["batt"]["old_battery"]
        top = [max(top[i], bb["lift"][i]) for i in range(2)]
        rise = max(rise, Scene(ctx).p0[2] - z0)
    bb = ctx.task._STATE["batt"]["old_battery"]
    pa, _ = tcp_actual(arm)
    ctx.flags["straight_lift"] = dict(hand_up_mm=round((pa[2] - p[2]) * 1000, 1),
                                      batt_max_lift_mm=[round(v * 1000, 1) for v in top],
                                      batt_lift_held_mm=[round(v * 1000, 1) for v in bb["lift"]],
                                      in_pocket=bb["in_pocket"], alarm_rise_mm=round(rise * 1000, 1),
                                      snapped=ctx.task._STATE.get("snapped"))
    yield from arm.set_grip(grip_for(0.042), 0.5)
    yield from arm.move(arm.pos + [0, 0, 0.04], R, 0.8)


def take_old(ctx, arm):
    """Pinch the old battery across its width on the part outside the shroud, tip its snap end up about its bottom
    end (under the shroud) until it is off its snaps and clear of the contact block, slide it out along its tipped
    axis from under the shroud, lift it away and lay it on the table."""
    sc = Scene(ctx)
    th = sc.hinge
    # the hand pitched -TILT about the compartment's y (its top toward the free end): once the battery is tipped up
    # TILT it stands upright, so its fingers never lean over the shroud
    R = down(sc.yaw0 + th - math.pi / 2)
    p = sc.tray_point((PINCH_X, g.POCKET_YC), PINCH_Z + TIP, th)
    yield from arm.move_joints(p + [0, 0, 0.04], R, 1.5, grip=grip_for(0.042))
    yield from settle_to(arm, p + [0, 0, 0.01], R)
    yield from settle_to(arm, p, R)
    yield from arm.set_grip(grip_for(PINCH_W), 0.6)
    yield from wait(0.3)
    # tip its snap end up about its bottom end's lower edge until TILT: the hand rises along that arc without
    # turning (the battery turns in the pinch), so its fingers stay upright, clear of the shroud
    p0, R0h = arm.pos.copy(), arm.rot.copy()
    pivot = sc.tray_point((g.BLOCK_FACE + g.BODY_L, g.POCKET_YC), g.FLOOR_TOP, th)
    axis = sc.tray_rot(th) @ np.array([0.0, 1.0, 0.0])          # turning about +y_tray lifts the snap end (-x_tray)
    a, unsnap = 0.0, None
    clear = g.BLOCK_TOP - g.FLOOR_TOP + 0.0015                  # the snap end's case edge this far up clears the block
    for k in range(700):
        bb = ctx.task._STATE["batt"]["old_battery"]
        if unsnap is None and ctx.task._STATE.get("snapped") != "old_battery":
            unsnap = a
        if bb["lift"][0] >= clear:
            break
        if a < math.radians(40.0):                              # the arm lags: the command runs ahead
            a += math.radians(0.25)
        Rr = turn(axis, a)
        arm.aim(pivot + Rr @ (p0 - pivot), R0h)
        yield
        if TRACE and k % 8 == 0:
            s_ = Scene(ctx)
            pa, _ = tcp_actual(arm)
            print(f"TIP a {math.degrees(a):.1f} lift {np.round(np.asarray(bb['lift']) * 1000, 1).tolist()} gap {bb['gap'] * 1000:.1f} "
                  f"ang {math.degrees(bb['ang']):.1f} alarm {np.round((s_.p0 - [g.ALARM_X, g.ALARM_Y, g.TOP]) * 1000, 1).tolist()} "
                  f"tcp_err {np.round((arm.pos - pa) * 1000, 1).tolist()} contacts {hand_contacts(ctx)}", flush=True)
    ctx.flags["unsnap_deg"] = round(math.degrees(unsnap), 1) if unsnap is not None else None
    yield from wait(0.2)
    b = ctx.task._STATE["batt"]["old_battery"]
    ctx.flags["old_tipped"] = dict(lift_mm=[round(v * 1000, 2) for v in b["lift"]], ang_deg=round(math.degrees(b["ang"]), 1))
    # out along its tipped axis (its top face slides along itself, so the shroud's clearance over it stays the same)
    tilt = ctx.task._STATE["batt"]["old_battery"]["ang"]
    d = sc.tray_rot(th) @ np.array([-math.cos(tilt), 0.0, math.sin(tilt)])
    p_act, _ = tcp_actual(arm)
    cmd = p_act + d * 0.004
    arm.pos = cmd
    n = 40
    for k in range(n):
        cmd = cmd + d * (SLIDE / n)
        arm.aim(cmd, arm.rot)
        yield
        if TRACE and k % 4 == 0:
            bb = ctx.task._STATE["batt"]["old_battery"]
            s_ = Scene(ctx)
            pa, _ = tcp_actual(arm)
            print(f"SLIDE k {k} lift {np.round(np.asarray(bb['lift']) * 1000, 1).tolist()} p {np.round(bb['p'] * 1000, 1).tolist()} "
                  f"alarm {np.round((s_.p0 - [g.ALARM_X, g.ALARM_Y, g.TOP]) * 1000, 1).tolist()} tcp_err "
                  f"{np.round((cmd - pa) * 1000, 1).tolist()}", flush=True)
    yield from wait(0.2)
    p_act, _ = tcp_actual(arm)
    arm.pos = p_act
    yield from arm.move(p_act + [0, 0, 0.045], R0h, 1.2)
    p_act, _ = tcp_actual(arm)
    place = np.array([0.195, -0.045, g.TOP + g.BATT_T / 2 + 0.002])
    pb, Rb = Scene(ctx).batt("old_battery")
    off = p_act - pb
    ctx.flags["old_out_mm"] = np.round((pb - [0, 0, g.TOP]) * 1000, 1).tolist()
    yield from arm.move_joints(place + off + [0, 0, 0.03], R0h, 1.5)
    yield from arm.move(place + off, arm.rot, 0.8)
    yield from arm.set_grip(grip_for(0.045), 0.5)
    yield from arm.move(arm.pos + [0, 0, 0.05], arm.rot, 0.6)


def battery_in_hand(ctx, arm, name):
    p, R = tcp_actual(arm)
    pb, Rb = Scene(ctx).batt(name)
    return R.T @ (pb - p), R.T @ Rb


def hand_for_battery(rel, pb, Rb):
    """The hand pose that puts the held battery at (pb, Rb)."""
    t_in, R_in = rel
    Rh = Rb @ R_in.T
    return pb - Rh @ t_in, Rh


def insert_new(ctx, arm):
    """Pinch the new battery on its card (where the old one was pinched: the fingers end up outside the shroud), turn
    it so its snaps point at the contacts at the free end, its snap end tipped up TILT; bring it in bottom end first
    along its tipped axis until its bottom end is under the shroud by the hinge-end wall, lower its snap end onto the
    contacts (the hand turns about the bottom end), slide it onto them and press it down until it snaps."""
    sc = Scene(ctx)
    pn, Rn = sc.batt("new_battery")
    flip = bool(Rn[2, 2] < 0.0)                                  # as it lies: label down (turned over) or up
    off_b = -(PINCH_X - g.SEAT[0])                               # the pinch along the battery from its centre (+x: snaps)
    pick = pn + Rn @ np.array([off_b, 0.0, 0.0])
    yaw = math.atan2(Rn[1, 0], Rn[0, 0]) + math.pi / 2           # closing axis across its width (from its x axis)
    yaw = math.remainder(yaw - math.radians(PICK_YAW0), math.pi) + math.radians(PICK_YAW0)
    # pitched about the battery's y so that, with the battery tipped TILT (snap end up) in the compartment, the hand
    # stands upright: +TILT about the battery's y lifts its bottom end... so the hand takes -TILT about the axis that
    # lifts the snap end
    by = Rn[:, 1] * (1.0 if not flip else -1.0)
    R = turn(by, TILT) @ down(yaw)
    p = np.array([pick[0], pick[1], g.TOP + g.CARD[2] + PICK_UP + TIP])
    yield from arm.move_joints(p + [0, 0, 0.05], R, 2.0, grip=grip_for(0.042))
    yield from settle_to(arm, p + [0, 0, 0.012], R)
    yield from settle_to(arm, p, R)
    yield from arm.set_grip(grip_for(PINCH_W_IN), 0.6)
    yield from wait(0.3)
    yield from arm.move(arm.pos + [0, 0, 0.06], R, 1.0)
    rel = battery_in_hand(ctx, arm, "new_battery")
    ctx.flags["new_in_hand_mm"] = np.round(rel[0] * 1000, 1).tolist()
    sc = Scene(ctx)
    Rt = sc.tray_rot()
    flat = Rt @ g.SEAT_ROT @ (np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3))
    axis = Rt @ np.array([0.0, 1.0, 0.0])

    def pose(tilt, out, lift=0.0, yaw=0.0):
        """The battery's centre and axes: its bottom end's lower edge (the pivot) 0.5 mm short of seated, tipped
        ``tilt`` about the pivot, then moved ``out`` along the tipped axis and ``lift`` up, and turned ``yaw`` about
        the vertical through its bottom end (its snap end toward the alarm's middle)."""
        s = Scene(ctx)
        piv_t = np.array([g.BLOCK_FACE + 0.0005 + g.BODY_L, g.POCKET_YC, g.FLOOR_TOP])
        piv = s.p0 + s.R0 @ g.tray_to_alarm(piv_t, s.hinge)
        Rr = turn(axis, tilt)
        Rb = Rr @ flat
        c = piv - Rb @ np.array([-g.BODY_L / 2, 0.0, -g.BATT_T / 2])
        if flip:
            c = piv - Rb @ np.array([-g.BODY_L / 2, 0.0, g.BATT_T / 2])
        d = Rt @ np.array([-math.cos(tilt), 0.0, math.sin(tilt)])
        sh = d * out + np.array([0.0, 0.0, lift])
        c, pv = c + sh, piv + sh
        if yaw:
            Rz = g.rot_z(-yaw)
            c, Rb = pv + Rz @ (c - pv), Rz @ Rb
        return c, Rb, piv

    def yaw_at(out):
        return ENTRY_YAW * min(1.0, max(0.0, (out - 0.008) / 0.008))

    cb, Rb, piv = pose(TILT, SLIDE_IN, ENTRY_LIFT, yaw_at(SLIDE_IN))
    ph, Rh = hand_for_battery(rel, cb + [0, 0, 0.03], Rb)
    yield from arm.move_joints(ph, Rh, 2.0)
    ph, Rh = hand_for_battery(rel, cb, Rb)
    yield from arm.move(ph, Rh, 1.0)
    cmd = ph.copy()
    for r in range(8):                                           # settle on the battery's measured position
        yield from wait(0.3)
        pb, _ = Scene(ctx).batt("new_battery")
        err = cb - pb
        if np.linalg.norm(err) < 0.0004:
            break
        cmd = cmd + err
        yield from arm.move(cmd, Rh, 0.3)
    ctx.flags["hover_err_mm"] = np.round(err * 1000, 2).tolist()
    # in along its tipped axis, bottom end first, under the shroud; closed loop on its sideways place
    corr = cmd - ph
    ylat = Rt[:, 1]
    lat, n = 0.0, 60
    for k in range(1, n + 1):
        out = SLIDE_IN * (1 - k / n)
        cbk, Rbk, _ = pose(TILT, out, 0.001 + (ENTRY_LIFT - 0.001) * (1 - k / n), yaw_at(out))
        phk, Rhk = hand_for_battery(rel, cbk, Rbk)
        pb, _ = Scene(ctx).batt("new_battery")
        e = float(ylat @ (cbk - pb))
        lat = max(-0.003, min(0.003, lat + max(-0.0003, min(0.0003, 0.3 * e))))
        arm.aim(phk + corr + ylat * lat, Rhk)
        yield
    ctx.flags["slide_lat_mm"] = round(lat * 1000, 2)
    yield from wait(0.3)
    b = ctx.task._STATE["batt"]["new_battery"]
    ctx.flags["slid_under"] = dict(gap_mm=round(b["gap"] * 1000, 2), side_mm=round(b["side"] * 1000, 2),
                                   lift_mm=[round(v * 1000, 2) for v in b["lift"]])
    # lower the snap end onto the contacts: the hand goes down along the arc about the bottom end without turning
    # (the battery turns in the pinch)
    p0h, R0h = arm.pos.copy(), arm.rot.copy()
    a = 0.0
    while a < TILT:
        a = min(TILT, a + math.radians(0.3))
        Rr = turn(axis, -a)
        arm.aim(piv + Rr @ (p0h - piv), R0h)
        yield
    yield from wait(0.3)
    cmd = arm.pos.copy()
    b = ctx.task._STATE["batt"]["new_battery"]
    ctx.flags["placed"] = dict(gap_mm=round(b["gap"] * 1000, 2), side_mm=round(b["side"] * 1000, 2),
                               lift_mm=[round(v * 1000, 2) for v in b["lift"]])
    # slide it onto its contacts on the floor (toward the free end) until the case is 0.5 mm from the block, then
    # press it down through the pinch: "push the battery in until it snaps"
    xt = -Scene(ctx).tray_rot()[:, 0]
    Rh = arm.rot.copy()
    for k in range(80):
        b = ctx.task._STATE["batt"]["new_battery"]
        if b["gap"] <= 0.0005:
            break
        cmd[:2] += 0.0001 * xt[:2]
        cmd[2] -= 0.00003
        arm.aim(cmd, Rh)
        yield
    ctx.flags["slid_in_mm"] = round(ctx.task._STATE["batt"]["new_battery"]["gap"] * 1000, 2)
    z0 = cmd[2]
    fmax = 0.0
    for k in range(int(PRESS_DEPTH / 0.0004) + 12):           # down 8 mm/s, then hold
        if cmd[2] > z0 - PRESS_DEPTH:
            cmd[2] -= 0.0004
        arm.aim(cmd, Rh)
        yield
        b = ctx.task._STATE["batt"]["new_battery"]
        fmax = max(fmax, b["force"])
        if TRACE and k % 4 == 0:
            print(f"PRESS k {k} F {b['force']:.2f} lift {np.round(np.asarray(b['lift']) * 1000, 2).tolist()} "
                  f"gap {b['gap'] * 1000:.2f} snapped {ctx.task._STATE['snapped']}", flush=True)
    ctx.flags["press_fmax"] = round(fmax, 2)
    yield from wait(0.2)
    yield from arm.set_grip(grip_for(0.036), 0.4)
    p_act, _ = tcp_actual(arm)
    arm.pos = p_act
    yield from arm.move(p_act + [0, 0, 0.03], Rh, 0.8)


def flip_new(ctx):
    """Variant setup: the new battery turned over on its card (label down)."""
    import torch

    from scenekit import env

    base = env().unwrapped
    a = base.scene["new_battery"]
    o = base.scene.env_origins[0]
    h = g.NEW_YAW / 2                                          # xyzw: turned to NEW_YAW, then half a turn about its x
    pose = torch.tensor([[g.NEW_X, g.NEW_Y, g.TOP + g.CARD[2] + g.BATT_T / 2 + 0.0003,
                          math.cos(h), math.sin(h), 0.0, 0.0]], dtype=torch.float32, device=a.device)
    pose[0, :3] += o
    for _ in range(20):                                        # until it reads label down (a reset can overwrite it)
        a.write_root_link_pose_to_sim_index(root_pose=pose)
        a.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=a.device))
        yield
        if Scene(ctx).batt("new_battery")[1][2, 2] < -0.99:
            break
    ctx.flags["new_label_down"] = bool(Scene(ctx).batt("new_battery")[1][2, 2] < -0.99)


# -- both arms: the alarm onto its bracket -----------------------------------------------------------------------------
SQUEEZE = float(os.environ.get("SA_SQUEEZE", "0.030"))     # each hand commanded this far into the band
GRIP_A = {"left": math.radians(130.0), "right": math.radians(310.0)}  # where each hand holds the band (alarm frame),
                                     # clear of the compartment's opening at the rim (236 to 297 degrees)
BETA = {"left": math.radians(70.0), "right": math.radians(-80.0)}     # each hand's roll from fingers-down (see side_grip)
FLIP = {"left": True, "right": False}
HC = 0.040                           # the fingertips' tapered side, its middle this far above the alarm's front
SEED = {"left": (-1.2806, 1.8996, 0.9325, -0.6419, -1.2465, -0.1722),  # IK seeds for the first approach (offline IK)
        "right": (0.2835, 2.0345, 0.5645, 1.2406, -0.7801, 1.4722)}
TWIST = math.radians(float(os.environ.get("SA_TWIST", "23")))
LEVEL_GAIN = float(os.environ.get("SA_LEVEL", "0.08"))   # levelling: each hand moves this fraction of its side's height
NF_ANG = math.atan2(5.5, 14.0)       # the closed fingertips' tapered inner side: 21.4 degrees off the approach axis
PF = np.array([0.0, 0.01045, 0.005])  # its middle, from the tool point (gripper frame)


def side_grip(aw: float, beta: float, flip: bool):
    """A side grip on the band at angle ``aw`` (world): the closed fingertips' tapered inner side flat on the band, the
    hand rolled ``beta`` about the band's normal from fingers-down (70 degrees: the fingers along the band's tangent,
    20 degrees below level). Returns the rotation and the tool point's offset from the contact."""
    u = np.array([math.cos(aw), math.sin(aw), 0.0])
    t = np.array([math.sin(aw), -math.cos(aw), 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    y = math.cos(NF_ANG) * (-u) + math.sin(NF_ANG) * ez
    z = -math.cos(NF_ANG) * ez - math.sin(NF_ANG) * u
    R = turn(-u, beta) @ np.column_stack([t, y, z])
    off = R @ PF
    if flip:                         # the other fingertip's side: the same contact, the hand turned over
        R = R @ np.diag([-1.0, -1.0, 1.0])
    return R, off


def ikmark(ctx, what):
    """The log: each arm's worst IK error so far (mm), and how far its tool point is from its command now."""
    out = {}
    for arm in (ctx.left, ctx.right):
        pa, _ = tcp_actual(arm)
        out[arm.side] = (round(arm.worst_mm, 1), round(float(np.linalg.norm(arm.pos - pa)) * 1000, 1))
    ctx.flags.setdefault("ik_marks", []).append((what, ctx.task._STATE["step"], out))


def mtrace(ctx, what):
    """Mount progress: the alarm's pose and each hand's distance from its axis (the log)."""
    s = Scene(ctx)
    hands = {arm.side: tcp_actual(arm)[0] for arm in (ctx.left, ctx.right)}
    r = {k: round(float(np.linalg.norm((v - s.pf)[:2])) * 1000, 1) for k, v in hands.items()}
    z = {k: round(float(v[2] - g.TOP) * 1000, 1) for k, v in hands.items()}
    f = {}
    try:
        V = ctx.flags.get("_hand_view")
        if V is None:
            from yam_arena.robot import asset_link_paths
            from scenekit import env
            import yam_arena.constants as C
            psv = env().unwrapped.sim.physics_sim_view
            ns = "/World/envs/env_*"
            lp = asset_link_paths()
            hands = [f"{ns}/{arm}{lp[link]}" for arm in ("LeftArm", "RightArm") for link in ("gripper", "tip_left", "tip_right")]
            V = (psv.create_rigid_contact_view(f"{ns}/smoke_alarm/body", filter_patterns=hands), float(C.PHYSICS_DT))
            ctx.flags["_hand_view"] = V
        m = V[0].get_contact_force_matrix(V[1])
        m = np.asarray(m.numpy() if hasattr(m, "numpy") else m).reshape(-1, 3)
        f = {k: round(float(np.linalg.norm(m[i])), 2) for i, k in enumerate(("Lg", "Ltl", "Ltr", "Rg", "Rtl", "Rtr"))}
    except Exception as e:                                  # debugging aid only
        f = {"err": repr(e)[:80]}
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, s.Rf[2, 2]))))
    print(f"MOUNT {what}: alarm {np.round((s.pf - [0, 0, g.TOP]) * 1000, 1).tolist()} yaw {math.degrees(s.yaw0):.2f} "
          f"tilt {tilt:.2f} hand r {r} z {z} forces {f}", flush=True)
    return f


def both(ctx, fn, seconds):
    """Both arms to the poses fn(arm) -> (p, R), together, in straight lines."""
    yield from together(*(arm.move(*fn(arm), seconds) for arm in (ctx.left, ctx.right)))


def mount(ctx):
    """Grip the alarm's band from opposite sides with both hands' closed fingertips (side grips, the hands along the
    band's tangent), lift it, carry it under the bracket, raise it until its back meets the ring's face, turn it
    clockwise seen from below until it locks, let go and draw the hands out under the board."""
    ctx.flags["view"] = "mount"
    sc = Scene(ctx)
    c, yaw = sc.pf.copy(), math.remainder(sc.yaw0, g.ENTRY_STEP)   # face down: the grips as for any of the six angles
    b = np.array([g.BRACKET_X, g.BRACKET_Y])

    def at(centre, front, dr, phi=0.0, dz=None):
        """Each hand's pose: its contact dr outside the band (negative: pressed in), HC above the alarm's front at
        height ``front`` (plus dz[side])."""
        def fn(arm):
            aw = yaw + phi + GRIP_A[arm.side]
            R, off = side_grip(aw, BETA[arm.side], FLIP[arm.side])
            r = g.ALARM_R + dr
            cp = np.array([centre[0] + r * math.cos(aw), centre[1] + r * math.sin(aw),
                           front + HC + (dz or {}).get(arm.side, 0.0)])
            return cp - off, R
        return fn

    f0 = c[2]
    # each hand up 80 mm from where it is, then (joint space) to 60 mm out from the band and 80 mm up, then straight in
    yield from together(*(arm.move(arm.pos + [0.0, 0.0, 0.080], arm.rot, 1.0, grip=0.0) for arm in (ctx.left, ctx.right)))
    yield from together(*(arm.move_joints(*at(c, f0 + 0.080, 0.060)(arm), 2.5, seeds=(SEED[arm.side],))
                          for arm in (ctx.left, ctx.right)))
    yield from both(ctx, at(c, f0 + 0.030, 0.015), 1.2)
    yield from both(ctx, at(c, f0, 0.015), 1.0)
    yield from both(ctx, at(c, f0, 0.002), 0.6)
    yield from both(ctx, at(c, f0, -SQUEEZE), 1.5)
    yield from wait(0.3)
    ctx.flags["squeeze_N"] = mtrace(ctx, "squeezed")
    ikmark(ctx, "squeezed")
    yield from both(ctx, at(c, f0 + 0.005, -SQUEEZE), 0.8)
    yield from wait(0.3)
    mtrace(ctx, "lifted")
    ikmark(ctx, "lifted")
    z = f0 + 0.005
    hand_c = c[:2].copy()
    dz = {"left": 0.0, "right": 0.0}

    def steer(phi, centre_gain=0.1):
        s = Scene(ctx)
        n = s.Rf[:, 2]
        topc = s.pf + s.Rf @ np.array([0, 0, g.ALARM_H])
        for side in dz:
            a = yaw + phi + GRIP_A[side]
            h = -(n[0] * math.cos(a) + n[1] * math.sin(a)) / n[2] * g.ALARM_R   # its back's height there vs its middle
            dz[side] = float(np.clip(dz[side] - LEVEL_GAIN * h, -0.010, 0.010))
        if centre_gain:
            hand_c[:] += centre_gain * (b - topc[:2])
            d = hand_c - b
            if np.linalg.norm(d) > 0.012:
                hand_c[:] = b + d * 0.012 / np.linalg.norm(d)
        return topc, math.degrees(math.acos(min(1.0, n[2])))

    # up to 60 mm, then carry it under the ring, its back 15 mm below the ring's face, levelling it as it goes
    z_carry = g.TOP + g.MOUNT_H - g.ALARM_H - 0.015
    for k in range(30):
        steer(0.0, 0.0)
        z = f0 + 0.005 + (0.060 - 0.005) * ik_smooth((k + 1) / 30)
        for arm in (ctx.left, ctx.right):
            arm.aim(*at(hand_c, z, -SQUEEZE, 0.0, dz)(arm))
        yield
    start, n = hand_c.copy(), 100
    p0 = math.remainder(round(yaw / g.ENTRY_STEP) * g.ENTRY_STEP - yaw, 2 * math.pi)  # to the nearest entry position
    ctx.flags["entry_turn_deg"] = round(math.degrees(p0), 2)
    for k in range(n):
        s_ = ik_smooth((k + 1) / n)
        steer(p0 * s_, 0.0)
        hand_c[:] = start + (b - start) * s_
        zz = z + (z_carry - z) * s_
        for arm in (ctx.left, ctx.right):
            arm.aim(*at(hand_c, zz, -SQUEEZE, p0 * s_, dz)(arm))
        yield
    z = z_carry
    mtrace(ctx, "carried")
    ikmark(ctx, "carried")
    for k in range(3):
        yield from wait(0.3)
        s = Scene(ctx)
        topc = s.pf + s.Rf @ np.array([0, 0, g.ALARM_H])
        err = b - topc[:2]
        mtrace(ctx, f"under the ring {k}")
        if np.linalg.norm(err) < 0.0007 or np.linalg.norm(err) > 0.03:
            break
        hand_c[:] += err
        yield from both(ctx, at(hand_c, z, -SQUEEZE, p0, dz), 0.5)
    s = Scene(ctx)
    ctx.flags["under_ring_mm"] = np.round(((s.pf + s.Rf @ np.array([0, 0, g.ALARM_H]))[:2] - b) * 1000, 2).tolist()
    # up until its back meets the ring's face, read off its pose: each step the hands level it and keep the middle of
    # its back under the ring's middle
    for k in range(400):
        topc, tilt = steer(p0)
        if topc[2] >= g.TOP + g.MOUNT_H - 0.0003 and tilt < 1.5:
            break
        if topc[2] < g.TOP + g.MOUNT_H - 0.0003:
            z += 0.0001
        for arm in (ctx.left, ctx.right):
            arm.aim(*at(hand_c, z, -SQUEEZE, p0, dz)(arm))
        yield
    mtrace(ctx, "at the ring")
    ikmark(ctx, "at the ring")
    s = Scene(ctx)
    yaw_start = s.yaw0
    topc = s.pf + s.Rf @ np.array([0, 0, g.ALARM_H])
    ctx.flags["seat"] = dict(gap_mm=round((g.TOP + g.MOUNT_H - topc[2]) * 1000, 2),
                             off_mm=round(float(np.linalg.norm(topc[:2] - b)) * 1000, 2),
                             tilt_deg=round(math.degrees(math.acos(min(1.0, s.Rf[2, 2]))), 2))
    # 1 mm more, then turn it clockwise seen from below (+z) about its axis until it has turned TWIST - 1.5 degrees
    z += 0.001
    phi, turned = p0, 0.0
    for k in range(240):
        s = Scene(ctx)
        turned = math.remainder(s.yaw0 - yaw_start, 2 * math.pi)
        if turned >= TWIST - math.radians(1.5) or phi - p0 >= TWIST + math.radians(10.0):
            break
        if k >= 6:
            phi += math.radians(0.5)
        steer(phi)
        for arm in (ctx.left, ctx.right):
            arm.aim(*at(hand_c, z, -SQUEEZE, phi, dz)(arm))
        yield
    ctx.flags["turned_deg"] = round(math.degrees(turned), 2)
    ctx.flags["hands_turned_deg"] = round(math.degrees(phi - p0), 2)
    mtrace(ctx, "turned")
    ikmark(ctx, "turned")
    yield from wait(0.3)
    # let go: out 20 mm, down 60 mm; then the left hand down 50 mm and 200 mm toward the robot, out from under the
    # board, and up above it (its ready pose is over the board) before both go home
    yield from both(ctx, at(hand_c, z, 0.020, phi, dz), 0.6)
    yield from both(ctx, at(hand_c, z - 0.060, 0.020, phi, dz), 1.0)
    ctx.flags["view"] = "mounted"
    ikmark(ctx, "let go")
    pl, Rl = ctx.left.pos.copy(), ctx.left.rot.copy()
    yield from together(ctx.left.move(pl + [0.0, 0.0, -0.050], Rl, 0.8))
    ikmark(ctx, "left down 50")
    yield from together(ctx.left.move(pl + [-0.200, 0.0, -0.050], Rl, 1.5))
    ikmark(ctx, "left 200 toward the robot")
    up = np.array([pl[0] - 0.200, pl[1], g.TOP + g.BOARD_Z + g.BOARD_T + 0.070])
    yield from together(ctx.left.move_joints(up, ctx.left.home[1], 1.5))
    ikmark(ctx, "left up above the board")


# -- right arm: turn the alarm over --------------------------------------------------------------------------------------
# One hand pinches the alarm's rim across its thickness (the rim overhangs the box on the robot's side), lifts it,
# turns it face down by turning its wrist about its own axis (which points at the alarm's middle) and lays it face
# down on the box again, its rim overhanging as before, and lets go.
RIM_IN = float(os.environ.get("SA_RIM_IN", "0.025"))   # the fingertips this far in from the rim
RIM_ZC = 0.022                       # the pinch's middle this far from the alarm's back
FLIP_LIFT = 0.135                    # the tool point this high above the table while it turns over
FLIP_BACK = 0.035                    # and drawn this far back toward the robot
FLIP_SEED = tuple(math.radians(v) for v in (91, 106, 47, 24, 90, -55))  # the approach's arm branch (offline IK: the
                                     # wrist is at its limit there; this branch reaches the pinch, another one does not)


def rim_pose(centre, side_angle, z, closing):
    """The hand pinching the rim at world angle ``side_angle`` of an alarm centred at ``centre`` (x, y): it points at
    the alarm's middle (approach horizontal), the jaws close along ``closing`` (+-vertical), the tool point at height
    z, the fingertips RIM_IN inside the rim."""
    u = np.array([math.cos(side_angle), math.sin(side_angle), 0.0])
    p = np.array([centre[0], centre[1], z]) + u * (g.ALARM_R - RIM_IN + TIP)
    a = -u
    c = np.array([0.0, 0.0, closing])
    return p, np.column_stack([c, np.cross(a, c), a])


def ftrace(ctx, what):
    s = Scene(ctx)
    pa, _ = tcp_actual(ctx.right)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, s.R0[2, 2]))))
    print(f"FLIP {what}: alarm {np.round((s.p0 - [0, 0, g.TOP]) * 1000, 1).tolist()} yaw {math.degrees(s.yaw0):.2f} "
          f"turned {tilt:.1f} hand {np.round((pa - [0, 0, g.TOP]) * 1000, 1).tolist()} err "
          f"{np.round((ctx.right.pos - pa) * 1000, 1).tolist()}", flush=True)


def flip(ctx):
    """Right arm: pinch the rim on the robot's side, lift it, turn it over on the wrist, lay it face down on the box
    (its rim over the box's edge, as before) and let go."""
    arm = ctx.right
    ctx.flags["view"] = "flip"
    sc = Scene(ctx)
    c = sc.p0.copy()                                   # the middle of its back (face up, on the box)
    side = math.pi                                     # the robot's side
    z = c[2] + RIM_ZC
    p, R = rim_pose(c, side, z, 1.0)
    u = np.array([math.cos(side), math.sin(side), 0.0])
    # come in from the front: well in front of it and above, then down, then straight in
    yield from arm.move_joints(p + u * 0.110 + [0, 0, 0.080], R, 2.0, grip=1.0, seeds=(FLIP_SEED,), seed_tol=0.006)
    yield from arm.move(p + u * 0.070, R, 1.2)
    yield from arm.move(p, R, 1.2)
    sign = -1.0 if arm.q[5] > 0 else 1.0               # turn the way that takes the wrist roll (j6) through zero
    ctx.flags["flip_j6_start_deg"] = round(math.degrees(arm.q[5]), 1)
    yield from arm.set_grip(0.0, 0.8)
    yield from wait(0.4)
    ftrace(ctx, "pinched")
    up = np.array([p[0] - FLIP_BACK, p[1], g.TOP + FLIP_LIFT])
    yield from arm.move(p + [0, 0, 0.012], R, 0.8)
    yield from arm.move(up, R, 1.5)
    ftrace(ctx, "lifted")
    a = -u
    n = 80
    for k in range(1, n + 1):
        arm.aim(up, turn(a, sign * math.pi * ik_smooth(k / n)) @ R)
        yield
    Rf = turn(a, sign * math.pi) @ R
    yield from wait(0.3)
    ftrace(ctx, "turned over")
    # lay it face down on the box where it lay: its front 1 mm above the box, then let go
    s = Scene(ctx)
    pa, _ = tcp_actual(arm)
    front_below = pa[2] - s.pf[2]                       # the front's middle below the tool point
    goal = np.array([p[0], p[1], g.TOP + g.BOX_H + front_below + 0.001])
    yield from arm.move(goal + [0, 0, 0.030], Rf, 1.5)
    yield from arm.move(goal, Rf, 1.0)
    yield from wait(0.2)
    yield from arm.set_grip(1.0, 0.6)
    yield from arm.move(goal + u * 0.060, Rf, 1.0)
    yield from arm.move(arm.pos + [0, 0, 0.060], Rf, 0.6)
    yield from wait(0.5)
    s = Scene(ctx)
    ctx.flags["flip"] = dict(face_down=bool(s.R0[2, 2] < -0.98), pos_mm=np.round((s.pf - [0, 0, g.TOP]) * 1000, 1).tolist(),
                             yaw_deg=round(math.degrees(s.yaw0), 1), worst_ik_mm=round(arm.worst_mm, 1))
    ftrace(ctx, "laid face down")
    ikmark(ctx, "turn-over done")


def ik_smooth(s: float) -> float:
    return s * s * (3 - 2 * s)


def right_script(ctx, variant):
    arm = ctx.right
    while "pin_state" not in ctx.flags:
        yield
    if variant == "mount_only":                      # a check of the turn-over and mount alone (the old battery stays in)
        return
    yield from open_door(ctx, arm)
    if variant == "open_only":
        return
    yield from set_pin(ctx, True)
    if variant == "lift_straight":                   # a check: the front cover stops a battery lifted straight up
        yield from lift_straight(ctx, arm)
        return
    yield from take_old(ctx, arm)
    if variant == "pull_leave":
        return
    if variant in ("latch_empty", "relatch"):
        yield from shut_door(ctx, arm)
        if variant == "latch_empty":
            return
        yield from set_pin(ctx, False)
        yield from open_door(ctx, arm)
        yield from set_pin(ctx, True)
    yield from insert_new(ctx, arm)
    if variant == "no_latch":
        return
    yield from shut_door(ctx, arm)


def script(ctx, variant: str):
    print(f"MODE variant={variant}", flush=True)
    if variant == "reversed":
        yield from flip_new(ctx)

    def right():
        yield from right_script(ctx, variant)
        ctx.flags["release_pin"] = True
        while not ctx.flags.get("left_free"):
            yield
        if variant in ("full", "relatch", "mount_only", "turned_over"):
            if variant != "mount_only":                 # both arms home first, so the turn-over starts as it does there
                yield from together(ctx.left.go_home(2.0), ctx.right.go_home(2.0))
            yield from together(ctx.left.go_home(2.0), ctx.right.move(ctx.right.pos + [0, 0, 0.05], ctx.right.rot, 1.0))
            yield from flip(ctx)
        if variant in ("full", "relatch", "mount_only"):
            yield from mount(ctx)
        yield from together(ctx.left.go_home(2.0), ctx.right.go_home(2.0))
        yield from wait(0.5)
        ikmark(ctx, "home")
        ctx.flags["view"] = "side"

    yield from together(right(), pin_script(ctx, ctx.left))
    print("FLAGS", json.dumps({k: v for k, v in ctx.flags.items() if k not in ("pin_state", "pin_want", "release_pin", "left_free", "_hand_view", "_cview")}), flush=True)
    yield from wait(1.0)


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    bs = " ".join(f"{n[:3]} gap {b['gap'] * 1000:.1f} lift {b['lift'][0] * 1000:.1f}/{b['lift'][1] * 1000:.1f} "
                  f"side {b['side'] * 1000:.1f} F {b['force']:.1f}" for n, b in S.get("batt", {}).items())
    sc = Scene(ctx)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, sc.R0[2, 2]))))
    return (f"tilt {tilt:.2f} hinge {math.degrees(S['hinge']):.2f} max {math.degrees(S['hinge_max']):.1f} tab {S['tab'] * 1000:.2f} "
            f"latched {S['latched']} snapped {S['snapped']} counter {S['on_counter']} mount gap {S['mount_gap'] * 1000:.1f} "
            f"off {S['mount_off'] * 1000:.1f} twist {math.degrees(S['twist']):.1f} seated {S['seated']} locked {S['locked']} "
            f"tier {S['tier']} "
            f"alarm {np.round((sc.p0 - [g.ALARM_X, g.ALARM_Y, g.TOP]) * 1000, 1).tolist()} {bs} events {S['events'][-3:]}")


def close_view(ctx, gen):
    """Recorded runs: once the compartment is open, the top camera moves to a close view of it (from the robot's right
    and above), so the video shows the battery swap. Only the recording changes; the demo reads object poses."""
    import torch

    base = ctx.emb._env.unwrapped
    cam = base.scene["top_cam"]
    o = base.scene.env_origins[0:1].to(torch.float32)
    moved = mount_view = side_view = flip_view = False

    def look(eye, t):
        cam.set_world_poses_from_view(torch.tensor(np.array([eye]), dtype=torch.float32, device=o.device) + o,
                                      torch.tensor(np.array([t]), dtype=torch.float32, device=o.device) + o)

    for _ in gen:
        yield
        if not moved and Scene(ctx).hinge > math.radians(45):
            # from outside the rim, beyond the hinge: the front cover over the battery's hinge end in front, the
            # hand at its free end behind it
            t = np.array([g.ALARM_X + 0.0, g.ALARM_Y - 0.078, g.TOP + g.BOX_H + 0.018])
            look(t + np.array([0.10, -0.13, 0.075]), t)
            moved = True
            ctx.flags["close_view_step"] = ctx.task._STATE["step"]
            print("top camera: close view", flush=True)
        if not flip_view and ctx.flags.get("view") == "flip":
            # the turn-over: the alarm and both hands, from the front right, a little above
            t = np.array([g.ALARM_X - 0.03, g.ALARM_Y, g.TOP + 0.110])
            look(np.array([g.ALARM_X + 0.34, g.ALARM_Y - 0.30, g.TOP + 0.24]), t)
            flip_view = True
            ctx.flags["flip_view_step"] = ctx.task._STATE["step"]
            print("top camera: turn-over view", flush=True)
        if not mount_view and ctx.flags.get("view") == "mount":
            # the second close view: the alarm and the bracket, from the far right, below the board
            t = np.array([g.BRACKET_X - 0.05, g.BRACKET_Y - 0.05, g.TOP + 0.110])
            look(np.array([g.BRACKET_X + 0.42, g.BRACKET_Y - 0.30, g.TOP + 0.150]), t)
            mount_view = True
            ctx.flags["mount_view_step"] = ctx.task._STATE["step"]
            print("top camera: mount view", flush=True)
        if not side_view and ctx.flags.get("view") == "side":
            # the arms home: a low side view of the alarm hanging on its bracket, from the far right, near table height
            t = np.array([g.BRACKET_X, g.BRACKET_Y, g.TOP + 0.120])
            look(np.array([g.BRACKET_X + 0.30, g.BRACKET_Y - 0.30, g.TOP + 0.050]), t)
            side_view = True
            ctx.flags["side_view_step"] = ctx.task._STATE["step"]
            print("top camera: side view", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--close-view", action="store_true")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds after the script ends (inf: to the step limit)")
    ap.add_argument("--variant", default="full",
                    choices=["full", "pull_leave", "no_latch", "open_only", "reversed", "latch_empty", "relatch",
                             "no_mount", "turned_over", "mount_only", "lift_straight", "nothing"])
    a = ap.parse_args()
    if a.variant == "nothing":
        def scr(ctx):
            print("MODE variant=nothing", flush=True)
            yield from wait(5.0)
    else:
        def scr(ctx):
            if a.close_view and not a.no_record:
                yield from close_view(ctx, script(ctx, a.variant))
            else:
                yield from script(ctx, a.variant)
    res = run(a.task, a.out, scr, max_steps=a.max_steps, record=not a.no_record, log=log,
              log_every=int(os.environ.get("SA_LOG_EVERY", "20")), hold_s=a.hold)
    S = sys.modules["task"]._STATE if "task" in sys.modules else {}
    extra = {"variant": a.variant, "tier": S.get("tier"), "snapped": S.get("snapped"), "latched": S.get("latched"),
             "hinge_deg": round(math.degrees(S.get("hinge", 0.0)), 3),
             "hinge_max_deg": round(math.degrees(S.get("hinge_max", 0.0)), 2), "on_counter": S.get("on_counter"), "mounted": S.get("mounted"),
             "events": S.get("events"), "snap_record": S.get("snap_record"),
             "batt": {n: {k: (np.round(np.asarray(v) * 1000, 2).tolist() if k in ("p", "lift") else
                              (round(v * 1000, 2) if k in ("gap", "side") else v)) for k, v in b.items()}
                      for n, b in S.get("batt", {}).items()}}
    print("EXTRA " + json.dumps(extra, default=str), flush=True)
    (pathlib.Path(a.out) / "extra.json").write_text(json.dumps(extra, indent=2, default=str))
