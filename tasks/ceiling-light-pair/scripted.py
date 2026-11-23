"""The scripted demo: one generator script per station (yam_pair CONTRACT §5), each in its own station frame,
reading object poses from the simulator (privileged). The two stations share no state: each runs on its own clock and
on what it reads of the housing and the screws.

Each station's outer arm (A's left at world y +0.805, B's right at -0.805) pinches the rubber trim on its end wall's
lower edge and holds that end. A's inner arm takes A's screwdriver before the raise and never touches the housing.
B's inner arm pinches B's end wall too; once A's screw is in, it hands its share of the load to B's outer arm and takes
B's screwdriver. Each inner arm drives the screw at its own end. The supports are closed loops on the housing's
measured height at each hand (they make up for the arm sagging under the load) and, for the outer hand, on its end's
slot lined up under the pilot hole. Variants (CLP_VARIANT) are the scorer checks of the README.
"""

from __future__ import annotations

import math
import os

import numpy as np
from yam_arena import ik
from yam_pair.demo import together, wait

import geometry as g

CEIL = g.CEIL
INNER_ELEV = 45.0
RAISE_S = 5.0
DEBUG = bool(os.environ.get("CLP_DEBUG"))
OUTER_ELEV = 60.0
ALIGN_GAP = 0.003             # raised to 3 mm under the ceiling, lined up there, then pressed
PART_TURNS = 6                # the checks that leave a screw short: 6 of its 10.4 turns
EXTRA_PRESS = 0.025           # the outer hand presses harder before the inner one lets go
PRESS = 0.004                 # each hand 4 mm above where it would hold the housing just flush


def rotm(approach, finger) -> np.ndarray:
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    f = np.asarray(finger, float)
    f = f - a * (f @ a)
    f = f / np.linalg.norm(f)
    return np.column_stack([f, np.cross(a, f), a])


def elevated(elev_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    e, y = math.radians(elev_deg), math.radians(yaw_deg)
    return np.array([math.cos(e) * math.cos(y), math.cos(e) * math.sin(y), math.sin(e)])


def _rot_q(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def best_rot(arm, pos, approach, fingers) -> np.ndarray:
    """Of the finger directions given (the two signs of one axis grip the same), the one the arm reaches exactly."""
    best = None
    for f in fingers:
        R = rotm(approach, f)
        quat = ik.matrix_to_quat_xyzw(R)
        for q0 in [arm.q] + [np.array(s, float) for s in arm.SEEDS] + [np.array([0.6, 1.5, 1.4, 0.2, 0.0, 0.0]),
                                                                       np.array([-0.6, 1.5, 1.4, 0.2, 0.0, 0.0])]:
            q, err, rerr = ik.solve(pos, quat, q_init=q0, base_pos=arm.base, iters=200)
            key = err + 0.05 * rerr
            if best is None or key < best[0]:
                best = (key, R)
    return best[1]


def smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Station:
    """What one station's script knows: its own frame, its two arms, the object poses it reads."""

    def __init__(self, ctx, s: str, variant: str):
        self.ctx, self.s, self.variant = ctx, s, variant
        self.m = 1.0 if s == "a" else -1.0
        self.outer = ctx.left if s == "a" else ctx.right
        self.inner = ctx.right if s == "a" else ctx.left
        self.phase = "start"
        W, L, D, WT = g.HOUSING_W, g.HOUSING_L, g.HOUSING_D, g.WALL_T
        # support points on the housing's rims, in its own frame (origin: the back's top face middle)
        # both hands pinch this end's end wall from below, 140 mm apart along it (the outer one toward the far side)
        self.contact = {"outer": np.array([0.10, self.m * (L / 2 - WT / 2), -D + 0.014]),
                        "inner": np.array([-0.04, self.m * (L / 2 - WT / 2), -D + 0.014])}
        self.elev = {"outer": OUTER_ELEV, "inner": INNER_ELEV}
        self.zoff = {"outer": 0.0, "inner": 0.0}
        self.active = {"outer": False, "inner": False}
        self.unloading = {"outer": False, "inner": False}
        self.corr = np.zeros(2)
        self.t_raise = None
        self.press_t = None
        self.extra_t = None
        self.R = {}

    # -- reading the scene ---------------------------------------------------------------------------------------
    def housing(self):
        h = self.ctx.states()["housing"]
        return np.asarray(h.pos, float), _rot_q(h.quat_xyzw)

    def pz(self, s=None) -> float:
        return float(self.ctx.states()["screws"].joints[f"{s or self.s}_pz"])

    def slot_world(self) -> np.ndarray:
        p, R = self.housing()
        hx, hy = g.SCREW_XY[self.s]
        return p + R @ np.array([hx - g.HOUSING_X, hy, 0.0])

    def L(self, p) -> np.ndarray:
        return self.ctx.local(np.asarray(p, float))

    def arm(self, which):
        return self.outer if which == "outer" else self.inner

    # -- supports ------------------------------------------------------------------------------------------------------
    def top_target(self) -> float:
        """The housing's back height the supports aim for: up from its stands to 3 mm into the ceiling (pressing)."""
        if self.t_raise is None:
            return CEIL - g.START_DROP
        if self.press_t is not None:
            return CEIL
        s = smooth((self.ctx.t - self.t_raise) / RAISE_S)
        return CEIL - g.START_DROP + s * (g.START_DROP - ALIGN_GAP)

    def support_point(self, which: str, below: float = 0.0) -> np.ndarray:
        """Where this hand's fingers' top edge should be (world): under its rim point of the housing at its nominal
        place (the pencil outline) and the height aimed for, plus the integral height correction."""
        c = self.contact[which]
        nominal = np.array([g.HOUSING_X + c[0], c[1], self.top_target() + c[2]])
        nominal[:2] += self.corr
        press = PRESS * smooth((self.ctx.t - self.press_t) / 1.0) if self.press_t is not None else 0.0
        if self.extra_t is not None and which == "outer":
            press += EXTRA_PRESS * smooth((self.ctx.t - self.extra_t) / 1.0)
        return nominal + np.array([0.0, 0.0, self.zoff[which] + press]) - below * elevated(self.elev[which])

    def tcp_for(self, point_world, which: str = "inner") -> np.ndarray:
        return self.L(point_world)

    def measured_tcp(self, which: str) -> np.ndarray:
        """This hand's tool point (station frame) from the measured joints (URDF FK)."""
        st = self.ctx.state
        arm = self.arm(which)
        q = st[0:6] if arm is self.ctx.left else st[7:13]
        return ik.tcp_pose(np.asarray(q, float), arm.base)[0]

    def support(self, which: str):
        """Closed loop while active: this hand at its pinch point on the housing at its nominal place, its height
        corrected by the housing's measured height there (the arm gives under the load) without ever winding the press
        up against the ceiling; the outer hand also keeps this end's slot lined up under its pilot hole."""
        arm = self.arm(which)
        while self.active[which]:
            p, R = self.housing()
            if self.unloading[which]:                # letting go: lower this hand's share slowly
                self.zoff[which] -= 0.0004
            else:
                rim = p + R @ self.contact[which]
                err = self.top_target() + self.contact[which][2] - rim[2]
                if rim[2] >= CEIL + self.contact[which][2] - 0.0004 and not getattr(self, "keep_press", False):
                    # flush here: bleed the press off slowly
                    err = min(err, 0.0) - 0.0008
                self.zoff[which] = float(np.clip(self.zoff[which] + 0.25 * err, -0.03, 0.20))
            if which == "outer":                     # keep this end's slot lined up under its pilot hole
                e = self.slot_world()[:2] - np.array(g.SCREW_XY[self.s])
                self.corr = np.clip(self.corr - 0.06 * e, -0.15, 0.15)
            arm.aim(self.tcp_for(self.support_point(which), which), self.R[which])
            if DEBUG and which == "outer" and round(self.ctx.t * 20) % 40 == 0:
                e = self.slot_world()[:2] - np.array(g.SCREW_XY[self.s])
                print(f"[scr] {self.s} t {self.ctx.t:.2f} {self.phase} slot err {np.round(e * 1000, 2).tolist()} corr "
                      f"{np.round(self.corr * 1000, 1).tolist()} zoff {self.zoff['outer'] * 1000:.1f}/"
                      f"{self.zoff['inner'] * 1000:.1f}", flush=True)
            yield

    def line_up(self, timeout: float = 5.0):
        """Ease off the ceiling 3 mm, wait for this end's slot to sit under its pilot hole (0.4 mm), press again."""
        self.phase = "line up"
        self.press_t = None
        ok, t0 = 0, self.ctx.t
        while self.ctx.t - t0 < timeout:
            e = self.slot_world()[:2] - np.array(g.SCREW_XY[self.s])
            ok = ok + 1 if np.linalg.norm(e) < 0.0004 else 0
            if ok >= 10:
                break
            yield
        self.phase = "press"
        self.press_t = self.ctx.t
        yield from wait(1.2)

    # -- the script ----------------------------------------------------------------------------------------------------
    def run(self):
        v = self.variant
        if v == "nothing" or (v == "alone" and self.s == "b"):
            yield from wait(5.0)
            return
        # A's inner hand takes its screwdriver first, so A's end is never let go of before its screw is in; B's end
        # is pinched by both of B's hands until then
        self.holders = ("outer",) if self.s == "a" else ("outer", "inner")
        if self.s == "a":
            yield from together(self.reach(self.holders), self.fetch())
        else:
            yield from self.reach(self.holders)
        self.t_raise = self.ctx.t
        for w in self.holders:
            self.active[w] = True
        yield from together(*(self.support(w) for w in self.holders), self.conductor())

    def reach(self, hands=("outer", "inner")):
        """The hands open across this end's end wall, from below along the wall, then in, then closed."""
        self.phase = "reach"
        pre = {w: self.tcp_for(self.support_point(w, below=0.05), w) for w in hands}
        for w in hands:
            self.R[w] = best_rot(self.arm(w), pre[w], elevated(self.elev[w]), ((0, 1, 0), (0, -1, 0)))
        low = {w: pre[w] - np.array([0.0, 0.0, 0.10]) - 0.04 * self.R[w][:, 2] for w in hands}
        yield from together(*(self.arm(w).move(low[w], self.R[w], 3.0, grip=1.0) for w in hands))
        yield from together(*(self.arm(w).move(pre[w], self.R[w], 1.2) for w in hands))
        yield from together(*(self.arm(w).move(self.tcp_for(self.support_point(w), w), self.R[w], 1.0) for w in hands))
        yield from together(*(self.arm(w).set_grip(0.0, 0.8) for w in hands))
        yield from wait(0.3)

    def conductor(self):
        """The sequence on top of the running supports."""
        v = self.variant
        self.phase = "raise"
        yield from wait(RAISE_S + 0.3)
        yield from self.line_up(4.0)
        if v == "drop":
            self.active = {"outer": False, "inner": False}
            yield
            yield from together(*(self.arm(w).set_grip(1.0, 0.3) for w in self.holders))
            yield from together(*(self.arm(w).move(self.arm(w).pos - [0, 0, 0.08], self.R[w], 0.4) for w in self.holders))
            yield from wait(4.0)
            return
        if v in ("lift", "alone"):                                     # raise and hold; no screw is driven
            yield from wait(6.0 if v == "lift" else 20.0)
            self.active = {"outer": False, "inner": False}
            return
        if self.s == "a":
            yield from self.drive(depth="part" if v == "unseated" else "full")
        else:
            self.phase = "hold while A drives"
            a_done = PART_TURNS * g.PITCH - 0.0005 if v == "unseated" else g.TRAVEL - 0.0008
            while self.pz("a") < a_done and self.ctx.t < 70.0:
                yield
            yield from wait(1.0)
            if v != "one_end":
                self.phase = "hand over the load"
                self.unloading["inner"] = True
                yield from wait(3.0)
                yield from self.inner.set_grip(0.55, 1.0)
                yield from wait(0.5)
                self.active["inner"] = False
                yield
                self.phase = "inner hand clear"
                yield from self.inner.move(self.inner.pos - [0.0, 0.0, 0.04], self.R["inner"], 0.8)
                yield from self.inner.set_grip(1.0, 0.3)
                yield from self.inner.move(self.inner.pos - 0.06 * self.R["inner"][:, 2] - [0.0, 0.0, 0.06],
                                           self.R["inner"], 0.8)
                yield from self.inner.go_home(2.0)                      # from where A's inner hand starts its fetch
                yield from self.fetch()
                yield from wait(1.0)
                yield from self.drive(depth="part" if v in ("unseated", "one_tight") else "full")
        self.phase = "hold until both are in"
        while (self.pz("a") < g.TRAVEL - 0.0008 or self.pz("b") < g.TRAVEL - 0.0008) and self.ctx.t < 160.0:
            if v in ("unseated", "one_tight", "slip", "one_end") and self.ctx.t > 110.0:
                break
            yield
        yield from wait(1.5)
        if v == "still_held":                           # the outer hand keeps pressing its end up, to the step limit
            self.keep_press, self.extra_t = True, self.ctx.t
            while True:
                yield
        if v == "one_end":
            yield from wait(8.0)
            self.active = {"outer": False, "inner": False}
            return
        self.phase = "let go"
        self.active = {"outer": False, "inner": False}
        yield
        o = self.outer
        yield from o.set_grip(1.0, 0.6)
        yield from o.move(o.pos - 0.06 * self.R["outer"][:, 2] - [0.0, 0.0, 0.02], self.R["outer"], 1.0)
        yield from o.go_home(2.5)
        yield from wait(4.0)

    def servo_to(self, pos_local, R, seconds: float):
        """Move the inner hand there, then correct by its measured tool point (FK of the measured joints)."""
        inner = self.inner
        off = np.zeros(3)
        for _ in range(max(1, round(seconds * 20))):
            st = self.ctx.state
            q = st[0:6] if inner is self.ctx.left else st[7:13]
            meas = ik.tcp_pose(np.asarray(q, float), inner.base)[0]
            off = np.clip(off + 0.3 * (pos_local - meas), -0.03, 0.03)
            inner.aim(pos_local + off, R)
            yield

    def driver_rot(self, target_world) -> np.ndarray:
        """The driver upright between the fingers, the approach level and turned toward the target (fingers across)."""
        base_w = self.ctx.world(self.inner.base)
        d = np.asarray(target_world, float) - base_w
        yaw = math.atan2(d[1], d[0])
        return rotm(elevated(0.0, math.degrees(yaw)), (math.sin(yaw), -math.cos(yaw), 0.0))

    def fetch(self):
        """Take the screwdriver out of its holder (A grips it 45 mm above its bottom, B 30 mm) and hold it up."""
        inner = self.inner
        self.phase = "fetch driver"
        hx, hy = g.HOLDER_XY[self.s]
        grip_w = np.array([hx, hy, g.TOP + g.HOLDER_FLOOR[self.s] + g.DRV_GRIP[self.s]])
        Rd = self.driver_rot(grip_w)
        self.Rd = Rd
        a = Rd[:, 2]
        out = self.m * 0.30                                     # away from the driver first, then back and up, then over
        yield from inner.move(self.L(np.array([0.30, out, g.TOP + 0.30])), inner.rot, 1.2, grip=1.0)
        yield from inner.move(self.L(np.array([0.15, out, g.TOP + 0.50])), Rd, 1.5)
        yield from inner.move(self.L(grip_w) + [0.0, 0.0, 0.25], Rd, 1.5)
        yield from inner.move(self.L(grip_w) + [0.0, 0.0, 0.03], Rd, 2.5, grip=1.0)          # down around it
        yield from self.servo_to(self.L(grip_w), Rd, 1.0)
        if DEBUG:
            st = self.ctx.state
            q = st[0:6] if inner is self.ctx.left else st[7:13]
            meas, Rm = ik.tcp_pose(np.asarray(q, float), inner.base)
            d = self.ctx.states()[f"driver_{self.s}"]
            print(f"[scr] {self.s} pick: want {np.round(self.L(grip_w) * 1000, 1).tolist()} meas {np.round(meas * 1000, 1).tolist()} "
                  f"approach {np.round(Rm[:, 2], 2).tolist()} fingers {np.round(Rm[:, 0], 2).tolist()} driver "
                  f"{np.round(self.L(np.asarray(d.pos)) * 1000, 1).tolist()} grip {st[6]:.2f}/{st[13]:.2f}", flush=True)
        # the grip-slip check: B's hand closes only to 33.3 mm on the 36 mm grip: it carries the driver, but is too
        # loose for the motor's torque
        yield from inner.set_grip(0.355 if self.variant == "slip" and self.s == "b" else 0.0, 0.8)
        yield from inner.move(inner.pos + [0.0, 0.0, 0.05], Rd, 0.8)

    def drive(self, depth: str = "full"):
        inner = self.inner
        hx, hy = g.HOLDER_XY[self.s]
        Rd = self.Rd
        # under the slot, the screw's tip 25 mm below the rim; then up into the slot and onto the pilot hole, servoing
        # the bit tip (read from the driver's pose) under the slot, then pushing and following the screw in
        self.phase = "to slot"
        slot = self.slot_world()
        bit_up = g.DRV_TIP - g.DRV_GRIP[self.s]                 # bit tip above the tool point
        stage_w = slot - np.array([0.0, 0.0, g.HOUSING_D + 0.025 + bit_up + g.HEAD_H + g.SCREW_L])
        Rs = self.driver_rot(slot)
        side = self.L(np.array([0.18, self.m * 0.30, g.TOP + 0.26]))                         # round the stand's end
        yield from inner.move(side, Rs, 2.0)
        yield from inner.move(self.L(stage_w), Rs, 2.0)
        contact_z = CEIL - g.SCREW_L - g.HEAD_H                 # bit tip height with the screw's tip at the ceiling
        goal_z = stage_w[2] + bit_up
        off = np.zeros(3)
        press, t_push, stall_t, last, bitten = 0.0016, None, None, -1.0, False
        seat_t, spin_ref, spin_t = None, 0.0, 0.0
        while self.ctx.t < 150.0:
            d = self.ctx.states()[f"driver_{self.s}"]
            Rd_now = _rot_q(d.quat_xyzw)
            bit = np.asarray(d.pos, float) + Rd_now[:, 2] * g.DRV_TIP
            pz = self.pz()
            j = self.ctx.states()["screws"].joints
            if pz > 0.0003:
                bitten = True
            hole = np.array(g.SCREW_XY[self.s])
            tip_xy = hole + np.array([j[f"{self.s}_px"], j[f"{self.s}_py"]])
            goal_xy = hole                                      # the command's xy base: under the pilot hole
            if bitten:                                          # keep the bit in the head (the screw is fixed now)
                err_xy = tip_xy - bit[:2]
            else:                                               # bring the screw's tip under the pilot hole,
                err_xy = hole - tip_xy
                if pz > -0.0015 and np.linalg.norm(tip_xy - bit[:2]) > 0.001:
                    err_xy = tip_xy - bit[:2]                   # but never drag the bit off a tip held by the ceiling
            self.err_f = 0.7 * getattr(self, "err_f", err_xy) + 0.3 * err_xy     # smoothed lateral error
            if pz < -0.0008 and not bitten:
                self.phase = "into the slot"
                if pz < -0.03:                                  # quickly up to 30 mm under the ceiling, the goal
                    goal_z = min(goal_z + 0.002, contact_z - 0.03, bit[2] + 0.008)    # kept close ahead of the bit
                elif np.linalg.norm(self.err_f) < (0.0015 if pz < -0.008 else 0.001):   # then slowly, under the hole
                    goal_z = min(goal_z + (0.001 if pz < -0.004 else 0.0003), contact_z, bit[2] + 0.004)
                off += np.r_[0.12 * err_xy, 0.25 * (goal_z - bit[2])]
            else:                                               # pressing the bit into the head, then following
                self.phase = "push and drive"
                t_push = t_push if t_push is not None else self.ctx.t
                if not bitten and self.ctx.t - t_push > 1.5 and press < 0.004:
                    press, t_push = press + 0.0005, self.ctx.t  # press harder if it doesn't bite
                goal_z = contact_z + min(max(pz, 0.0), g.TRAVEL) + press     # the bit this far into the head: the push
                lat = float(np.linalg.norm(tip_xy - bit[:2]))
                if bitten and (lat > g.HEAD_R or getattr(self, "reseat", False)):
                    # the bit has come off the head: back off 4 mm under it, centre again slowly, then push again
                    self.reseat = lat > 0.001
                    goal_z = contact_z + min(max(pz, 0.0), g.TRAVEL) - 0.004
                ez = goal_z - bit[2]
                if seat_t is not None:                          # seated: hold the push, no winding up
                    ez = min(ez, 0.0) + 0.2 * max(ez, 0.0)
                # near the end the motor's reaction torque changes fast, and the wrist gives sideways: follow harder
                kxy = 0.25 if bitten and pz > g.TRAVEL - 0.0015 and not getattr(self, "reseat", False) else 0.12
                e_int = np.clip(err_xy, -0.002, 0.002) if bitten else err_xy      # the screw is fixed now: no chasing
                off += np.r_[kxy * e_int, 0.25 * ez]
            off = np.clip(off, -0.10, 0.10)
            axis = Rd_now[:, 2]                                  # the driver's axis; turn the hand to bring it upright
            tilt_v = np.cross(axis, [0.0, 0.0, 1.0])
            s = float(np.linalg.norm(tilt_v))
            if s > 1e-6:
                ang = math.asin(min(1.0, s))
                k = tilt_v / s
                Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                corr_R = np.eye(3) + math.sin(0.2 * ang) * Kx + (1 - math.cos(0.2 * ang)) * Kx @ Kx
                self.tilt_R = corr_R @ getattr(self, "tilt_R", np.eye(3))
            Rcmd = getattr(self, "tilt_R", np.eye(3)) @ Rs
            q_before = inner.q.copy()
            inner.aim(self.L(np.r_[goal_xy, goal_z - bit_up]) + off, Rcmd)
            if DEBUG and (np.abs(inner.q - q_before).max() > 0.08 or round(self.ctx.t * 20) % 10 == 0):
                print(f"[drv] {self.s} t {self.ctx.t:.2f} dq {np.abs(inner.q - q_before).max():.3f} q {np.round(inner.q, 2).tolist()} "
                      f"off {np.round(off * 1000, 1).tolist()} err {np.round(err_xy * 1000, 2).tolist()} pz {pz * 1000:.1f} "
                      f"goalz {goal_z:.4f} bitz {bit[2]:.4f} ik {inner.worst_mm:.1f}", flush=True)
            yield
            if depth == "part" and pz >= PART_TURNS * g.PITCH:     # stopped after 6 of the 10.4 turns
                break
            raw = float(j[f"{self.s}_spin"])                  # the screw's turn, unwrapped
            spin = getattr(self, "spin_u", raw) + math.remainder(raw - getattr(self, "spin_raw", raw), 2 * math.pi)
            self.spin_u, self.spin_raw = spin, raw
            if bitten and pz >= g.TRAVEL - 0.0003:              # the head on the housing: drive on until the clutch slips
                if seat_t is None:
                    seat_t, spin_ref, spin_t = self.ctx.t, spin, self.ctx.t
                    spin_seat = spin
                    self.phase = "seat"
                head_z = CEIL + pz - g.SCREW_L - g.HEAD_H
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(Rd_now[2, 2])))))
                engaged = (np.linalg.norm(tip_xy - bit[:2]) <= 0.0015 and -0.0015 <= bit[2] - head_z <= 0.004
                           and tilt <= 6.0)                     # the bit in the head, square, pressed: the motor runs
                tightened = spin - spin_seat > 10.0             # 1.6 turns since the head met the housing: snug
                if tightened and not engaged:
                    break                                       # snug, and the bit has come out of the head
                if abs(spin - spin_ref) > 0.3 or not engaged:   # still turning, or not back in the head yet
                    spin_ref, spin_t = spin, self.ctx.t
                elif self.ctx.t - spin_t > (0.3 if tightened else 0.5):
                    break                                       # the clutch slips: it has stopped turning the screw
                if self.ctx.t - seat_t > 6.0:
                    break
            elif pz > last + 0.00005:
                stall_t, last = None, pz
            elif bitten:                                        # no progress: give up after 3 s near the end, else 6 s
                stall_t = stall_t or self.ctx.t
                if self.ctx.t - stall_t > (3.0 if pz > g.TRAVEL - 0.003 else 6.0):
                    break
        self.phase = "park the driver"
        meas = self.measured_tcp("inner")
        yield from inner.move(meas, inner.rot, 0.6)               # ease the push off, where the hand is
        yield from inner.move(meas - [0.0, 0.0, 0.08], inner.rot, 1.2)
        park = self.L(np.array([0.13, self.m * 0.33, g.TOP + 0.30]))
        yield from inner.move(park, Rd, 4.0)                      # out of the way, still holding it
        self.parked = True


def station_script(s: str, variant: str = ""):
    def script(ctx):
        st = Station(ctx, s, variant)
        ctx.flags["station"] = st
        yield from st.run()

    return script
