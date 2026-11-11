"""On the station table: a red steel toolbox on the robot's left with a hand compass lying on its lid, and a tabletop
equatorial telescope mount on the robot's right, pinned at its centre so that it can only turn.

The compass is a round liquid-filled hand compass (black housing 7.5 cm across, black face with white ticks and no
letters, a 6.5 cm needle whose north half is red and south half white). The mount is a 16 cm aluminium column on three
low legs, with an equatorial head on top; one leg is painted yellow and marked N. North lies at a bearing of 115 degrees
(clockwise from straight ahead, seen from above): toward the robot's right and a little back toward it. Steel pulls a
compass needle off north: on the toolbox's lid the needle points 50 degrees anticlockwise of north, and it reads within
2.5 degrees of north only once it is 25 cm or more from the box. The magnetic field is computed (field.py), not simulated: Earth's
horizontal field plus one dipole for the toolbox; the needle's joint is driven toward it every control step.
"""

from __future__ import annotations

import collections
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import field as F  # noqa: E402
import scene_usd as S  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = S.build()


def _yaw_xyzw(heading: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(heading / 2), math.cos(heading / 2))


PIVOT_START = (S.COMPASS_XY[0], S.COMPASS_XY[1], S.COMPASS_START[2] + S.PIVOT_Z)
NEEDLE_START = F.reading_heading(PIVOT_START)        # the lid's reading; the housing is turned so joint zero shows it
NEEDLE_TAU = 0.35                                    # s: the needle follows the field with this lag (liquid damping)
NEEDLE_K, NEEDLE_D = 2.5e-3, 3.3e-4                  # its joint drive: about 15 rad/s, critically damped

TOOLBOX = UsdBody("toolbox", usd=USD["toolbox"], pos=(0.0, 0.0, TOP), size=tuple(F.BOX_HI - F.BOX_LO))
COMPASS = UsdBody("compass", usd=USD["compass"], pos=(S.COMPASS_START[0], S.COMPASS_START[1], TOP + S.COMPASS_START[2] + 0.0005),
                  rot_xyzw=_yaw_xyzw(NEEDLE_START), size=(0.075, 0.075, 0.025), articulation=True,
                  joint_pos=(("needle_joint", 0.0),), joint_stiffness=NEEDLE_K, joint_damping=NEEDLE_D,
                  friction=(0.8, 0.7), solver_iters=16)
MOUNT = UsdBody("mount", usd=USD["mount"], pos=(S.MOUNT_AXIS[0], S.MOUNT_AXIS[1], TOP),
                rot_xyzw=_yaw_xyzw(S.MOUNT_START_HEADING), size=(0.28, 0.28, 0.26), articulation=True,
                joint_pos=(("turn", 0.0),), joint_damping=0.5, friction=(0.9, 0.8), solver_iters=16)
OBJECTS = [TOOLBOX, COMPASS, MOUNT]

# -- scoring ----------------------------------------------------------------------------------------------------------
NORTH_BEARING = F.NORTH_BEARING     # 115 degrees
FULL_DEG = 8.0                      # 1.0 within this of north
HALF_DEG = 20.0                     # 0.5 within this, at the run limit
RELEASE_M = 0.05                    # each TCP at least this far from every part of the mount
STILL_DEG = 0.5                     # the mount turned less than this ...
STILL_STEPS = 20                    # ... over the last 1 s (20 control steps at 20 Hz)

_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    _T.clear()
    _T.update(step=0, history=collections.deque(maxlen=STILL_STEPS + 1), verdict=None, needle=0.0, needle_set=False)


_reset()


def bearing_of_heading(h: float) -> float:
    return (-math.degrees(h)) % 360.0


def n_leg_bearing(turn: float) -> float:
    """Bearing of the N leg (from the axis to its foot), given the mount's joint angle (rad, anticlockwise)."""
    return bearing_of_heading(S.MOUNT_START_HEADING + turn)


def error_deg(turn: float) -> float:
    d = (n_leg_bearing(turn) - NORTH_BEARING + 180.0) % 360.0 - 180.0
    return abs(d)


def _obb_distance(p, centre, rot, size) -> float:
    q = rot.T @ (np.asarray(p, float) - np.asarray(centre, float))
    half = np.asarray(size, float) / 2
    return float(np.linalg.norm(q - np.clip(q, -half, half)))


def mount_distance(p, turn: float) -> float:
    """Distance from a point (station frame, height above the table) to the mount's nearest part: legs, column, head."""
    h = S.MOUNT_START_HEADING + turn
    c, s = math.cos(-h), math.sin(-h)
    rel = np.asarray(p, float) - (S.MOUNT_AXIS[0], S.MOUNT_AXIS[1], 0.0)
    local = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
    parts = [(c_, r_, s_) for _, c_, r_, s_ in S.collider_boxes()]
    return min(_obb_distance(local, c_, r_, s_) for c_, r_, s_ in parts)


def judge(turn: float, tcps, history) -> dict:
    err = error_deg(turn)
    released = all(mount_distance(t, turn) >= RELEASE_M for t in tcps) if tcps else True
    still = len(history) > STILL_STEPS and math.degrees(max(history) - min(history)) < STILL_DEG
    return {"bearing": n_leg_bearing(turn), "error": err, "released": released, "still": still,
            "full": err <= FULL_DEG and released and still, "half": err <= HALF_DEG and released and still}


def _tcps(environment) -> list:
    from yam_arena.runtime import tcp_world
    base = environment.unwrapped
    origin = base.scene.env_origins[0]
    origin = getattr(origin, "torch", origin).detach().cpu().numpy()
    out = []
    for side in ("left", "right"):
        t = tcp_world(base.scene[f"{side}_robot"]) - origin
        out.append(np.array([t[0], t[1], t[2] - TOP]))
    return out


def _needle(environment, compass) -> dict:
    """Drive the needle toward the computed field, projected on the compass face, with a first-order lag."""
    import torch

    rot = compass.rotation()
    pivot = np.asarray(compass.pos, float) + rot @ np.array([0.0, 0.0, S.PIVOT_Z])
    p = np.array([pivot[0], pivot[1], pivot[2] - TOP])
    b_local = rot.T @ F.field(p)
    aim = math.atan2(b_local[1], b_local[0])                 # where the needle should point, in the housing's frame
    if not _T["needle_set"]:
        _T["needle"], _T["needle_set"] = aim, True
    q = _T["needle"]
    target = q + (aim - q + math.pi) % (2 * math.pi) - math.pi
    q += (target - q) * (1.0 - math.exp(-1.0 / steptime.CONTROL_HZ / NEEDLE_TAU))
    _T["needle"] = q
    asset = environment.unwrapped.scene["compass"]
    jid = asset.find_joints(["needle_joint"])[0]
    asset.actuators.target_command.set_position_index(value=torch.tensor([[q]], dtype=torch.float32, device=asset.device),
                                                      joint_ids=jid)
    up = rot[:, 2]
    return {"aim": aim, "target": q, "pivot": p, "tilt": math.degrees(math.acos(max(-1.0, min(1.0, up[2])))),
            "reading_bearing": bearing_of_heading(math.atan2(*(F.field(p)[[1, 0]]))),
            "reading_error": F.error_deg(p)}


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    states = object_states(environment, ["compass", "mount"])
    nd = _needle(environment, states["compass"])
    turn = float(states["mount"].joints["turn"])
    tcps = _tcps(environment)
    _T["history"].append(turn)
    v = judge(turn, tcps, _T["history"])
    _T["verdict"] = v
    _T["step"] += 1
    folder = os.environ.get("TASK_SCORE_DIR")
    if folder:
        os.makedirs(folder, exist_ok=True)
        q_needle = float(states["compass"].joints.get("needle_joint", float("nan")))
        rec = {"t": round(_T["step"] / steptime.CONTROL_HZ, 2), "bearing": round(v["bearing"], 2), "error": round(v["error"], 2),
               "released": v["released"], "still": v["still"], "full": v["full"], "half": v["half"],
               "turn_deg": round(math.degrees(turn), 2), "needle_q": round(math.degrees(q_needle), 2),
               "needle_target": round(math.degrees(nd["target"]), 2), "needle_aim": round(math.degrees(nd["aim"]), 2),
               "reading_bearing": round(nd["reading_bearing"], 2), "reading_error": round(nd["reading_error"], 2),
               "compass": [round(float(c), 4) for c in nd["pivot"]], "compass_tilt": round(nd["tilt"], 1),
               "tcp": [[round(float(c), 4) for c in t] for t in tcps],
               "mount_dist": [round(mount_distance(t, turn), 3) for t in tcps]}
        with open(os.path.join(folder, "trace.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")


def success(objects) -> bool:
    """The N leg within 8 degrees of north, the mount released and still; the harness ends the trial at 1.0 after 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    v = _T["verdict"]
    return bool(v and v["full"])


def score(objects) -> float:
    """At the run limit: 0.5 if the N leg is within 20 degrees of north with the mount released and still, else 0."""
    v = _T["verdict"]
    return 0.5 if v and v["half"] else 0.0


# -- scripted demo (open loop: it reads no simulation state) ------------------------------------------------------
# North is fixed in the setup, so the demo turns the mount to it directly. The left arm still moves the compass first,
# so that the video shows the needle swing from the lid's wrong reading to north.
PARK = {"left": (0.25, 0.30, 0.17), "right": (0.27, -0.30, 0.17)}
COMPASS_HOLD = -(2 * S.COMPASS_R - 0.008) / 2       # pads 8 mm narrower than the housing
LEG_HOLD = -(S.LEG_W - 0.008) / 2
GRIP_R = 0.10                                       # the leg is gripped 10 cm from the axis
GRIP_Z = 0.012 + PAD_TIP_PAST_TCP                   # TCP height: fingertips 1.2 cm above the table, 3 mm over the feet
ABOVE_LEG = 0.10
READ_AT = (0.16, -0.08)                             # where the compass is set down, 30 cm from the toolbox
CARRY_Z = 0.155


def _wp(t, xyz, finger, label, yaw=0.0) -> Waypoint:
    x, y, z = xyz
    return Waypoint(t, (float(x), float(y), TOP + float(z)), finger, label, float(yaw))


def park(side: str, t: float, label: str = "park") -> list[Waypoint]:
    return [_wp(t, PARK[side], FINGER_OPEN, label)]


def stay(wps: list[Waypoint], t: float, label: str = "wait") -> list[Waypoint]:
    """Hold the last waypoint's pose for t seconds."""
    last = wps[-1]
    return wps + [Waypoint(t, last.pos, last.finger, label, last.yaw)]


def move_compass(dst=READ_AT) -> list[Waypoint]:
    """Left arm: grip the compass housing on the lid, lift it clear, carry it to dst and set it down, then park."""
    (cx, cy), lid = S.COMPASS_XY, S.COMPASS_START[2]
    grip = lid + 0.003 + PAD_TIP_PAST_TCP
    place = 0.002 + 0.003 + PAD_TIP_PAST_TCP
    dx, dy = dst
    return [
        _wp(2.0, (cx, cy, CARRY_Z), FINGER_OPEN, "over the compass"),
        _wp(1.0, (cx, cy, grip + 0.02), FINGER_OPEN, "near"),
        _wp(0.6, (cx, cy, grip), FINGER_OPEN, "down"),
        _wp(0.3, (cx, cy, grip), FINGER_OPEN, "settle"),
        _wp(0.6, (cx, cy, grip), COMPASS_HOLD, "grip"),
        _wp(1.0, (cx, cy, CARRY_Z), COMPASS_HOLD, "lift off the lid"),
        _wp(2.6, (dx, dy, CARRY_Z), COMPASS_HOLD, "carry clear of the steel"),
        _wp(1.2, (dx, dy, place + 0.01), COMPASS_HOLD, "lower"),
        _wp(0.5, (dx, dy, place), COMPASS_HOLD, "set down"),
        _wp(0.6, (dx, dy, place), FINGER_OPEN, "let go"),
        _wp(0.8, (dx, dy, 0.12), FINGER_OPEN, "rise"),
    ] + park("left", 1.5)


def _grip_point(heading: float):
    return (S.MOUNT_AXIS[0] + GRIP_R * math.cos(heading), S.MOUNT_AXIS[1] + GRIP_R * math.sin(heading))


def swing(leg_heading_deg: float, delta_deg: float, yaw0_deg: float, release: bool = True) -> list[Waypoint]:
    """Right arm: grip a leg 10 cm out, fingers closing across it, and swing it delta degrees about the axis
    (anticlockwise seen from above for delta > 0) along waypoints 10 degrees apart, the wrist turning with it."""
    h0 = math.radians(leg_heading_deg)
    gx, gy = _grip_point(h0)
    yaw0 = math.radians(yaw0_deg)
    out = [_wp(1.6, (gx, gy, ABOVE_LEG), FINGER_OPEN, "over the leg", yaw0),
           _wp(1.0, (gx, gy, GRIP_Z + 0.02), FINGER_OPEN, "near", yaw0),
           _wp(0.6, (gx, gy, GRIP_Z), FINGER_OPEN, "down beside the leg", yaw0),
           _wp(0.3, (gx, gy, GRIP_Z), FINGER_OPEN, "settle", yaw0),
           _wp(0.6, (gx, gy, GRIP_Z), LEG_HOLD, "grip the leg", yaw0)]
    n = max(1, int(math.ceil(abs(delta_deg) / 10.0)))
    for k in range(1, n + 1):
        d = math.radians(delta_deg * k / n)
        x, y = _grip_point(h0 + d)
        out.append(_wp(0.5, (x, y, GRIP_Z), LEG_HOLD, f"swing {delta_deg * k / n:.0f} deg", yaw0 + d))
    out.append(_wp(0.5, (x, y, GRIP_Z), LEG_HOLD, "settle", yaw0 + math.radians(delta_deg)))
    if release:
        out += [_wp(0.6, (x, y, GRIP_Z), FINGER_OPEN, "let go", yaw0 + math.radians(delta_deg)),
                _wp(0.8, (x, y, ABOVE_LEG), FINGER_OPEN, "rise", yaw0 + math.radians(delta_deg))]
    return out


GRIP_LEG = 295.0      # the right arm swings the black leg that starts at heading 295 (bearing 65): its wrist stays outside the head


def retreat(h_deg: float, yaw_deg: float) -> list[Waypoint]:
    """From over a leg at heading h, leave without sweeping through the head: close the fingers, move out from the
    axis, then go round the mount's far side along -y, or back toward the robot along -x and round its near side, to
    the park pose. Where neither is clear, stay out there."""
    from yam_arena.constants import FINGER_CLOSED
    yaw = math.radians(yaw_deg)
    h = math.radians(h_deg)
    x, y = _grip_point(h)
    ox = min(S.MOUNT_AXIS[0] + 0.15 * math.cos(h), 0.43)
    oy = S.MOUNT_AXIS[1] + 0.15 * math.sin(h)
    out = [_wp(0.4, (x, y, ABOVE_LEG), FINGER_CLOSED, "close the fingers", yaw),
           _wp(0.8, (ox, oy, ABOVE_LEG + 0.02), FINGER_CLOSED, "out from the mount", yaw)]
    if ox >= S.MOUNT_AXIS[0] + 0.09:
        out.append(_wp(1.0, (ox, -0.31, 0.12), FINGER_CLOSED, "past the far side", yaw))
    elif ox <= S.MOUNT_AXIS[0] - 0.05 or oy >= S.MOUNT_AXIS[1] + 0.09:
        out += [_wp(0.8, (0.18, oy, 0.12), FINGER_CLOSED, "toward the robot", yaw),
                _wp(1.0, (0.18, -0.30, 0.12), FINGER_CLOSED, "past the near side", yaw)]
    else:
        return out
    return out + park("right", 1.5)


def turn_mount(delta_deg: float, release: bool = True) -> list[Waypoint]:
    """Turn the mount delta degrees anticlockwise in swings of at most 100 degrees, re-gripping between them."""
    out, h, left = [], GRIP_LEG, delta_deg
    yaw_end = 0.0
    while abs(left) > 1e-6:
        step = max(-100.0, min(100.0, left))
        yaw0 = ((h + 90.0) % 180.0) - 90.0             # fingers across the leg: yaw = the leg's heading, mod 180
        last = abs(left - step) < 1e-6
        out += swing(h, step, yaw0, release=release or not last)
        h += step
        left -= step
        yaw_end = yaw0 + step
    if release:
        out += retreat(h, yaw_end)
    return out


def demo():
    """Open loop. TASK_DEMO picks a variant for checking the scorer:
    success  the compass carried 30 cm clear of the toolbox, then the N leg turned to north (bearing 115)
    lid      the N leg turned to the lid's reading (bearing 65), the compass left on the lid
    above5   the N leg turned to the reading 5 cm above the lid (about 15 degrees off: bearing 100)
    nothing  the arms only park
    ahead    the N leg turned straight ahead (bearing 0)
    right    the N leg turned to the robot's right (bearing 90)
    held     as success, but the gripper keeps hold of the leg to the end
    """
    mode = os.environ.get("TASK_DEMO", "success")
    start = n_leg_bearing(0.0)                                   # 185

    def delta_to(bearing: float) -> float:                       # anticlockwise turn = bearing decreasing
        d = (start - bearing) % 360.0
        return d

    if mode == "nothing":
        return {"left": park("left", 2.0) + park("left", 4.0, "wait"), "right": park("right", 2.0) + park("right", 4.0, "wait")}
    if mode in ("lid", "above5", "ahead", "right"):
        target = {"lid": NORTH_BEARING - 50.0, "above5": NORTH_BEARING - abs(F.error_deg((S.COMPASS_XY[0], S.COMPASS_XY[1],
                  S.COMPASS_START[2] + S.PIVOT_Z + 0.05))), "ahead": 0.0, "right": 90.0}[mode]
        right = stay(park("right", 2.0) + turn_mount(delta_to(target)), 3.0)
        return {"left": park("left", 2.0) + park("left", 2.0, "wait"), "right": right}
    left = move_compass()
    busy = sum(w.duration for w in left)
    right = park("right", 2.0) + park("right", busy, "wait while the needle settles")
    if mode == "held":
        hold = turn_mount(delta_to(NORTH_BEARING), release=False)
        last = hold[-1]
        hold.append(Waypoint(3.0, last.pos, last.finger, "keep holding the leg", last.yaw))
        return {"left": left + park("left", 6.0, "wait"), "right": right + hold}
    right = stay(right + turn_mount(delta_to(NORTH_BEARING)), 3.0)
    return {"left": left + park("left", busy, "wait"), "right": right}
