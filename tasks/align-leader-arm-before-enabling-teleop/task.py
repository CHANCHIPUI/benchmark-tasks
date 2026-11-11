"""On the station table: a leader-follower teleoperation rig, with the follower working beside a glass.

In front of the left arm, on a black rubber mat, stands the leader: a yellow 0.75-scale replica of the follower, two
links on vertical joint axes, folded at the start, with a black rubber knob over its tip. Its joints have friction, so
it stays where it is pushed. Further out, beyond the robot's reach, on a light grey board, stands the follower: a black
two-link arm of the same shape, mounted the same way round, parked reaching out to the right. A blue-tinted glass
tumbler stands loose on the board in front of it, and a green dot is printed on the board to the left of the glass.
In front of the right arm sits a black ENABLE box with a blue push button. While teleoperation is off the follower
holds still; once ENABLE is pressed, the button lights, and from then on the follower copies the leader's joint angles,
moving at up to 3 rad/s per joint.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import rig_usd as R  # noqa: E402
import scenekit  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = R.build()

# The follower's drives: stiff, capped in effort, and fed targets that move at most VMAX toward the leader each step.
STIFFNESS, DAMPING, EFFORT, VMAX = 60.0, 2.0, 12.0, 3.0   # N m/rad, N m s/rad, N m, rad/s
LEADER_DAMPING, KNOB_DAMPING = 0.05, 0.0005                # N m s/rad; the knob turns freely
# The leader's joint friction, as stick-slip: each joint is held by a weak spring to a set point that follows the joint
# with HOLD_BAND of play. A released link stays where it is left (springing back at most HOLD_BAND), and a push of more
# than about HOLD_K * HOLD_BAND (0.01 N m) moves it.
HOLD_K, HOLD_BAND = 2.0, math.radians(0.3)                 # N m/rad, rad
BUTTON_SPRING, BUTTON_DAMPING = 300.0, 2.0                 # N/m, N s/m


def _rel(angles, authored) -> tuple[float, float]:
    return tuple(math.radians(a - b) for a, b in zip(angles, authored))


_LS, _LE = _rel(R.LEADER_START, R.LEADER_AUTHORED)
MATS = UsdBody("rig_mats", usd=USD["mats"], pos=(0.0, 0.0, TOP), size=(0.41, 0.56, 0.006))
FOLLOWER = UsdBody("follower", usd=USD["follower"], pos=(0.0, 0.0, TOP), size=(0.30, 0.30, 0.09), articulation=True,
                   joint_pos=(("shoulder", 0.0), ("elbow", 0.0)), solver_iters=32,
                   actuators=((("shoulder", "elbow"), STIFFNESS, DAMPING, EFFORT, VMAX + 0.5),))
LEADER = UsdBody("leader", usd=USD["leader"], pos=(0.0, 0.0, TOP), size=(0.23, 0.23, 0.09), articulation=True,
                 joint_pos=(("shoulder", _LS), ("elbow", _LE), ("knob_spin", 0.0)), solver_iters=32, friction=(1.0, 0.9),
                 actuators=((("shoulder", "elbow"), HOLD_K, LEADER_DAMPING, None, None),
                            (("knob_spin",), 0.0, KNOB_DAMPING, None, None)))
ENABLE_BOX = UsdBody("enable_box", usd=USD["enable_box"], pos=(0.0, 0.0, TOP), size=(0.08, 0.08, 0.053),
                     articulation=True, joint_pos=(("press", 0.0),), solver_iters=16,
                     actuators=((("press",), BUTTON_SPRING, BUTTON_DAMPING, None, None),))
GLASS = UsdBody("glass", usd=USD["glass"], pos=(R.GLASS_XY[0], R.GLASS_XY[1], TOP + R.BOARD_T + 0.0005),
                size=(0.074, 0.074, 0.10), mass=R.GLASS_MASS, friction=(0.4, 0.35), solver_iters=16)
OBJECTS = [MATS, FOLLOWER, LEADER, ENABLE_BOX, GLASS]

# -- scoring --------------------------------------------------------------------------------------------------------
PRESSED = 0.004         # m: the button pushed down this far is an enable
CLEAN_GAP = 15.0        # deg: the largest gap, at either joint, for a clean enable
ON_DOT = 0.02           # m: the tip's axis within this of the dot's centre
GLASS_MOVED = 0.01      # m: the glass's base centre moving this far disturbs it ...
GLASS_TILT = 10.0       # deg: ... or tilting this far

_T: dict = {}


def _clear() -> None:
    _T.clear()
    _T.update(enabled=False, gap=None, enable_step=None, failed=None, step=0, target=[0.0, 0.0], lamp=False,
              worst_glass_mm=0.0, worst_tilt=0.0, hold=[_LS, _LE])


@on_reset
def _reset(env=None) -> None:
    """Failures stay failed for the whole trial: only a reset clears them."""
    _clear()
    if env is not None:
        _lamp(env, False)


_clear()


def angles(joints: dict, authored) -> tuple[float, float]:
    """(shoulder, elbow) in degrees, from an arm's joint readings."""
    return (authored[0] + math.degrees(float(joints["shoulder"])), authored[1] + math.degrees(float(joints["elbow"])))


def gap(leader: tuple, follower: tuple) -> float:
    return max(abs(leader[0] - follower[0]), abs(leader[1] - follower[1]))


def tip_to_dot(follower: tuple) -> float:
    return float(np.linalg.norm(R.follower_tip(*follower) - np.asarray(R.DOT_XY)))


def _tilt_deg(q) -> float:
    x, y, _, _ = q
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


# -- the rig's controller --------------------------------------------------------------------------------------------
def _ids(asset, names) -> list[int]:
    return [list(asset.joint_names).index(n) for n in names]


def _lamp(env, on: bool) -> None:
    """Show the lit or the unlit button cap."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        found = []
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):    # env 0's button caps (a USD visibility switch)
            if prim.GetName() in ("cap_on", "cap_off") and "env_0/enable_box" in str(prim.GetPath()):
                show = (prim.GetName() == "cap_on") == on
                UsdGeom.Imageable(prim).MakeVisible() if show else UsdGeom.Imageable(prim).MakeInvisible()
                found.append(str(prim.GetPath()))
        _T["lamp"] = on if found else None
    except Exception as exc:  # the lamp is only a picture; never let it stop a trial
        print(f"lamp: {exc!r}")


@on_step
def _control(env) -> None:
    """Every control step: detect the enable, then aim the follower's drives (start angles until the enable, then the
    leader's angles, approached at no more than VMAX)."""
    import torch

    scene = env.unwrapped.scene
    follower, leader, box = scene["follower"], scene["leader"], scene["enable_box"]
    fq = scenekit._np(follower.data.joint_pos)[0]
    lq = scenekit._np(leader.data.joint_pos)[0]
    bq = scenekit._np(box.data.joint_pos)[0]
    fi, li = _ids(follower, ("shoulder", "elbow")), _ids(leader, ("shoulder", "elbow"))
    f_ang = angles(dict(shoulder=fq[fi[0]], elbow=fq[fi[1]]), R.FOLLOWER_AUTHORED)
    l_ang = angles(dict(shoulder=lq[li[0]], elbow=lq[li[1]]), R.LEADER_AUTHORED)
    depth = -float(bq[_ids(box, ("press",))[0]])
    if not _T["enabled"] and depth >= PRESSED:
        _T.update(enabled=True, gap=gap(l_ang, f_ang), enable_step=_T["step"] + 1)
        _lamp(env, True)
    if _T["enabled"]:
        want = np.array(_rel(l_ang, R.FOLLOWER_AUTHORED))
        dt = float(getattr(env.unwrapped, "step_dt", 0.05))
        now = np.array(_T["target"])
        _T["target"] = list(now + np.clip(want - now, -VMAX * dt, VMAX * dt))
    target = torch.tensor([_T["target"]], dtype=torch.float32, device=env.unwrapped.device)
    follower.actuators.target_command.set_position_index(value=target, joint_ids=fi)
    # the leader's stick-slip friction: its set points follow its joints with HOLD_BAND of play
    lnow = np.array([lq[li[0]], lq[li[1]]])
    _T["hold"] = list(np.clip(np.array(_T["hold"]), lnow - HOLD_BAND, lnow + HOLD_BAND))
    hold = torch.tensor([_T["hold"]], dtype=torch.float32, device=env.unwrapped.device)
    leader.actuators.target_command.set_position_index(value=hold, joint_ids=li)


def _update(objects) -> None:
    """Latches the first failure."""
    g = objects["glass"]
    moved = float(np.linalg.norm(np.asarray(g.pos[:2]) - np.asarray(R.GLASS_XY)))
    tilt = _tilt_deg(g.quat_xyzw)
    _T["worst_glass_mm"] = max(_T["worst_glass_mm"], moved * 1000)
    _T["worst_tilt"] = max(_T["worst_tilt"], tilt)
    if _T["failed"]:
        return
    if moved > GLASS_MOVED or tilt > GLASS_TILT:
        _T["failed"] = f"glass disturbed: moved {moved * 1000:.0f} mm, tilted {tilt:.0f} deg"
    elif _T["enabled"] and _T["gap"] > CLEAN_GAP:
        _T["failed"] = f"enabled with a gap of {_T['gap']:.0f} deg"


def score(objects) -> float:
    """0 for the whole trial after a failure: the glass moved more than 1 cm or tilted more than 10 deg, or ENABLE was
    pressed with a gap over 15 deg at either joint. Otherwise 0.5 after a clean enable, and 0 if ENABLE was never
    pressed. (1.0 comes from success(), held for 1 s.)"""
    _update(objects)
    if _T["failed"] or not _T["enabled"]:
        return 0.0
    return 0.5


def success(objects) -> bool:
    """A clean enable, and the follower's tip within 2 cm of the dot's centre (the harness holds this for 1 s)."""
    steptime.tick()  # the harness calls success() once per control step
    _T["step"] += 1
    s = score(objects)
    follower = angles(objects["follower"].joints, R.FOLLOWER_AUTHORED)
    d = tip_to_dot(follower)
    ok = s == 0.5 and d <= ON_DOT
    _log(objects, s, follower, d, ok)
    return ok


def _tcp() -> dict:
    """Each arm's TCP (gripper link origin moved 13.4 cm along a straight-down approach), for debugging only."""
    env = scenekit.env()
    if env is None:
        return {}
    links = scenekit.robot_links(env)
    return {side: [round(float(v), 4) for v in links[side]["gripper"] - (0.0, 0.0, 0.134)] for side in links}


def _log(objects, s: float, follower: tuple, d: float, ok: bool) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    leader = angles(objects["leader"].joints, R.LEADER_AUTHORED)
    g = objects["glass"]
    row = {"step": _T["step"], "score": s, "success_cond": ok, "failed": _T["failed"], "enabled": _T["enabled"],
           "gap_at_enable": None if _T["gap"] is None else round(_T["gap"], 2), "enable_step": _T["enable_step"],
           "button_mm": round(-float(objects["enable_box"].joints["press"]) * 1000, 2),
           "leader": [round(a, 2) for a in leader], "follower": [round(a, 2) for a in follower],
           "gap_now": round(gap(leader, follower), 2), "tip_to_dot_mm": round(d * 1000, 1),
           "glass_xy": [round(float(v), 4) for v in g.pos[:2]], "glass_tilt": round(_tilt_deg(g.quat_xyzw), 2),
           "worst_glass_mm": round(_T["worst_glass_mm"], 2), "lamp": _T["lamp"], "tcp": _tcp()}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Open loop, from the layout the cameras show. The left arm grips the leader's knob (yaw 0: the fingers close along y,
# the knob turns in its bearing as the links swing under it) and drags it; the right arm presses ENABLE.
GRIP_Z = TOP + 0.082        # TCP: the fingertips 13 mm lower, over the knob's top 2.5 cm
ABOVE_Z = TOP + 0.15
PRESS_Z = TOP + R.BOX_SIZE[2] + R.BEZEL_PROUD - 0.001 + 0.013   # fingertips 1 mm below the bezel's top
R_IN = 0.13 * R.SCALE       # the folded route: the leader's tip within 9.75 cm of its axis (13 cm on the follower)


def _lw(t, xy, z, finger, label) -> Waypoint:
    return Waypoint(float(t), (float(xy[0]), float(xy[1]), float(z)), finger, label, 0.0)


def _drag(points, t, label) -> list[Waypoint]:
    return [_lw(t, p, GRIP_Z, FINGER_CLOSED, label) for p in points]


def _polar(r, deg):
    return np.asarray(R.L0) + r * np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def grip_leader() -> list[Waypoint]:
    k = R.leader_tip(*R.LEADER_START)
    return [_lw(2.5, k, ABOVE_Z, FINGER_OPEN, "above the knob"),
            _lw(1.5, k, GRIP_Z, FINGER_OPEN, "down round the knob"),
            _lw(0.8, k, GRIP_Z, FINGER_CLOSED, "grip the knob")]


def pose_leader(q_goal) -> list[Waypoint]:
    """Folded, swing the shoulder to the goal's shoulder angle, then unfold the elbow to the goal's elbow angle."""
    q1a, q2a = R.LEADER_START
    q1, q2 = q_goal
    swing = [R.leader_tip(q1a + (q1 - q1a) * k / 8, q2a) for k in range(1, 9)]
    unfold = [R.leader_tip(q1, q2a + (q2 - q2a) * k / 6) for k in range(1, 7)]
    return _drag(swing, 0.7, "swing the folded leader") + _drag(unfold, 0.6, "unfold the leader")


def steer_folded() -> list[Waypoint]:
    """Tip in towards the leader's axis, round inside the glass's place, and out to the dot's pose."""
    b, d = R.leader_tip(*R.FOLLOWER_START), R.leader_tip(*R.DOT_POSE)
    ab = math.degrees(math.atan2(b[1] - R.L0[1], b[0] - R.L0[0]))
    ad = math.degrees(math.atan2(d[1] - R.L0[1], d[0] - R.L0[0]))
    r_b = float(np.linalg.norm(b - np.asarray(R.L0)))
    inward = [_polar(r_b + (R_IN - r_b) * k / 2, ab) for k in (1, 2)]
    arc = [_polar(R_IN, ab + (ad - ab) * k / 6) for k in range(1, 7)]
    outward = [_polar(R_IN + (r_b - R_IN) * k / 2, ad) for k in (1, 2)]
    return _drag(inward, 0.8, "fold the arm in") + _drag(arc, 0.7, "round inside the glass") + _drag(outward, 0.8, "reach out to the dot")


def steer_straight() -> list[Waypoint]:
    b, d = R.leader_tip(*R.FOLLOWER_START), R.leader_tip(*R.DOT_POSE)
    return _drag([b + (d - b) * k / 6 for k in range(1, 7)], 0.7, "straight to the goal")


def let_go(xy) -> list[Waypoint]:
    return [_lw(0.8, xy, GRIP_Z, FINGER_OPEN, "let go"), _lw(1.2, xy, TOP + 0.12, FINGER_OPEN, "up")]


def _rw(t, xy, z, finger, label) -> Waypoint:
    return Waypoint(float(t), (float(xy[0]), float(xy[1]), float(z)), finger, label, 0.0)


PARK = {"left": (0.25, 0.25), "right": (0.25, -0.25)}


def park(side, t, label="wait") -> list[Waypoint]:
    return [Waypoint(float(t), (PARK[side][0], PARK[side][1], TOP + 0.16), FINGER_CLOSED, label, 0.0)]


def press_enable(wait_s: float) -> list[Waypoint]:
    """The right arm waits, then presses ENABLE with closed fingers (the bezel stops them at full travel) and goes back."""
    xy = R.BUTTON_XY
    return (park("right", 2.0) + park("right", max(0.1, wait_s - 2.0))
            + [_rw(2.0, xy, TOP + 0.12, FINGER_CLOSED, "above ENABLE"),
               _rw(1.2, xy, PRESS_Z, FINGER_CLOSED, "press ENABLE"),
               _rw(0.6, xy, PRESS_Z, FINGER_CLOSED, "hold it down"),
               _rw(1.0, xy, TOP + 0.12, FINGER_CLOSED, "release ENABLE")]
            + park("right", 2.0, "back"))


def _t(ws) -> float:
    return sum(w.duration for w in ws)


def demo():
    """TASK_DEMO picks a variant for checking the scorer:
    success (default): match the leader to the follower, press ENABLE, steer round inside the glass to the dot.
    enable_first: press ENABLE at once, with the leader folded; the left arm only hovers.
    goal_then_enable: pose the leader at the dot's pose, then press ENABLE.
    straight: match, press ENABLE, then drag the knob straight to the dot's pose.
    stop: match, press ENABLE, then let go and stop.
    nothing: both arms hover."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"left": park("left", 3.0) + park("left", 12.0), "right": park("right", 3.0) + park("right", 12.0)}
    if mode == "enable_first":
        return {"left": park("left", 3.0) + park("left", 12.0), "right": press_enable(2.0) + park("right", 6.0)}
    if mode == "goal_then_enable":
        left = grip_leader() + pose_leader(R.DOT_POSE)
        b = R.leader_tip(*R.DOT_POSE)
        return {"left": left + _drag([b], 7.0, "hold") + let_go(b), "right": press_enable(_t(left) + 0.5) + park("right", 3.0)}
    left = grip_leader() + pose_leader(R.FOLLOWER_START)
    right = press_enable(_t(left) + 0.5)
    hold = _drag([R.leader_tip(*R.FOLLOWER_START)], _t(right) - _t(left) - 1.5, "hold while ENABLE is pressed")
    left = left + hold
    if mode == "stop":
        b = R.leader_tip(*R.FOLLOWER_START)
        return {"left": left + let_go(b) + park("left", 6.0), "right": right}
    if mode == "straight":
        return {"left": left + steer_straight() + let_go(R.leader_tip(*R.DOT_POSE)) + park("left", 3.0), "right": right}
    return {"left": left + steer_folded() + let_go(R.leader_tip(*R.DOT_POSE)) + park("left", 3.0), "right": right}
