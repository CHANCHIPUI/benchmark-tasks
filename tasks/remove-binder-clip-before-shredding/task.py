"""On an office desk, in front of the right arm: a light grey personal paper shredder (30 x 16 cm, 12 cm tall) with a
23 cm feed slot on top, a green ready light and a dark red jam light. In front of it, an oak letter rack holds a stack of
three folded invoices upright, 21 cm long and 9.9 cm tall. A black 41 mm binder clip grips the stack's top edge near its
left end, its two steel handles folded up. To the left stands a white ceramic dish holding three more binder clips.

The shredder starts by itself when paper is 1.5 cm into its slot and draws it in at 4 cm/s. A clip that reaches the slot
jams it: the feed stops for good, the red light comes on and the stack stays stuck. Pinching the clip's handles opens it.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import shred_usd as S  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, before_reset, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = S.build()

STACK_XY = (0.165, -0.23)          # the invoices' centre, standing in the rack, long side along y
CLIP_XY = (0.165, -0.17)           # the clip on the stack's top edge, 4.5 cm from its left end
STACK_T, STACK_L, STACK_H = S.STACK
STACK_Z = S.RACK_FLOOR + STACK_H / 2
EDGE_Z = S.RACK_FLOOR + STACK_H    # the stack's top edge: 10.4 cm up
SLOT_TOP = S.SHRED_TOP

SHREDDER = UsdBody("shredder", usd=USD["shredder"], pos=(0.0, 0.0, TOP), size=(0.16, 0.30, S.SHRED_TOP), friction=(0.6, 0.5))
DISH = UsdBody("clip_dish", usd=USD["dish"], pos=(0.0, 0.0, TOP), size=S.DISH, friction=(0.6, 0.5))
RACK = UsdBody("letter_rack", usd=USD["rack"], pos=(*STACK_XY, TOP + 0.0002), size=S.RACK, mass=S.RACK_MASS,
               friction=(0.6, 0.5))
INVOICES = UsdBody("invoices", usd=USD["invoices"], pos=(*STACK_XY, TOP + STACK_Z + 0.0002), size=S.STACK,
                   mass=S.STACK_MASS, friction=(0.6, 0.5), solver_iters=32, torsional_patch=0.008)
CLIP = UsdBody("binder_clip", usd=USD["clip"], pos=(*CLIP_XY, TOP + EDGE_Z + 0.0002), size=(0.02, S.CLIP_L, 0.045),
               articulation=True, joint_pos=(("jaw", 0.0),), drives=(("jaw", 5000.0, 5.0, 15.0),),
               friction=(0.5, 0.45), solver_iters=32)
OBJECTS = [SHREDDER, DISH, RACK, INVOICES, CLIP]

# -- the shredder and the clip, once per control step ---------------------------------------------------------------------
FEED_SPEED = 0.04                  # m/s
START_DEPTH = 0.015                # m into the slot: the auto-start sensor and the cutters
GONE_DEPTH = 0.02                  # m: the stack's upper edge this far below the top has passed the cutters
PINCH_OPEN = (0.004, 0.014)        # m: fingers closed on the 10 mm handle pair
def _clip_outline() -> np.ndarray:
    """Points on the clip's real shape (clip frame): its lips, sloped sides, back and handles, along its length."""
    xz = [(sx * 0.0023, z) for sx in (-1, 1) for z in (S.LIP_Z[0], -0.015, S.LIP_Z[1])]
    for f in np.linspace(0.0, 1.0, 6):
        xz += [(sx * (0.0028 + f * (S.SPINE_HW - 0.0028)), S.LIP_Z[1] + f * (S.SPINE_Z[1] - S.LIP_Z[1])) for sx in (-1, 1)]
    xz += [(x, S.SPINE_Z[1]) for x in np.linspace(-S.SPINE_HW, S.SPINE_HW, 5)]
    xz += [(sx * 0.004, z) for sx in (-1, 1) for z in (0.015, S.HANDLE_TOP)]
    ys = np.linspace(-S.CLIP_L / 2, S.CLIP_L / 2, 5)
    return np.array([[x, y, z] for x, z in xz for y in ys])


CLIP_PTS = _clip_outline()

_SH: dict = {}


def _shred_reset() -> None:
    _SH.clear()
    _SH.update(feeding=False, jammed=False, shredded=False, clip_on=True, started_with_clip=False, pinched=False,
               shown=None, events=[], drive=None)


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _rot(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _yaw_quat(q) -> np.ndarray:
    r = _rot(q)
    yaw = math.atan2(r[1, 1], r[0, 1]) - math.pi / 2      # the stack's long axis (local y) in the table plane
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _pose(environment, name: str):
    base = environment.unwrapped
    p = _np(base.scene[name].data.root_link_pose_w)[0]
    return p[:3] - _np(base.scene.env_origins[0]), p[3:7]


def _rotvec(r: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * math.sin(angle))
    return axis * angle


PHYS_DT = 1.0 / 240.0              # s per physics step
SETTLE_S = 0.25                    # s: the throat squares the stack up to the slot


def _lift(environment, on: bool, extra_down: float = 0.0) -> None:
    """A constant force cancelling the stack's weight (less `extra_down`), or none."""
    import torch

    asset = environment.unwrapped.scene["invoices"]
    comp = asset.permanent_wrench_composer
    comp.reset()
    if on:
        f = torch.tensor([[[0.0, 0.0, S.STACK_MASS * 9.81 - extra_down]]], dtype=torch.float32, device=asset.device)
        comp.add_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), is_global=True)


def _set_drive(environment, drive: dict | None) -> None:
    """Drive the stack along the feed or hold it (`drive`), every physics step, or let it go (None)."""
    import torch

    asset = environment.unwrapped.scene["invoices"]
    if not getattr(asset, "_shredder_drive", False):           # run _substep before every physics step
        origin = _np(environment.unwrapped.scene.env_origins[0])
        write = asset.write_data_to_sim

        def write_data_to_sim():
            write()
            _substep(asset, origin)

        asset.write_data_to_sim = write_data_to_sim
        asset._shredder_drive = True
    _SH["drive"] = drive
    _lift(environment, drive is not None)
    if drive is None:
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=asset.device))


def _substep(asset, origin) -> None:
    """Set the stack's velocity so that it follows the feed (or holds still) by the next physics step. It never jumps,
    so a clip gripping it is carried along by its grip."""
    d = _SH.get("drive")
    if not d:
        return
    import torch

    d["t"] += PHYS_DT
    pose = _np(asset.data.root_link_pose_w)[0]
    pos, quat = pose[:3] - origin, pose[3:7]
    p0, q0 = d["p0"], d["q0"]
    if d["mode"] == "feed":
        s = min(1.0, d["t"] / SETTLE_S)
        target = np.array([p0[0] + s * (S.SLOT_C[0] - p0[0]), p0[1], p0[2] - FEED_SPEED * d["t"]])
        tq, vff = _slerp(q0, _yaw_quat(q0), s), np.array([0.0, 0.0, -FEED_SPEED])
    else:
        target, tq, vff = np.asarray(p0), np.asarray(q0), np.zeros(3)
    v = vff + (target - pos) / (4 * PHYS_DT)
    w = _rotvec(_rot(tq) @ _rot(quat).T) / (4 * PHYS_DT)
    v *= min(1.0, 0.3 / max(np.linalg.norm(v), 1e-9))
    w *= min(1.0, 4.0 / max(np.linalg.norm(w), 1e-9))
    asset.write_root_velocity_to_sim_index(root_velocity=torch.tensor([[*v, *w]], dtype=torch.float32, device=asset.device))


def _release_stack(environment) -> None:
    _SH["drive"] = None
    environment.unwrapped.scene["invoices"].permanent_wrench_composer.reset()


def stack_corners(pos, quat) -> np.ndarray:
    h = np.array([STACK_T, STACK_L, STACK_H]) / 2
    local = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return local @ _rot(quat).T + np.asarray(pos)


def clip_points(pos, quat) -> np.ndarray:
    return CLIP_PTS @ _rot(quat).T + np.asarray(pos)


def in_slot(p, below: float = 0.0) -> bool:
    """A point inside the slot's outline and below the shredder's top surface (by `below`)."""
    return (S.SLOT_X[0] <= p[0] <= S.SLOT_X[1] and S.SLOT_Y[0] <= p[1] <= S.SLOT_Y[1]
            and p[2] < TOP + SLOT_TOP - below)


def grippers(environment) -> list[dict]:
    """Each arm's tool point, rotation, finger opening and fingertip pad points (station frame)."""
    from scenekit import arm_joints
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.station import BASES

    out = []
    for side in ("left", "right"):
        p, r = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), BASES[side])
        q7, q8 = arm_joints(environment, side, ["joint7", "joint8"])
        opening = float(-(q7 + q8))
        tip = p + 0.013 * r[:, 2]
        pads = [tip + sx * (opening / 2 + 0.005) * r[:, 0] + sy * 0.018 * r[:, 1] for sx in (-1, 1) for sy in (-1, 0, 1)]
        out.append({"side": side, "tcp": p, "rot": r, "opening": opening, "pads": pads})
    return out


def _pinching(g: dict, cpos, cquat) -> bool:
    """Fingers closed on the folded-up handle pair, across it (both pads on it)."""
    if not PINCH_OPEN[0] <= g["opening"] <= PINCH_OPEN[1]:
        return False
    rc = _rot(cquat)
    tcp = (g["tcp"] - cpos) @ rc
    tip = tcp + 0.013 * (g["rot"][:, 2] @ rc)
    across = abs(float(g["rot"][:, 0] @ rc[:, 0]))          # the fingers close along the clip's x (across the paper)
    hx, hy, hz = S.HANDLE_BOX
    return bool(across > 0.8 and abs(tcp[0]) < 0.012 and abs(tcp[1]) < 0.02 and tip[2] < hz[1] - 0.003
                and tcp[2] + 0.03 > hz[0])


def _show(names_on: set, names_off: set) -> None:
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    if not _SH.get("prims"):
        _SH["prims"] = {p.GetName(): p for p in stage.Traverse()
                        if p.GetName() in ("jam_on", "jam_off", "shreds") and "/shredder/" in str(p.GetPath())}
    for name, prim in _SH["prims"].items():
        if name in names_on:
            UsdGeom.Imageable(prim).MakeVisible()
        elif name in names_off:
            UsdGeom.Imageable(prim).MakeInvisible()


def _lights() -> None:
    state = (_SH["jammed"], _SH["shredded"])
    if _SH.get("shown") == state:
        return
    on, off = set(), set()
    (on if _SH["jammed"] else off).add("jam_on")
    (off if _SH["jammed"] else on).add("jam_off")
    (on if _SH["shredded"] else off).add("shreds")
    _show(on, off)
    _SH["shown"] = state


@before_reset
def _put_back(environment) -> None:
    from scenekit import set_joint_targets, write_joint_positions

    set_joint_targets(environment, "binder_clip", {"jaw": S.JAW_CLOSED})
    write_joint_positions(environment, "binder_clip", {"jaw": 0.0})
    _release_stack(environment)
    _shred_reset()
    _lights()


@on_step
def _run_shredder(environment) -> None:
    """The clip's handles and the shredder's sensor, feed and jam, once per control step."""
    from scenekit import set_joint_targets

    sh = _SH
    spos, squat = _pose(environment, "invoices")
    cpos, cquat = _pose(environment, "binder_clip")
    gs = grippers(environment)
    sh["grippers"] = gs
    # the clip: open while a gripper pinches its handles
    sh["pinched"] = any(_pinching(g, cpos, cquat) for g in gs)
    set_joint_targets(environment, "binder_clip", {"jaw": S.JAW_OPEN if sh["pinched"] else S.JAW_CLOSED})
    # on or off the stack: off for good once lifted clear of the paper's edge
    if sh["clip_on"]:
        rel = (cpos - spos) @ _rot(squat)
        on = (abs(rel[0]) < 0.008 and abs(rel[1]) < STACK_L / 2 + S.CLIP_L / 2
              and STACK_H / 2 - 0.012 < rel[2] < STACK_H / 2 - S.LIP_Z[0] + 0.001)
        if not on:
            sh["clip_on"] = False
            sh["events"].append(("clip off", _T.get("step", 0) + 1))
    # a clip in the slot jams the shredder: the feed stops for good, and the stack stays stuck where it is
    if not sh["jammed"] and any(in_slot(p) for p in clip_points(cpos, cquat)):
        sh["jammed"] = True
        sh["events"].append(("jam", _T.get("step", 0) + 1))
        if sh["feeding"] or any(in_slot(p) for p in stack_corners(spos, squat)):
            _set_drive(environment, {"mode": "hold", "p0": spos.copy(), "q0": squat.copy(), "t": 0.0})
        sh["feeding"] = False
    # the sensor: paper 1.5 cm into the slot starts the cutters, which draw it in at 4 cm/s
    if not (sh["feeding"] or sh["jammed"] or sh["shredded"]):
        low = min(stack_corners(spos, squat), key=lambda p: p[2])
        if in_slot(low, START_DEPTH):
            sh["feeding"] = True
            sh["started_with_clip"] = sh["clip_on"]
            sh["events"].append(("start", _T.get("step", 0) + 1, sh["clip_on"]))
            _set_drive(environment, {"mode": "feed", "p0": spos.copy(), "q0": squat.copy(), "t": 0.0})
    if sh["feeding"] and spos[2] + STACK_H / 2 <= TOP + SLOT_TOP - GONE_DEPTH:
        sh["feeding"], sh["shredded"] = False, True       # past the cutters: it settles out of sight under the throat
        sh["events"].append(("shredded", _T.get("step", 0) + 1))
        _set_drive(environment, None)
    if os.environ.get("TASK_DEMO") == "pulltest":
        _pull_test(environment, spos, squat)
    _lights()


def _pull_test(environment, spos, squat) -> None:
    """Debug only (TASK_DEMO=pulltest): hold the stack where it starts and pull the clip straight up with a force rising
    5 N/s from step 20, until it comes off. Measures the force the clip's grip resists."""
    import torch

    sh = _SH
    step = _T.get("step", 0)
    force = max(0.0, (step - 20) * 0.25) if sh["clip_on"] else 0.0
    sh["pull_force"] = force
    if not sh.get("drive"):
        _set_drive(environment, {"mode": "hold", "p0": np.asarray(INVOICES.pos), "q0": np.array([0.0, 0.0, 0.0, 1.0]), "t": 0.0})
    _lift(environment, True)
    stack = environment.unwrapped.scene["invoices"]       # the hand holding the stack: an equal force down, under the clip
    cpos, _ = _pose(environment, "binder_clip")
    origin = _np(environment.unwrapped.scene.env_origins[0])
    down = torch.tensor([[[0.0, 0.0, -force]]], dtype=torch.float32, device=stack.device)
    at = torch.tensor([[[*(cpos + origin)]]], dtype=torch.float32, device=stack.device)
    stack.permanent_wrench_composer.add_forces_and_torques_index(forces=down, torques=torch.zeros_like(down), positions=at,
                                                                 is_global=True)
    asset = environment.unwrapped.scene["binder_clip"]
    comp = asset.permanent_wrench_composer
    comp.reset()
    if force > 0.0:
        body = asset.find_bodies("body")[0][0]
        f = torch.tensor([[[0.0, 0.0, force]]], dtype=torch.float32, device=asset.device)
        comp.add_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), body_ids=[body], is_global=True)


def _slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a @ b < 0:
        b = -b
    q = (1 - t) * a + t * b
    return q / np.linalg.norm(q)


# -- scoring ----------------------------------------------------------------------------------------------------------------
CLEAR = 0.03                       # m: fingertips this far from the slot's opening are clear
NEAR = 0.03                        # m: a tool point this near the clip holds it
REST = 0.01                        # m/s

_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Failures stay failed for the whole trial: only a reset clears them."""
    _T.clear()
    _T.update(failed=None, step=0)


_reset()
_shred_reset()


def _slot_distance(p) -> float:
    dx = max(S.SLOT_X[0] - p[0], 0.0, p[0] - S.SLOT_X[1])
    dy = max(S.SLOT_Y[0] - p[1], 0.0, p[1] - S.SLOT_Y[1])
    dz = p[2] - (TOP + SLOT_TOP)
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def clip_on_shredder(clip) -> bool:
    for p in clip_points(clip.pos, clip.quat_xyzw):
        if (S.SHRED_X[0] <= p[0] <= S.SHRED_X[1] and S.SHRED_Y[0] <= p[1] <= S.SHRED_Y[1]
                and p[2] > TOP + SLOT_TOP - 0.005):
            return True
    return False


def clip_held(clip) -> bool:
    return any(np.linalg.norm(g["tcp"] - np.asarray(clip.pos)) < NEAR for g in _SH.get("grippers", []))


def clip_put_down(clip) -> bool:
    return bool(not _SH["clip_on"] and np.linalg.norm(clip.lin_vel) < REST and not clip_held(clip)
                and not clip_on_shredder(clip))


def grippers_clear() -> bool:
    return all(_slot_distance(p) >= CLEAR for g in _SH.get("grippers", []) for p in g["pads"])


def _update(objects) -> None:
    if _T["failed"]:
        return
    if _SH["started_with_clip"]:
        _T["failed"] = "the shredder started with the clip on the stack"
    elif _SH["jammed"]:
        _T["failed"] = "a clip entered the slot (jam)"
    else:
        for g in _SH.get("grippers", []):
            if any(in_slot(p, 0.001) for p in g["pads"]):
                _T["failed"] = f"a {g['side']} fingertip entered the slot"


def score(objects) -> float:
    """0 for the whole trial if the shredder starts with the clip on the stack, a clip enters the slot, or a fingertip
    enters the slot. Otherwise 1.0 for the stack shredded, the clip put down off the shredder and every fingertip 3 cm
    or more from the slot; 0.3 for the clip off the stack with the stack still on the desk or in the shredder; else 0."""
    _update(objects)
    if _T["failed"]:
        return 0.0
    clip = objects["binder_clip"]
    if _SH["shredded"] and clip_put_down(clip) and grippers_clear():
        return 1.0
    on_desk = _SH["shredded"] or _SH["feeding"] or objects["invoices"].pos[2] > TOP - 0.02
    if not _SH["clip_on"] and on_desk:
        return 0.3
    return 0.0


def success(objects) -> bool:
    """The stack shredded, the clip put down and the grippers clear, with no failure; held 1 s, then the trial ends."""
    steptime.tick()  # the harness calls success() once per control step
    _T["step"] += 1
    s = score(objects)
    _log(objects, s)
    return s == 1.0


def _log(objects, s: float) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    st, clip = objects["invoices"], objects["binder_clip"]
    gs = _SH.get("grippers", [])
    row = {"step": _T["step"], "score": s, "failed": _T["failed"], "clip_on": _SH["clip_on"], "pinched": _SH["pinched"],
           "feeding": _SH["feeding"], "jammed": _SH["jammed"], "shredded": _SH["shredded"],
           "jaw": round(clip.joints.get("jaw", 0.0), 5),
           "stack": [round(float(v), 4) for v in st.pos], "stack_q": [round(float(v), 4) for v in st.quat_xyzw],
           "clip": [round(float(v), 4) for v in clip.pos], "clip_q": [round(float(v), 4) for v in clip.quat_xyzw],
           "clip_v": round(float(np.linalg.norm(clip.lin_vel)), 4), "rack": [round(float(v), 4) for v in objects["letter_rack"].pos],
           "put_down": clip_put_down(clip), "on_shredder": clip_on_shredder(clip), "clear": grippers_clear(),
           "slot_dist": [round(min(_slot_distance(p) for p in g["pads"]), 4) for g in gs],
           "tcp": [[round(float(v), 4) for v in g["tcp"]] for g in gs], "opening": [round(g["opening"], 4) for g in gs],
           "events": _SH["events"], "pull_force": _SH.get("pull_force")}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------------
YAW = math.pi / 2                  # the right arm's fingers close along x
OPEN = -0.0469
HIGH = 0.16                        # TCP while moving over the desk
PINCH_H = EDGE_Z + 0.0105 + 0.013  # fingertips 1 cm above the paper edge, round the handles
GRIP_H = EDGE_Z - 0.06 + 0.013     # fingertips 6 cm below the stack's top
CARRY_H = GRIP_H + (SLOT_TOP + 0.015 - S.RACK_FLOOR)        # the stack's lower edge 1.5 cm above the shredder
FEED_H = GRIP_H + (SLOT_TOP - 0.013 - S.RACK_FLOOR)         # the lower edge 1.3 cm into the slot
DISH_DROP = (S.DISH_C[0], S.DISH_C[1], 0.065)
SLOT_XY = S.SLOT_C
AWAY = (0.22, -0.33)


def _wp(t, xy, h, finger, label, yaw=YAW) -> Waypoint:
    return Waypoint(t, (float(xy[0]), float(xy[1]), TOP + h), finger, label, yaw)


def _take_clip_off() -> list[Waypoint]:
    return [_wp(2.0, CLIP_XY, 0.18, OPEN, "above the clip"),
            _wp(1.5, CLIP_XY, PINCH_H, OPEN, "down round the handles"),
            _wp(0.8, CLIP_XY, PINCH_H, FINGER_CLOSED, "pinch the handles"),
            _wp(0.4, CLIP_XY, PINCH_H, FINGER_CLOSED, "the jaws open"),
            _wp(1.2, CLIP_XY, HIGH, FINGER_CLOSED, "lift the clip off")]


def _clip_to(xy, h, label, high=HIGH) -> list[Waypoint]:
    return [_wp(2.0, xy, high, FINGER_CLOSED, f"carry the clip {label}"),
            _wp(1.0, xy, h, FINGER_CLOSED, "lower it"),
            _wp(0.6, xy, h, OPEN, "let go"),
            _wp(0.8, xy, high, OPEN, "up")]


def _grip_stack() -> list[Waypoint]:
    return [_wp(2.0, STACK_XY, HIGH, OPEN, "above the stack"),
            _wp(1.5, STACK_XY, GRIP_H, OPEN, "down round the stack"),
            _wp(0.8, STACK_XY, GRIP_H, FINGER_CLOSED, "grip it"),
            _wp(0.3, STACK_XY, GRIP_H, FINGER_CLOSED, "settle"),
            _wp(2.0, STACK_XY, CARRY_H, FINGER_CLOSED, "lift it out of the rack")]


def _feed(fingers_in: bool = False) -> list[Waypoint]:
    h = FEED_H if not fingers_in else GRIP_H + (SLOT_TOP - 0.065 - S.RACK_FLOOR)
    return [_wp(2.0, SLOT_XY, CARRY_H, FINGER_CLOSED, "carry it over the slot"),
            _wp(1.5, SLOT_XY, h, FINGER_CLOSED, "lower it into the slot"),
            _wp(0.6, SLOT_XY, h, OPEN, "let the shredder take it"),
            _wp(1.0, SLOT_XY, CARRY_H + 0.01, OPEN, "up"),
            _wp(1.5, AWAY, HIGH + 0.03, OPEN, "move clear")]


def _wait(t) -> list[Waypoint]:
    return [_wp(t, AWAY, HIGH + 0.03, OPEN, "wait clear")]


def demo():
    """Open loop, so it uses nothing a policy could not see. The right arm pinches the clip's handles, lifts the clip off
    and drops it in the dish, then grips the stack, lifts it out of the rack, lowers it into the slot and lets go.
    TASK_DEMO picks a variant for checking the scorer: failure (feed the stack with the clip on), nothing, cliponly (the
    clip into the dish, the stack left in the rack), clipslot (the clip dropped into the slot), clipontop (the clip left
    on the shredder's top, then the stack fed), byclip (the clip gripped by its body and lifted: the stack comes with it),
    fingers (the clip off, then the stack pushed in with the fingertips inside the slot), poke (the clip off, the
    closed fingertips poked 1 cm into the empty slot and out again, then the job finished), pulltest (debug: the clip's grip)."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "pulltest":
        return {"right": [_wp(2.0, AWAY, HIGH + 0.03, OPEN, "hover"), _wp(8.0, AWAY, HIGH + 0.03, OPEN, "wait")]}
    if mode == "nothing":
        return {"right": [_wp(2.0, AWAY, HIGH + 0.03, OPEN, "hover"), _wp(8.0, AWAY, HIGH + 0.03, OPEN, "wait")]}
    if mode == "failure":
        return {"right": _grip_stack() + _feed() + _wait(5.0)}
    if mode == "byclip":
        body_h = EDGE_Z + 0.001 + 0.013                     # fingertips on the clip's spine, below the handles
        return {"right": [_wp(2.0, CLIP_XY, 0.18, OPEN, "above the clip"),
                          _wp(1.5, CLIP_XY, body_h, OPEN, "down round its body"),
                          _wp(0.8, CLIP_XY, body_h, FINGER_CLOSED, "grip its body"),
                          _wp(1.5, CLIP_XY, body_h + 0.05, FINGER_CLOSED, "lift: the stack comes too"),
                          _wp(2.0, CLIP_XY, body_h + 0.05, FINGER_CLOSED, "hold"),
                          _wp(1.5, CLIP_XY, body_h + 0.002, FINGER_CLOSED, "back down"),
                          _wp(0.6, CLIP_XY, body_h + 0.002, OPEN, "let go")] + _wait(2.0)}
    plan = _take_clip_off()
    if mode == "clipslot":
        return {"right": plan + _clip_to(SLOT_XY, SLOT_TOP + 0.035, "over the slot", HIGH + 0.03) + _wait(3.0)}
    if mode == "clipontop":
        return {"right": plan + _clip_to((0.29, -0.095), SLOT_TOP + 0.042, "onto the shredder", HIGH + 0.03) + _grip_stack() + _feed()
                + _wait(5.0)}
    plan += _clip_to(DISH_DROP[:2], DISH_DROP[2], "to the dish")
    if mode == "cliponly":
        return {"right": plan + _wait(3.0)}
    if mode == "poke":
        poke = [_wp(1.5, SLOT_XY, HIGH, FINGER_CLOSED, "over the empty slot"),
                _wp(1.0, SLOT_XY, SLOT_TOP + 0.013 - 0.01, FINGER_CLOSED, "fingertips 1 cm into the slot"),
                _wp(1.0, SLOT_XY, HIGH, FINGER_CLOSED, "out again")]
        return {"right": plan + poke + _grip_stack() + _feed() + _wait(5.0)}
    if mode == "fingers":
        return {"right": plan + _grip_stack() + _feed(fingers_in=True) + _wait(3.0)}
    return {"right": plan + _grip_stack() + _feed() + _wait(5.0)}
