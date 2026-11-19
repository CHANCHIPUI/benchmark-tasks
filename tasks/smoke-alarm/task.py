"""On the station table: a First Alert SMI100 battery smoke alarm lies face down, taken off its ceiling bracket, its
low-battery chirp sounding once a minute. Its swing-out battery compartment, set into the back at the rim on the
robot's right, is shut and latched, the old 9 V (a band of red tape round its bottom end) snapped onto its contacts.
A new 9 V lies label up on its torn-open card beside it. The instruction is only "The smoke alarm keeps chirping. Make
it stop."

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
         --scorer task_score --max-steps 3600 ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scenekit import UsdBody, before_reset, on_reset, on_step, physics  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "The smoke alarm keeps chirping. Make it stop."

physics(g.PHYSICS_HZ)

TOP = g.TOP
ALARM_POS = np.array([g.ALARM_X, g.ALARM_Y, TOP + 0.0003])
_seat = g.tray_to_alarm(g.SEAT, 0.0)
OLD_POS = tuple(float(v) for v in ALARM_POS + _seat + [0.0, 0.0, 0.0002])
NEW_POS = (g.NEW_X, g.NEW_Y, TOP + g.CARD[2] + g.BATT_T / 2 + 0.0003)

ALARM = UsdBody("smoke_alarm", usd="assets/alarm.usda", pos=tuple(ALARM_POS), size=(0.142, 0.142, g.ALARM_H),
                articulation=True, joint_pos=(("hinge", 0.0), ("tab", 0.0)),
                drives=(("hinge", 0.0, g.HINGE_DAMP), ("tab", g.TAB_K, g.TAB_DAMP)),
                friction=(0.5, 0.45), solver_iters=32)
OLD = UsdBody("old_battery", usd="assets/battery_old.usda", pos=OLD_POS, size=(g.BATT_L, g.BATT_W, g.BATT_T),
              mass=g.BATT_MASS, friction=(0.6, 0.5), solver_iters=32, torsional_patch=0.005)
NEW = UsdBody("new_battery", usd="assets/battery_new.usda", pos=NEW_POS, size=(g.BATT_L, g.BATT_W, g.BATT_T),
              mass=g.BATT_MASS, friction=(0.6, 0.5), solver_iters=32, torsional_patch=0.005)
CARD = UsdBody("battery_card", usd="assets/card.usda", pos=(g.CARD_X, g.CARD_Y, TOP), size=g.CARD, friction=(0.6, 0.5))
OBJECTS = [ALARM, OLD, NEW, CARD]
BATTERIES = ("old_battery", "new_battery")

_STATE: dict = {}
_SUB: dict = {}
_VIEWS: dict = {}
_DEBUG = bool(os.environ.get("SA_DEBUG"))


def _rotation(xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), np.float64)
    return np.asarray(x, np.float64)


def _t(x):
    return getattr(x, "torch", x)


# -- geometry of the live scene ------------------------------------------------------------------------------------------
def tray_pose(alarm) -> tuple[np.ndarray, np.ndarray]:
    """The compartment's frame (origin on the hinge axis at the alarm's base height), world: (position, rotation)."""
    p, R = np.asarray(alarm.pos, float), _rotation(alarm.quat_xyzw)
    q = float(alarm.joints["hinge"])
    return p + R @ np.r_[g.HINGE, 0.0], R @ g.rot_z(q)


def battery_in_tray(battery, tray) -> dict:
    """Where a battery lies in the compartment: its snap-end face's gap to the contact block, each end's case bottom
    above the floor, its centre line off the pocket's, its angle off seated, and whether it is turned + to +."""
    pt, Rt = tray
    p = Rt.T @ (np.asarray(battery.pos, float) - pt)
    R = Rt.T @ _rotation(battery.quat_xyzw)
    ex, ez = R[:, 0], R[:, 2]
    half = g.BODY_L / 2
    snap_end, bottom_end = p + ex * half, p - ex * half
    lift = [float(e[2] - abs(ez[2]) * g.BATT_T / 2 - abs(ex[2]) * 0.0 - g.FLOOR_TOP) for e in (snap_end, bottom_end)]
    return dict(p=p, gap=float(g.BLOCK_FACE - snap_end[0]), lift=lift, side=float(p[1] - g.POCKET_YC),
                ang=float(math.acos(max(-1.0, min(1.0, ex[0])))), up=float(ez[2]),
                plus_to_plus=bool(ex[0] > math.cos(g.SNAP_ANG) and ez[2] > math.cos(g.SNAP_ANG)),
                in_pocket=bool(g.FREE_WALL[1] - 0.003 < p[0] < g.BLOCK_FACE + 0.003 and abs(p[1] - g.POCKET_YC) < 0.012
                               and g.FLOOR_TOP - 0.003 < p[2] < g.FLOOR_TOP + 0.035))


def seated(b: dict) -> bool:
    return bool(b["plus_to_plus"] and -0.0005 <= b["gap"] <= g.SNAP_GAP and max(abs(v) for v in b["lift"]) <= g.SNAP_FLAT
                and abs(b["side"]) <= g.SNAP_SIDE and b["ang"] <= g.SNAP_ANG)


def unseated(b: dict) -> bool:
    return bool(max(b["lift"]) >= g.UNSNAP_LIFT or b["gap"] >= g.UNSNAP_GAP or abs(b["side"]) > 0.002
                or b["ang"] > math.radians(10.0))


# -- the latch and the snap, in the simulation ------------------------------------------------------------------------
def _write_limits(environment, hi: float) -> None:
    import torch

    asset = environment.unwrapped.scene["smoke_alarm"]
    ids = asset.find_joints(["hinge"])[0]
    asset.write_joint_position_limit_to_sim_index(
        limits=torch.tensor([[[0.0, float(hi)]]], dtype=torch.float32, device=asset.device), joint_ids=ids,
        warn_limit_violation=False)


def _write_drive(environment, mode: str, target: float = 0.0) -> None:
    """The hinge's drive: "hump", the catch (over the last HUMP shut a spring pushes the compartment back open, HUMP_F
    at the free end at the catch); "drag", the hinge's friction (a stiff spring to a rest angle that follows the
    compartment once it pulls harder than HINGE_DRAG, so it stays where it is left); "off" while latched."""
    import torch

    from scenekit import set_joint_targets

    asset = environment.unwrapped.scene["smoke_alarm"]
    if mode != _SUB.get("mode"):
        ids = asset.find_joints(["hinge"])[0]
        k = {"hump": g.HUMP_K, "drag": g.HINGE_DRAG_K}.get(mode, 0.0)
        asset.write_joint_stiffness_to_sim_index(
            stiffness=torch.tensor([[k]], dtype=torch.float32, device=asset.device), joint_ids=ids)
        _SUB["mode"] = mode
    set_joint_targets(environment, "smoke_alarm", {"hinge": g.HUMP if mode == "hump" else target})


def _set_latch(environment, latched: bool) -> None:
    _SUB["latched"] = latched
    _write_limits(environment, g.LATCH_AT if latched else g.SWING)
    if latched:
        _write_drive(environment, "off")


def _install(environment) -> None:
    """Hook the scene's per-substep write so the snapped battery's hold is applied before every physics substep."""
    base = environment.unwrapped
    scene = base.scene
    alarm = scene["smoke_alarm"]
    _SUB.update(alarm=alarm, i_tray=alarm.body_names.index("tray"), i_hinge=alarm.joint_names.index("hinge"),
                batt={n: scene[n] for n in BATTERIES})
    if getattr(scene, "_snap_hook", False):
        return
    write = scene.write_data_to_sim

    def write_and_hold(*a, **k):
        try:
            _substep()
        except Exception as e:                          # never take the simulation down
            _SUB["err"] = repr(e)
            if _DEBUG:
                raise
        return write(*a, **k)

    scene.write_data_to_sim = write_and_hold
    scene._snap_hook = True


def _quat_rot(q):
    """Rotation matrix of an xyzw quaternion (torch, 4) -> (3, 3)."""
    import torch

    x, y, z, w = q[0], q[1], q[2], q[3]
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
                        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
                        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])])


def _substep() -> None:
    """The snap's hold, while a battery is snapped: task code holds it down on its contacts (CLAMP_F, half at each end,
    so lifting either end takes about 3 N) and onto them (CLAMP_AX), with the equal and opposite pull on the
    compartment."""
    import torch

    S = _SUB
    who = S.get("snapped")
    if who != S.get("applied"):
        for n, a in S["batt"].items():
            a.permanent_wrench_composer.reset()
        S["alarm"].permanent_wrench_composer.reset()
        S["applied"] = who
    if who is None:
        return
    batt, alarm = S["batt"][who], S["alarm"]
    Rt = _quat_rot(_t(alarm.data.body_link_quat_w)[0, S["i_tray"]])
    f = -g.CLAMP_F * Rt[:, 2] + g.CLAMP_AX * Rt[:, 0]
    pos = _t(batt.data.root_link_pose_w)[0, :3]
    batt.permanent_wrench_composer.set_forces_and_torques_index(
        forces=f.reshape(1, 1, 3), positions=pos.reshape(1, 1, 3), body_ids=[0], is_global=True)
    alarm.permanent_wrench_composer.set_forces_and_torques_index(
        forces=(-f).reshape(1, 1, 3), positions=pos.reshape(1, 1, 3), body_ids=[S["i_tray"]], is_global=True)


@before_reset
def _restart(environment) -> None:
    """The compartment shut and latched, the tab up, the old battery snapped on (the reset moves the alarm only by its
    root, and the batteries back to their start poses)."""
    from scenekit import set_joint_targets, write_joint_positions

    _install(environment)
    write_joint_positions(environment, "smoke_alarm", {"hinge": 0.0, "tab": 0.0})
    set_joint_targets(environment, "smoke_alarm", {"tab": g.TAB_PRELOAD})
    _SUB.update(snapped="old_battery", applied="none", opened_since=False, rest=0.0, mode=None)
    _set_latch(environment, True)
    _substep()


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, hinge=0.0, hinge_max=0.0, tab=0.0, latched=True, snapped="old_battery", on_table=True,
                  tier=0.0, batt={}, press={}, events=[], why=[], snap_record=None)
    _VIEWS.clear()
    if environment is not None and _SUB.get("alarm") is not None:
        _SUB.update(snapped="old_battery", opened_since=False)
        _STATE["latched"] = _SUB.get("latched", True)


_reset_state()


def _views(environment) -> dict:
    """PhysX contact views: each battery against the compartment (its floor, walls and contact block)."""
    if not _VIEWS:
        import yam_arena.constants as C

        psv = environment.unwrapped.sim.physics_sim_view
        ns = "/World/envs/env_*"
        for n in BATTERIES:
            _VIEWS[n] = psv.create_rigid_contact_view(f"{ns}/{n}", filter_patterns=[f"{ns}/smoke_alarm/tray"])
        _VIEWS["dt"] = float(C.PHYSICS_DT)
    return _VIEWS


def press_force(environment, name: str, tray_R: np.ndarray) -> float:
    """The compartment's push on a battery along the compartment's up (N): its weight plus any press on it."""
    V = _views(environment)
    m = _np(V[name].get_contact_force_matrix(V["dt"])).reshape(-1, 3)
    f = m.sum(axis=0) if m.size else np.zeros(3)
    return float(abs(f @ tray_R[:, 2]))


def tier(snapped: str | None, latched: bool, opened: bool, on_table: bool) -> float:
    """The score of an end state (see score())."""
    if not on_table:
        return 0.0
    if snapped == "new_battery":
        return 1.0 if latched else 0.5
    if snapped == "old_battery" and opened:
        return 0.25
    return 0.0


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    S["step"] += 1
    st = object_states(environment, ["smoke_alarm", *BATTERIES])
    alarm = st["smoke_alarm"]
    q = alarm.joints
    S["hinge"], S["tab"] = float(q["hinge"]), float(q["tab"])
    S["hinge_max"] = max(S["hinge_max"], S["hinge"])
    tray = tray_pose(alarm)
    # the latch: pressed 1.5 mm the tab frees the compartment; shut past the catch (tab up) it latches again
    if _SUB["latched"]:
        if -S["tab"] >= g.TAB_RELEASE:
            _set_latch(environment, False)
            _SUB["opened_since"] = False
            S["events"].append((S["step"], "released"))
    else:
        if S["hinge"] >= math.radians(1.0):
            _SUB["opened_since"] = True
        if S["hinge"] <= g.LATCH_AT and -S["tab"] < 0.0005 and _SUB["opened_since"]:
            _set_latch(environment, True)
            S["events"].append((S["step"], "latched"))
        elif S["hinge"] < g.HUMP:
            _write_drive(environment, "hump")
            _SUB["rest"] = g.HUMP
        else:
            lim = g.HINGE_DRAG / g.HINGE_DRAG_K
            rest = _SUB.get("rest", S["hinge"])
            if abs(S["hinge"] - rest) > lim:
                rest = S["hinge"] - math.copysign(lim, S["hinge"] - rest)
            _SUB["rest"] = rest
            _write_drive(environment, "drag", rest)
    S["latched"] = bool(_SUB["latched"])
    # the snap
    S["batt"] = {}
    for n in BATTERIES:
        b = battery_in_tray(st[n], tray)
        b["force"] = press_force(environment, n, tray[1])
        S["batt"][n] = b
    who = _SUB.get("snapped")
    if who is not None and unseated(S["batt"][who]):
        _SUB["snapped"] = None
        S["events"].append((S["step"], f"unsnapped {who}"))
    elif who is None:
        for n in BATTERIES:
            b = S["batt"][n]
            if seated(b) and b["force"] >= g.SNAP_F:
                _SUB["snapped"] = n
                S["events"].append((S["step"], f"snapped {n} {b['force']:.1f}N"))
                S["snap_record"] = dict(battery=n, step=S["step"], force=round(b["force"], 2),
                                        gap_mm=round(b["gap"] * 1000, 2), side_mm=round(b["side"] * 1000, 2),
                                        lift_mm=[round(v * 1000, 2) for v in b["lift"]],
                                        ang_deg=round(math.degrees(b["ang"]), 2))
                break
    S["snapped"] = _SUB.get("snapped")
    # the alarm on the table
    centre = np.asarray(alarm.pos, float) + _rotation(alarm.quat_xyzw) @ np.array([0.0, 0.0, g.ALARM_H / 2])
    S["on_table"] = bool(centre[2] - TOP < g.ON_TABLE_Z and g.TABLE_X[0] < centre[0] < g.TABLE_X[1]
                         and g.TABLE_Y[0] < centre[1] < g.TABLE_Y[1])
    S["tier"] = tier(S["snapped"], S["latched"], S["hinge_max"] >= g.OPENED, S["on_table"])
    if _DEBUG and S["step"] % 10 == 0:
        bs = " ".join(f"{n[:3]} gap {b['gap'] * 1000:.1f} lift {b['lift'][0] * 1000:.1f}/{b['lift'][1] * 1000:.1f} "
                      f"side {b['side'] * 1000:.1f} ang {math.degrees(b['ang']):.1f} F {b['force']:.1f}"
                      for n, b in S["batt"].items())
        print(f"[sa] step {S['step']} hinge {math.degrees(S['hinge']):.2f} tab {S['tab'] * 1000:.2f} latched "
              f"{S['latched']} snapped {S['snapped']} {bs} tier {S['tier']} err {_SUB.get('err')}", flush=True)


def success(objects) -> bool:
    """Never: the trial runs to its step limit and is scored on its end state."""
    return False


def score(objects) -> float:
    """From the end state, read every control step (20 Hz) from the alarm's joints and the batteries' poses:
    - snapped: a battery whose case lies on the compartment's floor (both ends within 1 mm), its snap end within 1.5 mm
      of the contact block, turned + to + (the + stud in the + contact's slot), when the compartment's floor pushes on
      it with 6 N or more; it stays snapped until one end is lifted 1.5 mm, it is drawn 2.5 mm off the block, pushed
      2 mm sideways off the pocket's centre line or turned 10 degrees off line (unseated());
    - latched: the compartment pushed shut past its catch (within 1 degree) after the tab freed it;
    - opened: the compartment swung out 45 degrees or more at some time in the run;
    - on the table: the alarm's centre less than 60 mm above the table top, over the table.
    1.0: the new battery snapped, the compartment latched, the alarm on the table; 0.5: the new battery snapped, the
    compartment not latched; 0.25: the old battery still snapped and the compartment was opened; 0 otherwise (no
    battery snapped, the compartment never opened, or the alarm off the table)."""
    return float(_STATE.get("tier", 0.0))
