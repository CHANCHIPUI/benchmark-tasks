"""Two bimanual YAM stations stand side by side under a plywood panel that serves as the ceiling, its underside 580 mm
above the table. The bare steel housing of a Commercial Electric WR484080LSL 4-ft LED wrap light (lens and channel
cover off, 1,218 x 280 x 55 mm, 3.7 kg) rests with its back up on two padded stands, 100 mm below the ceiling, under
its pencil outline. Two pilot holes are drilled in the ceiling, one at each end, above a diagonal pair of the
housing's keyhole slots. Each station has a push-to-start cordless screwdriver standing on a holder, a #8 x 3/4 in
pan-head screw on its magnetic bit. Each robot is told to raise the light with the other, hold it flat against the
ceiling with a slot lined up under each pilot hole, drive its own screw up through the slot at its end into the pilot
hole until it is tight, then let go.

Task code (fidelity note): the screw model (magnetic bit, bite, motor and clutch, thread, seating) in _substep() and
on_step(); scoring in score(). Run with the yam_pair embodiment (CONTRACT.md in yam-pair-kit):

    inspect-robots run --embodiment yam_pair -E task=task.py --policy yam_pair_scripted --instruction "$INSTR" \\
        --scorer pair_task_score --max-steps 6000
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geometry as g  # noqa: E402
from yam_pair.task import Layout, Shape, UsdBody  # noqa: E402

LAYOUT = Layout("side_by_side", separation=g.SEPARATION)
PHYSICS_HZ = 240
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
_TASK = json.loads((HERE / "task.json").read_text()) if (HERE / "task.json").exists() else {}
INSTRUCTIONS = dict(_TASK.get("instructions", {}))
DEBUG = bool(os.environ.get("CLP_DEBUG"))
VARIANT = os.environ.get("CLP_VARIANT", "")

TOP, CEIL = g.TOP, g.CEIL
WOOD, WOOD_DARK, FOAM, PLY = (0.74, 0.60, 0.42), (0.62, 0.48, 0.32), (0.18, 0.18, 0.20), (0.80, 0.68, 0.50)
HOUSING_START = (g.HOUSING_X, 0.0, CEIL - g.START_DROP + 0.0005)
DRIVER_START = {s: (x, y, TOP + g.HOLDER_FLOOR[s] + 0.0005) for s, (x, y) in g.HOLDER_XY.items()}


def _screw_start(s: str) -> dict:
    """The screw sitting on its driver's bit in the holder: joint values from its pilot hole."""
    hx, hy = g.SCREW_XY[s]
    dx, dy, dz = DRIVER_START[s]
    tip_z = dz + g.DRV_TIP + g.HEAD_H + g.SCREW_L
    return {f"{s}_px": dx - hx, f"{s}_py": dy - hy, f"{s}_pz": tip_z - CEIL, f"{s}_spin": 0.0}


SCREW_JOINTS = {**_screw_start("a"), **_screw_start("b")}


def _panel() -> list:
    """The ceiling panel's collider as boxes, open (10 mm square) over each pilot hole."""
    rects = [(g.PANEL_X[0], g.PANEL_X[1], g.PANEL_Y[0], g.PANEL_Y[1])]
    for hx, hy in g.SCREW_XY.values():
        rects = g.subtract(rects, (hx - g.HOLE_OPEN, hx + g.HOLE_OPEN, hy - g.HOLE_OPEN, hy + g.HOLE_OPEN))
    out = []
    for i, (x0, x1, y0, y1) in enumerate(rects):
        out.append(Shape(f"panel_{i}", pos=((x0 + x1) / 2, (y0 + y1) / 2, CEIL + g.PANEL_T / 2),
                         size=(x1 - x0, y1 - y0, g.PANEL_T), color=PLY))
    return out


def _frame() -> list:
    out = []
    for i, (x, y) in enumerate(g.LEG_XY):
        out.append(Shape(f"leg_{i}", pos=(x, y, (TOP + CEIL) / 2), size=(g.LEG[0], g.LEG[1], CEIL - TOP), color=WOOD))
    bar_top = TOP + 0.425 - g.FOAM_T
    for i, y in enumerate(g.STAND_Y):
        post_h = bar_top - g.CROSSBAR[2] - TOP
        out.append(Shape(f"stand_post_{i}", pos=(g.HOUSING_X, y, TOP + post_h / 2), size=(g.STAND_POST[0], g.STAND_POST[1],
                                                                                       post_h), color=WOOD))
        out.append(Shape(f"stand_bar_{i}", pos=(g.HOUSING_X, y, bar_top - g.CROSSBAR[2] / 2), size=g.CROSSBAR,
                         color=WOOD_DARK))
        out.append(Shape(f"stand_foam_{i}", pos=(g.HOUSING_X, y, bar_top + g.FOAM_T / 2),
                         size=(g.CROSSBAR[0], g.CROSSBAR[1], g.FOAM_T), color=FOAM))
        out.append(Shape(f"stand_foot_{i}", pos=(g.HOUSING_X, y, TOP + 0.006), size=(0.20, 0.20, 0.012), color=WOOD_DARK))
    for s, (x, y) in g.HOLDER_XY.items():
        fl = TOP + g.HOLDER_FLOOR[s]                      # the driver's bottom, on the peg
        rt = fl - g.PEG_H
        out.append(Shape(f"holder_{s}_riser", pos=(x, y, (TOP + rt) / 2), size=(0.089, 0.089, rt - TOP), color=WOOD))
        # the peg is turned square to the hand's approach (from the inner arm's base), so the fingers pass its sides
        yaw = math.atan2(y - (0.195 if s == "a" else -0.195), x)
        out.append(Shape(f"holder_{s}_peg", pos=(x, y, rt + g.PEG_H / 2), size=(g.PEG_W, g.PEG_W, g.PEG_H),
                         rot_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), color=WOOD_DARK))
    return out


OBJECTS = [
    UsdBody("housing", usd="assets/housing.usda", pos=HOUSING_START, size=(g.HOUSING_W, g.HOUSING_L, g.HOUSING_D),
            mass=g.HOUSING_MASS, friction=(0.8, 0.7), solver_iters=32),
    UsdBody("driver_a", usd="assets/driver.usda", pos=DRIVER_START["a"], size=(0.045, 0.045, g.DRV_TIP),
            mass=g.DRV_MASS, friction=(0.8, 0.7), solver_iters=32, torsional_patch=0.01),
    UsdBody("driver_b", usd="assets/driver.usda", pos=DRIVER_START["b"], size=(0.045, 0.045, g.DRV_TIP),
            mass=g.DRV_MASS, friction=(0.8, 0.7), solver_iters=32, torsional_patch=0.01),
    UsdBody("screws", usd="assets/screws.usda", pos=(0.0, 0.0, 0.0), size=(0.01, 0.01, 0.02), articulation=True,
            joint_pos=tuple(SCREW_JOINTS.items()),
            drives=(("[ab]_p[xyz]", g.BIT_K, g.BIT_C, g.MAGNET_F), ("[ab]_spin", 0.0, 0.05, g.CLUTCH)),
            friction=(0.7, 0.6), solver_iters=32),
    *_panel(),
    *_frame(),
]

# -- live state ------------------------------------------------------------------------------------------------------
_S: dict = {}       # scoring and monitor state (control rate)
_SUB: dict = {}     # substep state
_V: dict = {}       # PhysX views


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), np.float64)
    return np.asarray(x, np.float64)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _ids(asset, names):
    return asset.find_joints(list(names), preserve_order=True)[0]


def _write(env, fn: str, s: str, values: dict) -> None:
    """Write per-joint values (limits as pairs) to the screws articulation."""
    import torch

    asset = env.unwrapped.scene["screws"]
    names = [f"{s}_{k}" for k in values]
    ids = _ids(asset, names)
    vals = list(values.values())
    if fn == "limits":
        t = torch.tensor([[list(v) for v in vals]], dtype=torch.float32, device=asset.device)
        asset.write_joint_position_limit_to_sim_index(limits=t, joint_ids=ids, warn_limit_violation=False)
        return
    t = torch.tensor([vals], dtype=torch.float32, device=asset.device)
    {"stiffness": lambda: asset.write_joint_stiffness_to_sim_index(stiffness=t, joint_ids=ids),
     "damping": lambda: asset.write_joint_damping_to_sim_index(damping=t, joint_ids=ids),
     "effort": lambda: asset.write_joint_effort_limit_to_sim_index(limits=t, joint_ids=ids),
     "armature": lambda: asset.write_joint_armature_to_sim_index(armature=t, joint_ids=ids),
     "pos_target": lambda: asset.actuators.target_command.set_position_index(value=t, joint_ids=ids),
     "vel_target": lambda: asset.actuators.target_command.set_velocity_index(value=t, joint_ids=ids)}[fn]()


def _pre_bite(env, s: str) -> None:
    """A screw on its bit: free to move with it (the magnet's capped spring), its tip stopped at the ceiling."""
    _write(env, "limits", s, {"px": (-0.6, 0.6), "py": (-0.6, 0.6), "pz": (-0.8, 0.0)})
    _write(env, "stiffness", s, {"px": g.BIT_K, "py": g.BIT_K, "pz": g.BIT_K, "spin": 0.0})
    _write(env, "damping", s, {"px": g.BIT_C, "py": g.BIT_C, "pz": g.BIT_C, "spin": 0.05})
    _write(env, "effort", s, {"px": g.MAGNET_F, "py": g.MAGNET_F, "pz": g.MAGNET_F, "spin": g.CLUTCH})
    _write(env, "armature", s, {"px": 0.05, "py": 0.05, "pz": 0.05, "spin": 2e-5})
    _write(env, "vel_target", s, {"spin": 0.0})


# -- the screw model, applied before every physics substep ------------------------------------------------------------
def _install(env) -> None:
    base = env.unwrapped
    scene = base.scene
    art = scene["screws"]
    _SUB.update(art=art, drivers={s: scene[f"driver_{s}"] for s in "ab"},
                jid={s: {k: int(_ids(art, [f"{s}_{k}"])[0]) for k in ("px", "py", "pz", "spin")} for s in "ab"},
                origin=_np(scene.env_origins[0]), dt=float(base.physics_dt))
    if getattr(scene, "_clp_hook", False):
        return
    write = scene.write_data_to_sim

    def write_and_model(*a, **k):
        try:
            _substep(env)
        except Exception as e:                       # never take the simulation down
            _SUB["err"] = repr(e)
            if DEBUG:
                raise
        return write(*a, **k)

    scene.write_data_to_sim = write_and_model
    scene._clp_hook = True


def _driver_pose(s: str):
    d = _SUB["drivers"][s]
    pose = _np(d.data.root_link_pose_w)[0]
    w = _np(d.data.root_ang_vel_w)[0]
    return pose[:3] - _SUB["origin"], _rot(pose[3:7]), w


def _substep(env) -> None:
    """Before every physics substep, for each screw:
    - on its bit (before it bites): the bit's contact and magnet, as a stiff spring between the bit's tip and the
      screw's head (task code; the bit has no collider): it pushes the head with up to PUSH_MAX and pulls it back with
      at most MAGNET_F (sideways MAGNET_F); the push also acts back on the driver;
    - bitten: the head pushes back on the bit pressed into it; the thread ties advance to turn (the z joint's drive
      follows the spin angle); the motor turns the screw while the bit sits in the head and pushes it PUSH_START or more; the torque
      the screw takes (thread drag plus the clamp through the thread), capped by the clutch, acts back on the
      driver."""
    import torch

    S, art = _SUB, _SUB["art"]
    if "state" not in S:
        return
    q = _np(art.data.joint_pos)[0]
    for s in "ab":
        st, j = S["state"][s], S["jid"][s]
        hx, hy = g.SCREW_XY[s]
        p_d, R_d, w_d = _driver_pose(s)
        axis = R_d[:, 2]
        bit = p_d + axis * g.DRV_TIP
        tip = np.array([hx + q[j["px"]], hy + q[j["py"]], CEIL + q[j["pz"]]])
        head = tip - np.array([0.0, 0.0, g.SCREW_L + g.HEAD_H])
        rel = bit - head
        st["bit_lat"] = float(math.hypot(rel[0], rel[1]))
        st["bit_ax"] = float(rel[2])
        st["tilt"] = float(math.degrees(math.acos(max(-1.0, min(1.0, axis[2])))))
        push = 0.0
        if not st["bitten"]:
            rng = 3.0 * g.MAGNET_RANGE if st["magnet"] else g.MAGNET_RANGE     # held until pulled 18 mm away
            near = st["bit_lat"] <= rng and abs(st["bit_ax"]) <= rng and st["tilt"] < 30.0
            if near != st["magnet"]:
                k = g.BIT_K if near else 0.0
                _write(env, "stiffness", s, {"px": k, "py": k, "pz": k})
                st["magnet"] = near
            if near:
                want = bit + np.array([0.0, 0.0, g.HEAD_H + g.SCREW_L])
                pz_t = want[2] - CEIL
                at_stop = float(q[j["pz"]]) >= -0.0005                         # its tip stopped at the ceiling
                v_ax = float(_SUB["drivers"][s].data.root_lin_vel_w.torch[0, 2].item()) if hasattr(
                    _SUB["drivers"][s].data.root_lin_vel_w, "torch") else float(_np(_SUB["drivers"][s].data.root_lin_vel_w)[0, 2])
                push = max(0.0, g.BIT_K * (pz_t - float(q[j["pz"]])) + g.BIT_D * v_ax) if at_stop else 0.0
                cap = g.PUSH_MAX if push > 0.0 else g.MAGNET_F
                if cap != st.get("cap"):
                    _write(env, "effort", s, {"pz": cap})
                    st["cap"] = cap
                push = min(push, g.PUSH_MAX)
                _write(env, "pos_target", s, {"px": want[0] - hx, "py": want[1] - hy, "pz": pz_t})
            st["push_sum"] = st.get("push_sum", 0.0) + push
            st["push_n"] = st.get("push_n", 0) + 1
            st["push"] = push
            _driver_wrench(s, push, 0.0, axis, bit)
            continue
        # bitten: the bit's contact with the head, the thread, the motor
        if st["bit_lat"] <= g.HEAD_R and st["bit_ax"] > 0.0:                     # the bit on the head
            v_ax = float(_np(_SUB["drivers"][s].data.root_lin_vel_w)[0, 2])
            push = min(max(0.0, g.BIT_K * st["bit_ax"] + g.BIT_D * v_ax), g.PUSH_MAX)
        st["push_sum"] = st.get("push_sum", 0.0) + push
        st["push_n"] = st.get("push_n", 0) + 1
        st["push"] = push
        th = float(q[j["spin"]])
        st["theta"] += math.remainder(th - st["theta_prev"], 2 * math.pi)
        st["theta_prev"] = th
        target = st["z_bite"] + g.PITCH * (st["theta"] - st["theta_bite"]) / (2 * math.pi)
        z = float(q[j["pz"]])
        force = g.THREAD_K * (target - z)
        st["thread_f"] = force
        need = g.THREAD_DRAG + max(force, 0.0) * g.PITCH / (2 * math.pi * g.THREAD_EFF)
        seated_bit = st["bit_lat"] <= 0.0015 and -0.0015 <= st["bit_ax"] <= 0.004 and st["tilt"] <= 6.0
        on = seated_bit and push >= g.PUSH_START
        w_ax = float(w_d @ axis)
        clutch = S.get("clutch", g.CLUTCH)
        if on and need < clutch:
            omega, delivered, st["slipping"] = max(0.0, g.RPM * 2 * math.pi / 60 + w_ax), need, False
        elif on:
            omega, delivered, st["slipping"] = 0.0, clutch, True
        else:
            omega, delivered, st["slipping"] = 0.0, 0.0, False
        st["motor"] = on
        st["delivered"] = delivered
        st["w_driver"] = w_ax
        if on:
            st["turn"] += math.degrees(w_ax) / PHYSICS_HZ                       # the driver turning in the hand
        _write(env, "vel_target", s, {"spin": omega})
        _write(env, "pos_target", s, {"pz": min(target, g.TRAVEL + 0.004)})
        _driver_wrench(s, push, delivered, axis, bit)


def _driver_wrench(s: str, push: float, torque: float, axis, bit) -> None:
    """The head pushing back on the bit (along the driver's axis, at its tip) and the motor's reaction torque."""
    import torch

    st, d = _SUB["state"][s], _SUB["drivers"][s]
    if push <= 0.0 and torque <= 0.0:
        if st.get("wrench_on"):
            d.permanent_wrench_composer.reset()
            st["wrench_on"] = False
        return
    f = torch.tensor((-push * axis).reshape(1, 1, 3), dtype=torch.float32, device=d.device)
    tq = torch.tensor((-torque * axis).reshape(1, 1, 3), dtype=torch.float32, device=d.device)
    pos = torch.tensor(np.asarray(bit + _SUB["origin"]).reshape(1, 1, 3), dtype=torch.float32, device=d.device)
    d.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=tq, positions=pos, body_ids=[0],
                                                             is_global=True)
    st["wrench_on"] = True


def _bite(env, s: str) -> None:
    import torch  # noqa: F401

    st, j = _S["state"][s], _SUB["jid"][s]
    q = _np(_SUB["art"].data.joint_pos)[0]
    px, py, pz, th = (float(q[j[k]]) for k in ("px", "py", "pz", "spin"))
    # the pilot hole takes the tip: the screw is drawn onto the hole's axis and held there
    _write(env, "limits", s, {"pz": (pz - 0.001, g.TRAVEL + 0.004)})
    _write(env, "stiffness", s, {"px": 2.0e5, "py": 2.0e5, "pz": g.THREAD_K})
    _write(env, "damping", s, {"px": 200.0, "py": 200.0, "pz": g.THREAD_C})
    _write(env, "effort", s, {"px": 1000.0, "py": 1000.0, "pz": 3000.0})
    _write(env, "pos_target", s, {"px": 0.0, "py": 0.0, "pz": pz})
    st.update(bitten=True, z_bite=pz, theta=th, theta_prev=th, theta_bite=th, magnet=False, bite_step=_S["step"])
    _S["events"].append((_S["step"], f"{s} screw bit at ({px * 1000:.1f}, {py * 1000:.1f}) mm from its hole, "
                                     f"push {st['push']:.1f} N"))


# -- views and monitor --------------------------------------------------------------------------------------------------
def _views(env) -> dict:
    if not _V:
        from yam_arena.robot import asset_link_paths

        psv = env.unwrapped.sim.physics_sim_view
        ns = "/World/envs/env_*"
        lp = asset_link_paths()
        hands = {f"{st}_{side}": [f"{ns}/{st.upper()}{side.title()}Arm{lp[k]}" for k in ("gripper", "tip_left", "tip_right")]
                 for st in "ab" for side in ("left", "right")}
        _V["hands"] = hands
        for arm, paths in hands.items():
            _V[f"housing_{arm}"] = psv.create_rigid_contact_view(f"{ns}/housing", filter_patterns=paths)
            for s in "ab":
                _V[f"driver_{s}_{arm}"] = psv.create_rigid_contact_view(f"{ns}/driver_{s}", filter_patterns=paths)
        for s in "ab":
            _V[f"push_{s}"] = psv.create_rigid_contact_view(f"{ns}/screws/{s}_screw", filter_patterns=[f"{ns}/driver_{s}"])
            _V[f"clamp_{s}"] = psv.create_rigid_contact_view(f"{ns}/screws/{s}_screw", filter_patterns=[f"{ns}/housing"])
            _V[f"drvhous_{s}"] = psv.create_rigid_contact_view(f"{ns}/driver_{s}", filter_patterns=[f"{ns}/housing"])
        _V["dt"] = float(env.unwrapped.physics_dt)
    return _V


def _force(view, dt) -> float:
    m = _np(view.get_contact_force_matrix(dt)).reshape(-1, 3)
    return float(np.linalg.norm(m.sum(axis=0))) if m.size else 0.0


def _clamp(view, dt) -> float:
    """The housing's push down on the screw's head (the head clamping the housing up against the ceiling), N."""
    m = _np(view.get_contact_force_matrix(dt)).reshape(-1, 3)
    return float(max(0.0, -m.sum(axis=0)[2])) if m.size else 0.0


def _reset_state(env=None) -> None:
    _S.clear()
    _S.update(step=0, dropped=False, drop_step=None, events=[], last_contact_step=0, held=False,
              peak={}, tier=0.0, report={})
    _SUB["state"] = {s: dict(bitten=False, magnet=True, push=0.0, push_steps=0, theta=0.0, theta_prev=0.0, theta_bite=0.0,
                             z_bite=0.0, thread_f=0.0, slipping=False, motor=False, delivered=0.0, w_driver=0.0,
                             bit_lat=0.0, bit_ax=0.0, tilt=0.0, torque_on=False, bite_step=None, seated=False,
                             fastened=False, tight=False, flush=False, clamp=0.0, gap=1.0, edge_gap=1.0, turn=0.0)
                     for s in "ab"}
    _S["state"] = _SUB["state"]
    _SUB["clutch"] = float(os.environ.get("CLP_CLUTCH", g.CLUTCH))


def on_build(env) -> None:
    """Visual only: the two plywood plugs with their pilot holes, and the pencil outline, on the panel's underside."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()

    def mat(name, color, rough=0.7):
        path = f"/World/CeilingDecor/Looks/{name}"
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return m

    plug, hole, pencil = mat("plug", (0.86, 0.74, 0.55)), mat("hole", (0.05, 0.04, 0.03)), mat("pencil", (0.15, 0.15, 0.17))
    root = "/World/envs/env_0/CeilingDecor"
    for s, (x, y) in g.SCREW_XY.items():
        for name, r, m, dz in ((f"plug_{s}", g.PLUG_R, plug, 0.0004), (f"hole_{s}", 0.0016, hole, 0.0008)):
            c = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
            c.CreateRadiusAttr(r)
            c.CreateHeightAttr(0.0004)
            c.AddTranslateOp().Set(Gf.Vec3d(x, y, CEIL - dz))
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(m)
    x0, x1 = g.HOUSING_X - g.HOUSING_W / 2, g.HOUSING_X + g.HOUSING_W / 2
    y0, y1 = -g.HOUSING_L / 2, g.HOUSING_L / 2
    for k, (cx, cy, sx, sy) in enumerate((((x0 + x1) / 2, y0, x1 - x0 + 0.03, 0.0012), ((x0 + x1) / 2, y1, x1 - x0 + 0.03, 0.0012),
                                          (x0, 0.0, 0.0012, y1 - y0 + 0.03), (x1, 0.0, 0.0012, y1 - y0 + 0.03))):
        c = UsdGeom.Cube.Define(stage, f"{root}/line_{k}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(cx, cy, CEIL - 0.0003))
        c.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.0003))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(pencil)


VIEWS = {"front": ((1.55, 0.35, TOP + 0.30), (0.30, 0.0, TOP + 0.47)),
         "side": ((0.42, -1.75, TOP + 0.42), (0.42, 0.0, TOP + 0.47)),
         "a_end": ((0.95, 0.85, TOP + 0.30), (0.40, 0.45, TOP + 0.52)),
         "b_end": ((0.95, -0.85, TOP + 0.30), (0.45, -0.45, TOP + 0.52)),
         "a_slot": ((0.26, 0.40, TOP + 0.47), (0.37, 0.508, TOP + 0.565)),
         "holders": ((0.75, 0.0, TOP + 0.45), (0.20, 0.10, TOP + 0.22)),
         "b_close": ((0.62, -0.95, TOP + 0.36), (0.42, -0.58, TOP + 0.46)),
         "b_under": ((0.20, -0.75, TOP + 0.20), (0.40, -0.45, TOP + 0.47))}


def place_overview(env, view: str | None = None) -> None:
    """Put yam_pair's overview camera (not seen by either robot) where the ceiling does not hide the work."""
    import torch

    scene = env.unwrapped.scene
    if "overview_cam" not in scene.keys():
        return
    eye, target = VIEWS[view or os.environ.get("CLP_VIEW", "front")]
    origin = scene.env_origins[0:1]
    cam = scene["overview_cam"]
    cam.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=origin.device) + origin,
                                  torch.tensor([target], dtype=torch.float32, device=origin.device) + origin)


def before_reset(env) -> None:
    _install(env)
    _reset_state(env)
    for s in "ab":
        _pre_bite(env, s)
        _SUB["drivers"][s].permanent_wrench_composer.reset()


def on_reset(env) -> None:
    _install(env)
    place_overview(env)
    _reset_state(env)
    _V.clear()
    from yam_pair.sim import set_joint_targets

    set_joint_targets(env, "screws", {k: v for k, v in SCREW_JOINTS.items() if not k.endswith("spin")})


def _housing(env):
    h = env.unwrapped.scene["housing"]
    pose = _np(h.data.root_link_pose_w)[0]
    vel = _np(h.data.root_vel_w)[0]
    return pose[:3] - _SUB["origin"], _rot(pose[3:7]), vel[:3], vel[3:6]


def slot_frame(p_h, R_h, s: str):
    """The screw spot of station s's slot in the world, from the housing's pose."""
    hx, hy = g.SCREW_XY[s]
    return p_h + R_h @ np.array([hx - g.HOUSING_X, hy, 0.0])


def on_step(env) -> None:
    S, V = _S, _views(env)
    S["step"] += 1
    dt = V["dt"]
    art = _SUB["art"]
    q = _np(art.data.joint_pos)[0]
    p_h, R_h, v_h, w_h = _housing(env)
    # dropped: any corner of the housing moving down at 0.3 m/s or faster, more than 25 mm below where it is flush
    # (so settling onto screws left short is not a drop)
    L, W, D = g.HOUSING_L, g.HOUSING_W, g.HOUSING_D
    corners = [np.array([sx * W / 2, sy * L / 2, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (0.0, -D)]
    vz = min((float((v_h + np.cross(w_h, R_h @ c))[2]) for c in corners
              if (p_h + R_h @ c)[2] < CEIL + c[2] - g.DROP_BELOW), default=0.0)
    if vz <= -g.DROP_V and not S["dropped"]:
        S["dropped"], S["drop_step"] = True, S["step"]
        S["events"].append((S["step"], f"dropped (a corner moving down at {-vz:.2f} m/s)"))
    # who touches the housing / the drivers (per arm)
    held = False
    S["hand_f"] = {}
    for arm in V["hands"]:
        f = _force(V[f"housing_{arm}"], dt)
        S["hand_f"][arm] = f
        S["peak"][f"housing_{arm}"] = max(S["peak"].get(f"housing_{arm}", 0.0), f)
        held = held or f > g.HELD_F
    S["held"] = held
    if held:
        S["last_contact_step"] = S["step"]
    for s in "ab":
        st, j = S["state"][s], _SUB["jid"][s]
        hx, hy = g.SCREW_XY[s]
        st["clamp"] = _clamp(V[f"clamp_{s}"], dt)
        st["push_avg"] = st.get("push_sum", 0.0) / max(1, st.get("push_n", 0))       # over this control step
        st["push_sum"], st["push_n"] = 0.0, 0
        S["peak"][f"push_{s}"] = max(S["peak"].get(f"push_{s}", 0.0), st["push_avg"])
        tip = np.array([hx + q[j["px"]], hy + q[j["py"]], CEIL + q[j["pz"]]])
        # the bite: tip at the ceiling, at its pilot hole, square, pushed BITE_PUSH for BITE_HOLD
        if not st["bitten"]:
            ok = (q[j["pz"]] >= -0.0005 and math.hypot(q[j["px"]], q[j["py"]]) <= 0.0015 and st["tilt"] <= g.BITE_TILT
                  and st["push_avg"] >= g.BITE_PUSH)
            st["push_steps"] = st["push_steps"] + 1 if ok else 0
            if st["push_steps"] >= max(1, round(g.BITE_HOLD * 20)):
                _bite(env, s)
        # fastened: bitten, its shank in the narrow part of the housing's slot
        loc = R_h.T @ (tip - p_h)
        kx = hx - g.HOUSING_X
        out = 1.0 if s == "a" else -1.0
        lo, hi = sorted((hy - out * g.SLOT_IN, hy + out * (g.SLOT_TO_ROUND - g.ROUND_R)))
        in_slot = abs(loc[0] - kx) <= g.SLOT_W / 2 and lo <= loc[1] <= hi and loc[2] < 0.03
        st["fastened"] = bool(st["bitten"] and in_slot)
        # seated: head within 0.3 mm of the housing's back, pressing it 50 N
        under_back = p_h + R_h @ np.array([loc[0], loc[1], -g.SHEET_T])
        head_top = tip[2] - g.SCREW_L
        st["gap"] = float(under_back[2] - head_top)
        st["seated"] = bool(st["fastened"] and st["gap"] <= g.SEAT_GAP and st["clamp"] >= g.SEAT_F)
        # flush: the back's nearest long edge, in line with the screw, within 2 mm of the ceiling
        edge_x = -W / 2 if s == "a" else W / 2
        edge = p_h + R_h @ np.array([edge_x, loc[1], 0.0])
        st["edge_gap"] = float(CEIL - edge[2])
        st["flush"] = bool(st["edge_gap"] <= g.FLUSH_GAP)
        st["tight"] = bool(st["seated"] and st["flush"])
        if st["tight"] and not st.get("was_tight") and not st.get("tight_logged"):
            st["tight_logged"] = True
            S["events"].append((S["step"], f"{s} end tight: clamp {st['clamp']:.0f} N, head gap {st['gap'] * 1000:.2f} mm, "
                                            f"edge gap {st['edge_gap'] * 1000:.2f} mm"))
        st["was_tight"] = st["tight"]
    hangs = (S["step"] - S["last_contact_step"]) >= round(g.HANG_S * 20) and not held
    a, b = S["state"]["a"], S["state"]["b"]
    n_fast = int(a["fastened"]) + int(b["fastened"])
    n_tight = int(a["tight"]) + int(b["tight"])
    if S["dropped"] or n_fast == 0:
        tier = 0.0
    elif n_fast == 2 and hangs:
        tier = (0.5, 0.75, 1.0)[n_tight]
    else:
        tier = 0.25
    S["tier"] = tier
    if hangs and not S.get("hangs") and n_fast:
        S["events"].append((S["step"], f"hangs on its own (no gripper contact for {g.HANG_S:.0f} s); tier {tier}"))
    S["hangs"] = hangs
    _peaks(env)
    if DEBUG:
        for step, text in S["events"][S.get("printed", 0):]:
            print(f"[clp] event step {step}: {text}", flush=True)
        S["printed"] = len(S["events"])
        if S["step"] % 100 == 0:
            print("[rep] step", S["step"], json.dumps({k: round(v, 4) for k, v in report(None, None).items()}), flush=True)
    if DEBUG and os.environ.get("CLP_TRACE"):
        s_t = os.environ["CLP_TRACE"]
        lo, hi = (int(v) for v in os.environ.get("CLP_TRACE_STEPS", "0-100000").split("-"))
        if lo <= S["step"] <= hi:
            pd, Rd, wd = _driver_pose(s_t)
            vd = _np(_SUB["drivers"][s_t].data.root_lin_vel_w)[0]
            qd = _np(art.data.joint_vel)[0]
            jt, st_t = _SUB["jid"][s_t], S["state"][s_t]
            print(f"[clp-trace] step {S['step']} drv {np.round(pd * 1000, 1).tolist()} v {np.round(vd * 1000, 0).tolist()} "
                  f"w {np.round(wd, 2).tolist()} axis {np.round(Rd[:, 2], 3).tolist()} "
                  f"q {[round(float(q[jt[k]]) * 1000, 2) for k in ('px', 'py', 'pz')]} spin_v {float(qd[jt['spin']]):.1f} "
                  f"push {st_t['push']:.1f} mot {st_t['motor']} deliv {st_t['delivered']:.2f} F {st_t['thread_f']:.0f} "
                  f"lat {st_t['bit_lat'] * 1000:.2f} ax {st_t['bit_ax'] * 1000:.2f} "
                  f"grip {_force(V[f'driver_{s_t}_{s_t}_' + ('right' if s_t == 'a' else 'left')], dt):.1f} "
                  f"drv-housing {_force(V[f'drvhous_{s_t}'], dt):.1f}", flush=True)
    if DEBUG and S["step"] % 10 == 0:
        print(f"[clp] step {S['step']} tier {tier} drop {S['dropped']} held {held} hz {p_h[2] - CEIL:+.4f} "
              f"hx {p_h[0]:.4f} hy {p_h[1]:+.4f} "
              + " ".join(f"{s}: bit {st['bitten']} mag {st['magnet']} push {st['push']:.1f} pz {q[_SUB['jid'][s]['pz']] * 1000:+.1f} "
                         f"pxy ({q[_SUB['jid'][s]['px']] * 1000:+.1f},{q[_SUB['jid'][s]['py']] * 1000:+.1f}) "
                         f"lat {st['bit_lat'] * 1000:.1f} ax {st['bit_ax'] * 1000:+.1f} tilt {st['tilt']:.1f} "
                         f"mot {st['motor']} slip {st['slipping']} F {st['thread_f']:.0f} clamp {st['clamp']:.0f} "
                         f"gap {st['gap'] * 1000:+.2f} edge {st['edge_gap'] * 1000:+.2f} fast {st['fastened']} "
                         f"tight {st['tight']} wdrv {st['w_driver']:+.1f}"
                         for s, st in S["state"].items()) + f" hands {{{', '.join(f'{k}: {v:.1f}' for k, v in S['hand_f'].items())}}} err {_SUB.get('err')}", flush=True)


def _peaks(env) -> None:
    """Per-arm peaks: joint torques (applied by the arm's drives) and hold forces."""
    scene = env.unwrapped.scene
    for st in "ab":
        for side in ("left", "right"):
            robot = scene[f"{st}_{side}_robot"]
            tau = getattr(robot.data, "applied_torque", None)
            if tau is None:
                continue
            t = np.abs(_np(tau)[0][:6])
            key = f"torque_{st}_{side}"
            prev = _S["peak"].get(key, np.zeros(6))
            _S["peak"][key] = np.maximum(prev, t)


def _state():
    return _S


def report(objects, pair) -> dict:
    S = _S
    out = {"tier": S.get("tier", 0.0), "dropped": float(S.get("dropped", False)), "hangs": float(S.get("hangs", False))}
    for s, st in S.get("state", {}).items():
        for k in ("bitten", "fastened", "seated", "flush", "tight", "slipping"):
            out[f"{s}_{k}"] = float(st[k])
        for k in ("gap", "edge_gap", "clamp", "push", "thread_f", "turn"):
            out[f"{s}_{k}"] = float(st[k])
    for k, v in S.get("peak", {}).items():
        if isinstance(v, np.ndarray):
            for i, t in enumerate(v):
                out[f"peak_{k}_j{i + 1}"] = float(t)
        else:
            out[f"peak_{k}"] = float(v)
    out["cross_contact_peak_n"] = float(getattr(pair, "contact_peak_n", 0.0))
    return out


def success(objects, pair) -> bool:
    """Never: every trial runs to its step limit and is scored on its end state."""
    return False


def score(objects, pair) -> float:
    """1.0: both ends fastened (screw up through the narrow part of its slot into its pilot hole, biting) and tight
    (seated: head within 0.3 mm of the housing pressing 50 N; flush: the nearest long edge of the back, in line with
    the screw, within 2 mm of the ceiling), and it hangs on its own (no gripper contact for the last 3 s). 0.75: both
    fastened, hangs on its own, one end tight. 0.5: both fastened, hangs on its own, neither tight. 0.25: at least one
    end fastened and not dropped, otherwise less. 0: dropped at any time (a corner moving down at 0.3 m/s or faster,
    more than 25 mm below flush), or no end fastened."""
    return float(_S.get("tier", 0.0))


def demo(layout):
    from scripted import station_script

    return {"a": station_script("a", VARIANT), "b": station_script("b", VARIANT)}
