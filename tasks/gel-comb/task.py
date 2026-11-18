"""On the station table: a 1% agarose gel set in a Bio-Rad 7 x 7 cm UV-transparent tray, with an 8-well, 1.5 mm comb
still standing in it (run axis along x, the comb at the end toward the robot); behind it an empty, dry Bio-Rad Mini-Sub
Cell GT base (long axis along y, black cathode at +y, red anode at -y); to the robot's left a folded lab tissue for the
comb.

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

from scenekit import Shape, UsdBody, before_reset, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Take the comb out of this gel without damaging the wells, then put the gel on its tray into the gel box, "
               "ready to run DNA.")

TOP = g.TOP
WHITE = (0.95, 0.95, 0.94)

# -- objects -------------------------------------------------------------------------------------------------------------
TRAY_POS = np.array([g.TRAY_X, g.TRAY_Y, TOP + 0.0002])
TANK = UsdBody("tank", usd="assets/tank.usda", pos=(g.TANK_X, g.TANK_Y, TOP), size=(g.BASE_W, g.BASE_L, g.BASE_H),
               friction=(0.5, 0.45))
GUARDS = UsdBody("guards", usd="assets/guards.usda", pos=(g.TANK_X, g.TANK_Y, TOP), size=(g.TANK_INNER_W, 0.25, 0.02),
                 friction=(0.5, 0.45))
TRAY = UsdBody("tray", usd="assets/tray.usda", pos=tuple(TRAY_POS), size=(g.TRAY_L, g.TRAY_W, g.WALL_H),
               articulation=True, joint_pos=(("hold", 0.0),), drives=(("hold", g.CATCH_K, 10.0, 1.1 * g.F_REL),),
               friction=(0.6, 0.5), solver_iters=32, torsional_patch=0.006)
GEL = UsdBody("gel", usd="assets/gel.usda", pos=tuple(TRAY_POS + [0, 0, g.GEL_Z0 + 0.0001]),
              size=(g.GEL_L, g.GEL_W, g.GEL_T), mass=g.GEL_MASS, friction=(g.GEL_FRICTION, g.GEL_FRICTION),
              friction_min=True, solver_iters=32)
COMB = UsdBody("comb", usd="assets/comb.usda", pos=tuple(TRAY_POS + g.COMB_START + [0, 0, 0.0001]),
               size=(g.BAR_T, g.BAR_L, g.BAR_H), mass=g.COMB_MASS, friction=(0.8, 0.7), solver_iters=32,
               torsional_patch=0.004)
TISSUE = Shape("tissue", pos=(g.TISSUE_X, g.TISSUE_Y, TOP + g.TISSUE_SIZE[2] / 2), size=g.TISSUE_SIZE, color=WHITE)
OBJECTS = [TANK, GUARDS, TRAY, GEL, COMB, TISSUE]


# -- scoring geometry -------------------------------------------------------------------------------------------------------
HOLD_STEPS = 20                       # the final 1 s at 20 Hz
DT = 0.05
HAND_LINKS = ("gripper", "tip_left", "tip_right")
HOLD_F = 0.3                          # N: a pad pressing on the tray (or its gel) counts toward holding it
REST_STEPS = 10
STAGE_Z = TOP + g.STAGE_TOP
BLACK_Y = g.TANK_Y + g.PLUG_Y          # the black (cathode) plug's end; red at -y
RAMP_STEPS = max(1, round(g.RELEASE_S / DT))
TIP_LOCAL = np.array([[0.0, y, -g.TOOTH_L] for y in g.WELL_Y])     # tooth tips in the comb frame
ROOT_LOCAL = np.array([[0.0, y, 0.0] for y in g.WELL_Y])           # where each tooth meets the bar
CAST_TIP_Z = g.GEL_T - g.CAST_DEPTH   # a tooth tip's cast height in the gel frame (1 mm)
GEL_SURF = g.GEL_T                    # the gel's top surface in the gel frame

_STATE: dict = {}
_VIEWS: dict = {}
_DEBUG = bool(os.environ.get("GEL_DEBUG"))


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


# -- the comb's hold in the set gel --------------------------------------------------------------------------------------
def _tray(environment):
    return environment.unwrapped.scene["tray"]


def _hold_ids(environment):
    return _tray(environment).find_joints(["hold"])[0]


def _set_effort(environment, limit: float) -> None:
    _tray(environment).write_joint_effort_limit_to_sim_index(limits=float(limit), joint_ids=_hold_ids(environment))


def _park(environment) -> None:
    """The hold has let go: the catch drops inside the side wall's solid lower part, where nothing can touch it."""
    from scenekit import set_joint_targets, write_joint_positions

    _set_effort(environment, 5.0)
    write_joint_positions(environment, "tray", {"hold": g.CATCH_PARK})
    set_joint_targets(environment, "tray", {"hold": g.CATCH_PARK})


def _slick_catch(environment) -> None:
    """The catch's two shapes (the tray articulation's last two: links in order tray, catch) get zero friction, so the
    comb slides freely under it and the hold resists only the pull. (Isaac Lab binds the tray's material over all its
    colliders.)"""
    import torch
    import warp as wp

    view = _tray(environment).root_view
    m = wp.to_torch(view.get_material_properties()).clone()       # (1, shapes, 3) on the CPU
    m[0, -2:, 0:2] = 0.0
    view.set_material_properties(wp.from_torch(m, dtype=wp.float32),
                                 wp.from_torch(torch.tensor([0], dtype=torch.int32), dtype=wp.int32))


@before_reset
def _restart(environment) -> None:
    from scenekit import set_joint_targets, write_joint_positions

    _slick_catch(environment)
    _set_effort(environment, 1.1 * g.F_REL)
    write_joint_positions(environment, "tray", {"hold": 0.0})
    set_joint_targets(environment, "tray", {"hold": 0.0})


def _hold_step(environment, q_hold: float) -> None:
    """The comb is held in the set gel until the upward pull on it (read off the catch's stiff slide) reaches F_REL for
    two control steps; then the hold fades to nothing over RELEASE_S (the wells' seal letting go) and the catch parks.
    An unrestrained tray (lighter than F_REL with its gel) rises with the comb instead."""
    S = _STATE
    S["hold_f"] = g.CATCH_K * q_hold
    if S["hold"] == "on":
        S["hold_peak"] = max(S["hold_peak"], min(S["hold_f"], 1.1 * g.F_REL))
    if S["hold"] == "on":
        S["over"] = S["over"] + 1 if S["hold_f"] >= g.F_REL else 0
        if S["over"] >= 2:
            S["hold"], S["rel_step"] = "releasing", S["step"]
            S["rel_force"] = min(S["hold_f"], 1.1 * g.F_REL)
    if S["hold"] == "releasing":
        k = S["step"] - S["rel_step"]
        if k >= RAMP_STEPS:
            _park(environment)
            S["hold"] = "off"
        else:
            _set_effort(environment, 1.1 * g.F_REL * (1.0 - k / RAMP_STEPS))


# -- contact views ----------------------------------------------------------------------------------------------------------
def _views(environment) -> dict:
    if not _VIEWS:
        from yam_arena import constants as C
        from yam_arena.robot import asset_link_paths

        psv = environment.unwrapped.sim.physics_sim_view
        ns = "/World/envs/env_*"
        lp = asset_link_paths()
        pads = [f"{ns}/{arm}{lp[tip]}" for arm in ("LeftArm", "RightArm") for tip in ("tip_left", "tip_right")]
        links = [f"{ns}/{arm}{lp[n]}" for arm in ("LeftArm", "RightArm") for n in lp]
        _VIEWS["gel"] = psv.create_rigid_contact_view(f"{ns}/gel", filter_patterns=pads + [f"{ns}/comb"])
        _VIEWS["tray"] = psv.create_rigid_contact_view(f"{ns}/tray/base", filter_patterns=pads)
        _VIEWS["comb"] = psv.create_rigid_contact_view(f"{ns}/comb", filter_patterns=[f"{ns}/gel", f"{ns}/tray/base",
                                                                                         f"{ns}/tank"])
        _VIEWS["guards"] = psv.create_rigid_contact_view(f"{ns}/guards", filter_patterns=links + [
            f"{ns}/tray/base", f"{ns}/gel", f"{ns}/comb"])
        _VIEWS["dt"] = float(C.PHYSICS_DT)
    return _VIEWS


def _forces(environment):
    V = _views(environment)
    dt = V["dt"]
    return {k: _np(V[k].get_contact_force_matrix(dt))[0] for k in ("gel", "tray", "comb", "guards")}


# -- hands ------------------------------------------------------------------------------------------------------------------
def _edge_points(size, step: float = 0.004) -> np.ndarray:
    half, pts = np.asarray(size) / 2, []
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        t = np.linspace(-half[a], half[a], max(2, int(np.ceil(size[a] / step)) + 1))
        for sb in (-1.0, 1.0):
            for sc in (-1.0, 1.0):
                p = np.zeros((len(t), 3))
                p[:, a], p[:, b], p[:, c] = t, sb * half[b], sc * half[c]
                pts.append(p)
    return np.concatenate(pts)


def _hand_gap(environment, boxes) -> float:
    """The smallest gap from either gripper (its body and fingers' collision hulls) to these boxes (R, centre, size)."""
    from yam_arena.analysis import box_signed_distance, world_hulls

    from scenekit import robot_link_poses

    hulls = []
    for poses in robot_link_poses(environment).values():
        hulls.extend(h for n, h in world_hulls(poses) if n in HAND_LINKS)
    best = 1.0
    verts = np.concatenate([h.vertices for h in hulls])
    for R, centre, size in boxes:
        best = min(best, float(box_signed_distance(verts, centre, R, size).min()))
        edges = centre + _edge_points(size) @ R.T
        lo, hi = edges.min(0) - 0.04, edges.max(0) + 0.04
        for h in hulls:
            if np.all(h.vertices.max(0) >= lo) and np.all(h.vertices.min(0) <= hi):
                p = h.planes
                best = min(best, float((edges @ p[:, :3].T - p[:, 3]).max(axis=1).min()))
    return best


# -- state ------------------------------------------------------------------------------------------------------------------
@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, why=[], zero=False, hold="on", over=0, rel_step=None, rel_force=None, hold_f=0.0,
                  hold_peak=0.0, torn=[False] * g.N_WELLS, torn_why=[None] * g.N_WELLS, tip_prev=None, in_prev=None,
                  tray_z0=None, ref_z=None, rest=0, held=False, comb_out=False, seated=False, black=False,
                  slide=0.0, rung=0.0, rungs=[], hand_gap=1.0, max_tilt_t=0.0, max_tilt_e=0.0, max_off=0.0,
                  max_up=0.0, press=np.zeros(4), max_top=0.0, max_end=0.0, max_lift=0.0, in_gel=0, corners=np.zeros(4), last=None)
    _VIEWS.clear()


_reset_state()


def _fail(why: str) -> None:
    if why not in _STATE["why"]:
        _STATE["why"].append(why)
    _STATE["zero"] = True


def _pose(st):
    return np.asarray(st.pos, float), _rotation(st.quat_xyzw)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    S["step"] += 1
    st = object_states(environment, ["tray", "gel", "comb"])
    to, tR = _pose(st["tray"])
    go, gR = _pose(st["gel"])
    co, cR = _pose(st["comb"])
    _hold_step(environment, float(st["tray"].joints["hold"]))
    F = _forces(environment)
    if S["tray_z0"] is None:
        S["tray_z0"] = to[2]
    # -- the comb in the gel (gel frame) --
    rel = gR.T @ cR                                         # the comb's rotation in the gel frame
    tips = (co + TIP_LOCAL @ cR.T - go) @ gR                # tooth tips, gel frame
    roots = (co + ROOT_LOCAL @ cR.T - go) @ gR
    in_gel = ((tips[:, 2] < GEL_SURF) & (np.abs(tips[:, 0] - g.WELL_X) < 0.005)     # each tooth in its own well
              & (np.abs(tips[:, 1] - g.WELL_Y) < 0.006))
    S["in_gel"] = int(in_gel.sum())
    tilt_t = abs(math.atan2(rel[0, 2], rel[2, 2]))          # about the comb's long axis (y)
    tilt_e = abs(math.atan2(rel[1, 2], rel[2, 2]))          # end-first, in the comb's plane (about x)
    prev_in = S["in_prev"] if S["in_prev"] is not None else np.zeros(g.N_WELLS, bool)
    for i in range(g.N_WELLS):
        if S["torn"][i] or not (in_gel[i] or prev_in[i]):
            continue
        why = None
        a, b = tips[i], roots[i]
        if in_gel[i]:
            # the tooth's centre line from its tip up to the gel surface (or its root): its offset across its thickness
            s = 1.0 if b[2] <= GEL_SURF else (GEL_SURF - a[2]) / max(b[2] - a[2], 1e-9)
            top = a + min(max(s, 0.0), 1.0) * (b - a)
            off = max(abs(a[0] - g.WELL_X), abs(top[0] - g.WELL_X))
            S["max_off"] = max(S["max_off"], off)
            if off > g.D_TEAR:
                why = f"offset {off * 1000:.2f} mm"
            elif tilt_t > g.THETA_T:
                why = f"tilt {math.degrees(tilt_t):.1f} deg"
            elif tilt_e > g.THETA_E:
                why = f"end tilt {math.degrees(tilt_e):.1f} deg"
            elif a[2] <= CAST_TIP_Z - g.PUSH_DOWN + 1e-4:
                why = "pushed down"
        if why is None and prev_in[i]:
            # its rise over this step, counted on the step it leaves the gel too (a yank out in one step)
            up = (a[2] - S["tip_prev"][i]) / DT
            S["max_up"] = max(S["max_up"], up)
            if up > g.V_TEAR:
                why = f"speed {up * 1000:.0f} mm/s"
        if why:
            S["torn"][i], S["torn_why"][i] = True, why
    if in_gel.any():
        S["max_tilt_t"] = max(S["max_tilt_t"], tilt_t)
        S["max_tilt_e"] = max(S["max_tilt_e"], tilt_e)
    S["tip_prev"], S["in_prev"] = tips[:, 2].copy(), in_gel.copy()
    n_torn = sum(S["torn"])
    if n_torn >= 2:
        _fail("two or more wells torn")
    # -- zero 1: the comb lifts the tray, or the gel off the tray floor, while a tooth is in its well --
    gel_in_tray = tR.T @ (go - to)
    if in_gel.any():
        lift = max(to[2] - S["tray_z0"], gel_in_tray[2] - g.GEL_Z0)
        S["max_lift"] = max(S["max_lift"], lift)
        if lift >= g.LIFT_MAX:
            _fail("comb lifted the tray or gel")
    # -- zero 2: the gel slides 10 mm or more along the tray, or leaves it --
    S["slide"] = abs(gel_in_tray[0])
    if S["slide"] >= g.SLIDE_ZERO:
        _fail("gel slid 10 mm or more")
    if abs(gel_in_tray[1]) > 0.004 or gel_in_tray[2] - g.GEL_Z0 > 0.006 or (tR[:, 2] @ gR[:, 2]) < math.cos(
            math.radians(20)):
        _fail("gel left the tray")
    # -- zero 3: dropped, tipped, set down elsewhere, guard touched --
    pads_tray = np.linalg.norm(F["tray"], axis=-1)              # (4 pads)
    pads_gel = np.linalg.norm(F["gel"][:4], axis=-1)
    press = pads_tray + pads_gel
    held = bool(press[0:2].min() >= HOLD_F or press[2:4].min() >= HOLD_F)
    S["held"], S["press"] = held, press
    if held:
        S["ref_z"] = None
    else:
        S["ref_z"] = to[2] if S["ref_z"] is None else max(S["ref_z"], to[2])
        if S["ref_z"] - to[2] >= g.DROP:
            _fail("tray dropped")
    if tR[2, 2] < math.cos(g.TIP_MAX):
        _fail("tray tipped past 30 deg")
    v = st["tray"]
    resting = bool(np.linalg.norm(v.lin_vel) < 0.01 and np.linalg.norm(v.ang_vel) < 0.2)
    S["rest"] = S["rest"] + 1 if (resting and not held) else 0
    if S["rest"] >= REST_STEPS:
        in_tank = abs(to[0] - g.TANK_X) <= g.BASE_W / 2 and abs(to[1] - g.TANK_Y) <= g.BASE_L / 2
        on_tissue = (abs(to[0] - g.TISSUE_X) <= g.TISSUE_SIZE[0] / 2 + g.TRAY_L / 2 and
                     abs(to[1] - g.TISSUE_Y) <= g.TISSUE_SIZE[1] / 2 + g.TRAY_W / 2 and to[2] > TOP + 0.001)
        on_bench = to[2] <= TOP + 0.003 and not on_tissue
        if not (in_tank or on_bench):
            _fail("tray set down elsewhere")
    if np.linalg.norm(F["guards"], axis=-1).max() > 0.0:
        _fail("electrode guard touched")
    # -- zero 5: dents (normal forces, gel frame): a pad or the comb on the top surface, a pad on an end face --
    fg = F["gel"] @ gR                                          # (5, 3): 4 pads, comb
    top = np.maximum(0.0, -fg[:, 2])
    end = np.abs(fg[:4, 0])
    S["max_top"] = max(S["max_top"], float(top.max()))
    S["max_end"] = max(S["max_end"], float(end.max()))
    if top.max() >= g.F_DENT:
        _fail("gel top dented")
    if end.max() >= g.F_END:
        _fail("gel end face crushed")
    # -- the end state, this step --
    clear_tips = (tips[:, 2] >= GEL_SURF + g.COMB_CLEAR) | (np.abs(tips[:, 0]) > g.GEL_L / 2) | (
        np.abs(tips[:, 1]) > g.GEL_W / 2)
    comb_touch = float(np.linalg.norm(F["comb"], axis=-1).max())
    S["comb_out"] = bool(clear_tips.all() and comb_touch < 1e-3)
    corners = to + g.tray_corners() @ tR.T
    S["corners"] = corners[:, 2] - STAGE_Z
    local = corners[:, :2] - [g.TANK_X, g.TANK_Y]
    footprint = bool(np.all(np.abs(local[:, 0]) <= g.TANK_INNER_W / 2) and np.all(np.abs(local[:, 1]) <= g.STAGE_L / 2))
    flat = bool(np.all(np.abs(S["corners"]) <= g.CORNER_TOL))
    square = abs(tR[1, 0]) >= math.cos(g.SQUARE_TOL)
    wells_end = to + tR @ [-g.TRAY_L / 2, 0.0, 0.0]
    S["black"] = bool(wells_end[1] > g.TANK_Y)
    on_stage = flat and footprint and square
    rung = 0.0
    if S["comb_out"]:
        rung = 0.25
        if on_stage:
            boxes = [(tR, to + tR @ [0, 0, g.WALL_H / 2], np.array([g.TRAY_L, g.TRAY_W, g.WALL_H])),
                     (cR, co + cR @ [0, 0, (g.BAR_H - g.TOOTH_L) / 2], np.array([g.BAR_T, g.BAR_L, g.BAR_H + g.TOOTH_L]))]
            S["hand_gap"] = _hand_gap(environment, boxes)
            S["seated"] = S["hand_gap"] >= g.HANDS_OFF
            if S["seated"]:
                defects = int(n_torn == 1) + int(S["slide"] >= g.SLIDE_DEFECT)
                rung = (1.0, 0.75, 0.5)[defects] if S["black"] else (0.5, 0.25, 0.25)[defects]
        else:
            S["seated"] = False
    S["rung"] = rung
    S["rungs"] = (S["rungs"] + [rung])[-HOLD_STEPS:]


def success(objects) -> bool:
    """The full-score state (comb out, tray seated flat on the stage with the wells at the black end, both grippers 3 cm
    or more away, no torn well, gel slid under 3 mm) with no zero condition; yam_arena ends the trial once it has held
    for 1 s."""
    S = _STATE
    return bool(not S["zero"] and S["rung"] == 1.0)


def score(objects) -> float:
    """0 for the run (latched) if, at any control step (20 Hz): while a tooth is in its well the comb lifts the tray, or
    the gel off the tray floor, by 3 mm or more; the gel slides 10 mm or more along the tray or leaves it; the tray falls
    3 mm or more in free fall, tips past 30 degrees, or comes to rest anywhere but the bench or the tank; anything touches
    an electrode guard or wire; two or more wells are torn; a pad or the comb presses the gel's top surface with F_DENT
    or more, or a pad presses a gel end face with F_END or more.
    Otherwise, from the final 1 s (the lowest rung any of its steps reached), with the comb out (all teeth 5 mm or more
    above the gel or off it, and the comb touching none of gel, tray and tank; else 0):
    - seated (every bottom corner within 1 mm of the stage, the footprint inside the stage outline, the run axis along
      the tank within 5 degrees, both grippers 3 cm or more from tray, gel and comb), wells at the black end: 1.0 with no
      defect, 0.75 with one, 0.5 with two (defects: one torn well; the gel slid 3-10 mm);
    - seated, wells at the red end: 0.5 with no defect, 0.25 with any;
    - not seated: 0.25."""
    S = _STATE
    if S["zero"] or not S["rungs"]:
        return 0.0
    return float(min(S["rungs"]))
