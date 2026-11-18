"""On the station table, in front of the left arm: an Applied Biosystems GeneAmp PCR System 9700 thermal cycler with its
96-well sample block, front toward the arms, powered off. Its heated cover is slid back and its lever is up, so the
block is open. To the robot's left of it, a MicroAmp Optical 96-Well Reaction Plate, capped with a full plate cover,
sits in its black Splash-Free Support Base with well A1 at the back left.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
         --scorer task_score --max-steps 3600 ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scenekit import UsdBody, before_reset, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Load the capped PCR plate onto the thermal cycler's block with well A1 at the back left, then close the "
               "heated cover.")

TOP = g.TOP

# -- objects -----------------------------------------------------------------------------------------------------------
CYCLER = UsdBody("cycler", usd="assets/cycler.usda", pos=(g.FRONT_X, g.CENTRE_Y, TOP), size=(g.DEPTH, g.WIDTH, g.HEIGHT),
                 articulation=True, joint_pos=(("slide", 0.0), ("platen", 0.0), ("lever", 0.0)),
                 drives=(("slide", 0.0, 8.0), ("platen", g.PLATEN_K, 400.0, g.PLATEN_MAX),
                         ("lever", g.LEVER_K, g.LEVER_DAMP)),
                 friction=(0.4, 0.3), solver_iters=32)
BLOCK_POS = (g.FRONT_X + g.BLOCK_U, g.CENTRE_Y, TOP + g.BLOCK_TOP)
BLOCK = UsdBody("block", usd="assets/block.usda", pos=BLOCK_POS, size=(g.BLOCK_SIZE[0], g.BLOCK_SIZE[1], 0.02),
                friction=(0.3, 0.25))
SUPPORT_POS = (g.SUPPORT_X, g.SUPPORT_Y, TOP + g.SUPPORT_TOP)
SUPPORT = UsdBody("support", usd="assets/support.usda", pos=SUPPORT_POS, size=(0.098, 0.14, g.SUPPORT_TOP),
                  friction=(0.3, 0.25))
PLATE_START = (g.SUPPORT_X, g.SUPPORT_Y, TOP + g.SUPPORT_TOP + g.SEAT + 0.0003)
PLATE = UsdBody("plate", usd="assets/plate.usda", pos=PLATE_START, size=(g.PLATE_X, g.PLATE_Y, 0.021),
                mass=g.PLATE_MASS, friction=(0.5, 0.4), solver_iters=32)
OBJECTS = [CYCLER, BLOCK, SUPPORT, PLATE]

# -- scoring geometry (station frame) ------------------------------------------------------------------------------------
BLOCK_XY = np.array(BLOCK_POS[:2])
BLOCK_HALF = np.array(g.BLOCK_SIZE) / 2
SUPPORT_XY = np.array(SUPPORT_POS[:2])
SUPPORT_HALF = np.array(g.SUPPORT_TOP_SIZE) / 2
BACK_LEFT_WELL = BLOCK_XY + g.well_xy(0, 0)              # the block's A1 well (back left)
FRONT_RIGHT_WELL = BLOCK_XY + g.well_xy(g.ROWS - 1, g.COLS - 1)
A1_LOCAL = np.array([g.A1_X, g.A1_Y, 0.0])
SEATED_Z = TOP + g.SEATED_DECK_H
# the plate's footprint (deck rectangle) sampled, plate frame
_fx, _fy = np.meshgrid(np.linspace(-g.PLATE_X / 2, g.PLATE_X / 2, 9), np.linspace(-g.PLATE_Y / 2, g.PLATE_Y / 2, 13))
FOOTPRINT = np.c_[_fx.ravel(), _fy.ravel(), np.zeros(_fx.size)]
# the plate as one box for the hands-off check (plate frame): caps' top to the wells' tips
PLATE_BOX = (np.array([0.0, 0.0, (g.CAP_H - g.WELL_TIP) / 2]), np.array([g.PLATE_X, g.PLATE_Y, g.CAP_H + g.WELL_TIP]))
HOLD_STEPS = 20                     # the final 1 s at 20 Hz
HAND_LINKS = ("gripper", "tip_left", "tip_right")

_STATE: dict = {}
_VIEWS: dict = {}
_DEBUG = bool(__import__("os").environ.get("PCR_DEBUG"))


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


# -- the cycler's mechanism: end stop, interlock, latch, cam -------------------------------------------------------------
def _asset(environment, name):
    return environment.unwrapped.scene[name]


def _limits(environment, joint: str, lo: float, hi: float) -> None:
    import torch

    asset = _asset(environment, "cycler")
    ids = asset.find_joints([joint])[0]
    asset.write_joint_position_limit_to_sim_index(
        limits=torch.tensor([[[lo, hi]]], dtype=torch.float32, device=asset.device), joint_ids=ids,
        warn_limit_violation=False)


def _mechanism(environment, slide: float, lever: float) -> None:
    """The cover's slide runs to its end stop (the joint's limit). The lever is blocked (its joint held at up) until
    the cover is within 1 mm of the stop; once the lever is down from up, the cover can't slide. Until it latches the
    lever's drive pulls it back toward up (a spring: the cam's resistance, about 1.4 N m at the stop), so a lever let
    go of part-way returns up; within 3 degrees of its stop it goes over centre and latches there (held within 0.5
    degrees, the drive targeting the stop). The lever's cam lowers the platen (its drive target)."""
    from scenekit import set_joint_targets

    S = _STATE
    at_stop = slide <= -g.COVER_TRAVEL + g.LEVER_UNLOCK
    if not S["lever_free"] and at_stop:
        _limits(environment, "lever", g.LEVER_DOWN, 0.0)
        S["lever_free"] = True
    elif S["lever_free"] and not at_stop and lever > -math.radians(0.5):
        _limits(environment, "lever", -0.005, 0.0)
        S["lever_free"] = False
    lever_down = lever < -math.radians(1.0)
    if lever_down and not S["slide_locked"]:
        _limits(environment, "slide", slide - 0.0002, slide + 0.0002)
        S["slide_locked"] = True
    elif not lever_down and S["slide_locked"]:
        _limits(environment, "slide", -g.COVER_TRAVEL, 0.0)
        S["slide_locked"] = False
    if not S["latched"] and lever <= g.LEVER_DOWN + g.LEVER_LATCH:
        _limits(environment, "lever", g.LEVER_DOWN, g.LEVER_DOWN + g.LEVER_CATCH)
        S["latched"] = True
    set_joint_targets(environment, "cycler", {"platen": -g.platen_drop(lever),
                                              "lever": g.LEVER_DOWN if S["latched"] else g.LEVER_PRELOAD})


@before_reset
def _restart(environment) -> None:
    """Cover back, lever up and blocked, platen raised (the reset moves articulations only by their roots)."""
    from scenekit import set_joint_targets, write_joint_positions

    _limits(environment, "slide", -g.COVER_TRAVEL, 0.0)
    _limits(environment, "lever", -0.005, 0.0)
    write_joint_positions(environment, "cycler", {"slide": 0.0, "platen": 0.0, "lever": 0.0})
    set_joint_targets(environment, "cycler", {"platen": 0.0, "lever": g.LEVER_PRELOAD})


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(lever_free=False, slide_locked=False, latched=False, why=[], zero=False, held={}, pads={},
                  ref_z=None, rest=0, cover_moved=False, lever_moved=False, last=None, rungs=[], rung=0.0,
                  corners=np.zeros(4), engaged=False, oriented=False, turned=False, on_block=False, on_support=False,
                  centre_on_block=False, moving_max=-1.0, close_ref=None,
                  cover_fwd=False, lever_down=False, hand_gap=1.0, slide=0.0, lever=0.0, platen=0.0, step=0)
    _VIEWS.clear()


_reset_state()


def _fail(why: str) -> None:
    if why not in _STATE["why"]:
        _STATE["why"].append(why)
    _STATE["zero"] = True


def _views(environment) -> dict:
    """PhysX contact views: the plate against each of the four finger pads."""
    if not _VIEWS:
        from yam_arena import constants as C
        from yam_arena.robot import asset_link_paths

        psv = environment.unwrapped.sim.physics_sim_view
        ns = "/World/envs/env_*"
        lp = asset_link_paths()
        pads = [f"{ns}/{arm}{lp[tip]}" for arm in ("LeftArm", "RightArm") for tip in ("tip_left", "tip_right")]
        _VIEWS["plate"] = psv.create_rigid_contact_view(f"{ns}/plate", filter_patterns=pads)
        _VIEWS["dt"] = float(C.PHYSICS_DT)
    return _VIEWS


def _holders(environment) -> dict:
    """Each hand holding the plate: both its pads pressing on it with HOLD_F or more."""
    V = _views(environment)
    f = np.linalg.norm(_np(V["plate"].get_contact_force_matrix(V["dt"]))[0], axis=-1)    # (4 pads)
    return {side: bool(f[2 * k:2 * k + 2].min() >= g.HOLD_F) for k, side in enumerate(("left", "right"))}, f


def _overlaps(points_xy: np.ndarray, centre: np.ndarray, half: np.ndarray) -> bool:
    d = np.abs(points_xy - centre)
    return bool(np.any(np.all(d <= half, axis=1)))


def cover_parts(slide: float, lever: float) -> list:
    """The cover and the lever as boxes (rotation, centre, size), station frame, for the hands-off check."""
    o = np.array([g.FRONT_X, g.CENTRE_Y, TOP])
    f = g.cover_front_u(slide)
    parts = [(np.eye(3), o + [f + g.COVER_L / 2, 0, (g.COVER_UNDER + g.COVER_TOP) / 2],
              np.array([g.COVER_L, g.COVER_W, g.COVER_TOP - g.COVER_UNDER]))]
    c, s = math.cos(lever), math.sin(lever)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pivot = o + [f, 0, g.COVER_TOP]
    parts.append((Ry, pivot + Ry @ [0, 0, g.LEVER_R], np.array([2 * g.LEVER_BAR_R, g.LEVER_BAR_L, 2 * g.LEVER_BAR_R])))
    for sv in (-1, 1):
        parts.append((Ry, pivot + Ry @ [0, sv * g.LEVER_ARM_V, (g.LEVER_R + 0.008) / 2],
                      np.array([0.010, 0.008, g.LEVER_R + 0.008])))
    return parts


def _edge_points(size, step: float = 0.005) -> np.ndarray:
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
    """The smallest gap from either gripper (its body and fingers' collision hulls) to these boxes."""
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
        lo, hi = edges.min(0) - 0.03, edges.max(0) + 0.03
        for h in hulls:
            if np.all(h.vertices.max(0) >= lo) and np.all(h.vertices.min(0) <= hi):
                p = h.planes
                best = min(best, float((edges @ p[:, :3].T - p[:, 3]).max(axis=1).min()))
    return best


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    S["step"] += 1
    st = object_states(environment, ["plate", "cycler"])
    p, c = st["plate"], st["cycler"]
    slide, lever = float(c.joints["slide"]), float(c.joints["lever"])
    S.update(slide=slide, lever=lever, platen=float(c.joints["platen"]))
    _mechanism(environment, slide, lever)
    if _DEBUG:
        lim = _np(_asset(environment, "cycler").data.joint_pos_limits)[0]
        names = _asset(environment, "cycler").joint_names
        S["limits"] = {n: tuple(np.round(lim[i], 4).tolist()) for i, n in enumerate(names)}
    o, R = np.asarray(p.pos, float), _rotation(p.quat_xyzw)
    held, pads = _holders(environment)
    S["held"], S["pads"] = held, pads
    any_held = any(held.values())
    corners = o + g.DECK_CORNERS @ R.T
    S["corners"] = corners[:, 2] - SEATED_Z
    foot = (o + FOOTPRINT @ R.T)[:, :2]
    S["on_block"] = _overlaps(foot, BLOCK_XY, BLOCK_HALF)            # its footprint overlaps the block's top (zero 2)
    # set down on the block or the base: the footprint's centre over the block's top or the base's top (zero 1)
    S["centre_on_block"] = _overlaps(o[None, :2], BLOCK_XY, BLOCK_HALF)
    S["on_support"] = _overlaps(o[None, :2], SUPPORT_XY, SUPPORT_HALF)
    resting = bool(np.linalg.norm(p.lin_vel) < g.REST_SPEED and np.linalg.norm(p.ang_vel) < g.REST_SPIN)
    S["rest"] = S["rest"] + 1 if (resting and not any_held) else 0
    # zero 1: dropped (a free fall of more than 1 cm), upside down, or set down anywhere but the base or the block
    if any_held:
        S["ref_z"] = None
    else:
        S["ref_z"] = o[2] if S["ref_z"] is None else max(S["ref_z"], o[2])
        if S["ref_z"] - o[2] > g.DROP:
            _fail("dropped")
    if S["rest"] >= 1 and R[2, 2] < 0.0:
        _fail("upside down")
    if S["rest"] >= g.REST_STEPS and not (S["centre_on_block"] or S["on_support"]):
        _fail("set down elsewhere")
    # zero 2: on any step the cover slides forward or the lever is pulled down, the plate overlapping the block but
    # not engaged (a deck corner HIGH or more above seated)
    # (moving forward/down: more than MOVED_STEP past a reference that follows each such move, and any move back, so a
    # slow push is caught too)
    high = bool(S["on_block"] and S["corners"].max() >= g.HIGH)
    ref = S["close_ref"] or (slide, lever)
    fwd, pulled = ref[0] - slide > g.MOVED_STEP[0], ref[1] - lever > g.MOVED_STEP[1]
    S["close_ref"] = (slide if (fwd or slide > ref[0]) else ref[0], lever if (pulled or lever > ref[1]) else ref[1])
    if (fwd or pulled) and S["on_block"]:              # for the log: the highest corner (mm) while closing
        S["moving_max"] = max(S["moving_max"], round(float(S["corners"].max()) * 1000, 2))
    if fwd and high:
        _fail("cover slid forward, plate not engaged")
    if pulled and high:
        _fail("lever pulled, plate not engaged")
    if not S["cover_moved"] and slide <= -g.COVER_MOVED:
        S["cover_moved"] = True
        S["cover_check"] = (round(float(S["corners"].max()) * 1000, 2), S["on_block"])
    if not S["lever_moved"] and lever <= -g.LEVER_MOVED:
        S["lever_moved"] = True
        S["lever_check"] = (round(float(S["corners"].max()) * 1000, 2), S["on_block"])
    # zero 3: the cover or the lever moving while a hand holds the plate
    if S["last"] is not None and any_held:
        ds, dl = abs(slide - S["last"][0]), abs(lever - S["last"][1])
        if ds > g.MOVED_STEP[0] or dl > g.MOVED_STEP[1]:
            _fail("cover/lever moved while the plate was held")
    S["last"] = (slide, lever)
    # the end-state ladder, this step
    a1 = (o + R @ A1_LOCAL)[:2]
    S["engaged"] = bool(np.all(np.abs(S["corners"]) <= g.ENGAGED_TOL))
    S["oriented"] = bool(np.linalg.norm(a1 - BACK_LEFT_WELL) <= g.ORIENT_TOL)
    S["turned"] = bool(np.linalg.norm(a1 - FRONT_RIGHT_WELL) <= g.ORIENT_TOL)
    S["cover_fwd"] = bool(slide <= -g.COVER_TRAVEL + g.COVER_STOP)
    S["lever_down"] = bool(lever <= g.LEVER_DOWN + g.LEVER_STOP)
    rung = 0.0
    if S["engaged"] and S["turned"]:
        rung = 0.25
    elif S["engaged"] and S["oriented"]:
        if not S["cover_fwd"]:
            rung = 0.5
        elif not S["lever_down"]:
            rung = 0.75
        else:
            boxes = [(R, o + R @ PLATE_BOX[0], PLATE_BOX[1])] + cover_parts(slide, lever)
            S["hand_gap"] = _hand_gap(environment, boxes)
            rung = 1.0 if S["hand_gap"] >= g.HANDS_OFF else 0.75
    S["rung"] = rung
    S["rungs"] = (S["rungs"] + [rung])[-HOLD_STEPS:]


def success(objects) -> bool:
    """The full-score state (engaged, A1 at the back left, cover fully forward, lever down, both grippers 1 cm or more
    from plate, cover and lever) with no zero condition; yam_arena ends the trial once it has held for 1 s."""
    S = _STATE
    return bool(not S["zero"] and S["rung"] == 1.0)


def score(objects) -> float:
    """0 for the run (latched) if, at any control step: the plate fell more than 1 cm in free fall, came to rest upside
    down, or was set down anywhere but its support base or the block (came to rest with its footprint's centre over
    neither one's top); the cover slid forward or the lever was pulled down, at any step, while the plate's footprint
    overlapped the block's top with any deck corner 5 mm or more above its seated height; or the cover or the lever
    moved while a hand held the plate (both its pads pressing on it).
    Otherwise, from the final 1 s (the lowest rung any of its steps reached):
    - 0.25: engaged (all four deck corners within 1 mm of seated) but turned 180 degrees (A1 in the front-right well);
    - 0.5: engaged with A1 in the block's back-left well, the cover not fully forward (more than 1 mm from its stop);
    - 0.75: the same, cover fully forward, lever not down (more than 2 degrees from its stop), or both down with a
      gripper within 1 cm of plate, cover or lever;
    - 1.0: the same, lever down, both grippers 1 cm or more from plate, cover and lever;
    - 0 for every other end state (on the ridges, outside the block, untouched)."""
    S = _STATE
    if S["zero"] or not S["rungs"]:
        return 0.0
    return float(min(S["rungs"]))
