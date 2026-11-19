"""On the station table: a white 6-outlet power strip lies flat, sockets up, switched off. A GE 24-hour plug-in
mechanical timer (model 15119) is plugged into its third socket from the switch end, dial up, its top edge (the OUTLET
ON / TIMER ON slide, at TIMER ON) away from the robot; a table lamp beyond it is plugged into the timer. Every one of
the dial's 48 half-hour segments is up and the pointer shows 9:00 AM. A used-up ballpoint pen stands point down in a
mug at the robot's right. Program the timer for 6:00 to 11:00 PM with the pen and set it to 2:30 PM.

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

INSTRUCTION = ("It is 2:30 PM. Set the plug-in timer so the lamp comes on at 6:00 PM and goes off at 11:00 PM: with "
               "the pen, push down the ten segments from 6:00 PM to 10:30 PM and no others, and turn the dial "
               "clockwise until its pointer shows 2:30 PM.")

physics(g.PHYSICS_HZ)               # PhysX substeps per 20 Hz control step: 24 (small, light segments)

TOP = g.TOP
SEGS = tuple(f"seg_{i:02d}" for i in range(g.SEG_N))
UP_TARGET = 0.0001                  # an up segment's spring target: 0.1 mm past its upper stop (0.3 N preload)
DOWN_TARGET = -0.0030               # a latched segment's: 1 mm past its lower stop (3 N on the stop)
SWITCH_TARGETS = (g.SWITCH_STOPS[0] - 0.001, g.SWITCH_STOPS[1] + 0.001)
SWITCH_FLIP = (g.SWITCH_STOPS[0] + g.SWITCH_STOPS[1]) / 2


def _quat_xyzw(R) -> tuple:
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return tuple(float(v) for v in ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                                    (R[1, 0] - R[0, 1]) / (4 * w), w))


def _pen_start():
    tip, u = g.pen_rest_pose()
    x = np.cross([0.0, 1.0, 0.0], u) if abs(u[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], u)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(u, x), u])
    origin = tip - g.PEN_BALL_R * u + np.array([0.0, 0.0, 0.0004])
    return tuple(float(v) for v in origin), _quat_xyzw(R)


TIMER = UsdBody("timer", usd="assets/timer.usda", pos=(g.TIMER_X, g.TIMER_Y, TOP + 0.0003),
                size=(g.STRIP_W, g.STRIP_L, g.SEG_TOP_H), articulation=True,
                joint_pos=(("dial", 0.0), ("switch", 0.0), *((s, 0.0) for s in SEGS)),
                drives=(("seg_.*", g.SEG_K_UP, g.SEG_DAMP), ("dial", g.DIAL_K, g.DIAL_DAMP),
                        ("switch", g.SWITCH_K, 0.5)),
                friction=(0.5, 0.45), solver_iters=16)
_pp, _pq = _pen_start()
PEN = UsdBody("pen", usd="assets/pen.usda", pos=_pp, rot_xyzw=_pq, size=(0.008, 0.008, g.PEN_L), mass=g.PEN_MASS,
              friction=(0.6, 0.5), solver_iters=32, torsional_patch=0.003)
MUG = UsdBody("mug", usd="assets/mug.usda", pos=(g.MUG_X, g.MUG_Y, TOP + 0.0003), size=(0.08, 0.08, g.MUG_H),
              mass=g.MUG_MASS, friction=(0.5, 0.45))
LAMP = UsdBody("table_lamp", usd="assets/table_lamp.usda", pos=(g.LAMP_X, g.LAMP_Y, TOP), size=(0.19, 0.19, g.LAMP_H))
OBJECTS = [TIMER, PEN, MUG, LAMP]

HAND_LINKS = ("gripper", "tip_left", "tip_right")
_STATE: dict = {}
_VIEWS: dict = {}
_SUB: dict = {}
_DEBUG = bool(os.environ.get("PT_DEBUG"))


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


# -- the mechanism, run before every physics substep --------------------------------------------------------------------
def _install(environment) -> None:
    """Hook the scene's per-substep write: the segment detents, the dial's stick-slip drag and the switch's two stops
    are updated from the joints' positions before every physics substep."""
    base = environment.unwrapped
    scene = base.scene
    asset = scene["timer"]
    names = list(asset.joint_names)
    _SUB.update(asset=asset, i_dial=names.index("dial"), i_switch=names.index("switch"),
                i_segs=[names.index(s) for s in SEGS], n=len(names))
    if getattr(scene, "_timer_hook", False):
        return
    write = scene.write_data_to_sim

    def write_and_step(*a, **k):
        _substep()
        return write(*a, **k)

    scene.write_data_to_sim = write_and_step
    scene._timer_hook = True


def _substep() -> None:
    """Segments: an up one is held by a spring (3 N at 1 mm); pushed 1 mm it latches down for good (its target moves
    past the lower stop). The dial: a stiff spring to a rest angle that is dragged along once the spring's torque passes
    the drag (DIAL_DRAG), so it turns under a hand and stays where it is left. The switch: a spring to the stop on the
    side of the middle it is on."""
    import torch

    if not _SUB.get("on"):
        return
    asset = _SUB["asset"]
    q = _t(asset.data.joint_pos)[0]
    S = _SUB
    segq = q[S["i_segs"]]
    newly = (segq <= -g.SEG_LATCH) & ~S["latched"]
    if bool(newly.any()):
        S["latched"] = S["latched"] | newly
        S["latch_event"] = True
    dial = float(q[S["i_dial"]])
    rest = S["rest"]
    lim = g.DIAL_DRAG / g.DIAL_K
    err = math.remainder(dial - rest, 2 * math.pi)
    if abs(err) > lim:
        rest = math.remainder(dial - math.copysign(lim, err), 2 * math.pi)
        S["rest"] = rest
    sw = float(q[S["i_switch"]])
    if sw > SWITCH_FLIP:
        S["switch_on"] = False
    elif sw < SWITCH_FLIP:
        S["switch_on"] = True
    tgt = S["targets"]
    tgt[S["i_segs"]] = torch.where(S["latched"], torch.full_like(segq, DOWN_TARGET), torch.full_like(segq, UP_TARGET))
    tgt[S["i_dial"]] = dial + math.remainder(rest - dial, 2 * math.pi)
    tgt[S["i_switch"]] = SWITCH_TARGETS[0] if S["switch_on"] else SWITCH_TARGETS[1]
    asset.actuators.target_command.set_position_index(value=tgt.reshape(1, -1), joint_ids=None)


@before_reset
def _restart(environment) -> None:
    """Every segment up, the dial at 9:00 AM (angle 0), the switch at TIMER ON; the detents and the drag reset."""
    import torch

    from scenekit import write_joint_positions

    write_joint_positions(environment, "timer", {"dial": 0.0, "switch": 0.0, **{s: 0.0 for s in SEGS}})
    _install(environment)
    asset = _SUB["asset"]
    dev = asset.device
    _SUB["latched"] = torch.zeros(g.SEG_N, dtype=torch.bool, device=dev)
    _SUB["rest"] = 0.0
    _SUB["switch_on"] = True
    _SUB["latch_event"] = False
    _SUB["targets"] = torch.zeros(_SUB["n"], dtype=torch.float32, device=dev)
    _SUB["on"] = True
    _substep()


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, why=[], zero=False, touched=False, touch_who=None, pushed=False, push_who=None,
                  touch_start={}, contact_prev=set(), pulled=False, start_z=None, dial=0.0,
                  dial_prev=None, dial_min=0.0, ccw=False, down=[], rung=0.0, rungs=[], gap=1.0, pen_gap=1.0,
                  clear=False, switch_on=True, time=g.START_TIME, time_set=False, tier=0.0, presses=[],
                  nb_max=0.0, nb_who=None, strip_moved=0.0, strip_turn=0.0, start_pose=None, lamp=[])
    _VIEWS.clear()


_reset_state()


def _views(environment) -> dict:
    """PhysX contact views: the 48 segments against both hands' gripper bodies and fingers."""
    if not _VIEWS:
        from yam_arena.robot import asset_link_paths

        import yam_arena.constants as C

        psv = environment.unwrapped.sim.physics_sim_view
        ns = "/World/envs/env_*"
        lp = asset_link_paths()
        hands = [f"{ns}/{arm}{lp[link]}" for arm in ("LeftArm", "RightArm") for link in HAND_LINKS]
        _VIEWS["segs"] = psv.create_rigid_contact_view(f"{ns}/timer/seg_*", filter_patterns=hands)
        _VIEWS["hands"] = hands
        _VIEWS["dt"] = float(C.PHYSICS_DT)
    return _VIEWS


def _edge_points(size, step: float = 0.005) -> np.ndarray:
    half, pts = np.asarray(size) / 2, []
    for a in range(3):
        b, cc = [i for i in range(3) if i != a]
        t = np.linspace(-half[a], half[a], max(2, int(np.ceil(size[a] / step)) + 1))
        for sb in (-1.0, 1.0):
            for sc in (-1.0, 1.0):
                p = np.zeros((len(t), 3))
                p[:, a], p[:, b], p[:, cc] = t, sb * half[b], sc * half[cc]
                pts.append(p)
    return np.concatenate(pts)


def timer_boxes(origin_pose) -> list:
    """The timer as two boxes (rotation, centre, size), world: its housing, and the dial with the centre cap."""
    p, R = origin_pose
    hx, hy, hz = g.HOUSING
    housing = (R, p + R @ np.array([0.0, 0.0, g.HOUSING_Z0 + hz / 2]), np.array([hx, hy, hz]))
    dz = g.CAP_H - g.FACE_H
    dial = (R, p + R @ np.array([0.0, 0.0, g.FACE_H + dz / 2]), np.array([2 * g.RIB_R, 2 * g.RIB_R, dz]))
    return [housing, dial]


def _hand_gap(environment, boxes) -> float:
    """The smallest gap from either gripper (its body's and fingers' collision hulls) to these boxes."""
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
                pl = h.planes
                best = min(best, float((edges @ pl[:, :3].T - pl[:, 3]).max(axis=1).min()))
    return best


def pen_tip(states) -> tuple[np.ndarray, np.ndarray]:
    """The pen's ball centre and its axis (tip to top), world, from the pen's pose."""
    pen = states["pen"]
    R = _rotation(pen.quat_xyzw)
    return np.asarray(pen.pos, float) + R[:, 2] * g.PEN_BALL_R, R[:, 2]


def _pen_gap(states, boxes) -> float:
    from yam_arena.analysis import box_signed_distance

    tip, u = pen_tip(states)
    pts = tip[None, :] + np.linspace(0.0, g.PEN_L - g.PEN_BALL_R, 40)[:, None] * u[None, :]
    rad = np.where(np.linspace(0.0, g.PEN_L, 40) < g.PEN_CONE, g.PEN_BALL_R, g.PEN_R)
    return float(min((box_signed_distance(pts, c, R, s) - rad).min() for R, c, s in boxes))


def lamp_test(down) -> list[tuple[float, float]]:
    """The timer's output over one turn: the times (h) the lamp goes on and off, one pair per unbroken run of down
    segments (a run through midnight is one pair)."""
    d = [i in set(down) for i in range(g.SEG_N)]
    if all(d):
        return [(0.0, 24.0)]
    runs = []
    for i in range(g.SEG_N):
        if d[i] and not d[i - 1]:
            j = i
            while d[(j + 1) % g.SEG_N]:
                j += 1
            runs.append((i * 0.5, ((j + 1) * 0.5) % 24.0))
    return runs


def tier(down, time_set: bool) -> float:
    """The score of a set of down segments (the spec's tiers), before the end conditions and the gripper cap."""
    down = set(down)
    ten = set(g.TARGETS)
    if down == ten:
        return 1.0 if time_set else 0.75
    near = (set(range(g.ON_SEG - 1, g.OFF_SEG)), set(range(g.ON_SEG + 1, g.OFF_SEG)),
            set(range(g.ON_SEG, g.OFF_SEG - 1)), set(range(g.ON_SEG, g.OFF_SEG + 1)))
    if down in near:
        return 0.5
    if len(down & ten) >= 8 and len(down - ten) <= 1:
        return 0.25
    return 0.0


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    S["step"] += 1
    V = _views(environment)
    st = object_states(environment, ["timer", "pen"])
    tm = st["timer"]
    p0, R0 = np.asarray(tm.pos, float), _rotation(tm.quat_xyzw)
    if S["start_pose"] is None:
        S["start_pose"] = (p0.copy(), R0.copy())
    sp, sR = S["start_pose"]
    S["strip_moved"] = max(S["strip_moved"], float(np.linalg.norm((p0 - sp)[:2])))
    S["strip_turn"] = max(S["strip_turn"], math.degrees(math.acos(min(1.0, (np.trace(sR.T @ R0) - 1) / 2))))
    # pulled out of its socket: the timer raised more than 3 mm (latched)
    if p0[2] - sp[2] > g.RAISED:
        S["pulled"] = True
        if "pulled out of its socket" not in S["why"]:
            S["why"].append("pulled out of its socket")
    q = tm.joints
    # the dial, unwrapped; a counterclockwise turn beyond CCW_TOL from the most clockwise angle reached
    raw = float(q["dial"])
    if S["dial_prev"] is None:
        S["dial_prev"] = raw
    S["dial"] += math.remainder(raw - S["dial_prev"], 2 * math.pi)
    S["dial_prev"] = raw
    S["dial_min"] = min(S["dial_min"], S["dial"])
    if S["dial"] - S["dial_min"] > g.CCW_TOL:
        S["ccw"] = True
    S["time"] = g.pointer_time(S["dial"])
    S["time_set"] = bool(abs(math.remainder(S["dial"] - g.TARGET_ANGLE, 2 * math.pi)) <= g.TIME_TOL and not S["ccw"])
    # segments: latched ones are down; record each new latch with the pen tip's offset from its centre line
    prev_down = set(S["down"])
    latched = _np(_SUB["latched"]).astype(bool) if "latched" in _SUB else np.zeros(g.SEG_N, bool)
    down = [i for i in range(g.SEG_N) if latched[i]]
    segq = np.array([float(q[s]) for s in SEGS])
    tip, _ = pen_tip(st)
    rel = tip - (p0 + R0 @ np.array([0.0, 0.0, g.SEG_TOP_H]))
    rel = R0.T @ rel
    rtip, atip = math.hypot(rel[0], rel[1]), math.atan2(rel[1], rel[0])
    for i in set(down) - set(S["down"]):
        off = math.remainder(atip - g.seg_angle_world(i, raw), 2 * math.pi) * rtip
        half = 0.5 * g.SEG_ANG_W * rtip
        gap = (g.SEG_ANG - g.SEG_ANG_W) * rtip
        S["presses"].append(dict(seg=i, step=S["step"], off_mm=round(off * 1000, 3), r_mm=round(rtip * 1000, 2),
                                 margin_mm=round((half + gap - g.PEN_BALL_R) * 1000, 3),
                                 clear_mm=round((half + gap - g.PEN_BALL_R - abs(off)) * 1000, 3),
                                 nb_mm=round(max(-segq[(i - 1) % g.SEG_N], -segq[(i + 1) % g.SEG_N]) * 1000, 3)))
    S["down"] = down
    # the neighbours of the segment under the pen: how far they are pushed (unlatched)
    if rtip < g.SEG_R_OUT + 0.003 and tip[2] < TOP + g.SEG_TOP_H + 0.003:
        k = int(round((math.remainder(atip - raw, 2 * math.pi) - g.seg_angle_dial(0)) / g.SEG_ANG)) % g.SEG_N
        for j in ((k - 1) % g.SEG_N, (k + 1) % g.SEG_N):
            if not latched[j] and -segq[j] > S["nb_max"]:
                S["nb_max"], S["nb_who"] = float(-segq[j]), (j, S["step"])
    # a gripper pushing a segment down (0.5 mm or more, or latching it) caps the score at 0.25 (latched); a touch that
    # does not push it (fingers on the segments' ends while turning the dial) does not. A segment already down when a
    # gripper meets it is not counted.
    m = _np(V["segs"].get_contact_force_matrix(V["dt"]))
    fseg = np.linalg.norm(m, axis=-1).reshape(g.SEG_N, -1).max(axis=1) if m.size else np.zeros(g.SEG_N)
    contact = fseg > g.TOUCH_F
    ts = S["touch_start"]
    for j in range(g.SEG_N):
        was = j in S["contact_prev"]
        if contact[j] and j not in ts:
            ts[j] = j in prev_down                         # already down when the gripper met it
        if (contact[j] or was) and not ts.get(j, False):
            if -segq[j] >= g.PUSH_DOWN or (latched[j] and j not in prev_down):
                if not S["pushed"]:
                    S["push_who"] = (S["step"], j, round(-segq[j] * 1000, 2))
                S["pushed"] = True
        if not contact[j]:
            ts.pop(j, None)
    if contact.any():
        if not S["touched"]:
            S["touch_who"] = (S["step"], [int(j) for j in np.flatnonzero(contact)], round(float(fseg.max()), 3))
        S["touched"] = True
    S["contact_prev"] = set(int(j) for j in np.flatnonzero(contact))
    S["switch_on"] = bool(_SUB.get("switch_on", True))
    S["lamp"] = lamp_test(down)
    # end conditions
    boxes = timer_boxes((p0, R0))
    S["gap"] = _hand_gap(environment, boxes)
    S["pen_gap"] = _pen_gap(st, boxes)
    S["clear"] = S["gap"] >= g.CLEAR and S["pen_gap"] >= g.CLEAR
    t = tier(down, S["time_set"])
    S["tier"] = t
    rung = t if (S["clear"] and S["switch_on"] and not S["pulled"]) else 0.0
    if S["pushed"]:
        rung = min(rung, 0.25)
    S["rung"] = rung
    S["rungs"] = (S["rungs"] + [rung])[-g.HOLD_STEPS:]
    if _DEBUG and S["step"] % 20 == 0:
        print(f"[pt] step {S['step']} dial {math.degrees(S['dial']):.2f} time {g.fmt_time(S['time'])} down {down} "
              f"touched {S['touched']} pushed {S['pushed']} gap {S['gap'] * 1000:.0f} pen_gap {S['pen_gap'] * 1000:.0f} rung {rung}",
              flush=True)


def success(objects) -> bool:
    """The full-score state (yam_arena ends the trial once it has held for 1 s)."""
    return bool(_STATE["rung"] == 1.0)


def score(objects) -> float:
    """Every control step (20 Hz), from the joints and poses:
    - down: a segment pushed 1 mm or more latches (as the real detent clicks) and stays down; "the ten" are 6:00 PM to
      10:30 PM (segments 36..45; segment i covers i/2 to i/2 + 0.5 h);
    - time set: the pointer within 3.75 degrees (a quarter hour) of 2:30 PM, and the dial never turned back
      counterclockwise more than 15 degrees (an hour) from the furthest clockwise angle it reached;
    - 1.0: exactly the ten down and the time set; 0.75: exactly the ten, time not set; 0.5: one unbroken run that is
      the ten with one end moved by one segment (on at 5:30 or 6:30 PM, or off at 10:30 or 11:30 PM), nothing else;
      0.25: at least 8 of the ten down and at most one other; 0 otherwise;
    - every level needs, at every step of the final 1 s: both grippers (bodies and fingers, collision hulls) and the pen
      20 mm or more from the timer (its housing and dial), the slide switch at TIMER ON, and the timer never raised
      more than 3 mm (out of its socket);
    - a gripper pushing any segment down at any time (0.5 mm or more, or clicking it) caps the score at 0.25 (the
      segments are for the pen); a touch that does not push one down does not.
    The run's score is the lowest level any step of its final 1 s reached."""
    S = _STATE
    if not S["rungs"]:
        return 0.0
    return float(min(S["rungs"]))
