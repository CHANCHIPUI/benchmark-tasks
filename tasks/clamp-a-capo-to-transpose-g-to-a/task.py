"""On the station table, a full-size steel-string acoustic guitar lies on its back on a foam work mat, strings up, its neck
along y in a padded neck rest and its headstock toward the robot's right. A red cam-lever capo is clamped on the neck at
the 4th fret, left there from the previous song. The singer wants the next song in A but plays it with G chord shapes,
so the capo belongs at the 2nd fret: two semitones above G, counted from the nut. The guitar is fixed; the capo is a
free articulation (frame, jaw, lever).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import assets as a  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, before_reset, on_reset, on_step  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

a.build()

SCENE = "station"
TOP = TABLE_TOP[SCENE]


def space_mid(n: int) -> float:
    """The middle of fret space n (between wires n - 1 and n; wire 0 is the nut), metres from the nut."""
    return 0.5 * (a.fret_d(n - 1) + a.fret_d(n)) if n > 1 else 0.5 * a.fret_d(1)


# -- layout (station frame, metres: x forward, y to the robot's left) ---------------------------------------------
Z_S = TOP + a.STRING_TOP  # the pad's underside, on the strings
CAPO_START = (a.X, a.NUT_Y + a.D4, Z_S + 0.0002)  # clamped at the 4th fret, bar across the neck

JAW_CLAMP = 0.012  # the jaw's target when closed: above any neck it can meet, so it presses with the force limit
CLAMP_N = 25.0
OBJECTS = [
    UsdBody("capo", usd="assets/capo.usda", pos=CAPO_START, size=(0.066, 0.012, 0.06), articulation=True,
            joint_pos=(("jaw", 0.0), ("lever", 0.0)),
            drives=(("jaw", 2500.0, 20.0, CLAMP_N), ("lever", 0.03, 0.001, 0.2)),
            friction=a.CAPO_FRICTION, solver_iters=32, torsional_patch=0.006),
    UsdBody("guitar", usd="assets/guitar.usda", pos=(0.0, 0.0, TOP), mass=None, size=(0.46, 1.06, 0.12),
            friction=(0.6, 0.5)),
]

# -- the capo's cam and latch (logic, once per control step) -------------------------------------------------------
CLOSE_AT = -math.radians(4.0)  # the lever pressed to within 4 degrees of flat closes the capo
OPEN_AT = -math.radians(16.0)  # the lever turned past 16 degrees up (its tail pressed flat) opens it
_CAPO: dict = {}


def _capo_reset() -> None:
    _CAPO.clear()
    _CAPO.update(closed=True, q={"jaw": 0.0, "lever": 0.0})


_capo_reset()


@before_reset
def _put_back(environment) -> None:
    """Lever closed and jaw at the neck, before the scene is reset (a reset moves the capo only by its root)."""
    from scenekit import set_joint_targets, write_joint_positions

    set_joint_targets(environment, "capo", {"jaw": 0.0, "lever": 0.0})
    write_joint_positions(environment, "capo", {"jaw": 0.0, "lever": 0.0})
    _capo_reset()


@on_step
def _run_capo(environment) -> None:
    """The cam and its over-centre latch, once per control step (before the scorer)."""
    from scenekit import _np, set_joint_targets

    asset = environment.unwrapped.scene["capo"]
    names = ["jaw", "lever"]
    q = _np(asset.data.joint_pos)[0, asset.find_joints(names, preserve_order=True)[0]]
    _CAPO["q"] = {"jaw": float(q[0]), "lever": float(q[1])}
    if q[1] >= CLOSE_AT:
        _CAPO["closed"] = True
    elif q[1] <= OPEN_AT:
        _CAPO["closed"] = False
    closed = _CAPO["closed"]
    set_joint_targets(environment, "capo", {"jaw": JAW_CLAMP if closed else a.JAW_OPEN, "lever": 0.0 if closed else a.LEVER_OPEN})


# -- scoring ------------------------------------------------------------------------------------------------------
# Distances along the neck from the nut's body-side face. The 1st and 2nd fret wires' centre lines are at 36.5 and
# 70.9 mm; the bar is 12 mm wide and a wire's crown 2.7 mm, so the bar covers neither wire while its centre line is at
# least 7.5 mm from both.
CLEAN = (0.0440, 0.0634)  # clean 2nd fret: the bar wholly in the 2nd fret space
FRET2 = (0.0440, 0.0784)  # 2nd fret: in the space or on top of the 2nd wire
LEVER_FLAT = -math.radians(5.0)  # clamped: the lever within 5 degrees of flat
PAD_TOL = 0.002  # the pad's underside (middle) from the string tops
JAW_TOL = 0.002  # the jaw's top from the neck's back
TILT_MAX = 10.0  # degrees from level
SQUARE_MAX = 10.0  # degrees between the bar and square to the strings, seen from above
TCP_CLEAR = 0.06  # both TCPs from the middle of the pad's underside
STILL_MOVE, STILL_S = 0.001, 1.0  # still: moved less than 1 mm in the last 1 s
DRAG = 0.005  # moved this far along the neck while its lever is closed: dragged (0 for good)
GUITAR_MOVE, GUITAR_TURN = 0.01, 2.0  # m, degrees: the guitar knocked (only possible in a real trial)
PAD_HALF = 0.027  # the pad's half length along the bar
JAW_TOP = np.array([0.5 * (a.JAW_X[0] + a.JAW_X[1]), 0.0, a.JAW_Z0])  # the jaw pad's top middle, frame coordinates, joint at 0


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def crossing(o: np.ndarray, ax: np.ndarray, k: int) -> tuple[float, float] | None:
    """Where the bar's centre line (o + t ax, seen from above) crosses string k: (distance from the nut, t)."""
    c = (k - 3.5) / 5.0
    g = (0.054 - 0.035) / a.SCALE
    den = ax[0] - c * g * ax[1]
    if abs(den) < 1e-9:
        return None
    t = (a.X + c * 0.035 + c * g * (o[1] - a.NUT_Y) - o[0]) / den
    return float(o[1] + t * ax[1] - a.NUT_Y), float(t)


def capo_checks(o, quat, q_jaw: float, q_lever: float, tcps: list, moved_1s: float | None) -> dict:
    """Every measured quantity and condition of 'How it is scored'. o, quat: the frame's pose (its origin is the middle
    of the pad's underside); q_jaw, q_lever: the joints; moved_1s: how far the capo has moved in the last 1 s."""
    r = _rot(quat)
    o = np.asarray(o, float)
    ax = r[:, 0]
    tilt = math.degrees(math.acos(float(np.clip(r[2, 2], -1.0, 1.0))))
    square = math.degrees(math.atan2(abs(ax[1]), abs(ax[0]))) if math.hypot(ax[0], ax[1]) > 1e-9 else 90.0
    cross = [crossing(o, ax, k) for k in (1, 6)]
    on_pad = all(c is not None and abs(c[1]) <= PAD_HALF for c in cross)
    ds = [c[0] for c in cross] if on_pad else [float("nan"), float("nan")]
    d_mid = float(o[1] - a.NUT_Y)
    jaw = o + r @ (JAW_TOP + np.array([0.0, 0.0, q_jaw]))
    pad_gap = float(o[2] - (TOP + a.STRING_TOP))
    jaw_gap = float((TOP + a.neck_back_z(jaw[1] - a.NUT_Y)) - jaw[2])  # how far the jaw is below the neck's back
    on_neck = 0.0 < d_mid < a.FB_END - a.NUT_Y - 0.03 and abs(o[0] - a.X) < 0.02
    lever_closed = q_lever >= LEVER_FLAT
    clamped = (lever_closed and on_neck and on_pad and abs(pad_gap) <= PAD_TOL and abs(jaw_gap) <= JAW_TOL
               and tilt < TILT_MAX)
    released = all(float(np.linalg.norm(np.asarray(t, float) - o)) >= TCP_CLEAR for t in tcps)
    still = moved_1s is not None and moved_1s < STILL_MOVE
    in_win = lambda w: on_pad and all(w[0] <= d <= w[1] for d in ds)  # noqa: E731
    return {"lever_closed": bool(lever_closed), "clamped": bool(clamped), "square": bool(square < SQUARE_MAX), "released": bool(released), "still": bool(still),
            "clean": bool(in_win(CLEAN)), "fret2": bool(in_win(FRET2)), "d": ds, "d_mid": d_mid, "tilt": float(tilt),
            "skew": float(square), "pad_gap": pad_gap, "jaw_gap": jaw_gap, "lever_deg": math.degrees(q_lever)}


_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, latched=None, guitar0=None, history=[], anchor=None, success=False, half=False, checks=None)


_reset_state()


def update(o, quat, q_jaw: float, q_lever: float, guitar, tcps: list) -> dict:
    """One control step. Two failures latch until a reset: the guitar moved, and the capo dragged (moved more than 5 mm
    along the neck while its lever is closed, whatever its tilt or height: only the lever counts)."""
    _STATE["step"] += 1
    if _STATE["guitar0"] is None:
        _STATE["guitar0"] = (np.asarray(guitar.pos, float).copy(), _rot(guitar.quat_xyzw))
    p0, r0 = _STATE["guitar0"]
    turn = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ _rot(guitar.quat_xyzw)) - 1) / 2, -1, 1))))
    if _STATE["latched"] is None and (float(np.linalg.norm(np.asarray(guitar.pos, float) - p0)) > GUITAR_MOVE or turn > GUITAR_TURN):
        _STATE["latched"] = "guitar moved"
    hist = _STATE["history"]
    hist.append(np.asarray(o, float).copy())
    n = int(round(STILL_S * steptime.CONTROL_HZ)) + 1
    del hist[:-n]
    moved = max(float(np.linalg.norm(h - hist[-1])) for h in hist) if len(hist) == n else None
    c = capo_checks(o, quat, q_jaw, q_lever, tcps, moved)
    if c["lever_closed"]:
        if _STATE["anchor"] is None:  # where the capo was when the lever last closed
            _STATE["anchor"] = c["d_mid"]
        elif abs(c["d_mid"] - _STATE["anchor"]) > DRAG and _STATE["latched"] is None:
            _STATE["latched"] = "dragged with the lever closed"
    else:
        _STATE["anchor"] = None
    ok = _STATE["latched"] is None and c["clamped"] and c["released"] and c["still"]
    _STATE["success"] = bool(ok and c["square"] and c["clean"])
    _STATE["half"] = bool(ok and c["fret2"])
    _STATE["checks"] = c
    c["moved_1s"] = moved
    return c


def _trace(record: dict) -> None:
    folder = os.environ.get("TASK_SCORE_DIR")
    if not folder:
        return
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "trace.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


@on_step
def _monitor(environment) -> None:
    from yam_arena.runtime import tcp_world
    from yam_arena.scene import object_states

    base = environment.unwrapped
    origin = base.scene.env_origins[0]
    origin = getattr(origin, "torch", origin).detach().cpu().numpy()
    tcps = [tcp_world(base.scene[f"{side}_robot"]) - origin for side in ("left", "right")]
    states = object_states(environment, ["capo", "guitar"])
    st = states["capo"]
    c = update(st.pos, st.quat_xyzw, _CAPO["q"]["jaw"], _CAPO["q"]["lever"], states["guitar"], tcps)
    rec = {"t": round(_STATE["step"] / steptime.CONTROL_HZ, 2), "success": _STATE["success"], "half": _STATE["half"],
           "latched": _STATE["latched"], "latch_closed": bool(_CAPO["closed"]),
           "capo": [round(float(v), 5) for v in st.pos], "d_mm": [round(d * 1000, 1) for d in c["d"]],
           "d_mid_mm": round(c["d_mid"] * 1000, 2), "tilt": round(c["tilt"], 2), "skew": round(c["skew"], 2),
           "pad_gap_mm": round(c["pad_gap"] * 1000, 2), "jaw_gap_mm": round(c["jaw_gap"] * 1000, 2),
           "lever_deg": round(c["lever_deg"], 2), "jaw_mm": round(_CAPO["q"]["jaw"] * 1000, 2),
           "anchor_mm": None if _STATE["anchor"] is None else round(_STATE["anchor"] * 1000, 2),
           "moved_1s_mm": None if c["moved_1s"] is None else round(c["moved_1s"] * 1000, 3),
           "fail": [k for k in ("clamped", "square", "released", "still", "clean", "fret2") if not c[k]],
           "tcpR": [round(float(v), 4) for v in tcps[1]]}
    if os.environ.get("TASK_SCORE_DIR"):
        from scenekit import arm_joints
        rec["qR"] = [round(float(v), 4) for v in arm_joints(environment, "right", [f"joint{i}" for i in range(1, 7)])]
    _trace(rec)


def success(objects) -> bool:
    """Clamped, square, released, still and a clean 2nd fret, with nothing latched; the harness ends the trial at 1.0
    after 1 s of this."""
    steptime.tick()  # the harness calls success() once per control step
    return bool(_STATE["success"])


def score(objects) -> float:
    """At the run limit: 0.5 for the capo clamped, released and still at the 2nd fret (in the space or on its wire);
    0 otherwise, and 0 whatever happens once the guitar has moved or the capo has been dragged with its lever closed."""
    if _STATE["latched"] is not None:
        return 0.0
    return 0.5 if _STATE["half"] else 0.0


# -- scripted demo (open loop: it reads no simulation state) -----------------------------------------------------
FINGER_CLOSED = 0.0
PRESS_YAW = math.pi / 2  # right arm: the closed fingertips cover 13 mm along x and 36 mm along y
GRIP_YAW = 0.0  # fingers closing along y, on the frame's sides at the spine end
OPEN_W = -0.030  # pads 60 mm apart
HOLD = -0.004  # pads 8 mm apart: 4 mm narrower than the 12 mm frame
TIP = 0.013  # pad tips below the TCP
GRIP_DX = -0.030  # the grip, from the middle of the bar, toward the spine
TAIL_DX = -0.034  # the tail is pressed here
HANDLE_DX = 0.008  # the handle is pressed here, 30.5 mm from the hinge
PRESS_TO = 0.011  # fingertips commanded to this height above the strings (the flat lever's top is at 16 mm; the arm gives)
GRIP_AT = 0.002  # fingertips this far above the strings while gripping the frame
LIFT = 0.002  # the open capo is raised this much while it slides
PARK = (0.20, -0.42, TOP + 0.20)  # out of the top camera's view of the nut and the first frets
TARGET_D = 0.057  # the bar's centre line 13.9 mm behind the 2nd wire, 6.4 mm inside the clean window


def _wp(dt, x, y, z, finger, label, yaw):
    return Waypoint(dt, (x, y, z), finger, label, yaw)


def _at(d: float, dx: float, yaw: float = 0.0) -> tuple[float, float]:
    """A point dx along the bar from its middle, for the capo's bar centred d from the nut and turned yaw."""
    return a.X + dx * math.cos(yaw), a.NUT_Y + d + dx * math.sin(yaw)


def _press(xy, label) -> list[Waypoint]:
    x, y = xy
    z = Z_S + PRESS_TO + TIP
    return [
        _wp(1.5, x, y, Z_S + 0.07, FINGER_CLOSED, f"over the {label}", PRESS_YAW),
        _wp(0.8, x, y, Z_S + 0.045, FINGER_CLOSED, "near", PRESS_YAW),
        _wp(0.8, x, y, z, FINGER_CLOSED, f"press the {label}", PRESS_YAW),
        _wp(0.5, x, y, z, FINGER_CLOSED, "hold", PRESS_YAW),
        _wp(0.6, x, y, Z_S + 0.07, FINGER_CLOSED, "rise", PRESS_YAW),
    ]


def _grip(xy, yaw: float = GRIP_YAW) -> list[Waypoint]:
    x, y = xy
    return [
        _wp(1.8, x, y, Z_S + 0.06, OPEN_W, "over the frame", yaw),
        _wp(0.8, x, y, Z_S + GRIP_AT + TIP, OPEN_W, "down beside it", yaw),
        _wp(0.6, x, y, Z_S + GRIP_AT + TIP, HOLD, "grip", yaw),
    ]


def _let_go(xy, yaw: float = GRIP_YAW) -> list[Waypoint]:
    x, y = xy
    return [_wp(0.6, x, y, Z_S + GRIP_AT + TIP, OPEN_W, "let go", yaw), _wp(0.6, x, y, Z_S + 0.07, OPEN_W, "rise", yaw)]


def _park(t: float = 2.5) -> list[Waypoint]:
    return [_wp(2.0, *PARK, OPEN_W, "park", PRESS_YAW), _wp(t, *PARK, OPEN_W, "wait", PRESS_YAW)]


def _move(d_to: float, yaw_to: float = 0.0, clamp: bool = True, sideways: bool = False) -> list[Waypoint]:
    """Release the capo at the 4th fret, slide it to d_to (turned yaw_to), and clamp it there."""
    d0 = a.D4
    gz = Z_S + GRIP_AT + TIP
    g0 = _at(d0, GRIP_DX)
    g1 = _at(d_to, GRIP_DX, yaw_to)
    plan = _press(_at(d0, TAIL_DX), "tail") + _grip(g0)
    plan += [_wp(0.4, *g0, gz + LIFT, HOLD, "raise 2 mm", GRIP_YAW)]
    if sideways:
        off = (g0[0] - 0.06, g0[1])
        plan += [_wp(1.5, *off, gz + LIFT, HOLD, "slide off sideways", GRIP_YAW),
                 _wp(1.5, off[0], g1[1], gz + LIFT, HOLD, "along, beside the neck", GRIP_YAW),
                 _wp(1.5, *g1, gz + LIFT, HOLD, "slide back on", yaw_to)]
    else:
        plan += [_wp(1.6, *g1, gz + LIFT, HOLD, "slide along the neck", GRIP_YAW + yaw_to)]
    plan += [_wp(0.4, *g1, gz, HOLD, "lower onto the strings", GRIP_YAW + yaw_to)] + _let_go(g1, GRIP_YAW + yaw_to)
    if clamp:
        plan += _press(_at(d_to, HANDLE_DX, yaw_to), "handle")
    return plan


def demo():
    """The capo released at the 4th fret, slid to the 2nd and clamped. Variants (TASK_DEMO), to check the scorer:
      failure   slid to the 2nd fret but left open (lever up): 0
      fret3     clamped at the 3rd fret (G, G sharp, A counted as three frets): 0
      fret1     clamped at the 1st fret (one step read as one fret): 0
      fret6     clamped two frets up from the start: 0
      wire      clamped on top of the 2nd fret wire: 0.5
      skew      clamped in the 2nd fret space, turned 15 degrees: 0.5
      held      clamped in the 2nd fret space, then gripped again, the hand never 6 cm away, and held: 0
      nothing   the capo left clamped at the 4th fret: 0
      grip      a firm grip on the clamped capo, then the demo: 1.0 (the grip does not trip the drag rule)
      drag      the clamped capo pushed 6 cm along the neck without opening the lever: 0 (0 for good if it moves 5 mm)
      side      the demo, but slid off the neck sideways, along, and back on: 1.0"""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode == "nothing":
        return {"right": _park(3.0)}
    if mode == "drag":
        gz = Z_S + GRIP_AT + TIP
        g0, g1 = _at(a.D4, GRIP_DX), _at(TARGET_D, GRIP_DX)
        return {"right": _grip(g0) + [_wp(2.0, *g1, gz, HOLD, "drag along the neck", GRIP_YAW)] + _let_go(g1) + _park(3.0)}
    if mode == "grip":
        g0 = _at(a.D4, GRIP_DX)
        plan = _grip(g0) + [_wp(1.5, *g0, Z_S + GRIP_AT + TIP, HOLD, "hold firmly", GRIP_YAW)] + _let_go(g0)
        return {"right": plan + _move(TARGET_D) + _park()}
    if mode == "held":
        # Clamped right, then gripped again without the hand ever going 6 cm from the pad, and held to the end.
        g1, h = _at(TARGET_D, GRIP_DX), _at(TARGET_D, HANDLE_DX)
        z_near = Z_S + 0.030 + TIP  # fingertips 30 mm above the strings: the TCP stays under 6 cm from the pad
        plan = _move(TARGET_D)[:-1]  # the handle pressed flat, without the rise that ends the press
        plan += [_wp(0.6, *h, z_near, FINGER_CLOSED, "rise a little", PRESS_YAW),
                 _wp(1.8, *g1, z_near, OPEN_W, "over the frame, staying close", GRIP_YAW),
                 _wp(0.8, *g1, Z_S + GRIP_AT + TIP, OPEN_W, "down beside it", GRIP_YAW),
                 _wp(0.6, *g1, Z_S + GRIP_AT + TIP, HOLD, "grip", GRIP_YAW),
                 _wp(4.0, *g1, Z_S + GRIP_AT + TIP, HOLD, "keep holding", GRIP_YAW)]
        return {"right": plan}
    plans = {
        "failure": lambda: _move(TARGET_D, clamp=False),
        "fret3": lambda: _move(space_mid(3)),
        "fret1": lambda: _move(space_mid(1)),
        "fret6": lambda: _move(space_mid(6)),
        "wire": lambda: _move(a.fret_d(2)),
        "skew": lambda: _move(TARGET_D, yaw_to=math.radians(15.0)),
        "side": lambda: _move(TARGET_D, sideways=True),
    }
    return {"right": plans.get(mode, lambda: _move(TARGET_D))() + _park()}
