"""On the station table, in front of the right arm: a small four-wheeled robot car (20 x 14 cm, 8.5 cm tall) stands on
its wheels, facing away from the arms. Seven centimetres in front of its nose is a white rack of ten capped glass sample
vials. Two centimetres to the car's left stands its test stand: a beech block, 26 cm long, 6 cm wide and 6 cm tall,
with a ribbed grey rubber pad on top, its long side across the car's heading.

The car has two black acrylic decks with yellow gearmotors and a battery between them, a green controller board and an
ultrasonic sensor at the front, an aluminium carry handle on top in the middle, and a latching power button with a
green LED ring on top behind the handle. Its wheels are 6.5 cm across with black tyres on yellow hubs; the left and
right tyres are 9.2 cm apart inside. Switched on, it runs a wheel test: for 10 s it drives all four wheels forward.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import car_usd as C  # noqa: E402
import steptime  # noqa: E402
from scenekit import UsdBody, before_reset, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = C.build()

CAR_XY = (0.25, -0.27)            # facing +x
STAND_XY = (0.25, -0.05)          # long side along y
RACK_XY = (0.455, -0.27)          # long side along y
WHEEL_JOINTS = [f"axle_{n}" for n in C.WHEELS]

CAR = UsdBody("robot_car", usd=USD["car"], pos=(*CAR_XY, TOP + 0.0003), size=(C.LENGTH, 0.14, C.TOP_Z), articulation=True,
              joint_pos=tuple((j, 0.0) for j in WHEEL_JOINTS) + (("button", 0.0),),
              drives=(("axle_.*", 0.0, 0.05, 0.1), ("button", 500.0, 2.0, 100.0)),
              friction=(0.9, 0.8), solver_iters=32, torsional_patch=0.008)
STAND = UsdBody("test_stand", usd=USD["stand"], pos=(*STAND_XY, TOP + 0.0002), size=C.STAND_SIZE, mass=C.STAND_MASS,
                friction=(0.9, 0.8))
RACK = UsdBody("vial_rack", usd=USD["rack"], pos=(*RACK_XY, TOP + 0.0002), size=C.RACK_ENVELOPE, mass=C.RACK_MASS,
               friction=(0.5, 0.4))
OBJECTS = [CAR, STAND, RACK]

# -- the car: power switch, wheel test, LED -----------------------------------------------------------------------------
SWITCH_AT = -0.004                # m: the cap pressed this far switches the car over
REARM_AT = -0.002                 # m: ... and must come back above this before the next press counts
LATCHED = -0.0015                 # m: where the cap rests while the car is on
TEST_STEPS = 200                  # 10 s at 20 Hz
WHEEL_SPEED = 7.7                 # rad/s: 74 rpm, 0.25 m/s at the tread

_CAR: dict = {}


def _car_reset() -> None:
    _CAR.clear()
    _CAR.update(on=False, armed=True, testing=False, steps=0, tests=0, completed=False, driven=False, shown=None,
                switched=[])


@before_reset
def _put_back(environment) -> None:
    """Button up, wheels still, power off, before the scene is reset (a reset moves the car only by its root)."""
    from scenekit import set_joint_targets, set_joint_velocity_targets, write_joint_positions

    set_joint_targets(environment, "robot_car", {"button": 0.0})
    set_joint_velocity_targets(environment, "robot_car", {j: 0.0 for j in WHEEL_JOINTS})
    write_joint_positions(environment, "robot_car", {**{j: 0.0 for j in WHEEL_JOINTS}, "button": 0.0})
    _car_reset()
    _show_led(False)


def _show_led(lit: bool) -> None:
    """Show the lit or the unlit button cap and LED ring."""
    if _CAR.get("shown") is lit:
        return
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    if "prims" not in _CAR or not _CAR["prims"]:
        names = {"led_ring_on", "led_ring_off", "cap_on", "cap_off"}
        _CAR["prims"] = [p for p in stage.Traverse() if p.GetName() in names and "/robot_car/" in str(p.GetPath())]
    for prim in _CAR["prims"]:
        img = UsdGeom.Imageable(prim)
        if prim.GetName().endswith("_on") == lit:
            img.MakeVisible()
        else:
            img.MakeInvisible()
    _CAR["shown"] = lit


@on_step
def _run_car(environment) -> None:
    """The car's switch and firmware, once per control step: a press past 4 mm toggles the power; each switch-on runs
    a 10 s wheel test; the LED shows the power."""
    from scenekit import set_joint_targets, set_joint_velocity_targets

    asset = environment.unwrapped.scene["robot_car"]
    from scenekit import _np

    q = float(_np(asset.data.joint_pos)[0, asset.find_joints(["button"])[0][0]])
    S = _CAR
    S["driven"] = S["testing"]           # the wheels were driven during the step that just ran
    if S["testing"]:
        S["steps"] += 1
        if S["steps"] >= TEST_STEPS:
            S["testing"] = False
            S["completed"] = True
    if S["armed"] and q <= SWITCH_AT:
        S["armed"] = False
        S["on"] = not S["on"]
        S["testing"], S["steps"] = S["on"], 0
        S["tests"] += int(S["on"])
        S["switched"].append(("on" if S["on"] else "off", _T.get("step", 0) + 1))
    elif q >= REARM_AT:
        S["armed"] = True
    if os.environ.get("TASK_DEMO") == "ledtest":       # debug: light the LED from step 20 to 29 without a press
        S["on"] = 20 <= _T.get("step", 0) + 1 < 30
    set_joint_targets(environment, "robot_car", {"button": LATCHED if S["on"] else 0.0})
    set_joint_velocity_targets(environment, "robot_car", {j: WHEEL_SPEED if S["testing"] else 0.0 for j in WHEEL_JOINTS})
    _show_led(S["on"])


# -- scoring --------------------------------------------------------------------------------------------------------------
CLOSE = 0.005                     # m: a tyre this near anything is too close
RACK_MOVE, RACK_TURN = 0.005, 5.0  # m, degrees
TILT_MAX = 10.0                   # degrees
REST_SPEED = 0.01                 # m/s
HELD = 0.03                       # m: a tool point this near the car holds it
RACK_START = np.array([*RACK_XY, TOP + 0.0002])


def _tyre_points() -> np.ndarray:
    """Points on the four tyres' outer surfaces in the car's frame: the tread at three widths and both side faces."""
    th = np.radians(np.arange(0, 360, 10))
    pts = []
    for x, y in C.WHEELS.values():
        for a in (-C.WHEEL_W / 2, 0.0, C.WHEEL_W / 2):
            pts += [(x + C.WHEEL_R * math.cos(t), y + a, C.AXLE_Z + C.WHEEL_R * math.sin(t)) for t in th]
        for a in (-C.WHEEL_W / 2, C.WHEEL_W / 2):
            for r in (0.66 * C.WHEEL_R, 0.33 * C.WHEEL_R):
                pts += [(x + r * math.cos(t), y + a, C.AXLE_Z + r * math.sin(t)) for t in th[::2]]
            pts.append((x, y + a, C.AXLE_Z))
    return np.array(pts)


TYRE_PTS = _tyre_points()
N_PER_TYRE = len(TYRE_PTS) // 4


def _rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _world(state, local: np.ndarray) -> np.ndarray:
    return np.asarray(state.pos, float) + local @ _rot(state.quat_xyzw).T


def _box_distance(points: np.ndarray, state, size) -> float:
    """Smallest distance from the points to a box standing on its origin (0 inside)."""
    half = np.asarray(size, float) / 2
    centre = np.asarray(state.pos, float) + _rot(state.quat_xyzw) @ np.array([0.0, 0.0, half[2]])
    local = (points - centre) @ _rot(state.quat_xyzw)
    gap = np.maximum(np.abs(local) - half, 0.0)
    return float(np.linalg.norm(gap, axis=1).min())


def tyre_gaps(objects) -> dict:
    """Smallest gap from any tyre to the table top, the floor, the stand and the vial rack (with its vials)."""
    from yam_arena.station import FLOOR_Z

    pts = _world(objects["robot_car"], TYRE_PTS)
    return {"table": float(pts[:, 2].min() - TOP), "floor": float(pts[:, 2].min() - FLOOR_Z),
            "stand": _box_distance(pts, objects["test_stand"], C.STAND_SIZE),
            "rack": _box_distance(pts, objects["vial_rack"], C.RACK_ENVELOPE)}


def tilt_deg(state) -> float:
    return float(np.degrees(np.arccos(np.clip(_rot(state.quat_xyzw)[2, 2], -1.0, 1.0))))


def rack_disturbed(rack) -> bool:
    moved = np.linalg.norm(np.asarray(rack.pos) - RACK_START) > RACK_MOVE
    w = abs(float(np.asarray(rack.quat_xyzw)[3]))
    turned = np.degrees(2 * math.acos(min(1.0, w))) > RACK_TURN
    return bool(moved or turned)


def tool_points() -> list[np.ndarray]:
    """Both arms' tool points (between the finger pads), from their measured joints."""
    from scenekit import arm_joints, env
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.station import BASES

    environment = env()
    if environment is None:
        return []
    return [ik.tcp_pose(arm_joints(environment, s, ARM_JOINTS), BASES[s])[0] for s in ("left", "right")]


def held(car) -> bool:
    """A tool point within 3 cm of the car's outline (wheels and handle included)."""
    half = np.array([C.LENGTH / 2, 0.07, C.TOP_Z / 2])
    for p in tool_points():
        local = (np.asarray(p) - np.asarray(car.pos)) @ _rot(car.quat_xyzw) - np.array([0.0, 0.0, half[2]])
        if np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)) < HELD:
            return True
    return False


def on_the_stand(objects, gaps=None) -> bool:
    """The car's belly rests on the stand's top, tilted under 10 deg and still, not held, every tyre 5 mm clear of the
    table and the stand."""
    car, stand = objects["robot_car"], objects["test_stand"]
    gaps = gaps or tyre_gaps(objects)
    belly = _world(car, np.array([[0.0, 0.0, C.BELLY]]))[0]
    local = (belly - np.asarray(stand.pos)) @ _rot(stand.quat_xyzw)
    resting = (abs(local[2] - C.STAND_SIZE[2]) < 0.005 and abs(local[0]) <= C.STAND_SIZE[0] / 2
               and abs(local[1]) <= C.STAND_SIZE[1] / 2)
    still = np.linalg.norm(car.lin_vel) < REST_SPEED
    clear = gaps["table"] >= CLOSE and gaps["stand"] >= CLOSE
    return bool(resting and tilt_deg(car) < TILT_MAX and still and clear and not held(car))


_T: dict = {}


@on_reset
def _reset(_env=None) -> None:
    """Failures stay failed for the whole trial: only a reset clears them."""
    _T.clear()
    _T.update(failed=None, step=0, done=False)


_reset()
_car_reset()


def _update(objects) -> dict:
    gaps = tyre_gaps(objects)
    if not _T["failed"]:
        if _CAR["driven"] or _CAR["testing"]:
            near = {k: v for k, v in gaps.items() if v < CLOSE}
            if near:
                what, gap = min(near.items(), key=lambda kv: kv[1])
                _T["failed"] = f"a driven tyre {gap * 1000:.1f} mm from the {what}"
        if not _T["failed"] and rack_disturbed(objects["vial_rack"]):
            _T["failed"] = "the vial rack was disturbed"
        if not _T["failed"] and _CAR["completed"]:
            _T["done"] = True
    return gaps


def score(objects) -> float:
    """0 for the whole trial if a tyre comes within 5 mm of the table, the floor, the stand or the vial rack while a
    test drives the wheels, or if the vial rack moves more than 5 mm or turns or tilts more than 5 deg. Otherwise 1.0
    once a wheel test has run its full 10 s; 0.6 while a test is still running; 0.3 for the car off and on the stand;
    else 0."""
    gaps = _update(objects)
    _T["gaps"] = gaps
    if _T["failed"]:
        return 0.0
    if _T["done"]:
        return 1.0
    if _CAR["testing"]:
        return 0.6
    if not _CAR["on"] and on_the_stand(objects, gaps):
        return 0.3
    return 0.0


def success(objects) -> bool:
    """A wheel test has run its full 10 s with no failure; the harness holds it 1 s, then the trial ends."""
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
    car = objects["robot_car"]
    row = {"step": _T["step"], "score": s, "failed": _T["failed"], "on": _CAR["on"], "testing": _CAR["testing"],
           "test_steps": _CAR["steps"], "completed": _CAR["completed"], "button": round(car.joints.get("button", 0.0), 5),
           "car": [round(float(v), 4) for v in car.pos], "quat": [round(float(v), 4) for v in car.quat_xyzw],
           "tilt": round(tilt_deg(car), 2),
           "yaw": round(float(np.degrees(math.atan2(_rot(car.quat_xyzw)[1, 0], _rot(car.quat_xyzw)[0, 0]))), 2),
           "car_v": round(float(np.linalg.norm(car.lin_vel)), 4),
           "stand": [round(float(v), 4) for v in objects["test_stand"].pos],
           "rack": [round(float(v), 4) for v in objects["vial_rack"].pos],
           "gaps": {k: round(v, 4) for k, v in _T["gaps"].items()},
           "on_stand": on_the_stand(objects, _T["gaps"]), "held": held(car),
           "wheels": [round(car.joints.get(j, 0.0), 2) for j in WHEEL_JOINTS]}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------------
OPEN = -0.030                      # finger joint target: pads 60 mm apart
GRASP_H = 0.083                    # TCP above the car's ground: fingertips 3 mm below the handle bar
CARRY = 0.175                      # TCP height while carrying: the tyres about 8.7 cm up, clear of the stand
PLACE_DX = 0.005                   # m: the loaded arm sets the car this far short in x at the stand (from a logged run)
STAND_TOP = C.STAND_SIZE[2]
PLACE_H = STAND_TOP - C.BELLY + 0.002 + GRASP_H     # TCP with the belly 2 mm above the pad
CAP_TOP_ON_STAND = STAND_TOP - C.BELLY + C.DECK_TOP + C.CAP_H
PRESS_H = CAP_TOP_ON_STAND - 0.008 + 0.013          # tips commanded 8 mm past the cap's top
YAW0, YAW90 = 0.0, math.pi / 2
CLEAR_XY = (0.20, -0.33)


def _wp(t, xy, h, finger, label, yaw=YAW0) -> Waypoint:
    return Waypoint(t, (float(xy[0]), float(xy[1]), TOP + h), finger, label, yaw)


def _pick() -> list[Waypoint]:
    return [_wp(2.0, CAR_XY, 0.15, OPEN, "above the handle"),
            _wp(1.5, CAR_XY, GRASP_H + 0.0003, OPEN, "down to the handle"),
            _wp(0.8, CAR_XY, GRASP_H + 0.0003, FINGER_CLOSED, "grip the handle"),
            _wp(1.5, CAR_XY, CARRY, FINGER_CLOSED, "lift the car")]


def _place(yaw=YAW90, xy=None) -> list[Waypoint]:
    xy = xy or (STAND_XY[0] + PLACE_DX, STAND_XY[1])
    return [_wp(2.5, xy, CARRY, FINGER_CLOSED, "carry it over the stand", yaw),
            _wp(1.5, xy, PLACE_H, FINGER_CLOSED, "lower it onto the stand", yaw),
            _wp(0.5, xy, PLACE_H, FINGER_CLOSED, "settle", yaw),
            _wp(0.6, xy, PLACE_H, OPEN, "let go", yaw),
            _wp(1.0, xy, CARRY, OPEN, "up", yaw)]


def _press(xy, cap_top_h) -> list[Waypoint]:
    h = cap_top_h - 0.008 + 0.013
    return [_wp(1.5, xy, cap_top_h + 0.04, FINGER_CLOSED, "above the power button"),
            _wp(1.0, xy, cap_top_h + 0.016, FINGER_CLOSED, "just above the cap"),
            _wp(0.8, xy, h, FINGER_CLOSED, "press the button"),
            _wp(0.5, xy, h, FINGER_CLOSED, "hold"),
            _wp(0.8, xy, cap_top_h + 0.04, FINGER_CLOSED, "release")]


def _wait(t, label="wait clear of the car") -> list[Waypoint]:
    return [_wp(1.5, CLEAR_XY, 0.20, OPEN, "move clear"), _wp(t, CLEAR_XY, 0.20, OPEN, label)]


BUTTON_ON_STAND = (STAND_XY[0], STAND_XY[1] + C.BUTTON_X)      # nose toward +y: the button is toward -y
BUTTON_ON_TABLE = (CAR_XY[0] + C.BUTTON_X, CAR_XY[1])


def demo():
    """Open loop, so it uses nothing a policy could not see. The right arm lifts the car by its handle, turns it a
    quarter turn so that it straddles the stand lengthwise, sets it down, presses the power button, and waits clear
    while the 10 s test runs. TASK_DEMO picks a variant for checking the scorer:
    failure (press the button where the car stands), across_on / across_off (onto the stand without turning, then
    pressed / not), staged (on the stand, never switched on), offmid (switched off 8 s into the test), running (the
    trial ends 6 s into the test), setdown (the running car lifted off the stand onto the table), aloft (switched on on
    the stand, then held up clear for the rest of the test), nothing; and for debugging, physics (switch on in place)
    and ledtest (the LED lit from step 20 to 29 with no press)."""
    mode = os.environ.get("TASK_DEMO", "success")
    if mode in ("nothing", "ledtest"):
        return {"right": [_wp(2.0, CLEAR_XY, 0.20, OPEN, "hover"), _wp(8.0, CLEAR_XY, 0.20, OPEN, "wait")]}
    if mode in ("failure", "physics"):
        return {"right": _press(BUTTON_ON_TABLE, C.DECK_TOP + C.CAP_H) + _wait(6.0)}
    if mode.startswith("across"):
        plan = _pick() + _place(YAW0)
        if mode == "across_on":
            # Set across, the car comes to rest on its tyres on the stand's edges, nose down 13.5 deg; its cap's top is
            # then at (20.7, -5.55) cm, 11.8 cm up (from the across_off run's logged pose).
            plan += _press((0.207, -0.0555), 0.118)
        return {"right": plan + _wait(6.0)}
    plan = _pick() + _place()
    if mode == "staged":
        return {"right": plan + _wait(4.0)}
    plan += _press(BUTTON_ON_STAND, CAP_TOP_ON_STAND)
    if mode == "offmid":
        return {"right": plan + _wait(2.0) + _press(BUTTON_ON_STAND, CAP_TOP_ON_STAND) + _wait(4.0)}
    if mode == "running":
        return {"right": plan + _wait(0.5)}
    if mode in ("setdown", "aloft"):
        lift = [_wp(1.0, STAND_XY, CARRY, OPEN, "back over the handle", YAW90),
                _wp(1.2, STAND_XY, PLACE_H, OPEN, "down to the handle", YAW90),
                _wp(0.8, STAND_XY, PLACE_H, FINGER_CLOSED, "grip the handle", YAW90),
                _wp(1.2, STAND_XY, CARRY + 0.02, FINGER_CLOSED, "lift the running car", YAW90)]
        if mode == "aloft":
            return {"right": plan + lift + [_wp(12.0, STAND_XY, CARRY + 0.02, FINGER_CLOSED, "hold it up", YAW90)]}
        down = (0.25, -0.33)
        return {"right": plan + lift + [_wp(2.0, down, CARRY, FINGER_CLOSED, "over the table", YAW90),
                                        _wp(1.5, down, GRASP_H + 0.002, FINGER_CLOSED, "set it down", YAW90),
                                        _wp(3.0, down, GRASP_H + 0.002, FINGER_CLOSED, "hold", YAW90)]}
    return {"right": plan + _wait(12.0)}
