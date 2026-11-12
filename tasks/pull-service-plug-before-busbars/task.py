"""On the station table, in front of the arms: an opened battery pack on a dismantling bench for recycling, its insulated
tools and voltage tester, and a grey parts tray.

The pack's dark grey steel housing (16 x 37 cm, walls 2.5 cm) holds four brushed-aluminium modules side by side, each
15 x 7 x 4 cm (25.6 V), with two terminal posts on its lid (a nickel pad and an M8 stud): one 2 cm from the front end,
one 2 cm from the back end, with a red (+) or black (-) mark beside each. The modules are wired in series. At the back,
two copper busbars (11 x 2.5 x 0.5 cm) join the left pair and the right pair, each held by two M8 nuts. At the front,
orange cables are bolted to the posts: the middle two run to a black socket in the gap between the middle modules, and
the outer two are the pack's main leads. An orange service disconnect plug sits 2 cm deep in that socket and closes the
string between the two halves. In front of the pack: the grey tray, a multimeter leaning on its stand and facing the
robot, and on each side a rubber tool stand holding a 1000 V insulated nut driver and a test probe (black on the left,
red on the right) whose lead runs to the meter.

Simplified mechanics, each a rule in this file:
- A busbar nut is fitted (fixed) until an insulated driver's socket, seated over it, has turned it half a turn
  counter-clockwise in total; it then comes off its thread and is a loose nut that can be lifted off the stud.
- The meter shows the DC voltage between the two probe tips, from where they touch and from which links (busbars, plug)
  close the string.
- The leads are drawn from the meter's jacks to the probes every control step and have no colliders.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # lets this file import helpers beside it

import kit_usd as K  # noqa: E402
import pack_usd as P  # noqa: E402
import steptime  # noqa: E402
import scenekit  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
USD = {**P.build(), **K.build()}
HZ = 20.0

PACK = UsdBody("battery_pack", usd=USD["pack"], pos=(0.0, 0.0, TOP), size=(0.16, 0.37, 0.064))
TRAY = UsdBody("parts_tray", usd=USD["tray"], pos=(0.0, 0.0, TOP), size=(0.088, 0.268, 0.015))
BENCH = UsdBody("bench_kit", usd=USD["bench"], pos=(0.0, 0.0, TOP), size=(0.11, 0.63, 0.09))
BARS = {name: UsdBody(name, usd=USD["busbar"], pos=(x, y, TOP + P.BAR_Z), size=P.BAR_SIZE, mass=P.BAR_MASS,
                      friction=(0.9, 0.8), torsional_patch=0.008)
        for name, (x, y) in P.BARS.items()}
PLUG = UsdBody("service_plug", usd=USD["plug"], pos=(*P.SOCKET_CENTRE, TOP + P.PLUG_Z), size=P.PLUG_SIZE,
               mass=P.PLUG_MASS, friction=(0.9, 0.8), torsional_patch=0.01)
SEATS = {nut: (post, np.array((x, y))) for nut, post, x, y in P.nut_seats()}
FITTED = {nut: UsdBody(f"{nut}_fitted", usd=USD["nut_fitted"], pos=(xy[0], xy[1], TOP + P.NUT_Z),
                       size=(P.NUT_AF, P.NUT_AF, P.NUT_H)) for nut, (_, xy) in SEATS.items()}
PARK = {nut: (0.30 + 0.03 * k, 0.40, FLOOR_Z + P.NUT_H / 2 + 0.0005) for k, nut in enumerate(SEATS)}   # under the table
LOOSE = {nut: UsdBody(nut, usd=USD["nut_loose"], pos=PARK[nut], size=(P.NUT_AF, P.NUT_AF, P.NUT_H), mass=P.NUT_MASS,
                      friction=(0.9, 0.8), torsional_patch=0.004) for nut in SEATS}
DRIVERS = {name: UsdBody(name, usd=USD["driver"], pos=(*K.SLOTS[slot], TOP + K.DRIVER_START_Z), size=(0.03, 0.03, 0.105),
                         mass=K.DRIVER_MASS, friction=(1.0, 0.9), torsional_patch=0.010)
           for name, slot in (("driver_left", "driver_l"), ("driver_right", "driver_r"))}
PROBES = {name: UsdBody(name, usd=USD[name], pos=(*K.SLOTS[name], TOP + K.PROBE_START_Z), size=(0.016, 0.016, 0.1),
                        mass=K.PROBE_MASS, friction=(1.0, 0.9), torsional_patch=0.005)
          for name in ("probe_black", "probe_red")}
OBJECTS = [PACK, TRAY, BENCH, *BARS.values(), PLUG, *FITTED.values(), *LOOSE.values(), *DRIVERS.values(), *PROBES.values()]

# -- geometry used by the rules and the scorer ------------------------------------------------------------------------
POSTS = {name: (np.array((x, y)), pol) for name, x, y, pol in P.posts()}
MAIN = ("M1neg", "M4pos")                       # the pack's main terminals: the front posts of M1 (-) and M4 (+)
STUD_TIP_Z = TOP + P.STUD_TIP
BAR_TOP_Z = TOP + P.BAR_TOP
TRAY_FLOOR_Z = TOP + P.TRAY_FLOOR
RIM_Z = TOP + P.RIM

SEAT_XY = 0.005          # m: a driver is seated on a fitted nut when its socket's axis is within this of the stud's ...
SEAT_Z = 0.004           # ... its mouth within this above the busbar's top (on the nut's flange or past it) ...
SEAT_TILT = 10.0         # ... and it leans less than this many degrees
FREE_TURN = math.pi      # rad: half a turn counter-clockwise in total frees a nut from its thread
TURNED = math.radians(10.0)   # rad: a nut has been turned once this far
TOUCH = 0.001            # m: a probe tip touches a post when its lowest point is within this of the post's metal
NUT_R = 0.007            # m: an M8 nut's metal as a cylinder (13 mm across flats, 15 mm across corners)
READ_STEPS = 20          # a reading: both tips held on test points for 1 s
LIVE_V = 25.0            # V: a reading of at least this proves the tester
MOVED = 0.005            # m: a busbar has moved once its centre is this far from its start
CLEAR = 0.03             # m: a busbar must stay this far above the stud tip of any post it did not start on ...
NEAR = 0.02              # m: ... while any part of it, seen from above, is this close to that post's axis
PLUG_XY = 0.015          # m: the plug is in its socket within this of the socket's centre (top view) ...
IN_TRAY_LOW = 0.015      # m: a part in the tray has its lowest point within this of the tray's floor
LOST = 0.015             # m: a loose nut more than this from its own stud's axis and below the stud tips is in the pack
START = {name: np.array(spec.pos) for name, spec in BARS.items()}
OWN = {name: {p for p, (xy, _) in POSTS.items() if abs(xy[0] - START[name][0]) < 0.005
              and min(abs(xy[1] - (START[name][1] + s * P.BAR_HOLE_Y)) for s in (-1, 1)) < 0.005} for name in BARS}
assert all(len(v) == 2 for v in OWN.values()), OWN
BAR_LINKS = {name: tuple(sorted(OWN[name])) for name in BARS}
NUT_OF_BAR = {name: [n for n, (post, _) in SEATS.items() if post in OWN[name]] for name in BARS}

# -- world state (the simplified mechanics) and the scorer's tracker ---------------------------------------------------
_W: dict = {}
_T: dict = {}
_FACES: dict = {}        # each reading's face as authored (kept across resets)


@on_reset
def _reset(_env=None) -> None:
    """Every nut fitted, the display blank at 0.0, and the tracker cleared: only a reset clears a failure."""
    _W.clear()
    _W.update(fitted={n: True for n in SEATS}, turned={n: 0.0 for n in SEATS}, yaw={d: None for d in DRIVERS},
              shown=None, prims=None, step=0, freed_at={})
    _T.clear()
    _T.update(failed=None, step=0, run=0, run_kind=None, stage=0, events=[], worst={n: math.inf for n in BARS})
    if _env is not None:
        _show_nuts(_env)


_reset()


def _rotation(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _corners(state, size) -> np.ndarray:
    return np.asarray(state.pos, float) + (_SIGNS * np.asarray(size) / 2) @ _rotation(state.quat_xyzw).T


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(state) -> float:
    r = _rotation(state.quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def tilt_deg(state) -> float:
    return math.degrees(math.acos(float(np.clip(_rotation(state.quat_xyzw)[2, 2], -1.0, 1.0))))


def seated_on(driver, nut: str) -> bool:
    """The driver's socket is down over the fitted nut: axis within 5 mm of the stud's, mouth within 4 mm of the
    busbar's top, leaning less than 10 deg."""
    xy = SEATS[nut][1]
    p = np.asarray(driver.pos, float)
    return bool(np.linalg.norm(p[:2] - xy) < SEAT_XY and p[2] < BAR_TOP_Z + SEAT_Z and tilt_deg(driver) < SEAT_TILT)


def socket_over(driver, nut: str) -> bool:
    """A driver's socket is still round the nut: its axis within 15 mm of the stud's and its mouth below the nut's top
    plus 3 mm."""
    xy = SEATS[nut][1]
    p = np.asarray(driver.pos, float)
    return bool(np.linalg.norm(p[:2] - xy) < 0.015 and p[2] < TOP + P.BAR_TOP + P.NUT_H + 0.003)


# -- the meter ----------------------------------------------------------------------------------------------------------
def _post_metal(post: str) -> list[tuple[float, float, float]]:
    """A post's metal as upright cylinders on its axis: (radius, bottom, top), heights above the table top."""
    parts = [(P.PAD_R, P.LID, P.PAD_TOP), (P.STUD_R, P.PAD_TOP, P.STUD_TIP)]
    if abs(POSTS[post][0][0] - P.POST_X["front"]) < 1e-6:     # ring lug and plain nut
        parts += [(0.0085, P.PAD_TOP, P.PAD_TOP + 0.002), (NUT_R, P.PAD_TOP + 0.002, P.PAD_TOP + 0.002 + P.FRONT_NUT_H)]
    else:                                                     # a fitted flange nut on the busbar
        nut = f"nut_{post}"
        if _W["fitted"].get(nut, False):
            parts += [(P.FLANGE_R, P.BAR_TOP, P.BAR_TOP + P.FLANGE_H), (NUT_R, P.BAR_TOP + P.FLANGE_H, P.BAR_TOP + P.NUT_H)]
    return parts


def tip_post(probe) -> str | None:
    """The post a probe's tip touches: its lowest point within 1 mm of the post's metal (pad, stud, lug or nut)."""
    p = np.asarray(probe.pos, float) - np.array([0.0, 0.0, TOP])
    for name, (xy, _) in POSTS.items():
        r = float(np.linalg.norm(p[:2] - xy))
        if r > P.PAD_R + TOUCH:
            continue
        for radius, z0, z1 in _post_metal(name):
            if math.hypot(max(0.0, r - radius), max(0.0, z0 - p[2], p[2] - z1)) <= TOUCH:
                return name
    return None


def bar_on_studs(name, bar) -> bool:
    return bool(np.linalg.norm(np.asarray(bar.pos) - START[name]) <= MOVED)


def plug_in_socket(plug) -> bool:
    centred = np.linalg.norm(np.asarray(plug.pos[:2]) - P.SOCKET_CENTRE) < PLUG_XY
    return bool(centred and _corners(plug, P.PLUG_SIZE)[:, 2].min() < RIM_Z)


def potentials(objects) -> dict[str, tuple[int, float]]:
    """Each post's (group, volts) along the series string: modules are 25.6 V sources, a busbar on its own studs and a
    plug in its socket are links. Posts in different groups have no closed path between them."""
    links = [BAR_LINKS[n] for n in BARS if bar_on_studs(n, objects[n])]
    if plug_in_socket(objects["service_plug"]):
        links.append(("M2pos", "M3neg"))
    edges: dict[str, list[tuple[str, float]]] = {p: [] for p in POSTS}
    for i in range(4):                                   # each module: + is 25.6 V above -
        a, b = f"M{i + 1}neg", f"M{i + 1}pos"
        edges[a].append((b, P.MODULE_V))
        edges[b].append((a, -P.MODULE_V))
    for a, b in links:
        edges[a].append((b, 0.0))
        edges[b].append((a, 0.0))
    out: dict[str, tuple[int, float]] = {}
    for g, start in enumerate(POSTS):
        if start in out:
            continue
        out[start] = (g, 0.0)
        todo = [start]
        while todo:
            a = todo.pop()
            for b, dv in edges[a]:
                if b not in out:
                    out[b] = (g, out[a][1] + dv)
                    todo.append(b)
    return out


def meter_reading(objects) -> tuple[float, str | None, str | None]:
    """(volts shown, black tip's post, red tip's post): red minus black, 0.0 unless both tips touch posts joined by a
    closed path."""
    black, red = tip_post(objects["probe_black"]), tip_post(objects["probe_red"])
    if black is None or red is None or black == red:
        return 0.0, black, red
    pot = potentials(objects)
    (gb, vb), (gr, vr) = pot[black], pot[red]
    return (round(vr - vb, 1) if gb == gr else 0.0), black, red


# -- every control step: the nuts' threads, the display and the leads -------------------------------------------------
def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _prims() -> dict:
    """Env 0's prims that the rules show, hide or redraw."""
    if _W.get("prims"):
        return _W["prims"]
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    want = {*(f"{s}_fitted" for s in SEATS), *SEATS}
    found: dict = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0")):
        path, name = str(prim.GetPath()), prim.GetName()
        if name in want and (name not in found or len(path) < len(str(found[name].GetPath()))):
            found[name] = prim
        elif "/bench_kit/" in path and (name.startswith("read_") or name.startswith("lead_")) and prim.GetTypeName() == "Mesh":
            found[name] = prim
    _W["prims"] = found
    return found


def _visible(prim, show: bool) -> None:
    from pxr import UsdGeom

    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if show else img.MakeInvisible()


def _show_nuts(env) -> None:
    try:
        prims = _prims()
        for n in SEATS:
            _visible(prims[f"{n}_fitted"], _W["fitted"][n])
            _visible(prims[n], not _W["fitted"][n])
    except Exception as exc:  # pictures only: never stop a trial
        print(f"nuts: {exc!r}")


def _place(env, name: str, pos, quat=(0.0, 0.0, 0.0, 1.0)) -> None:
    import torch

    base = env.unwrapped
    asset = base.scene[name]
    origin = _np(base.scene.env_origins)[0]
    pose = torch.tensor([[pos[0] + origin[0], pos[1] + origin[1], pos[2] + origin[2], *quat]], dtype=torch.float32,
                        device=asset.device)
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=asset.device))


def _free(env, nut: str) -> None:
    """The nut comes off its thread: the fitted one goes, a loose one takes its place on the stud."""
    xy = SEATS[nut][1]
    _place(env, f"{nut}_fitted", (PARK[nut][0], PARK[nut][1] - 0.1, PARK[nut][2] + 0.02))
    _place(env, nut, (xy[0], xy[1], TOP + P.NUT_Z + 0.0003))
    _W["fitted"][nut] = False
    _W["freed_at"][nut] = _W["step"]
    _show_nuts(env)


@on_step
def _world(env) -> None:
    from yam_arena.scene import object_states

    _W["step"] += 1
    if os.environ.get("TASK_SCORE_DIR"):
        from yam_arena.runtime import tcp_world

        base = env.unwrapped
        origin = _np(base.scene.env_origins)[0]
        _W["tcp"] = {s: [round(float(v), 4) for v in tcp_world(base.scene[f"{s}_robot"]) - origin]
                     + [round(math.degrees(float(scenekit.arm_joints(env, s, ["joint6"])[0])), 1)] for s in ("left", "right")}
    names = [*DRIVERS, *PROBES, *BARS, "service_plug"]
    st = object_states(env, names)
    if _W["step"] == 1:
        _show_nuts(env)
    # Threads: a seated driver turns its nut; half a turn counter-clockwise frees it.
    for d in DRIVERS:
        yaw = yaw_of(st[d])
        prev, _W["yaw"][d] = _W["yaw"][d], yaw
        if prev is None:
            continue
        for n in SEATS:
            if _W["fitted"][n] and _W["turned"][n] < FREE_TURN and seated_on(st[d], n):
                _W["turned"][n] = min(FREE_TURN, max(0.0, _W["turned"][n] + _wrap(yaw - prev)))
    # A nut turned half a turn is off its thread; it becomes a free body once no socket is over it.
    for n in SEATS:
        if _W["fitted"][n] and _W["turned"][n] >= FREE_TURN and not any(socket_over(st[d], n) for d in DRIVERS):
            _free(env, n)
    # Display and leads.
    try:
        prims = _prims()
        v, _, _ = meter_reading(st)
        shown = min(K.READINGS, key=lambda r: abs(r - v))
        from pxr import UsdGeom, Vt

        if shown != _W["shown"]:     # bring one reading's face to the display; sink the others 6 mm into the meter
            sink = (K.meter_point((0.0, 0.0, -K.READING_SINK)) - K.meter_point((0.0, 0.0, 0.0))).astype(np.float32)
            for r in K.READINGS:
                attr = UsdGeom.Mesh(prims[K.reading_prim(r)]).GetPointsAttr()
                key = K.reading_prim(r)
                if key not in _FACES:     # the face where it shows (authored there for 0.0, sunk for the others)
                    _FACES[key] = np.array(attr.Get(), dtype=np.float32) - (sink if r != 0.0 else 0.0)
                pts = _FACES[key] if r == shown else _FACES[key] + sink
                attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts)))
            _W["shown"] = shown

        for colour in ("black", "red"):
            s = st[f"probe_{colour}"]
            r = _rotation(s.quat_xyzw)
            axis = r[:, 2]
            top = np.asarray(s.pos, float) - np.array([0.0, 0.0, TOP]) + K.PROBE_TOP * axis
            mesh = K.lead_mesh(K.lead_path(colour, top, axis))
            UsdGeom.Mesh(prims[f"lead_{colour}"]).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.v.astype(np.float32)))
    except Exception as exc:  # pictures only: never stop a trial
        if _W["step"] < 3:
            print(f"display: {exc!r}")


# -- scoring ------------------------------------------------------------------------------------------------------------
def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, np.round(points, 9)))

    def half(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2 and np.cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def _distance_to_outline(p: np.ndarray, poly: np.ndarray) -> float:
    """Distance from a point to a convex polygon in the plane; 0 inside."""
    n = len(poly)
    if all(np.cross(poly[(i + 1) % n] - poly[i], p - poly[i]) >= 0 for i in range(n)):
        return 0.0
    best = math.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def clearance_over_other_posts(name, bar) -> float:
    """Height of the bar's lowest point above the stud tips of the posts it did not start on that lie within NEAR of
    its outline seen from above; infinity if none is that close."""
    corners = _corners(bar, P.BAR_SIZE)
    outline = _hull(corners[:, :2])
    low = corners[:, 2].min()
    near = [p for p, (xy, _) in POSTS.items() if p not in OWN[name] and _distance_to_outline(xy, outline) < NEAR]
    return float(low - STUD_TIP_Z) if near else math.inf


def in_tray(state, half_height: float) -> bool:
    dx, dy = np.abs(np.asarray(state.pos[:2]) - P.TRAY_CENTRE)
    inside = dx < P.TRAY_INNER[0] / 2 and dy < P.TRAY_INNER[1] / 2
    return bool(inside and state.pos[2] - half_height < TRAY_FLOOR_Z + IN_TRAY_LOW)


def bar_in_tray(bar) -> bool:
    dx, dy = np.abs(np.asarray(bar.pos[:2]) - P.TRAY_CENTRE)
    inside = dx < P.TRAY_INNER[0] / 2 and dy < P.TRAY_INNER[1] / 2
    return bool(inside and _corners(bar, P.BAR_SIZE)[:, 2].min() < TRAY_FLOOR_Z + IN_TRAY_LOW)


def nut_lost_in_pack(nut: str, state) -> bool:
    p = np.asarray(state.pos, float)
    in_outline = P.HOUSING_X[0] < p[0] < P.HOUSING_X[1] and abs(p[1]) < P.HOUSING_HALF_Y
    return bool(in_outline and np.linalg.norm(p[:2] - SEATS[nut][1]) > LOST and p[2] < STUD_TIP_Z)


def _update(objects) -> None:
    """Called every control step: the readings, the verification and the latched failures."""
    if _T["failed"] or _T.get("seen") == _W["step"]:
        return
    _T["seen"] = _W["step"]                              # once per control step, however often the harness asks
    _T["step"] += 1
    plug_in = plug_in_socket(objects["service_plug"])
    # Readings: both tips on test points for 1 s. Live: 25 V or more. Dead: on the main terminals, showing 0 V.
    v, black, red = meter_reading(objects)
    kind = None
    if black and red and black != red:
        if abs(v) >= LIVE_V:
            kind = "live"
        elif {black, red} == set(MAIN) and not plug_in:
            kind = "dead"
    _T["run"] = _T["run"] + 1 if kind and kind == _T["run_kind"] else (1 if kind else 0)
    _T["run_kind"] = kind
    if kind and _T["run"] == READ_STEPS:
        _T["events"].append((_T["step"], kind))
        if kind == "live" and _T["stage"] in (0, 2):
            _T["stage"] += 1
        elif kind == "dead" and _T["stage"] == 1:
            _T["stage"] = 2
    if plug_in and _T["stage"] >= 2:                     # a refitted plug voids the dead test
        _T["stage"] = 1
    turned = any(t >= TURNED for t in _W["turned"].values()) or any(not f for f in _W["fitted"].values())
    moved = any(not bar_on_studs(n, objects[n]) for n in BARS)
    if (turned or moved) and (plug_in or _T["stage"] < 3):
        what = "a nut was turned" if turned else "a busbar moved"
        _T["failed"] = f"order: {what} " + ("with the plug in its socket" if plug_in else "before the pack was verified dead")
        return
    for n in BARS:
        if not bar_on_studs(n, objects[n]) and any(_W["fitted"][k] for k in NUT_OF_BAR[n]):
            _T["failed"] = f"forced: {n} moved off its studs with a nut still fitted"
            return
    for n in BARS:
        c = clearance_over_other_posts(n, objects[n])
        _T["worst"][n] = min(_T["worst"][n], c)
        if c < CLEAR:
            _T["failed"] = f"short: {n} {c * 100:.1f} cm above another post's stud tip"
            return
    for n in SEATS:
        if not _W["fitted"][n] and nut_lost_in_pack(n, objects[n]):
            _T["failed"] = f"short: loose {n} in the pack"
            return
    loose = [n for n in SEATS if not _W["fitted"][n]]
    for n in (*BARS, "service_plug", *DRIVERS, *PROBES, *loose):
        if objects[n].pos[2] < TOP:
            _T["failed"] = f"dropped: {n} fell off the table"
            return


def verified() -> bool:
    return _T["stage"] >= 3


def score(objects) -> float:
    """0 for the whole trial after a failure; 0 with the plug in its socket at the end. Otherwise 0.1, plus 0.2 once the
    pack was verified dead (prove, test, prove), 0.05 for each nut and 0.25 for each busbar in the tray."""
    _update(objects)
    if _T["failed"] or plug_in_socket(objects["service_plug"]):
        return 0.0
    nuts = sum(in_tray(objects[n], P.NUT_H / 2) for n in SEATS if not _W["fitted"][n])
    bars = sum(bar_in_tray(objects[n]) for n in BARS)
    return round(0.1 + 0.2 * verified() + 0.05 * nuts + 0.25 * bars, 3)


def success(objects) -> bool:
    """The 1.0 state with the plug, both busbars and the four nuts still (under 1 cm/s and 0.5 rad/s); the harness holds
    it 1 s."""
    steptime.tick()  # the harness calls success() once per control step
    s = score(objects)
    still = all(objects[n].resting() for n in (*BARS, "service_plug", *SEATS))
    _log(objects, s)
    return s == 1.0 and still


def _log(objects, s: float) -> None:
    out = os.environ.get("TASK_SCORE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    v, black, red = meter_reading(objects)
    r3 = lambda a: [round(float(x), 4) for x in a]  # noqa: E731
    row = {"step": _W["step"], "tcp": _W.get("tcp"), "score": s, "failed": _T["failed"], "stage": _T["stage"], "plug_in": plug_in_socket(objects["service_plug"]),
           "meter": v, "tips": [black, red], "turned": {n: round(math.degrees(t)) for n, t in _W["turned"].items()},
           "fitted": [n for n in SEATS if _W["fitted"][n]],
           **{n: r3(objects[n].pos) for n in (*BARS, "service_plug")},
           **{n: r3(objects[n].pos) + [round(tilt_deg(objects[n]), 1)] for n in PROBES},
           **{n: r3(objects[n].pos) + [round(math.degrees(yaw_of(objects[n])), 1), round(tilt_deg(objects[n]), 1)] for n in DRIVERS},
           **{n: r3(objects[n].pos) for n in SEATS if not _W["fitted"][n]},
           "clearance": {n: (None if math.isinf(c) else round(c, 4)) for n, c in
                         ((n, clearance_over_other_posts(n, objects[n])) for n in BARS)}}
    with open(os.path.join(out, "steps.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# -- scripted demo ------------------------------------------------------------------------------------------------------
HALF_OPEN = -0.025                 # finger joint target: pads 50 mm apart
YAW = {"left": -math.pi / 2, "right": math.pi / 2}   # fingers close along x
PLUG_GRASP = 0.063                 # TCP height: fingertips 7 mm above the socket rim
BAR_GRASP = 0.060                  # TCP height: fingertips 2 mm below the bar's underside, 4 mm above the lid
CARRY = 0.13                       # TCP height while a bar crosses the front posts: its underside about 12 cm up
LOW_CARRY = 0.081                  # lowcarry variant: the bar's underside about 6 mm above the front stud tips
PLUG_PARK = (0.10, -0.16)
PROBE_ON = P.STUD_TIP + K.PROBE_GRIP_Z - 0.0015     # TCP with the probe's tip pressed on a stud's tip
DRIVER_SEAT = P.BAR_TOP + K.GRIP_Z + 0.001          # TCP with the driver's mouth just over the nut's flange
DRIVER_PICK = K.WELL_FLOOR + K.GRIP_Z                # TCP gripping a driver standing in its stand's well
NUT_GRASP = P.BAR_TOP + 0.0035 + 0.013              # TCP: pad tips 3.5 mm above the busbar
BAR_DROP = {"busbar_a": (0.168, 0.068), "busbar_b": (0.168, -0.068)}
NUT_DROP = {"nut_M1pos": (0.138, 0.115), "nut_M2neg": (0.138, 0.08), "nut_M3pos": (0.138, -0.08), "nut_M4neg": (0.138, -0.115)}
# Each nut's counter-clockwise sweep (yaw from, to; degrees), inside the wrist's +-120 deg.
SWEEP = {"nut_M1pos": (-120, 90), "nut_M2neg": (-125, 80), "nut_M3pos": (-80, 125), "nut_M4neg": (-90, 120)}
SIDE_NUTS = {"left": ("nut_M1pos", "nut_M2neg"), "right": ("nut_M4neg", "nut_M3pos")}


def _wp(t, xy, h, finger, label, yaw) -> Waypoint:
    return Waypoint(t, (float(xy[0]), float(xy[1]), TOP + h), finger, label, float(yaw))


class Plan:
    """One arm's waypoints, with the time they add up to (to line the two arms up)."""

    def __init__(self, side):
        self.side, self.w, self.t, self.yaw = side, [], 0.0, YAW[side]
        self.xy, self.h, self.finger = None, 0.26, HALF_OPEN

    def go(self, t, xy=None, h=None, finger=None, label="", yaw=None):
        self.xy = self.xy if xy is None else xy
        self.h = self.h if h is None else h
        self.finger = self.finger if finger is None else finger
        self.yaw = self.yaw if yaw is None else yaw
        self.w.append(_wp(t, self.xy, self.h, self.finger, label, self.yaw))
        self.t += t
        return self

    def until(self, t, label="wait"):
        if t > self.t:
            self.go(t - self.t, label=label)
        return self


def _pick(p: Plan, xy, grasp, label, lift, yaw=None):
    p.go(2.0, xy, lift, HALF_OPEN, f"above the {label}", YAW[p.side] if yaw is None else yaw)
    p.go(1.4, h=grasp, label=f"down to the {label}")
    p.go(0.6, finger=FINGER_CLOSED, label=f"grip the {label}")
    p.go(1.2, h=lift, label=f"lift the {label}")


def _put(p: Plan, xy, h, label, up=0.13):
    p.go(2.0, xy, up, label=f"over the {label}")
    p.go(1.0, h=h + 0.02, label=f"down toward the {label}")
    p.go(0.6, label="settle")
    p.go(1.0, h=h, label=f"down into the {label}")
    p.go(0.5, finger=HALF_OPEN, label="let go")
    p.go(0.8, h=up, label="up")


def _pull_plug(p: Plan):
    s = P.SOCKET_CENTRE
    p.go(2.0, s, 0.12, HALF_OPEN, "above the plug", YAW[p.side])
    p.go(1.5, h=PLUG_GRASP, label="down to the plug")
    p.go(0.8, finger=FINGER_CLOSED, label="grip the plug")
    p.go(1.5, h=0.11, label="pull straight up")
    p.go(2.0, PLUG_PARK, label="carry the plug aside")
    p.go(1.2, h=0.044, label="set it on the bench")
    p.go(0.6, finger=HALF_OPEN, label="let go")
    p.go(1.0, h=0.12, label="up")


def _probe_on(p: Plan, post, hold, label):
    xy = POSTS[post][0]
    p.go(1.8, xy, PROBE_ON + 0.035, label=f"over {post}")
    p.go(1.0, h=PROBE_ON, label=f"tip on {post}")
    if hold:
        p.go(hold, label=label)


def _wrist(p: Plan, yaw, t=2.4, label="turn the wrist"):
    """Turn the wrist in two halves, so that each half takes the intended way round."""
    mid = (p.yaw + yaw) / 2
    p.go(t / 2, yaw=mid, label=label)
    p.go(t / 2, yaw=yaw, label=label)


def _unbolt(p: Plan, nut):
    xy = SEATS[nut][1]
    a, b = SWEEP[nut]
    _wrist(p, math.radians(a), label="turn the wrist back")
    p.go(1.8, xy, DRIVER_SEAT + 0.03, label=f"over {nut}")
    p.go(0.8, h=DRIVER_SEAT + 0.012, label="socket just above the nut")
    p.go(0.6, label="settle")
    p.go(0.8, h=DRIVER_SEAT, label=f"socket on {nut}")
    p.go(1.3, yaw=math.radians((a + b) / 2), label="turn")
    p.go(1.3, yaw=math.radians(b), label="turn")
    p.go(0.8, h=DRIVER_SEAT + 0.035, label="lift off")


LIVE = ("M1pos", "M2pos", "prove the tester on M2 (25.6 V)")    # M1+ at the back is joined to M2- by bar A
DEAD = ("M1neg", "M4pos", "test the main terminals (0 V)")
LIVE_AGAIN = ("M1pos", "M2pos", "prove the tester again")


def _probes_out(L: Plan, R: Plan):
    _pick(L, K.SLOTS["probe_black"], K.PROBE_START_Z + K.PROBE_GRIP_Z, "black probe", 0.13)
    _pick(R, K.SLOTS["probe_red"], K.PROBE_START_Z + K.PROBE_GRIP_Z, "red probe", 0.13)


def _readings(L: Plan, R: Plan, pairs):
    """Black probe (left arm) and red probe (right arm) on each pair of posts together, held 1.6 s."""
    for black, red, label in pairs:
        _probe_on(L, black, 0.0, "")
        _probe_on(R, red, 0.0, "")
        t = max(L.t, R.t)
        for p in (L, R):
            p.until(t)
            p.go(1.6, label=label)
            p.go(0.6, h=PROBE_ON + 0.035, label="up")


def _probe_back(p: Plan, slot: str):
    _put(p, K.SLOTS[slot], K.PROBE_START_Z + K.PROBE_GRIP_Z + 0.002, "probe's stand")


def _unbolt_all(L: Plan, R: Plan, inner=True):
    """Each arm takes its driver, turns its outer nut, then its inner nut (left first: the arms would meet in the
    middle), and stands the driver back in its stand."""
    t = max(L.t, R.t)
    for p, slot in ((L, "driver_l"), (R, "driver_r")):
        p.until(t)
        p.go(2.0, K.SLOTS[slot], DRIVER_PICK + 0.085, HALF_OPEN, "above the driver")
        _wrist(p, math.radians(SWEEP[SIDE_NUTS[p.side][0]][0]), t=2.4, label="wrist to the sweep's start")
        p.go(1.4, h=DRIVER_PICK, label="down to the driver")
        p.go(0.6, finger=FINGER_CLOSED, label="grip the driver")
        p.go(1.2, h=DRIVER_PICK + 0.085, label="lift the driver")
        _unbolt(p, SIDE_NUTS[p.side][0])
    if inner:
        _unbolt(L, SIDE_NUTS["left"][1])
        R.until(L.t - 1.0, "wait for the left arm to leave the middle")
        _unbolt(R, SIDE_NUTS["right"][1])
    for p, slot in ((L, "driver_l"), (R, "driver_r")):
        _wrist(p, YAW[p.side], label="wrist back")
        _put(p, K.SLOTS[slot], DRIVER_PICK + 0.001, "driver's stand", up=DRIVER_PICK + 0.085)


def _nuts_to_tray(L: Plan, R: Plan, drop_one_in_pack=False):
    for p in (L, R):
        for nut in SIDE_NUTS[p.side]:
            _pick(p, SEATS[nut][1], NUT_GRASP, nut, 0.10)
            if drop_one_in_pack and nut == "nut_M1pos":     # set it down on module M1's lid, between its posts
                _put(p, (0.30, 0.145), 0.065, "lid of M1", up=0.10)
            else:
                _put(p, NUT_DROP[nut], 0.036, "tray")


def _bars_to_tray(L: Plan, R: Plan, low=False):
    """The left bar first; the right arm lifts its bar meanwhile and waits for the left arm to leave the tray (the arms
    would meet over it). low: the right bar crosses the front posts 6 mm above their stud tips."""
    _pick(L, P.BARS["busbar_a"], BAR_GRASP, "busbar_a", CARRY)
    _pick(R, P.BARS["busbar_b"], BAR_GRASP, "busbar_b", CARRY if not low else 0.10)
    _put(L, BAR_DROP["busbar_a"], 0.019, "tray", up=CARRY)
    L.go(1.5, (0.25, 0.25), 0.15, label="clear of the tray")
    R.until(L.t + 0.5, "hold the bar over its own studs")
    _put(R, BAR_DROP["busbar_b"], 0.019, "tray", up=CARRY if not low else LOW_CARRY)
    R.go(1.5, (0.25, -0.25), 0.15, label="clear of the tray")


def _refit(p: Plan):
    """Take the plug from the bench and push it back into its socket."""
    s = P.SOCKET_CENTRE
    _pick(p, PLUG_PARK, 0.044, "plug", 0.11)
    p.go(2.0, s, 0.11, label="over the socket")
    p.go(1.5, h=PLUG_GRASP + 0.001, label="push it back in")
    p.go(0.6, finger=HALF_OPEN, label="let go")
    p.go(1.0, h=0.12, label="up")


def _rip(p: Plan, bar: str):
    """Grip a busbar whose nuts are still fitted and pull it straight up hard, then try to slide it."""
    x, y = P.BARS[bar]
    p.go(2.0, (x, y), CARRY, HALF_OPEN, f"above {bar}", YAW[p.side])
    p.go(1.4, h=BAR_GRASP, label=f"down to {bar}")
    p.go(0.6, finger=FINGER_CLOSED, label=f"grip {bar}")
    p.go(1.5, h=0.16, label="pull straight up")
    p.go(1.5, label="keep pulling")
    p.go(1.2, h=BAR_GRASP, label="back down")
    p.go(1.5, (x - 0.02, y), label="slide toward the robot")
    p.go(1.5, (x, y + 0.02), label="slide along the bar")
    p.go(1.5, (x, y), 0.16, label="pull straight up again")
    p.go(0.6, finger=HALF_OPEN, label="let go")
    p.go(1.0, h=CARRY + 0.03, label="up")


def demo():
    """Open loop, so it uses nothing a policy could not see. The right arm pulls the plug and sets it on the bench. The
    probes prove the tester across module M2 (black on M1+, red on M2+: 25.6 V), test the main terminals (black on M1-,
    red on M4+: 0 V) and prove it again. Each arm then unbolts its busbar's two nuts with its insulated driver (half a
    turn each), stands the driver back, lifts the two loose nuts off their studs into the tray, and carries its busbar
    high over the front posts into the tray. TASK_DEMO picks a variant for checking the scorer (see README.md)."""
    mode = os.environ.get("TASK_DEMO", "success")
    L, R = Plan("left"), Plan("right")
    done = lambda: {side: p.w for side, p in (("left", L), ("right", R)) if p.w}  # noqa: E731
    if mode == "nothing":
        L.go(2.0, (0.25, 0.25), 0.15, HALF_OPEN, "hover", YAW["left"])
        L.go(8.0, label="wait")
        return {"left": L.w}
    if mode == "failure":                       # the literal reading: unbolt straight away
        _pick(L, K.SLOTS["driver_l"], DRIVER_PICK, "driver", DRIVER_PICK + 0.085, yaw=math.radians(SWEEP["nut_M1pos"][0]))
        _unbolt(L, "nut_M1pos")
        return {"left": L.w}
    if mode == "livefirst":                     # test the main terminals with the plug still in (102.4 V: live), then isolate
        _probes_out(L, R)
        _readings(L, R, [("M1neg", "M4pos", "the main terminals, plug in (102.4 V)")])
        _probe_back(R, "probe_red")
        _pull_plug(R)
        _pick(R, K.SLOTS["probe_red"], K.PROBE_START_Z + K.PROBE_GRIP_Z, "red probe", 0.13)
        L.until(R.t, "hold the black probe")
        _readings(L, R, [DEAD, LIVE_AGAIN])
    else:
        _pull_plug(R)
        if mode == "plugonly":
            return done()
        if mode == "notest":                    # isolated, but not verified
            L.go(2.0, (0.25, 0.25), 0.15, HALF_OPEN, "wait clear of the pack", YAW["left"])
            L.until(R.t)
            _unbolt_all(L, R, inner=False)
            return done()
        _probes_out(L, R)
        pairs = {"deadonly": [DEAD], "noafter": [LIVE, DEAD]}.get(mode, [LIVE, DEAD, LIVE_AGAIN])
        _readings(L, R, pairs)
    _probe_back(L, "probe_black")
    _probe_back(R, "probe_red")
    if mode == "verified":
        return done()
    if mode == "rip":
        _rip(L, "busbar_a")
        return done()
    if mode == "refit":
        _refit(R)
    if mode in ("deadonly", "noafter", "refit"):
        _unbolt_all(L, R, inner=False)
        return done()
    _unbolt_all(L, R)
    _nuts_to_tray(L, R, drop_one_in_pack=mode == "nutdrop")
    _bars_to_tray(L, R, low=mode == "lowcarry")
    return done()
