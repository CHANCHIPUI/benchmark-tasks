"""The bench's tools: two insulated nut drivers, the voltage tester (a multimeter with two probes on leads) and the
rubber tool stands. Layout and USD files; positions in metres in the station frame (x forward, y left), heights above
the table top.

- Nut driver: a 1000 V insulated stubby nut driver, 13 mm socket (IEC 60900 style: red insulation over a yellow inner
  layer). Its origin is the centre of the socket's mouth, its axis +z.
- Probe: a CAT III test probe with a finger guard and a shrouded 6 mm tip. Its origin is the tip's lowest point, its
  axis +z.
- Bench kit (fixed): the multimeter leaning on its stand facing the robot, its display, and two tool stands, each
  holding one driver and one probe upright. The two leads are meshes in this body whose points task.py redraws every
  control step, from the meter's jacks to the probes; they have no colliders.
"""

from __future__ import annotations

import numpy as np

import pack_usd as P
import usdkit as U

BUILD = P.BUILD

# -- driver -----------------------------------------------------------------------------------------------------------
DRIVER_MASS = 0.08
SOCKET_BORE = 0.0100        # radius of the bore's walls (colliders): clear of a fitted nut's collider
SOCKET_DEPTH = 0.014        # the bore's depth: room for the stud's end
SLEEVE_R = 0.011
GRIP_Z = 0.070              # the TCP sits this far above the socket's mouth in a grip on the handle
DRIVER_TOP = 0.105

# -- probe ------------------------------------------------------------------------------------------------------------
PROBE_MASS = 0.03
PROBE_GRIP_Z = 0.050        # the TCP sits this far above the tip in a grip on the handle
PROBE_TOP = 0.100

# -- stands -----------------------------------------------------------------------------------------------------------
STAND_TOP = 0.015           # a driver stand's top: the driver stands on its mouth in a shallow square pocket
WELL = 0.048                # the pocket's side
WELL_FLOOR = 0.005
PROBE_STAND_TOP = 0.025     # a probe stand's top: the probe's guard rests on it, its tip down a hole
HOLE = 0.014
HOLE_FLOOR = 0.005
DRIVER_STAND = 0.058        # square block sides
PROBE_STAND = 0.032
SLOTS = {"driver_l": (0.29, 0.28), "probe_black": (0.17, 0.235),
         "driver_r": (0.29, -0.28), "probe_red": (0.17, -0.235)}
DRIVER_START_Z = WELL_FLOOR
PROBE_START_Z = PROBE_STAND_TOP - 0.020    # the guard's underside is 20 mm above the tip

# -- meter ------------------------------------------------------------------------------------------------------------
METER_TILT = 32.0           # degrees from horizontal, face toward the top camera
METER_L, METER_W, METER_T = 0.125, 0.085, 0.034
METER_FOOT = (-0.020, 0.0)  # where the meter's lower back edge meets the table
JACK_LOCAL = {"black": (-0.048, 0.016), "red": (-0.048, -0.016)}   # on the face: (along the meter, across)
DISPLAY_LOCAL = (0.033, 0.0)
DISPLAY_SIZE = (0.068, 0.034)   # across, along
READINGS = (0.0, 25.6, -25.6, 51.2, -51.2, 76.8, -76.8, 102.4, -102.4)
READING_SINK = 0.006        # a reading not shown sits this far inside the meter's face

LEAD_R, LEAD_N, LEAD_SIDES = 0.0022, 40, 8

LOOKS = {
    "ins_red": U.Look(color=(0.62, 0.03, 0.02), roughness=0.45),
    "ins_yellow": U.Look(color=(0.95, 0.70, 0.02), roughness=0.45),
    "socket_steel": U.Look(color=(0.62, 0.63, 0.65), metallic=1.0, roughness=0.3),
    "bore": U.Look(color=(0.02, 0.02, 0.02), roughness=0.6),
    "probe_red": U.Look(color=(0.60, 0.03, 0.03), roughness=0.4),
    "probe_black": U.Look(color=(0.015, 0.015, 0.017), roughness=0.4),
    "tip_steel": U.Look(color=(0.70, 0.70, 0.70), metallic=1.0, roughness=0.25),
    "rubber": U.Look(color=(0.05, 0.055, 0.06), roughness=0.85),
    "hole": U.Look(color=(0.005, 0.005, 0.005), roughness=0.9),
    "holster": U.Look(color=(0.85, 0.55, 0.02), roughness=0.7),
    "panel": U.Look(color=(0.05, 0.05, 0.055), roughness=0.5),
    "lcd": U.Look(color=(0.70, 0.75, 0.66), roughness=0.4, emissive=(0.08, 0.085, 0.075)),
    "dial": U.Look(color=(0.02, 0.02, 0.02), roughness=0.5),
    "mark_white": U.Look(color=(0.9, 0.9, 0.9), roughness=0.5),
    "jack_red": U.Look(color=(0.6, 0.02, 0.02), roughness=0.4),
    "jack_black": U.Look(color=(0.01, 0.01, 0.01), roughness=0.4),
    "bail": U.Look(color=(0.15, 0.15, 0.16), roughness=0.6),
    "lead_red": U.Look(color=(0.55, 0.02, 0.02), roughness=0.5),
    "lead_black": U.Look(color=(0.02, 0.02, 0.022), roughness=0.5),
}
def _lcd_look(v: float) -> str:
    return "lcd_" + f"{v:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


for _r in READINGS:
    LOOKS[_lcd_look(_r)] = U.Look(color=(1.0, 1.0, 1.0), roughness=0.4, texture=f"lcd_{_r:+.1f}.png", tile=1.0)


def reading_prim(v: float) -> str:
    """The display prim that shows reading v."""
    return "read_" + f"{v:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def _crossed(name, half, z0, z1, root) -> list[str]:
    """Two crossed square colliders (0 and 45 deg): a round part's outline to within 4 %."""
    size = (2 * half, 2 * half, z1 - z0)
    c = (0.0, 0.0, (z0 + z1) / 2)
    return [U.collider(f"{name}_a", size, c, root), U.collider(f"{name}_b", size, c, root, yaw_deg=45.0)]


def _driver() -> str:
    root = "InsulatedNutDriver"
    r = SLEEVE_R
    sleeve = U.lathe([(0.0068, 0.0), (0.0095, 0.0), (r, 0.0015), (r, 0.027), (0.0085, 0.031), (0.0072, 0.033), (0.0, 0.033)], 32)
    shaft = U.cylinder(0.0072, 0.015, (0, 0, 0.0385), segments=24)
    guard = U.lathe([(0.0, 0.045), (0.0145, 0.045), (0.015, 0.047), (0.015, 0.049), (0.0145, 0.050), (0.0, 0.050)], 32)
    handle = U.lathe([(0.0, 0.050), (0.0125, 0.050), (0.0138, 0.056), (0.0142, 0.075), (0.0138, 0.095), (0.0125, 0.103),
                      (0.009, 0.105), (0.0, 0.105)], 32)
    band = U.lathe([(0.0, 0.0575), (0.0141, 0.0575), (0.0143, 0.060), (0.0141, 0.0625), (0.0, 0.0625)], 32)
    ring = U.lathe([(0.0066, 0.0), (0.0094, 0.0), (0.0094, 0.0016), (0.0066, 0.0016)], 32)
    hexr = 0.0075
    bore = U.prism([(hexr * np.cos(np.pi / 6 + k * np.pi / 3), hexr * np.sin(np.pi / 6 + k * np.pi / 3)) for k in range(6)],
                   0.0004, 0.002)
    parts = [("sleeve", sleeve, "ins_red"), ("shaft", shaft, "ins_red"), ("guard", guard, "ins_yellow"),
             ("handle", handle, "ins_red"), ("band", band, "ins_yellow"), ("mouth", ring, "socket_steel"), ("bore", bore, "bore")]
    for k in range(6):   # grip ribs along the handle
        a = k * np.pi / 3
        rib = U.box((0.0016, 0.004, 0.030), (0, 0, 0)).turned(np.degrees(a)).moved((0.0141 * np.cos(a), 0.0141 * np.sin(a), 0.080))
        parts.append((f"rib{k}", rib, "ins_yellow"))
    cols = []
    rc, wall = SOCKET_BORE + 0.00125, 0.0025
    length = 2 * (SOCKET_BORE + wall) * np.tan(np.pi / 8)
    for k in range(8):
        phi = k * 45.0
        c = (rc * np.cos(np.radians(phi)), rc * np.sin(np.radians(phi)), SOCKET_DEPTH / 2)
        cols.append(U.collider(f"c_bore{k}", (wall, length, SOCKET_DEPTH), c, root, yaw_deg=phi))
    cols += _crossed("c_sleeve", 0.0085, SOCKET_DEPTH, 0.031, root)
    cols.append(U.collider("c_shaft", (0.012, 0.012, 0.015), (0, 0, 0.0385), root))
    cols += _crossed("c_guard", 0.0113, 0.045, 0.050, root)
    cols += _crossed("c_handle", 0.0105, 0.050, DRIVER_TOP, root)
    U.write_usda(str(BUILD / "driver.usda"), root, parts, LOOKS, cols, mass=DRIVER_MASS, friction=(1.0, 0.9))
    return "build/driver.usda"


def _probe(colour: str) -> str:
    root = f"TestProbe{colour.title()}"
    look = f"probe_{colour}"
    tip = U.lathe([(0.0, 0.0), (0.0006, 0.0002), (0.001, 0.001), (0.001, 0.0065), (0.0, 0.0065)], 16)
    shroud = U.lathe([(0.0, 0.006), (0.0028, 0.006), (0.0035, 0.008), (0.0035, 0.020), (0.0, 0.020)], 24)
    guard = U.lathe([(0.0, 0.020), (0.0075, 0.020), (0.008, 0.021), (0.008, 0.023), (0.0075, 0.024), (0.0, 0.024)], 32)
    handle = U.lathe([(0.0, 0.024), (0.0050, 0.024), (0.0055, 0.028), (0.0055, 0.080), (0.0050, 0.085), (0.0, 0.085)], 24)
    boot = U.lathe([(0.0, 0.085), (0.0045, 0.085), (0.0040, 0.093), (0.0030, PROBE_TOP), (0.0, PROBE_TOP)], 20)
    parts = [("tip", tip, "tip_steel"), ("shroud", shroud, look), ("guard", guard, look), ("handle", handle, look),
             ("boot", boot, look)]
    for k in range(4):   # grip rings
        z = 0.034 + 0.008 * k
        parts.append((f"ring{k}", U.lathe([(0.0, z), (0.0058, z), (0.0058, z + 0.003), (0.0, z + 0.003)], 24), look))
    cols = [U.collider("c_tip", (0.002, 0.002, 0.0065), (0, 0, 0.00325), root),
            U.collider("c_shroud", (0.006, 0.006, 0.014), (0, 0, 0.013), root)]
    cols += _crossed("c_guard", 0.0060, 0.020, 0.024, root)
    cols += _crossed("c_handle", 0.0041, 0.024, 0.085, root)
    cols.append(U.collider("c_boot", (0.007, 0.007, PROBE_TOP - 0.085), (0, 0, (0.085 + PROBE_TOP) / 2), root))
    U.write_usda(str(BUILD / f"probe_{colour}.usda"), root, parts, LOOKS, cols, mass=PROBE_MASS, friction=(1.0, 0.9))
    return f"build/probe_{colour}.usda"


# -- the meter's frame --------------------------------------------------------------------------------------------------
def _meter_rot() -> np.ndarray:
    """Meter frame to station frame: x up the face (away from the robot), y across, z out of the face."""
    t = np.radians(METER_TILT)
    return np.array([[np.cos(t), 0.0, -np.sin(t)], [0.0, 1.0, 0.0], [np.sin(t), 0.0, np.cos(t)]])


def _meter_origin() -> np.ndarray:
    """The centre of the meter's face in the station frame (heights above the table)."""
    t = np.radians(METER_TILT)
    # the body's lower back edge (face frame (-L/2, y, -T)) rests on the table at METER_FOOT
    return np.array([METER_FOOT[0] + METER_L / 2 * np.cos(t) - METER_T * np.sin(t), METER_FOOT[1],
                     METER_L / 2 * np.sin(t) + METER_T * np.cos(t)])


def meter_point(local) -> np.ndarray:
    """A point given in the meter's frame (x up the face, y across, z out of it), in the station frame."""
    return _meter_origin() + _meter_rot() @ np.asarray(local, float)


def jack(colour: str) -> np.ndarray:
    a, b = JACK_LOCAL[colour]
    return meter_point((a, b, 0.004))


def _to_meter(mesh: U.Mesh) -> U.Mesh:
    R, o = _meter_rot(), _meter_origin()
    return U.Mesh(mesh.v @ R.T + o, mesh.f, mesh.uv)


def _bench() -> str:
    root = "BenchKit"
    parts, cols = [], []
    # Multimeter in a yellow holster, leaning back on its bail.
    L, W, T = METER_L, METER_W, METER_T
    parts.append(("holster", _to_meter(U.box((L, W, T), (0, 0, -T / 2), bevel=0.004, round_xy=0.009)), "holster"))
    parts.append(("panel", _to_meter(U.box((L - 0.014, W - 0.012, 0.002), (0, 0, 0.0002), bevel=0.0005, round_xy=0.004)), "panel"))
    dx, dy = DISPLAY_LOCAL
    sw, sh = DISPLAY_SIZE
    parts.append(("lcd", _to_meter(U.box((sh + 0.004, sw + 0.004, 0.001), (dx, dy, 0.0012))), "lcd"))
    for v in READINGS:
        # image up = up the face, right = across to -y; all but 0.0 start sunk 6 mm inside the meter (task.py raises one)
        q = U.quad(sw, sh).turned(-90.0, "z").moved((dx, dy, 0.0019 - (READING_SINK if v != 0.0 else 0.0)))
        parts.append((reading_prim(v), _to_meter(q), _lcd_look(v)))
    parts.append(("dial", _to_meter(U.cylinder(0.0155, 0.004, (-0.015, 0.0, 0.002), segments=32, bevel=0.001)), "dial"))
    parts.append(("dial_mark", _to_meter(U.box((0.012, 0.0022, 0.0008), (-0.015 + 0.007, 0.0, 0.0044)).turned(0.0)), "mark_white"))
    for k, ang in enumerate(np.radians(np.arange(-120, 121, 30))):     # range marks round the dial
        x, y = -0.015 + 0.0195 * np.cos(ang), 0.0195 * np.sin(ang)
        parts.append((f"tick{k}", _to_meter(U.box((0.0028, 0.0010, 0.0006), (x, y, 0.0014))), "mark_white"))
    for colour, (a, b) in JACK_LOCAL.items():
        parts.append((f"jack_{colour}", _to_meter(U.cylinder(0.0042, 0.003, (a, b, 0.0015), segments=24)), f"jack_{colour}"))
        parts.append((f"jack_hole_{colour}", _to_meter(U.cylinder(0.0020, 0.0012, (a, b, 0.0031), segments=16)), "hole"))
    # Bail: from the holster's back to the table.
    R, o = _meter_rot(), _meter_origin()
    top_back = o + R @ np.array([L / 2 - 0.02, 0.0, -T])
    foot = np.array([top_back[0] + 0.02, 0.0, 0.002])
    for side in (-1, 1):
        a = top_back + np.array([0, side * 0.03, 0])
        b = foot + np.array([0, side * 0.03, 0])
        parts.append((f"bail{side + 1}", U.tube(np.linspace(a, b, 6), 0.0025, sides=10), "bail"))
    parts.append(("bail_bar", U.tube(np.linspace(foot + [0, -0.03, 0], foot + [0, 0.03, 0], 4), 0.0025, sides=10), "bail"))
    # Meter colliders: a staircase of three boxes under the leaning body.
    xs = [o + R @ np.array([s * L / 2, 0, -T / 2]) for s in (-0.66, 0.0, 0.66)]
    for k, c in enumerate(xs):
        top = c[2] + 0.5 * T
        cols.append(U.collider(f"c_meter{k}", (L / 3 * np.cos(np.radians(METER_TILT)), W, max(top, 0.005)),
                               (c[0], 0.0, max(top, 0.005) / 2), root))
    # Tool stands: black rubber blocks; a driver stands over a peg, a probe in a hole.
    for slot, (x, y) in SLOTS.items():
        if slot.startswith("driver"):
            h, g = DRIVER_STAND / 2, WELL / 2
            boxes = [((x - h, y - h, 0.0), (x - g, y + h, STAND_TOP)), ((x + g, y - h, 0.0), (x + h, y + h, STAND_TOP)),
                     ((x - g, y - h, 0.0), (x + g, y - g, STAND_TOP)), ((x - g, y + g, 0.0), (x + g, y + h, STAND_TOP)),
                     ((x - g, y - g, 0.0), (x + g, y + g, WELL_FLOOR))]
            for k, (lo, hi) in enumerate(boxes):
                parts.append((f"stand_{slot}{k}", P._box(lo, hi, bevel=0.0008), "rubber"))
                cols.append(P._col(f"c_stand_{slot}{k}", lo, hi, root))
            parts.append((f"well_{slot}", P._box((x - g, y - g, WELL_FLOOR), (x + g, y + g, WELL_FLOOR + 0.0005)), "hole"))
        else:
            h, g = PROBE_STAND / 2, HOLE / 2
            boxes = [((x - h, y - h, 0.0), (x - g, y + h, PROBE_STAND_TOP)), ((x + g, y - h, 0.0), (x + h, y + h, PROBE_STAND_TOP)),
                     ((x - g, y - h, 0.0), (x + g, y - g, PROBE_STAND_TOP)), ((x - g, y + g, 0.0), (x + g, y + h, PROBE_STAND_TOP)),
                     ((x - g, y - g, 0.0), (x + g, y + g, HOLE_FLOOR))]
            for k, (lo, hi) in enumerate(boxes):
                parts.append((f"stand_{slot}{k}", P._box(lo, hi, bevel=0.0006), "rubber"))
                cols.append(P._col(f"c_stand_{slot}{k}", lo, hi, root))
            parts.append((f"hole_{slot}", P._box((x - g, y - g, HOLE_FLOOR), (x + g, y + g, HOLE_FLOOR + 0.0005)), "hole"))
    # The two leads (points redrawn every control step; no colliders).
    for colour in ("black", "red"):
        parts.append((f"lead_{colour}", lead_mesh(rest_lead(colour)), f"lead_{colour}"))
    U.write_usda(str(BUILD / "bench.usda"), root, parts, {**P.LOOKS, **LOOKS}, cols, mass=2.0, friction=(0.8, 0.7))
    return "build/bench.usda"


# -- leads --------------------------------------------------------------------------------------------------------------
def _chaikin(p: np.ndarray, n: int = 3) -> np.ndarray:
    for _ in range(n):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.array(q)
    return p


def _resample(p: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], axis=1)


def floor_under(x: float, y: float) -> float:
    """The height a lead lying at (x, y) rests on: the table, the tray, the pack, a stand or the meter."""
    h = 0.0
    tx, ty = P.TRAY_CENTRE
    if abs(x - tx) < P.TRAY_INNER[0] / 2 + P.TRAY_WALL and abs(y - ty) < P.TRAY_INNER[1] / 2 + P.TRAY_WALL:
        h = max(h, P.TRAY_TOP)
    if P.HOUSING_X[0] < x < P.HOUSING_X[1] and abs(y) < P.HOUSING_HALF_Y:
        h = max(h, P.STUD_TIP + 0.004)
    for slot, (sx, sy) in SLOTS.items():
        half = (DRIVER_STAND if slot.startswith("driver") else PROBE_STAND) / 2
        if abs(x - sx) < half and abs(y - sy) < half:
            h = max(h, STAND_TOP if slot.startswith("driver") else PROBE_STAND_TOP)
    return h + LEAD_R


def lead_path(colour: str, top: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """A lead from the meter's jack to a probe whose back end is at `top` with axis `axis` (unit, toward the back end):
    out of the jack, down to the bench, along it, and up into the probe's boot, smoothed and redrawn as LEAD_N points."""
    j = jack(colour)
    n = _meter_rot() @ np.array([0.0, 0.0, 1.0])
    p0, p1 = j, j + 0.012 * n
    q1 = top + 0.018 * axis
    away = np.array([j[0] - top[0], j[1] - top[1], 0.0])
    d = np.linalg.norm(away)
    away = away / d if d > 1e-6 else np.array([-1.0, 0.0, 0.0])
    drop = top + away * min(0.6 * top[2] + 0.03, 0.5 * d) + np.array([0.0, 0.0, -top[2]])
    side = 1.0 if colour == "black" else -1.0
    near_meter = np.array([p1[0] - 0.01, p1[1] + side * 0.055, 0.0])
    pts = np.array([p0, p1, near_meter, 0.5 * (near_meter + drop), drop, q1, top])
    pts = _resample(_chaikin(pts, 3), LEAD_N)
    pts[0], pts[-1] = p0, top
    for k in range(1, LEAD_N - 1):
        pts[k, 2] = max(pts[k, 2], floor_under(pts[k, 0], pts[k, 1]))
    return pts


def rest_lead(colour: str) -> np.ndarray:
    xy = SLOTS["probe_black" if colour == "black" else "probe_red"]
    top = np.array([xy[0], xy[1], PROBE_START_Z + PROBE_TOP])
    return lead_path(colour, top, np.array([0.0, 0.0, 1.0]))


def lead_mesh(points: np.ndarray) -> U.Mesh:
    m = U.tube(points, LEAD_R, sides=LEAD_SIDES)
    m.smooth = True     # no stored normals: the renderer works them out as the points move
    return m


def build() -> dict[str, str]:
    BUILD.mkdir(exist_ok=True)
    return {"driver": _driver(), "probe_black": _probe("black"), "probe_red": _probe("red"), "bench": _bench()}
