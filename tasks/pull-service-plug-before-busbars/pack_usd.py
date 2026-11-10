"""The battery pack, its busbars, its service plug and the parts tray: layout and USD files.

Positions are metres in the station frame (x forward, y left); heights are above the table top. The fixed pack and tray
are written with their origin on the table under the arm bases' midpoint, so their parts sit at the station xy below.
The busbars and the plug have their origin at their own centre. build() writes the USD files into build/ (not
committed: they are generated each time the task file loads).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import usdkit as U

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# -- layout -----------------------------------------------------------------------------------------------------------
HOUSING_X = (0.255, 0.415)        # outer, along x
HOUSING_HALF_Y = 0.185            # outer half width
FLOOR = 0.003                     # housing floor top
WALL_T, WALL_TOP = 0.003, 0.025   # housing wall thickness and top

MODULE_X = (0.26, 0.41)           # 15 cm long
MODULE_W = 0.07                   # 7 cm wide (y)
MODULE_Y = (0.145, 0.065, -0.065, -0.145)   # M1..M4, robot's left to right
LID = 0.043                       # module lid height (4 cm tall on the 3 mm floor)

POST_X = {"front": 0.28, "back": 0.39}
PAD_R, PAD_TOP = 0.010, 0.049     # nickel-plated terminal pad, 6 mm tall
STUD_R, STUD_TIP = 0.003, 0.061   # M6 stud, 12 mm above the pad

# Each module's (+, -) terminals: modules alternate so that the series string runs
# M1- (front, main -) > M1+ back > bar A > M2- back > M2+ front > plug > M3- front > M3+ back > bar B > M4- back > M4+ front (main +).
POLARITY = {0: {"front": "-", "back": "+"}, 1: {"front": "+", "back": "-"},
            2: {"front": "-", "back": "+"}, 3: {"front": "+", "back": "-"}}

BAR_SIZE = (0.025, 0.11, 0.005)   # x (width), y (length), z (thickness)
BAR_HOLE, BAR_HOLE_Y = 0.008, 0.04
BAR_MASS = 0.12
BARS = {"busbar_a": (POST_X["back"], 0.105), "busbar_b": (POST_X["back"], -0.105)}
BAR_Z = PAD_TOP + BAR_SIZE[2] / 2

SOCKET_X, SOCKET_HALF = (0.275, 0.325), (0.025, 0.028)
SOCKET_CENTRE = (0.30, 0.0)
POCKET_HALF = (0.016, 0.023)      # 32 x 46 mm pocket
POCKET_FLOOR, RIM = 0.023, LID    # 2 cm deep
PLUG_SIZE = (0.030, 0.044, 0.045)
PLUG_MASS = 0.12
PLUG_Z = POCKET_FLOOR + PLUG_SIZE[2] / 2
HANDLE = (0.008, 0.030, 0.010)    # grip bar on the plug's top

TRAY_CENTRE = (0.18, 0.0)
TRAY_INNER = (0.09, 0.26)         # x, y
TRAY_WALL, TRAY_FLOOR, TRAY_TOP = 0.004, 0.003, 0.015


def posts():
    """Every terminal post: (name, x, y, polarity)."""
    out = []
    for i, y in enumerate(MODULE_Y):
        for end, x in POST_X.items():
            pol = POLARITY[i][end]
            out.append((f"M{i + 1}{'pos' if pol == '+' else 'neg'}", x, y, pol))
    return out


# -- looks ------------------------------------------------------------------------------------------------------------
LOOKS = {
    "steel_paint": U.Look(color=(0.055, 0.06, 0.065), roughness=0.55),
    "aluminium": U.Look(color=(1.0, 1.0, 1.0), metallic=0.85, roughness=0.38, texture="brushed_aluminium.png", tile=0.1),
    "end_cap": U.Look(color=(0.03, 0.03, 0.035), roughness=0.6),
    "nickel": U.Look(color=(0.62, 0.60, 0.56), metallic=1.0, roughness=0.3),
    "zinc": U.Look(color=(0.55, 0.56, 0.58), metallic=1.0, roughness=0.4),
    "tin": U.Look(color=(0.70, 0.70, 0.68), metallic=1.0, roughness=0.35),
    "copper": U.Look(color=(0.80, 0.36, 0.17), metallic=1.0, roughness=0.42),
    "hv_orange": U.Look(color=(0.90, 0.20, 0.01), roughness=0.5),
    "black_nylon": U.Look(color=(0.02, 0.02, 0.022), roughness=0.55),
    "red_mark": U.Look(color=(0.55, 0.02, 0.02), roughness=0.5),
    "black_mark": U.Look(color=(0.01, 0.01, 0.01), roughness=0.5),
    "label_socket": U.Look(color=(1.0, 1.0, 1.0), roughness=0.45, texture="hv_warning.png", tile=0.02),
    "label_wall": U.Look(color=(1.0, 1.0, 1.0), roughness=0.45, texture="hv_warning.png", tile=0.016),
    "grey_pp": U.Look(color=(0.16, 0.17, 0.18), roughness=0.7),
    "handle_grey": U.Look(color=(0.08, 0.08, 0.09), roughness=0.5),
}


def _box(lo, hi, bevel=0.0, round_xy=0.0) -> U.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return U.box(tuple(hi - lo), tuple((lo + hi) / 2), bevel=bevel, round_xy=round_xy)


def _col(name, lo, hi, root) -> str:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return U.collider(name, tuple(hi - lo), tuple((lo + hi) / 2), root)


def _pack() -> str:
    root = "BatteryPack"
    parts, cols = [], []
    x0, x1 = HOUSING_X
    hy = HOUSING_HALF_Y
    # Housing: floor and four walls.
    shells = {
        "floor": ((x0, -hy, 0.0), (x1, hy, FLOOR)),
        "wall_front": ((x0, -hy, FLOOR), (x0 + WALL_T, hy, WALL_TOP)),
        "wall_back": ((x1 - WALL_T, -hy, FLOOR), (x1, hy, WALL_TOP)),
        "wall_left": ((x0 + WALL_T, hy - WALL_T, FLOOR), (x1 - WALL_T, hy, WALL_TOP)),
        "wall_right": ((x0 + WALL_T, -hy, FLOOR), (x1 - WALL_T, -hy + WALL_T, WALL_TOP)),
    }
    for name, (lo, hi) in shells.items():
        parts.append((f"housing_{name}", _box(lo, hi, bevel=0.0006), "steel_paint"))
        cols.append(_col(f"c_housing_{name}", lo, hi, root))
    # Modules: brushed aluminium body between two black end caps.
    mx0, mx1 = MODULE_X
    for i, y in enumerate(MODULE_Y):
        hw = MODULE_W / 2
        parts.append((f"module{i + 1}_body", _box((mx0 + 0.008, y - hw + 0.0005, FLOOR), (mx1 - 0.008, y + hw - 0.0005, LID - 0.0005),
                                                  bevel=0.001), "aluminium"))
        for k, (a, b) in enumerate(((mx0, mx0 + 0.008), (mx1 - 0.008, mx1))):
            parts.append((f"module{i + 1}_cap{k}", _box((a, y - hw, FLOOR), (b, y + hw, LID), bevel=0.0015, round_xy=0.002), "end_cap"))
        cols.append(_col(f"c_module{i + 1}", (mx0, y - hw, FLOOR), (mx1, y + hw, LID), root))
    # Terminal posts: nickel pad and zinc-plated M6 stud; a red or black polarity mark on the lid beside each.
    for name, x, y, pol in posts():
        parts.append((f"pad_{name}", U.cylinder(PAD_R, PAD_TOP - LID, (x, y, (LID + PAD_TOP) / 2), segments=28, bevel=0.0008), "nickel"))
        parts.append((f"stud_{name}", U.cylinder(STUD_R, STUD_TIP - PAD_TOP, (x, y, (PAD_TOP + STUD_TIP) / 2), segments=16, bevel=0.0006), "zinc"))
        side = 1.0 if y > 0 else -1.0
        mark_y = y + side * 0.022
        parts.append((f"mark_{name}", U.cylinder(0.0045, 0.0008, (x, mark_y, LID + 0.0004), segments=20),
                      "red_mark" if pol == "+" else "black_mark"))
        cols.append(_col(f"c_pad_{name}", (x - 0.008, y - 0.008, LID), (x + 0.008, y + 0.008, PAD_TOP), root))
        cols.append(_col(f"c_stud_{name}", (x - 0.00275, y - 0.00275, PAD_TOP), (x + 0.00275, y + 0.00275, STUD_TIP), root))
    # Front posts carry cable lugs held by nuts: M2+ and M3- to the plug socket, M1- and M4+ out as the main leads.
    xf = POST_X["front"]
    for i, y in enumerate(MODULE_Y):
        toward = -1.0 if abs(y) < 0.1 else 1.0          # inner modules' lugs point to the middle, outer ones outward
        d = toward * np.sign(y)
        parts.append((f"lug_ring{i}", U.cylinder(0.0085, 0.002, (xf, y, PAD_TOP + 0.001), segments=24), "tin"))
        parts.append((f"lug_tongue{i}", _box((xf - 0.0055, min(y, y + d * 0.022), PAD_TOP), (xf + 0.0055, max(y, y + d * 0.022), PAD_TOP + 0.002)), "tin"))
        parts.append((f"nut{i}", U.cylinder(0.0058, 0.005, (xf, y, PAD_TOP + 0.0045), segments=6), "zinc"))
        cols.append(_col(f"c_lug{i}", (xf - 0.008, y - 0.008, PAD_TOP), (xf + 0.008, y + 0.008, PAD_TOP + 0.007), root))
        if abs(y) < 0.1:   # to the socket's front face, beside its label
            s = np.sign(y)
            path = [(xf, y + d * 0.020, 0.0505), (xf - 0.002, s * 0.036, 0.0505), (0.271, s * 0.025, 0.042), (0.266, s * 0.021, 0.028),
                    (0.268, s * 0.019, 0.016), (0.276, s * 0.019, 0.014)]
        else:              # main leads, over the side wall and back along the table
            s = np.sign(y)
            path = [(xf, y + d * 0.020, 0.0505), (xf, s * 0.180, 0.047), (xf + 0.004, s * 0.192, 0.030), (xf + 0.012, s * 0.198, 0.006),
                    (0.33, s * 0.205, 0.0036), (0.40, s * 0.210, 0.0036), (0.425, s * 0.210, 0.0036)]
        parts.append((f"cable{i}", U.tube(_smooth(path), 0.0035, sides=12), "hv_orange"))
        if abs(y) > 0.1:
            parts.append((f"lead_plug{i}", _box((0.425, np.sign(y) * 0.210 - 0.011, 0.0), (0.455, np.sign(y) * 0.210 + 0.011, 0.016),
                                                bevel=0.0015, round_xy=0.003), "hv_orange"))
    # Service plug socket: a black block with a 32 x 46 mm pocket 2 cm deep.
    sx0, sx1 = SOCKET_X
    sy = SOCKET_HALF[1]
    px, py = POCKET_HALF
    cx = SOCKET_CENTRE[0]
    blocks = {
        "base": ((sx0, -sy, FLOOR), (sx1, sy, POCKET_FLOOR)),
        "front": ((sx0, -sy, POCKET_FLOOR), (cx - px, sy, RIM)),
        "back": ((cx + px, -sy, POCKET_FLOOR), (sx1, sy, RIM)),
        "left": ((cx - px, py, POCKET_FLOOR), (cx + px, sy, RIM)),
        "right": ((cx - px, -sy, POCKET_FLOOR), (cx + px, -py, RIM)),
    }
    for name, (lo, hi) in blocks.items():
        parts.append((f"socket_{name}", _box(lo, hi, bevel=0.0005), "black_nylon"))
        cols.append(_col(f"c_socket_{name}", lo, hi, root))
    # High-voltage warning labels: on the socket's front face and on the housing's front wall (centred on tile lines).
    parts.append(("label_socket", _box((sx0 - 0.0005, -0.010, 0.010), (sx0, 0.010, 0.030)), "label_socket"))
    parts.append(("label_wall", _box((x0 - 0.0005, -0.008, 0.008), (x0, 0.008, 0.024)), "label_wall"))
    U.write_usda(str(BUILD / "pack.usda"), root, parts, LOOKS, cols, mass=5.0, friction=(0.6, 0.5))
    return "build/pack.usda"


def _smooth(points, n: int = 6) -> np.ndarray:
    """A polyline through the points, corners rounded with a Chaikin pass."""
    p = np.asarray(points, float)
    for _ in range(3):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.array(q)
    return p


def _busbar() -> str:
    root = "Busbar"
    hx, hy, hz = BAR_SIZE[0] / 2, BAR_SIZE[1] / 2, BAR_SIZE[2] / 2
    h, c = BAR_HOLE / 2, BAR_HOLE_Y
    pieces = {
        "mid": ((-hx, -(c - h), -hz), (hx, c - h, hz)),
        "end_p": ((-hx, c + h, -hz), (hx, hy, hz)),
        "end_n": ((-hx, -hy, -hz), (hx, -(c + h), hz)),
    }
    for sgn, tag in ((1, "p"), (-1, "n")):
        lo_y, hi_y = sorted((sgn * (c - h), sgn * (c + h)))
        pieces[f"side_{tag}0"] = ((h, lo_y, -hz), (hx, hi_y, hz))
        pieces[f"side_{tag}1"] = ((-hx, lo_y, -hz), (-h, hi_y, hz))
    parts, cols = [], []
    for name, (lo, hi) in pieces.items():
        parts.append((name, _box(lo, hi), "copper"))
        cols.append(_col(f"c_{name}", lo, hi, root))
    U.write_usda(str(BUILD / "busbar.usda"), root, parts, LOOKS, cols, mass=BAR_MASS, friction=(0.9, 0.8))
    return "build/busbar.usda"


def _plug() -> str:
    root = "ServicePlug"
    hx, hy, hz = (s / 2 for s in PLUG_SIZE)
    parts = [("body", U.box(PLUG_SIZE, (0, 0, 0), bevel=0.0015, round_xy=0.003), "hv_orange"),
             ("grip", U.box(HANDLE, (0, 0, hz + HANDLE[2] / 2), bevel=0.0015, round_xy=0.002), "handle_grey")]
    for k, y in enumerate(np.linspace(-0.015, 0.015, 5)):   # grip ribs on the faces the fingers hold
        for sgn in (-1, 1):
            parts.append((f"rib{k}_{'p' if sgn > 0 else 'n'}", U.box((0.0008, 0.003, 0.016), (sgn * (hx + 0.0004), y, hz - 0.012)), "hv_orange"))
    cols = [U.collider("c_body", PLUG_SIZE, (0, 0, 0), root),
            U.collider("c_grip", HANDLE, (0, 0, hz + HANDLE[2] / 2), root)]
    U.write_usda(str(BUILD / "plug.usda"), root, parts, LOOKS, cols, mass=PLUG_MASS, friction=(0.9, 0.8))
    return "build/plug.usda"


def _tray() -> str:
    root = "PartsTray"
    cx, cy = TRAY_CENTRE
    ix, iy = TRAY_INNER[0] / 2, TRAY_INNER[1] / 2
    w = TRAY_WALL
    pieces = {
        "floor": ((cx - ix - w, cy - iy - w, 0.0), (cx + ix + w, cy + iy + w, TRAY_FLOOR)),
        "front": ((cx - ix - w, cy - iy - w, TRAY_FLOOR), (cx - ix, cy + iy + w, TRAY_TOP)),
        "back": ((cx + ix, cy - iy - w, TRAY_FLOOR), (cx + ix + w, cy + iy + w, TRAY_TOP)),
        "left": ((cx - ix, cy + iy, TRAY_FLOOR), (cx + ix, cy + iy + w, TRAY_TOP)),
        "right": ((cx - ix, cy - iy - w, TRAY_FLOOR), (cx + ix, cy - iy, TRAY_TOP)),
    }
    parts, cols = [], []
    for name, (lo, hi) in pieces.items():
        parts.append((name, _box(lo, hi, bevel=0.0008), "grey_pp"))
        cols.append(_col(f"c_{name}", lo, hi, root))
    U.write_usda(str(BUILD / "tray.usda"), root, parts, LOOKS, cols, mass=1.0, friction=(0.6, 0.5))
    return "build/tray.usda"


def build() -> dict[str, str]:
    BUILD.mkdir(exist_ok=True)
    return {"pack": _pack(), "busbar": _busbar(), "plug": _plug(), "tray": _tray()}


if __name__ == "__main__":
    print(build())
