"""The model arch's parts as USD files in build/: eight voussoirs, the keystone, two abutments and the timber centering.

Geometry in the arch frame (metres): x across the arch's depth from its mid-plane, u along the span from the crown line
(world y), z up from the table top. The arch is a 120 degree segment of a circle whose centre lies below the table; its
nine stones each span 13.33 degrees, with radial joints and flat undersides (chords). Every stone's collider is its exact
convex wedge (8 corners); the visual mesh is the same wedge with 1.5 mm chamfers on the front and back arrises.

Pure Python and NumPy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as U

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

# -- the arch -------------------------------------------------------------------------------------------------------
R_I, R_E = 0.145, 0.180             # intrados and extrados radii
HALF = math.radians(60.0)           # half the angle the arch spans
N_STONES = 9
STEP = 2 * HALF / N_STONES          # 13.33 degrees per stone
SPRING_Z = 0.020                    # springing line above the table
Z0 = SPRING_Z - R_I * math.cos(HALF)  # circle centre (below the table top)
DEPTH = 0.050                       # voussoirs, front to back
KEY_DEPTH = 0.060                   # keystone: 5 mm proud of each face
KEY_TOP_Z = 0.1375                  # keystone's flat top when seated, 1 cm above the extrados at the crown
KEY_SOFFIT_UP = 0.0015              # the keystone's underside stops 1.5 mm short of the intrados: it hangs on its faces, clear of the centering
CHAMFER = 0.002

STONE_DENSITY = 2300.0              # sandstone, kg/m3
PINE_DENSITY = 500.0
BEECH_DENSITY = 700.0

# -- the centering (a solid turning piece) and its pull handle -----------------------------------------------------
CENTER_HALF_U = 0.1235              # 2 mm clear of each abutment's inner face (springing at u 0.1256)
CENTER_DEPTH = 0.054                # 2 mm proud of each face of the arch
BOARD = (0.080, 0.080, 0.015)       # x (toward the robot), u, z
PEG_R, PEG_H = 0.0125, 0.050
PEG_X = -(CENTER_DEPTH / 2 + BOARD[0]) + 0.012   # peg centre in x, 1.2 cm in from the board's front end

# -- abutments -------------------------------------------------------------------------------------------------------
ABUT_OUT = 0.186                    # outer face, from the crown line
ABUT_TOP = SPRING_Z + (R_E - R_I) * math.cos(HALF)   # 3.75 cm: the skewback's top edge


def _polar(r: float, a: float) -> tuple[float, float]:
    """(u, z) of radius r at angle a from the vertical (positive toward +u)."""
    return r * math.sin(a), Z0 + r * math.cos(a)


def stone_section(k: int) -> np.ndarray:
    """Stone k's cross-section (u, z), counter-clockwise: k = 0 is the keystone, -4..4 from -u to +u."""
    a0, a1 = (k - 0.5) * STEP, (k + 0.5) * STEP
    if k == 0:
        r_top = (KEY_TOP_Z - Z0) / math.cos(STEP / 2)
        r_bot = (Z0 + R_I * math.cos(STEP / 2) + KEY_SOFFIT_UP - Z0) / math.cos(STEP / 2)
        pts = [_polar(r_bot, a0), _polar(r_bot, a1), _polar(r_top, a1), _polar(r_top, a0)]
    else:
        pts = [_polar(R_I, a0), _polar(R_I, a1), _polar(R_E, a1), _polar(R_E, a0)]
    pts = np.array(pts)
    # counter-clockwise in (u, z)
    a, b = pts[1] - pts[0], pts[2] - pts[1]
    if a[0] * b[1] - a[1] * b[0] < 0:
        pts = pts[::-1]
    return pts


def _area_centroid(poly: np.ndarray) -> tuple[float, np.ndarray]:
    x, y = poly[:, 0], poly[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x * ys - xs * y
    area = cross.sum() / 2
    cx = ((x + xs) * cross).sum() / (6 * area)
    cy = ((y + ys) * cross).sum() / (6 * area)
    return float(area), np.array([cx, cy])


def stone_pose(k: int) -> tuple[np.ndarray, float]:
    """Stone k's built centroid (x, u, z) in the arch frame, and its tilt about x (radians, the body's rotation)."""
    _, c = _area_centroid(stone_section(k))
    return np.array([0.0, c[0], c[1]]), -k * STEP


def stone_mass(k: int) -> float:
    area, _ = _area_centroid(stone_section(k))
    return area * (KEY_DEPTH if k == 0 else DEPTH) * STONE_DENSITY


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def stone_local_section(k: int) -> np.ndarray:
    """Stone k's section in its own frame: centroid at the origin, radial axis up (local z), tangent along local y."""
    centre, tilt = stone_pose(k)
    sec = stone_section(k) - centre[1:]
    r = _rot_x(tilt)[1:, 1:]            # body -> arch frame in the (u, z) plane
    return sec @ r                       # arch -> body: multiply by r (r is orthonormal, rows/cols swapped)


def _extrude_x(section_uz: np.ndarray, depth: float, bevel: float = 0.0) -> U.Mesh:
    """A prism along x from a counter-clockwise (u, z) section: (u, z, x) -> (x, y, z) is a cyclic swap, so faces stay outward."""
    m = U.prism(section_uz, -depth / 2, depth / 2, bevel)
    v = m.v[:, [2, 0, 1]]
    return U.Mesh(v, m.f)


def stone_corners(k: int) -> np.ndarray:
    """Stone k's 8 corners in its own frame."""
    sec = stone_local_section(k)
    d = (KEY_DEPTH if k == 0 else DEPTH) / 2
    return np.array([(x, u, z) for x in (-d, d) for u, z in sec])


# -- looks ------------------------------------------------------------------------------------------------------------
LOOKS = {
    "stone_a": U.Look(color=(1.0, 1.0, 1.0), roughness=0.9, texture="sandstone_a.jpg", tile=0.12),
    "stone_b": U.Look(color=(1.0, 1.0, 1.0), roughness=0.9, texture="sandstone_b.jpg", tile=0.12),
    "stone_c": U.Look(color=(1.0, 1.0, 1.0), roughness=0.9, texture="sandstone_c.jpg", tile=0.12),
    "pine": U.Look(color=(1.0, 1.0, 1.0), roughness=0.7, texture="pine.jpg", tile=0.15),
    "beech": U.Look(color=(1.0, 1.0, 1.0), roughness=0.55, texture="beech.jpg", tile=0.05),
}
STONE_LOOK = {-4: "stone_a", -3: "stone_b", -2: "stone_c", -1: "stone_a", 0: "stone_b", 1: "stone_c", 2: "stone_b",
              3: "stone_a", 4: "stone_c"}

STONE_FRICTION = (0.7, 0.6)
PINE_FRICTION = (0.2, 0.15)          # planed, waxed pine; combined by "min", so it rules its contacts with stone and table


def _write_stone(k: int) -> str:
    name = "keystone" if k == 0 else f"voussoir_{k:+d}".replace("+", "p").replace("-", "m")
    depth = KEY_DEPTH if k == 0 else DEPTH
    sec = stone_local_section(k)
    look = STONE_LOOK[k]
    parts = [("stone", _extrude_x(sec, depth, CHAMFER), look)]
    colliders = [U.hull_collider("collider", stone_corners(k), name)]
    path = BUILD / f"{name}.usda"
    U.write_usda(str(path), name, parts, LOOKS, colliders, mass=stone_mass(k), friction=STONE_FRICTION, solver_iters=64)
    return path.relative_to(HERE).as_posix()


def abutment_section(side: int) -> np.ndarray:
    """The abutment under the arch's end on side +1 (+u) or -1 (-u), in (u, z), counter-clockwise."""
    ui, zi = _polar(R_I, HALF)
    ue, ze = _polar(R_E, HALF)
    pts = np.array([(ui, 0.0), (ABUT_OUT, 0.0), (ABUT_OUT, ze), (ue, ze), (ui, zi)])
    if side < 0:
        pts = pts * np.array([-1.0, 1.0])
        pts = pts[::-1]
    return pts


def _write_abutment(side: int) -> str:
    name = "abutment_left" if side > 0 else "abutment_right"
    sec = abutment_section(side)
    parts = [("block", _extrude_x(sec, DEPTH, CHAMFER), "stone_c" if side > 0 else "stone_a")]
    corners = np.array([(x, u, z) for x in (-DEPTH / 2, DEPTH / 2) for u, z in sec])
    colliders = [U.hull_collider("collider", corners, name)]
    path = BUILD / f"{name}.usda"
    # A rigid body in the file; the task loads it kinematic (fixed in place).
    U.write_usda(str(path), name, parts, LOOKS, colliders, mass=1.0, friction=STONE_FRICTION, solver_iters=64)
    return path.relative_to(HERE).as_posix()


def centering_section() -> np.ndarray:
    """The turning piece's section (u, z), counter-clockwise: a 2 cm plinth under the arch's undersides."""
    top = []
    for j in range(4, -5, -1):                       # joints at +-6.67, +-20, +-33.3, +-46.7 degrees
        if j == 0:
            continue
        a = (j - 0.5 if j > 0 else j + 0.5) * STEP
        top.append(_polar(R_I, a))
    # the end facets (46.7 to 60 degrees) cut at u = +-CENTER_HALF_U
    ue, ze = _polar(R_I, HALF)
    um, zm = _polar(R_I, 3.5 * STEP)
    z_end = ze + (zm - ze) * (ue - CENTER_HALF_U) / (ue - um)
    pts = [(-CENTER_HALF_U, 0.0), (CENTER_HALF_U, 0.0), (CENTER_HALF_U, z_end)] + top + [(-CENTER_HALF_U, z_end)]
    return np.array(pts)


def centering_points() -> np.ndarray:
    """Every corner of the centering's colliders in its own frame (origin: bottom, centre of the span, mid-depth)."""
    sec = centering_section()
    body = [(x, u, z) for x in (-CENTER_DEPTH / 2, CENTER_DEPTH / 2) for u, z in sec]
    bx0, bx1 = -CENTER_DEPTH / 2 - BOARD[0], -CENTER_DEPTH / 2
    board = [(x, u, z) for x in (bx0, bx1) for u in (-BOARD[1] / 2, BOARD[1] / 2) for z in (0.0, BOARD[2])]
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    peg = [(PEG_X + PEG_R * math.cos(t), PEG_R * math.sin(t), z) for t in th for z in (BOARD[2], BOARD[2] + PEG_H)]
    return np.array(body + board + peg)


def centering_mass() -> float:
    area, _ = _area_centroid(centering_section())
    return (area * CENTER_DEPTH + BOARD[0] * BOARD[1] * BOARD[2]) * PINE_DENSITY + math.pi * PEG_R ** 2 * PEG_H * BEECH_DENSITY


def _write_centering() -> str:
    name = "centering"
    sec = centering_section()
    body = _extrude_x(sec, CENTER_DEPTH, 0.001)
    bx0, bx1 = -CENTER_DEPTH / 2 - BOARD[0], -CENTER_DEPTH / 2
    board = U.box((BOARD[0], BOARD[1], BOARD[2]), ((bx0 + bx1) / 2, 0.0, BOARD[2] / 2), bevel=0.001)
    peg = U.cylinder(PEG_R, PEG_H, (PEG_X, 0.0, BOARD[2] + PEG_H / 2), segments=24, bevel=0.0015)
    parts = [("former", body, "pine"), ("board", board, "pine"), ("peg", peg, "beech")]
    body_pts = np.array([(x, u, z) for x in (-CENTER_DEPTH / 2, CENTER_DEPTH / 2) for u, z in sec])
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    peg_pts = np.array([(PEG_X + PEG_R * math.cos(t), PEG_R * math.sin(t), z) for t in th for z in (BOARD[2], BOARD[2] + PEG_H)])
    colliders = [U.hull_collider("former_collider", body_pts, name),
                 U.collider("board_collider", BOARD, ((bx0 + bx1) / 2, 0.0, BOARD[2] / 2), name),
                 U.hull_collider("peg_collider", peg_pts, name)]
    path = BUILD / f"{name}.usda"
    U.write_usda(str(path), name, parts, LOOKS, colliders, mass=centering_mass(), friction=PINE_FRICTION, solver_iters=64,
                 friction_combine="min")
    return path.relative_to(HERE).as_posix()


def build() -> dict[str, str]:
    """Write every part; return the USD paths (relative to this folder) by part name."""
    out = {"keystone": _write_stone(0), "centering": _write_centering(),
           "abutment_left": _write_abutment(+1), "abutment_right": _write_abutment(-1)}
    for k in (-4, -3, -2, -1, 1, 2, 3, 4):
        out[k] = _write_stone(k)
    return out


if __name__ == "__main__":
    for k in range(-4, 5):
        c, t = stone_pose(k)
        print(k, np.round(c, 4), round(math.degrees(t), 2), round(stone_mass(k), 3))
    print("centering", round(centering_mass(), 3))
    print(build())
