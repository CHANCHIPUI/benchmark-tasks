"""Geometry of the egg cartons, the ball racks and the balls, and the USD files built from it (into assets/, at load
time). Pure NumPy. Sizes in metres; station frame (x forward, y to the robot's left), heights above the table top.

- Board: dark grey painted hardboard under the cartons.
- Egg cartons: three half-dozen moulded-pulp carton bases, lids cut off, taped to the board. Each has 2 x 3 cups at 5 cm
  pitch, 46 mm across at the rim, tapering to 30 mm at the floor, 28 mm deep. Sized from a common half-dozen carton
  (about 156 x 106 mm).
- Ball racks: two pine boards with five drilled 24 mm sockets each.
- Balls: solid beech, 40 mm, painted red (beech about 0.72 g/cm3: 24 g).

The board, cartons and racks are one fixed body. Each cup collides through a floor box and eight sloped wall slabs, so a
ball released off-centre rolls to the cup's middle; each socket through a square ring of four boxes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- board --------------------------------------------------------------------------------------------------------
BOARD = (0.22, 0.48, 0.003)  # x, y, thickness
BOARD_XY = (0.31, 0.0)
BASE = BOARD[2]  # the cartons stand on the board

# -- egg cartons --------------------------------------------------------------------------------------------------
CARTON_X = 0.31
CARTON_Y = (0.13, 0.0, -0.13)  # cartons 1 to 3, left to right as the reader (on the robot's side) sees them
OUTER = (0.156, 0.106)  # x, y
PITCH = 0.05
RIM = 0.032  # rim height above the carton's base
DEPTH = 0.028
R_RIM, R_FLOOR = 0.023, 0.015  # cup radius at the rim and at the floor
FLOOR = RIM - DEPTH
SLAB_T = 0.003  # thickness of the wall slabs (outward): thin, so they stay out of the neighbouring cups
FACETS = 8
POST_X = (PITCH / 2, -PITCH / 2)  # the two moulded posts on the centre line, where four cups meet
POST_R = (0.009, 0.0045)  # radius at the deck and at the top
POST_H = 0.006  # above the rim: low, so open fingers clear it
ROW_X = (0.05, 0.0, -0.05)  # top (far) row first: dots 1/4, 2/5, 3/6
COL_Y = (0.025, -0.025)  # left column (dots 1-3), right column (dots 4-6)

# -- racks ----------------------------------------------------------------------------------------------------------
RACK = (0.30, 0.06, 0.015)
RACK_X = 0.30
RACK_Y = (0.28, -0.28)
SOCKET_X = (0.18, 0.24, 0.30, 0.36, 0.42)
SOCKET_R = 0.012  # drilled 24 mm
SOCKET_DEPTH = 0.006
SOCKET_HALF = 0.011  # collider: a 22 mm square hole

# -- balls ----------------------------------------------------------------------------------------------------------
BALL_R = 0.020
BALL_MASS = 0.024
BALL_FRICTION = (0.6, 0.5)
ANGULAR_DAMPING = 5.0  # stands in for rolling resistance: a ball rolling at 20 cm/s on the board stops within ~15 cm
LINEAR_DAMPING = 0.05


def cup_centres() -> dict:
    """(carton 0..2, dot 1..6) -> (x, y) of the cup's centre."""
    out = {}
    for k, cy in enumerate(CARTON_Y):
        for dot in range(1, 7):
            col, row = (0 if dot <= 3 else 1), (dot - 1) % 3
            out[(k, dot)] = (CARTON_X + ROW_X[row], cy + COL_Y[col])
    return out


def seated_centre() -> float:
    """Height above the table of a ball's centre resting on a cup's wall slabs (inscribed cone), or on its floor."""
    slope = (R_RIM - R_FLOOR) / DEPTH  # radius gained per metre of height
    # Wall line r = R_FLOOR + (z - FLOOR) * slope; distance from (0, c) to it equals BALL_R.
    c = (BALL_R * math.sqrt(1 + slope ** 2) - R_FLOOR + FLOOR * slope) / slope
    return BASE + max(c, FLOOR + BALL_R)


def rack_centre() -> float:
    """Height above the table of a ball's centre in a rack socket (resting on the square hole's edges)."""
    return RACK[2] - (BALL_R - math.sqrt(BALL_R ** 2 - SOCKET_HALF ** 2)) + BALL_R


def balls_start() -> list:
    """Ten balls, five per rack, (x, y) of each."""
    return [(x, y) for y in RACK_Y for x in SOCKET_X]


# -- meshes ---------------------------------------------------------------------------------------------------------
def _orient(mesh: uk.Mesh, ref) -> uk.Mesh:
    """Turn every face so that it faces the point `ref` (all faces here are seen from it)."""
    v, f = mesh.v, mesh.f.copy()
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    c = v[f].mean(axis=1)
    flip = np.einsum("ij,ij->i", n, np.asarray(ref) - c) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return uk.Mesh(v, f)


def _square_ring(sx: float, sy: float, th: np.ndarray) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    k = np.minimum(sx / 2 / np.maximum(np.abs(c), 1e-9), sy / 2 / np.maximum(np.abs(s), 1e-9))
    return np.c_[k * c, k * s]


def _holed_tile(sx: float, sy: float, profile, n: int = 48) -> uk.Mesh:
    """A flat sx x sy tile at profile[0]'s height with a round hollow whose (radius, z) profile runs from the rim down
    to its floor centre (radius 0). Faces face up and into the hollow."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z_top = profile[0][1]
    rings = [np.c_[_square_ring(sx, sy, th), np.full(n, z_top)]]
    for r, z in profile:
        if r == 0:
            break
        rings.append(np.c_[r * np.cos(th), r * np.sin(th), np.full(n, z)])
    centre = np.array([[0.0, 0.0, profile[-1][1]]])
    v = np.vstack(rings + [centre])
    f = []
    for a in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            p, q, r2, s = a * n + i, a * n + j, (a + 1) * n + j, (a + 1) * n + i
            f += [(p, q, r2), (p, r2, s)]
    last, ci = (len(rings) - 1) * n, len(v) - 1
    f += [(last + i, last + (i + 1) % n, ci) for i in range(n)]
    return _orient(uk.Mesh(v, np.array(f)), (0.0, 0.0, z_top + 0.2))


def _cup_profile(lip: float = 0.0015) -> list:
    """The cup's inner surface, from the flat rim inward and down: a rounded lip, the straight taper the wall slabs
    follow, and a rounded floor."""
    slope = (R_RIM - R_FLOOR) / DEPTH

    def r_at(z):
        return R_FLOOR + (z - FLOOR) * slope

    pts = [(R_RIM + lip, RIM), (R_RIM + 0.35 * lip, RIM - 0.35 * lip), (r_at(RIM - lip), RIM - lip)]
    for z in np.linspace(RIM - 0.006, FLOOR + 0.004, 5):
        pts.append((r_at(z), z))
    pts += [(R_FLOOR - 0.0015, FLOOR + 0.0006), (R_FLOOR - 0.005, FLOOR), (0.0, FLOOR)]
    return pts


def _carton_mesh() -> uk.Mesh:
    tiles = []
    for rx in ROW_X:
        for cy in COL_Y:
            tiles.append(_holed_tile(PITCH, PITCH, _cup_profile()).moved((rx, cy, 0.0)))
    inner = uk.rounded_rect(3 * PITCH, 2 * PITCH, 0.0005)
    outer = uk.rounded_rect(OUTER[0], OUTER[1], 0.007)
    bottom = uk.rounded_rect(OUTER[0] - 0.012, OUTER[1] - 0.012, 0.007)
    flange = _tapered_rim(outer, bottom, inner)
    posts = [uk.lathe(_post_profile(), segments=24).moved((px, 0.0, 0.0)) for px in POST_X]
    return uk.join(*tiles, flange, *posts)


def _tapered_rim(outer, bottom, inner) -> uk.Mesh:
    """The carton's rim (a flat band at rim height between the cup tiles and the outer edge) and its outer wall,
    which tapers in toward the base as a moulded carton's sides do."""
    n = len(outer)
    v = np.vstack([np.c_[inner, np.full(n, RIM)], np.c_[outer, np.full(n, RIM)], np.c_[outer * 0.995, np.full(n, RIM - 0.002)],
                   np.c_[bottom, np.zeros(n)]])
    f = []
    for ring in range(3):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = ring * n + i, ring * n + j, (ring + 1) * n + j, (ring + 1) * n + i
            f += [(a, b, c), (a, c, d)]
    return _orient_out(uk.Mesh(v, np.array(f)))


def _orient_out(mesh: uk.Mesh) -> uk.Mesh:
    """Turn faces to face up or outward from the carton's vertical axis."""
    v, f = mesh.v, mesh.f.copy()
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    c = v[f].mean(axis=1)
    out = np.c_[c[:, 0], c[:, 1], np.zeros(len(c))]
    want = np.where(np.abs(n[:, 2]) > 0.7 * np.linalg.norm(n, axis=1), np.einsum("ij,ij->i", n, [[0, 0, 1]] * len(n)),
                    np.einsum("ij,ij->i", n, out))
    flip = want < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    return uk.Mesh(v, f)


def _post_profile() -> list:
    r0, r1 = POST_R
    top = RIM + POST_H
    return [(0.0, RIM - 0.001), (r0, RIM - 0.001), (r0 * 0.9, RIM + 0.001), (r1 + 0.0012, top - 0.0008), (r1, top), (0.0, top)]


def _rack_mesh() -> uk.Mesh:
    sx, sy, sz = RACK
    z_floor = sz - SOCKET_DEPTH
    tiles = []
    for x in SOCKET_X:
        prof = [(SOCKET_R + 0.0008, sz), (SOCKET_R, sz - 0.0008), (SOCKET_R, z_floor + 0.0005), (SOCKET_R - 0.002, z_floor),
                (0.0, z_floor)]
        tiles.append(_holed_tile(sx / len(SOCKET_X), sy, prof).moved((x - RACK_X, 0.0, 0.0)))
    body = uk.box((sx, sy, z_floor), (0, 0, z_floor / 2), round_xy=0.003)
    sides = uk.ring_prism(uk.rounded_rect(sx, sy, 0.003), uk.rounded_rect(sx - 0.002, sy - 0.002, 0.002), z_floor, sz)
    return uk.join(body, sides, *tiles)


def _ball_mesh() -> uk.Mesh:
    prof = [(BALL_R * math.sin(a), -BALL_R * math.cos(a)) for a in np.linspace(0, math.pi, 17)]
    prof[0], prof[-1] = (0.0, -BALL_R), (0.0, BALL_R)
    return uk.lathe(prof, segments=32)


# -- colliders ------------------------------------------------------------------------------------------------------
def _quat_xyzw_from_matrix(m: np.ndarray) -> tuple:
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return x, y, z, w


def _box_rot(name: str, size, center, rot: np.ndarray, root: str) -> str:
    """A box collider turned by the rotation matrix `rot` (not rendered: purpose "guide")."""
    x, y, z, w = _quat_xyzw_from_matrix(rot)
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = 0.003
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})
        quatf xformOp:orient = ({w:.7f}, {x:.7f}, {y:.7f}, {z:.7f})
        float3 xformOp:scale = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}"""


def _cup_colliders(tag: str, cx: float, cy: float, z0: float, root: str) -> list:
    """Floor box and eight sloped slabs whose inner faces are the cup's taper (inscribed), around (cx, cy)."""
    slope_ang = math.atan2(R_RIM - R_FLOOR, DEPTH)  # from vertical
    length = math.hypot(R_RIM - R_FLOOR, DEPTH)
    width = 2 * R_RIM * math.tan(math.pi / FACETS) + 0.004
    out = [uk.collider(f"col_{tag}_floor", (2 * R_FLOOR + 0.004, 2 * R_FLOOR + 0.004, 0.004),
                       (cx, cy, z0 + FLOOR - 0.002), root)]
    for i in range(FACETS):
        phi = 2 * math.pi * i / FACETS
        radial = np.array([math.cos(phi), math.sin(phi), 0.0])
        tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])
        # Along the slope, upward and outward; the slab's outward normal tilts down by the same angle.
        up = np.array([radial[0] * math.sin(slope_ang), radial[1] * math.sin(slope_ang), math.cos(slope_ang)])
        normal = np.cross(tangent, up)  # outward and down
        mid_inner = np.array([cx, cy, z0]) + radial * (R_FLOOR + R_RIM) / 2 + np.array([0, 0, (FLOOR + RIM) / 2])
        centre = mid_inner + normal * SLAB_T / 2
        rot = np.c_[normal, tangent, up]  # box x: thickness, y: width, z: along the slope
        out.append(_box_rot(f"col_{tag}_wall{i}", (SLAB_T, width, length), centre, rot, root))
    return out


def _carton_colliders(k: int, root: str) -> list:
    cy0, z0 = CARTON_Y[k], BASE
    ox, oy = OUTER
    cols = []
    for r, rx in enumerate(ROW_X):
        for c, ccy in enumerate(COL_Y):
            cols += _cup_colliders(f"c{k + 1}_r{r}{c}", CARTON_X + rx, cy0 + ccy, z0, root)
    deck_t = 0.003
    zd = z0 + RIM - deck_t / 2
    gap = PITCH - 2 * R_RIM  # 4 mm ridges between cups
    for i, xr in enumerate((PITCH / 2, -PITCH / 2)):
        cols.append(uk.collider(f"col_c{k + 1}_ridge_x{i}", (gap, oy, deck_t), (CARTON_X + xr, cy0, zd), root))
    cols.append(uk.collider(f"col_c{k + 1}_ridge_mid", (ox, gap, deck_t), (CARTON_X, cy0, zd), root))
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for i, px in enumerate(POST_X):
        ring = [(CARTON_X + px + r * np.cos(t), cy0 + r * np.sin(t), z0 + z) for r, z in ((POST_R[0], RIM - 0.001), (POST_R[1], RIM + POST_H)) for t in th]
        cols.append(uk.hull_collider(f"col_c{k + 1}_post{i}", ring, root))
    rim_x = (ox - 3 * PITCH) / 2 + gap / 2  # flange plus half a ridge
    rim_y = (oy - 2 * PITCH) / 2 + gap / 2
    taper = 0.006  # the outer wall comes in this much from rim to base
    for i, s in enumerate((1, -1)):
        # End and side walls: the rim band on top, the outer face tapering in toward the base (hulls of 8 corners).
        xo, xi = ox / 2, ox / 2 - rim_x
        end = [(CARTON_X + s * x, cy0 + t * y, z0 + z) for (x, y, z) in
               ((xo, oy / 2, RIM), (xi, oy / 2, RIM), (xo - taper, oy / 2 - taper, 0.0), (xi - taper, oy / 2 - taper, 0.0))
               for t in (1, -1)]
        yo, yi = oy / 2, oy / 2 - rim_y
        side = [(CARTON_X + t * x, cy0 + s * y, z0 + z) for (x, y, z) in
                ((ox / 2, yo, RIM), (ox / 2, yi, RIM), (ox / 2 - taper, yo - taper, 0.0), (ox / 2 - taper, yi - taper, 0.0))
                for t in (1, -1)]
        cols.append(uk.hull_collider(f"col_c{k + 1}_end{i}", end, root))
        cols.append(uk.hull_collider(f"col_c{k + 1}_side{i}", side, root))
    return cols


def _rack_colliders(k: int, root: str) -> list:
    sx, sy, sz = RACK
    y0 = RACK_Y[k]
    z_floor = sz - SOCKET_DEPTH
    cols = [uk.collider(f"col_rack{k}_body", (sx, sy, z_floor), (RACK_X, y0, z_floor / 2), root)]
    zt = z_floor + SOCKET_DEPTH / 2
    side_w = sy / 2 - SOCKET_HALF
    for i, s in enumerate((1, -1)):
        cols.append(uk.collider(f"col_rack{k}_side{i}", (sx, side_w, SOCKET_DEPTH), (RACK_X, y0 + s * (SOCKET_HALF + side_w / 2), zt), root))
    edges = [RACK_X - sx / 2] + [e for x in SOCKET_X for e in (x - SOCKET_HALF, x + SOCKET_HALF)] + [RACK_X + sx / 2]
    for i in range(0, len(edges), 2):
        a, b = edges[i], edges[i + 1]
        cols.append(uk.collider(f"col_rack{k}_bar{i // 2}", (b - a, 2 * SOCKET_HALF, SOCKET_DEPTH), ((a + b) / 2, y0, zt), root))
    return cols


def _bench() -> str:
    looks = {
        "pulp": uk.Look(texture="pulp.png", color=(1.0, 1.0, 1.0), roughness=0.95, tile=0.1),
        "pine": uk.Look(texture="pine.png", color=(1.0, 1.0, 1.0), roughness=0.7, tile=0.1),
        "board": uk.Look(texture="board.png", color=(1.0, 1.0, 1.0), roughness=0.85, tile=0.1),
    }
    root = "Bench"
    bx, by = BOARD_XY
    parts = [("board", uk.box(BOARD, (bx, by, BOARD[2] / 2), round_xy=0.004), "board")]
    carton = _carton_mesh()
    for k, cy in enumerate(CARTON_Y):
        parts.append((f"carton_{k + 1}", carton.moved((CARTON_X, cy, BASE)), "pulp"))
    rack = _rack_mesh()
    for k, ry in enumerate(RACK_Y):
        parts.append((f"rack_{k + 1}", rack.moved((RACK_X, ry, 0.0)), "pine"))
    cols = [uk.collider("col_board", BOARD, (bx, by, BOARD[2] / 2), root)]
    for k in range(len(CARTON_Y)):
        cols += _carton_colliders(k, root)
    for k in range(len(RACK_Y)):
        cols += _rack_colliders(k, root)
    # A rigid body for the loader; the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "bench.usda"), root, parts, looks, cols, mass=5.0, friction=(0.6, 0.5), crease=50.0)


def _ball() -> str:
    root = "Ball"
    looks = {"paint": uk.Look(color=(0.62, 0.025, 0.02), roughness=0.42)}
    col = f"""
    def Sphere "col_ball" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterial>
        uniform token purpose = "guide"
        double radius = {BALL_R}
        float physxCollision:contactOffset = 0.003
        float physxCollision:restOffset = 0
    }}"""
    path = uk.write_usda(str(OUT / "ball.usda"), root, [("ball", _ball_mesh(), "paint")], looks, [col], mass=BALL_MASS,
                         friction=BALL_FRICTION, crease=80.0)
    text = Path(path).read_text()
    text = text.replace("    float physxRigidBody:maxDepenetrationVelocity = 1\n",
                        "    float physxRigidBody:maxDepenetrationVelocity = 1\n"
                        f"    float physxRigidBody:angularDamping = {ANGULAR_DAMPING}\n"
                        f"    float physxRigidBody:linearDamping = {LINEAR_DAMPING}\n", 1)
    Path(path).write_text(text)
    return path


def build() -> None:
    _bench()
    _ball()


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")), "seated centre", round(seated_centre(), 4),
          "rack centre", round(rack_centre(), 4))
