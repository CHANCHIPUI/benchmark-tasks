"""Geometry of the Go set, and the USD files built from it (into assets/, at load time). Pure NumPy. Sizes in metres, in
the station frame (x forward from the arm bases, y to the robot's left), heights above the table top.

The set copies a 9x9 Go demonstration set (a 42 cm board with 4 cm stones, for teaching groups), laid flat on the table:
- Board: beech-faced, 47 x 47 x 2 cm, nine lines each way 50 mm apart, 5 mm wide, five star points, a 35 mm margin.
- Stones: single-convex melamine (flat underside, 5 mm vertical edge, domed top), 40 mm across, 12 mm thick, 16 g.
- Bowls: turned chestnut, 17 cm across and 5.7 cm tall, each with its lid upturned beside it as in play.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- board --------------------------------------------------------------------------------------------------------
BOARD = (0.47, 0.47, 0.020)  # x, y, thickness
BOARD_XY = (0.40, 0.085)  # centre: x 0.165 to 0.635, y -0.15 to +0.32
BOARD_TOP = BOARD[2]
PITCH = 0.050
LINE_T = 0.005
STAR_D = 0.014
COLS = "ABCDEFGHJ"  # from the robot's left (Go boards skip I)
ROW_X = {r: 0.20 + (r - 1) * PITCH for r in range(1, 10)}  # row 1 nearest the arms
COL_Y = {c: 0.285 - k * PITCH for k, c in enumerate(COLS)}  # A at +0.285 ... J at -0.115
STARS = ["C3", "G3", "C7", "G7", "E5"]
BOARD_X_RANGE = (BOARD_XY[0] - BOARD[0] / 2, BOARD_XY[0] + BOARD[0] / 2)
BOARD_Y_RANGE = (BOARD_XY[1] - BOARD[1] / 2, BOARD_XY[1] + BOARD[1] / 2)


def point(name: str) -> tuple[float, float]:
    """An intersection's (x, y), from its Go name such as 'J1'."""
    return ROW_X[int(name[1:])], COL_Y[name[0]]


# -- stones -------------------------------------------------------------------------------------------------------
STONE_R = 0.020
STONE_H = 0.012
STONE_EDGE = 0.005  # the vertical edge's height
STONE_BEVEL = 0.0008
DOME_R = (STONE_R**2 + (STONE_H - STONE_EDGE) ** 2) / (2 * (STONE_H - STONE_EDGE))  # the top's sphere radius
MELAMINE_KG_M3 = 1500.0
STONE_FRICTION = (0.6, 0.5)  # melamine on wood and on the gripper pads


def stone_profile(segments: int = 9) -> list[tuple[float, float]]:
    """(radius, z) from the underside's centre up: flat underside, small bevel, vertical edge, domed top."""
    prof = [(0.0, 0.0), (STONE_R - STONE_BEVEL, 0.0), (STONE_R, STONE_BEVEL), (STONE_R, STONE_EDGE)]
    zc = STONE_H - DOME_R  # the dome's sphere centre
    a0 = math.asin(STONE_R / DOME_R)
    for k in range(1, segments + 1):
        a = a0 * (1 - k / segments)
        prof.append((DOME_R * math.sin(a), zc + DOME_R * math.cos(a)))
    prof[-1] = (0.0, STONE_H)
    return prof


def stone_volume() -> float:
    cap_h = STONE_H - STONE_EDGE
    return math.pi * STONE_R**2 * STONE_EDGE + math.pi * cap_h**2 * (3 * DOME_R - cap_h) / 3


STONE_MASS = round(MELAMINE_KG_M3 * stone_volume(), 4)  # 0.016 kg

# -- bowls and lids -----------------------------------------------------------------------------------------------
BOWL_R = 0.085  # outside, at the belly
BOWL_H = 0.057
BOWL_BASE = 0.012  # the inside floor's height
BOWL_IN_R = 0.075  # the inside floor's radius
LID_R = 0.085
LID_H = 0.025
LID_FLOOR = 0.006
LID_WALL_R = 0.0785  # the lid's inner wall (15.7 cm across inside)
LID_IN_R = 0.075  # "in the lid": a stone's centre within this of the lid's centre
BLACK_BOWL_XY = (0.215, -0.24)
BLACK_LID_XY = (0.40, -0.24)
WHITE_BOWL_XY = (0.73, 0.20)
WHITE_LID_XY = (0.73, 0.01)
BOWL_STONES_X = (-0.045, 0.0, 0.045)  # three black stones in a row along x, from the bowl's centre: each one can be
# lifted straight down with the fingers closing across the row (along y), 4 mm inside the bowl's wall

# -- the problem --------------------------------------------------------------------------------------------------
BLACK = ["F3", "G2", "G5", "H1", "H3", "J3"]
WHITE = ["G3", "G4", "H2", "J2"]
CAPTURED = ["H2", "J2"]
ANSWER = "J1"


def _bowl_profile() -> list[tuple[float, float]]:
    """A squat goke: narrow foot, round belly, slightly narrower mouth; then the inside, down to a flat floor."""
    out = [(0.0, 0.0), (0.060, 0.0), (0.066, 0.003), (0.076, 0.010), (0.083, 0.020), (BOWL_R, 0.030), (0.084, 0.040),
           (0.081, 0.050), (0.0795, BOWL_H - 0.002), (0.078, BOWL_H)]
    inside = [(0.0745, BOWL_H), (0.0755, 0.045), (0.0760, 0.030), (0.0755, 0.020), (BOWL_IN_R - 0.004, BOWL_BASE + 0.001),
              (BOWL_IN_R - 0.008, BOWL_BASE), (0.0, BOWL_BASE)]
    return out + inside


def _lid_profile() -> list[tuple[float, float]]:
    """An upturned lid: a shallow dish standing on its (once top) face, open upward."""
    out = [(0.0, 0.0), (0.078, 0.0), (0.083, 0.003), (LID_R, 0.008), (LID_R, LID_H - 0.002), (0.0838, LID_H)]
    inside = [(LID_WALL_R, LID_H), (LID_WALL_R, LID_FLOOR + 0.001), (LID_WALL_R - 0.001, LID_FLOOR), (0.0, LID_FLOOR)]
    return out + inside


def _hull_of(mesh: uk.Mesh, name: str, root: str) -> str:
    """A convex-hull collider from a closed convex mesh's own triangles (usdkit.hull_collider without its brute-force search)."""
    text = uk.hull_collider(name, mesh.v[:4], root)  # the prim's template, from four points
    faces = ", ".join(str(int(i)) for t in mesh.f for i in t)
    points = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in mesh.v)
    text = re.sub(r"int\[\] faceVertexCounts = \[.*?\]", "int[] faceVertexCounts = [" + ", ".join("3" for _ in mesh.f) + "]", text)
    text = re.sub(r"int\[\] faceVertexIndices = \[.*?\]", "int[] faceVertexIndices = [" + faces + "]", text)
    return re.sub(r"point3f\[\] points = \[.*?\]", "point3f[] points = [" + points + "]", text)


def _lathe_open(profile, segments: int = 48) -> uk.Mesh:
    """usdkit.lathe for a profile that starts and ends on the axis (a closed solid, outside then inside)."""
    return uk.lathe(profile, segments)


def _ring_hulls(prefix: str, root: str, cx: float, cy: float, r_in: float, r_out: float, z0: float, z1: float,
                n: int = 16) -> list[str]:
    """A round wall as n convex wedges (PhysX colliders), each from r_in to r_out and z0 to z1."""
    out = []
    for k in range(n):
        a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
        pts = []
        for a in (a0, a1):
            for r in (r_in, r_out):
                for z in (z0, z1):
                    pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
        out.append(uk.hull_collider(f"{prefix}_{k:02d}", pts, root))
    return out


def _wood_looks() -> dict[str, uk.Look]:
    return {
        "board_print": uk.Look(texture="board_print.png", color=(1, 1, 1), roughness=0.5,
                               origin=(BOARD_XY[0] - BOARD[0] / 2, BOARD_XY[1] - BOARD[1] / 2, 0.0),
                               tile_uv=(BOARD[0], BOARD[1])),
        "board_side": uk.Look(texture="board_side.jpg", color=(1, 1, 1), roughness=0.55, tile=0.12),
        "chestnut": uk.Look(texture="chestnut.jpg", color=(1, 1, 1), roughness=0.45, tile=0.08),
        "white_stone": uk.Look(color=(0.80, 0.80, 0.76), roughness=0.28),
    }


def _set() -> str:
    """The board, both bowls and both lids: one fixed body whose origin is the table top under the arms' midpoint.
    White's stones in White's bowl are part of it (visual only): White's set is out of reach, on the far side."""
    bx, by, bz = BOARD
    cx, cy = BOARD_XY
    skin = 0.0004
    root = "GoSet"
    looks = _wood_looks()
    parts = [
        ("board", uk.box((bx, by, bz - skin), (cx, cy, (bz - skin) / 2), bevel=0.0015), "board_side"),
        ("board_print", uk.box((bx - 0.003, by - 0.003, skin), (cx, cy, bz - skin / 2)), "board_print"),
    ]
    cols = [uk.collider("col_board", BOARD, (cx, cy, bz / 2), root)]
    bowl = _lathe_open(_bowl_profile())
    lid = _lathe_open(_lid_profile())
    for tag, (x, y) in (("black", BLACK_BOWL_XY), ("white", WHITE_BOWL_XY)):
        parts.append((f"bowl_{tag}", bowl.moved((x, y, 0.0)), "chestnut"))
        cols.append(_hull_of(uk.cylinder(BOWL_IN_R, BOWL_BASE, (x, y, BOWL_BASE / 2), segments=24), f"col_bowl_{tag}_floor", root))
        cols += _ring_hulls(f"col_bowl_{tag}_wall", root, x, y, 0.0762, 0.0835, BOWL_BASE, BOWL_H)
    for tag, (x, y) in (("black", BLACK_LID_XY), ("white", WHITE_LID_XY)):
        parts.append((f"lid_{tag}", lid.moved((x, y, 0.0)), "chestnut"))
        cols.append(_hull_of(uk.cylinder(LID_R - 0.002, LID_FLOOR, (x, y, LID_FLOOR / 2), segments=24), f"col_lid_{tag}_floor", root))
        cols += _ring_hulls(f"col_lid_{tag}_wall", root, x, y, LID_WALL_R, LID_R, LID_FLOOR, LID_H)
    # White's stones in White's bowl: a loose layer with a few on top, visual only.
    stone = uk.lathe(stone_profile(), 32)
    wx, wy = WHITE_BOWL_XY
    rng = np.random.default_rng(3)
    layer = [(0.0, 0.0), (0.045, 0.0), (-0.045, 0.0), (0.0225, 0.039), (-0.0225, 0.039), (0.0225, -0.039),
             (-0.0225, -0.039)]
    for k, (dx, dy) in enumerate(layer):
        m = stone.turned(float(rng.uniform(0, 360)), "z").moved((wx + dx, wy + dy, BOWL_BASE))
        parts.append((f"white_stone_{k}", m, "white_stone"))
    for k, (dx, dy, tilt) in enumerate([(0.012, 0.018, 9.0), (-0.02, -0.012, 12.0), (0.03, -0.02, 7.0)]):
        m = stone.turned(tilt, "x").turned(float(rng.uniform(0, 360)), "z").moved((wx + dx, wy + dy, BOWL_BASE + 0.011))
        parts.append((f"white_stone_top_{k}", m, "white_stone"))
    # Written as a rigid body (the loader needs one); the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "go_set.usda"), root, parts, looks, cols, mass=6.0, friction=(0.5, 0.4))


def _stone(color: str) -> str:
    """One stone: origin at the centre of its flat underside, axis along z."""
    root = "BlackStone" if color == "black" else "WhiteStone"
    looks = {"black": uk.Look(color=(0.012, 0.012, 0.014), roughness=0.30),
             "white": uk.Look(color=(0.80, 0.80, 0.76), roughness=0.28)}
    mesh = uk.lathe(stone_profile(), 40)
    cols = [_hull_of(uk.lathe(stone_profile(5), 24), "col_stone", root)]
    return uk.write_usda(str(OUT / f"{color}_stone.usda"), root, [("stone", mesh, color)], looks, cols,
                         mass=STONE_MASS, friction=STONE_FRICTION, crease=50.0)


def build() -> None:
    _set()
    _stone("black")
    _stone("white")


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")), "stone mass", STONE_MASS, "dome radius", round(DOME_R, 4))
