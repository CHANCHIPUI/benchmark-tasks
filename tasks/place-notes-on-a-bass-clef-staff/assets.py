"""Geometry of the staff board, the trays and the note heads, and the USD files built from it (into assets/, at load
time). Pure NumPy. Sizes in metres, in the station frame (x forward from the arm bases, y to the robot's left), heights
above the table top.

Staff board: a white melamine-faced board lying flat, printed with a five-line staff, a bass clef at its left end, three
bar lines plus one after the clef, and a final double bar: four bars. Line spacing 4.5 cm, as on large classroom staff
boards. Note heads: black-lacquered beech discs, 40 mm across and 18 mm thick, like large wooden game counters. Trays:
shallow beech trays along the board's near edge, one in front of each half.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- staff board ----------------------------------------------------------------------------------------------------
BOARD = (0.25, 0.63, 0.012)  # x, y, thickness
BOARD_XY = (0.30, 0.065)  # centre: x 0.175 to 0.425, y -0.25 to +0.38
BOARD_TOP = BOARD[2]
SPACE = 0.045  # staff line spacing, centre to centre
LINE_X = [0.21 + k * SPACE for k in range(5)]  # lines 1 to 5, line 1 nearest the arms
LINE_T = 0.008
BAR_Y = [0.22, 0.11, 0.0, -0.11]  # bar lines; bar k runs from BAR_Y[k] to the next one (or the final thin line)
BAR_T = 0.006
FINAL_THIN_Y, FINAL_THIN_T = -0.22, 0.006
FINAL_THICK_Y, FINAL_THICK_T = -0.232, 0.012
STAFF_Y = (0.365, FINAL_THICK_Y - FINAL_THICK_T / 2)  # the lines run from the left end to the thick final line
CLEF_LEFT_Y = 0.356  # the clef glyph's origin; its ink spans about y +0.355 to +0.232
BAR_EDGES = BAR_Y + [FINAL_THIN_Y]  # bar k (0-based) lies between BAR_EDGES[k] and BAR_EDGES[k + 1]

# Every line and space of the staff by its centre x: line n is position 2(n - 1), space n is position 2n - 1.
POSITION_X = [LINE_X[0] + k * SPACE / 2 for k in range(9)]

# -- note heads -----------------------------------------------------------------------------------------------------
DISC_R = 0.020
DISC_H = 0.018
BEECH_KG_M3 = 710.0
DISC_MASS = round(BEECH_KG_M3 * np.pi * DISC_R**2 * DISC_H, 4)  # 0.016 kg
DISC_FRICTION = (0.6, 0.5)  # lacquered wood on melamine, beech and the gripper pads

# -- trays ----------------------------------------------------------------------------------------------------------
# Two shallow trays along the board's near edge, one in front of each half, where the top camera sees them between the
# arms. Each holds two heads lying flat, 10 cm apart along y.
TRAY_IN = (0.048, 0.18)  # inside, x and y
TRAY_FLOOR = 0.005
TRAY_RIM_W, TRAY_RIM_H = 0.005, 0.010  # rim width, and its height above the table (5 mm above the floor)
TRAY_XY = {"left": (0.145, 0.22), "right": (0.145, -0.22)}
TRAY_SLOTS = {"left": (0.27, 0.17), "right": (-0.27, -0.17)}  # y of the two heads in each tray, at the tray's centre x


def _bench() -> str:
    """The board and both trays: one fixed body whose origin is the table top under the arms' midpoint."""
    bx, by, bz = BOARD
    cx, cy = BOARD_XY
    skin = 0.0004  # the printed top's skin (visual)
    looks = {
        "melamine": uk.Look(color=(0.80, 0.80, 0.78), roughness=0.55),
        "print": uk.Look(texture="staff_print.png", color=(1, 1, 1), roughness=0.55,
                         origin=(cx - bx / 2, cy - by / 2, 0.0), tile_uv=(bx, by)),
        "beech": uk.Look(texture="beech.jpg", color=(1, 1, 1), roughness=0.6, tile=0.06),
    }
    parts = [
        ("board", uk.box((bx, by, bz - skin), (cx, cy, (bz - skin) / 2), bevel=0.001), "melamine"),
        ("board_print", uk.box((bx - 0.002, by - 0.002, skin), (cx, cy, bz - skin / 2)), "print"),
    ]
    cols = [uk.collider("col_board", BOARD, (cx, cy, bz / 2), "Bench")]
    ix, iy = TRAY_IN
    ox, oy = ix + 2 * TRAY_RIM_W, iy + 2 * TRAY_RIM_W
    for side, (tx, ty) in TRAY_XY.items():
        rim = uk.ring_prism(uk.rounded_rect(ox, oy, 0.004), uk.rounded_rect(ix, iy, 0.0015), 0.0, TRAY_RIM_H)
        parts += [
            (f"tray_{side}_rim", uk.Mesh(rim.v + [tx, ty, 0], rim.f), "beech"),
            (f"tray_{side}_floor", uk.box((ix + 0.001, iy + 0.001, TRAY_FLOOR), (tx, ty, TRAY_FLOOR / 2)), "beech"),
        ]
        zc = TRAY_RIM_H / 2
        cols += [
            uk.collider(f"col_tray_{side}_floor", (ix, iy, TRAY_FLOOR), (tx, ty, TRAY_FLOOR / 2), "Bench"),
            uk.collider(f"col_tray_{side}_far", (TRAY_RIM_W, oy, TRAY_RIM_H), (tx + ix / 2 + TRAY_RIM_W / 2, ty, zc), "Bench"),
            uk.collider(f"col_tray_{side}_near", (TRAY_RIM_W, oy, TRAY_RIM_H), (tx - ix / 2 - TRAY_RIM_W / 2, ty, zc), "Bench"),
            uk.collider(f"col_tray_{side}_left", (ix, TRAY_RIM_W, TRAY_RIM_H), (tx, ty + iy / 2 + TRAY_RIM_W / 2, zc), "Bench"),
            uk.collider(f"col_tray_{side}_right", (ix, TRAY_RIM_W, TRAY_RIM_H), (tx, ty - iy / 2 - TRAY_RIM_W / 2, zc), "Bench"),
        ]
    # Written as a rigid body (the loader needs one); the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "bench.usda"), "Bench", parts, looks, cols, mass=4.0, friction=(0.5, 0.4))


def _hull_of(mesh: uk.Mesh, name: str, root: str) -> str:
    """A convex-hull collider from a closed mesh's own triangles (usdkit.hull_collider without its brute-force search)."""
    text = uk.hull_collider(name, mesh.v[:4], root)  # the prim's template, from four points
    faces = ", ".join(str(int(i)) for t in mesh.f for i in t)
    points = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in mesh.v)
    text = re.sub(r"int\[\] faceVertexCounts = \[.*?\]", "int[] faceVertexCounts = [" + ", ".join("3" for _ in mesh.f) + "]", text)
    text = re.sub(r"int\[\] faceVertexIndices = \[.*?\]", "int[] faceVertexIndices = [" + faces + "]", text)
    return re.sub(r"point3f\[\] points = \[.*?\]", "point3f[] points = [" + points + "]", text)


def _disc() -> str:
    """One note head: origin at its centre, axis along z."""
    looks = {"lacquer": uk.Look(color=(0.018, 0.018, 0.02), roughness=0.32)}
    mesh = uk.cylinder(DISC_R, DISC_H, segments=32, bevel=0.0015)
    cols = [_hull_of(uk.cylinder(DISC_R, DISC_H, segments=24), "col_disc", "NoteHead")]
    return uk.write_usda(str(OUT / "note_head.usda"), "NoteHead", [("disc", mesh, "lacquer")], looks, cols,
                         mass=DISC_MASS, friction=DISC_FRICTION)


def build() -> None:
    _bench()
    _disc()


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")), "disc mass", DISC_MASS,
          "positions", [round(x, 4) for x in POSITION_X])
