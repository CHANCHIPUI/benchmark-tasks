"""Geometry of the map board and the two map markers, and the USD files built from them (into assets/, at load time).
Pure NumPy. Sizes in metres.

- Map board: a printed map, laminated in matte film and mounted on 3 mm hardboard, taped to the table (fixed). Its origin is
  the table top under the arms' midpoint, so the layout's numbers (layout.py) are its own coordinates.
- Map marker: a weighted acetal cylinder, 25 mm across and 40 mm tall, 40 g, with an 8 mm white dot centred on its top.
  Its origin is its centre.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import layout as L
import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

FILM = 0.0003  # the laminated print on top of the hardboard (visual skin)
MARKER_R = 0.0125
MARKER_H = 0.040
MARKER_KG = 0.040
DOT_R = 0.004
MARKER_FRICTION = (0.5, 0.4)  # acetal on laminate film
# sRGB colours turned linear for USD
RED = (0.56, 0.012, 0.010)
BLUE = (0.010, 0.085, 0.50)


def _board() -> str:
    x0, x1 = L.BOARD_X
    y0, y1 = L.BOARD_Y
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy, t = x1 - x0, y1 - y0, L.BOARD_T
    looks = {
        "hardboard": uk.Look(texture="hardboard.jpg", color=(1, 1, 1), roughness=0.85, tile=0.05),
        "map": uk.Look(texture="map.png", color=(1, 1, 1), roughness=0.72, origin=(x0, y0, 0.0), tile_uv=(sx, sy)),
    }
    parts = [
        ("hardboard", uk.box((sx, sy, t - FILM), (cx, cy, (t - FILM) / 2)), "hardboard"),
        ("map", uk.box((sx, sy, FILM), (cx, cy, t - FILM / 2)), "map"),
    ]
    cols = [uk.collider("col_board", (sx, sy, t), (cx, cy, t / 2), "MapBoard")]
    # Written as a rigid body (the loader needs one); the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "map_board.usda"), "MapBoard", parts, looks, cols, mass=2.0, friction=(0.5, 0.4))


def _marker(name: str, color) -> str:
    h = MARKER_H / 2
    looks = {
        "acetal": uk.Look(color=color, roughness=0.38),
        "dot": uk.Look(color=(0.85, 0.85, 0.83), roughness=0.45),
    }
    body = uk.cylinder(MARKER_R, MARKER_H, segments=40, bevel=0.0012)
    dot = uk.cylinder(DOT_R, 0.0004, (0, 0, h + 0.0002 - 0.0001), segments=24)
    parts = [("body", body, "acetal"), ("dot", dot, "dot")]
    ring = [(MARKER_R * np.cos(a), MARKER_R * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    hull = [(px, py, z) for px, py in ring for z in (-h, h)]
    root = name.capitalize()
    cols = [uk.hull_collider("col_marker", hull, root)]
    return uk.write_usda(str(OUT / f"{name}.usda"), root, parts, looks, cols, mass=MARKER_KG, friction=MARKER_FRICTION)


def build() -> None:
    _board()
    _marker("red_marker", RED)
    _marker("blue_marker", BLUE)


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")))
