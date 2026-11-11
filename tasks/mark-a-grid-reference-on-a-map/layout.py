"""Where the map's grid lies on the table, and the two references. Pure Python; shared by make_textures.py, assets.py and
task.py. Metres, station frame: x forward from the arm bases, y to the robot's left.

The map is an invented 1:10,000 sheet with a 1 km grid: each square is 10 cm on the map. North is the far side (+x), so
east is the robot's right (-y). Easting lines 31 to 35 run west to east, northing lines 32 to 35 south to north.
"""

from __future__ import annotations

EASTINGS = (31, 32, 33, 34, 35)
NORTHINGS = (32, 33, 34, 35)
SQUARE = 0.10  # one kilometre on the map
X_N32 = 0.16  # northing line 32
Y_E33 = 0.0  # easting line 33


def northing_x(n: float) -> float:
    return X_N32 + SQUARE * (n - 32)


def easting_y(e: float) -> float:
    return Y_E33 - SQUARE * (e - 33)


def point(e: float, n: float) -> tuple[float, float]:
    """The table position (x, y) of the map point at easting e, northing n (in km, fractions allowed)."""
    return northing_x(n), easting_y(e)


# The board and its printed face.
BOARD_X = (0.05, 0.57)
BOARD_Y = (-0.29, 0.29)
BOARD_T = 0.003  # hardboard with the laminated map on top
FACE_X = (0.13, 0.49)  # the face runs 3 cm past the outer northing lines
FACE_Y = (easting_y(35), easting_y(31))  # -0.20 .. +0.20
LINE_W = 0.004
TICK = 0.01  # grid lines run on into the margins as ticks
DIGIT_H = 0.06  # grid numbers' height
EAST_LABEL_X = (0.09, 0.53)  # easting numbers: near and far margins, centred on their lines
NORTH_LABEL_Y = 0.245  # northing numbers: both side margins, centred on their lines
ARROW_XY = (0.522, -0.268)  # north arrow, in the far margin's east corner

# The references in the instruction.
RED_SQUARE = (32, 34)  # grid square 3234: south-west corner at easting 32, northing 34
BLUE_REF = (34.6, 33.2)  # grid reference 346 332
BLUE_POINT = point(34.65, 33.25)  # the centre of the 100 m square it names
BLUE_SQUARE = (34, 33)  # the 1 km square it lies in, 3433
INSIDE = 0.015  # a marker inside a square: at least this far from each of its grid lines
POINT_TOL = 0.010  # blue within this of BLUE_POINT


def square_bounds(e: int, n: int) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of grid square e n (south-west corner at easting e, northing n)."""
    x0, x1 = northing_x(n), northing_x(n + 1)
    y1, y0 = easting_y(e), easting_y(e + 1)
    return x0, x1, y0, y1


def inside_square(x: float, y: float, e: int, n: int, margin: float = INSIDE) -> bool:
    x0, x1, y0, y1 = square_bounds(e, n)
    return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin
