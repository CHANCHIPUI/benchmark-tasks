"""Geometry of the lock-before-loading scene, and where it stands on the Dream station table.

The stand, platform, pin, cradle, blocks and hand are described here in their own frame ("build frame"), and
:func:`place` puts that frame on the station: turned 180 degrees about the vertical through the stand, so the pin
goes in from the robot's side, and moved within the arms' reach. Pure NumPy, so the task file can import it before
Isaac Sim starts.

Build frame: the pin's handle is at +x and it is pushed in along -x. The platform spans y -0.20..0.20 on an axle along
x at height ZA, through the plate's mid-thickness. With the pin out, the platform tips toward -y until its lug, hanging
under that half, rests on the table through a slot in the base. The hand lies on the table under the +y end.
"""

from __future__ import annotations

import math

import numpy as np

# -- build frame (metres) -------------------------------------------------------------------------------------------
XC, ZA = 0.45, 0.83                    # stand centre x, hinge height
BUILD_TABLE = 0.75                     # table top the build was made for
PL_HX, PL_HY, PL_HZ = 0.05, 0.20, 0.008
PL_TOP = ZA + PL_HZ
MARK_Y = 0.15                          # the white marks on the platform, either side of the hinge
BLK = 0.02                             # block half size
BLOCK_MASS = {"left": 0.25, "right": 0.15}   # "left" is the hand side in the build frame
BLOCK_START = {"left": (0.62, 0.28), "right": (0.62, -0.28)}

PIN_R = 0.006
PIN_Y, PIN_Z = -0.13, ZA - 0.024
PIN_X0, PIN_X1 = XC - 0.068, XC + 0.090          # shaft ends when inserted
PIN_LEN = PIN_X1 - PIN_X0
PIN_MASS = 0.03
TAB_HALF = (0.005, 0.015, 0.015)                 # flat handle at the shaft's +x end
TAB_IN = np.array([PIN_X1 + TAB_HALF[0], PIN_Y, PIN_Z])   # handle centre when inserted
TAB_OUT = np.array([0.70, -0.195, 0.79])         # handle centre when the pin lies in its cradle
CRADLE_X = (TAB_OUT[0] - 0.13, TAB_OUT[0] - 0.05)

HOLE = 0.0085                                    # half side of the side walls' square holes
CHEEK_X, CHEEK_HX = 0.058, 0.006                 # side walls at XC +- CHEEK_X
LUG_HX = CHEEK_X - CHEEK_HX - 0.015              # half length of the lug's bore (15 mm gaps to the walls)
LUG_HOLE = PIN_R + 0.00125
LUG_Y, HUB_Z0 = (PIN_Y - 0.020, PIN_Y + 0.020), -0.040

FINGER_FROM = np.array([XC + 0.032, 0.255, 0.7595])
FINGER_TO = np.array([XC + 0.032, 0.165, 0.7595])
FINGER_R = 0.0095
PALM = ((XC - 0.01, 0.30, 0.763), (0.084, 0.09, 0.026))
CURLED_X = (-0.032, -0.012, 0.012)
THUMB = ((XC + 0.0475, 0.2975, 0.762), math.radians(45.0))

# the lug's outer bottom corner touches the table at this hinge angle
FALLEN_TILT = float(next(t for t in np.radians(np.arange(0.0, 40.0, 0.01))
                         if ZA + LUG_Y[0] * np.sin(t) + HUB_Z0 * np.cos(t) <= BUILD_TABLE))

# -- placement on the station --------------------------------------------------------------------------------------
STATION_TABLE = 0.7475                 # yam_arena.task_file.TABLE_TOP["station"]
STAND_XY = (0.40, 0.0)                 # where the stand's centre goes on the station
YAW = math.pi                          # the build frame turned about the vertical through the stand
DZ = STATION_TABLE - BUILD_TABLE

_C, _S = math.cos(YAW), math.sin(YAW)
ROT = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
OFFSET = np.array([STAND_XY[0], STAND_XY[1], DZ]) - ROT @ np.array([XC, 0.0, 0.0])


def place(p) -> np.ndarray:
    """A build-frame point on the station (world frame)."""
    return ROT @ np.asarray(p, float) + OFFSET


def unplace(p) -> np.ndarray:
    """A station point in the build frame."""
    return ROT.T @ (np.asarray(p, float) - OFFSET)


def yaw_xyzw(extra: float = 0.0) -> tuple[float, float, float, float]:
    """The station orientation of a build-frame body turned ``extra`` about the vertical."""
    a = YAW + extra
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


# -- the lock test (build frame) -------------------------------------------------------------------------------------
def lug_hole_at(tilt_rad: float) -> tuple[float, float]:
    """Build-frame (y, z) of the lug hole's axis with the platform tilted by ``tilt_rad`` (right end down > 0)."""
    y, z = PIN_Y, PIN_Z - ZA
    c, s = math.cos(tilt_rad), math.sin(tilt_rad)
    return y * c - z * s, ZA + y * s + z * c


def pin_locks(base, axis, tilt_rad: float) -> bool:
    """Whether a shaft starting at ``base`` (its handle end) and running along the unit vector ``axis`` (build frame)
    locks the hinge, pushed in from either side. It must reach at least the middle of both side walls, and its axis
    must lie inside both walls' holes and the lug's hole. A shaft that stops short of the far wall hangs from one wall
    alone and lets the platform rock."""
    base, axis = np.asarray(base, float), np.asarray(axis, float)
    if abs(axis[0]) < 0.9:
        return False
    ends = sorted((base[0], base[0] + axis[0] * PIN_LEN))
    if ends[0] > XC - CHEEK_X or ends[1] < XC + CHEEK_X:
        return False

    def at(x):
        return base + axis * (x - base[0]) / axis[0]

    near, lug, far = at(XC + CHEEK_X), at(XC), at(XC - CHEEK_X)
    ly, lz = lug_hole_at(tilt_rad)
    return bool(all(abs(p[1] - PIN_Y) < HOLE and abs(p[2] - PIN_Z) < HOLE for p in (near, far))
                and abs(lug[1] - ly) < LUG_HOLE + 0.001 and abs(lug[2] - lz) < LUG_HOLE + 0.001)
