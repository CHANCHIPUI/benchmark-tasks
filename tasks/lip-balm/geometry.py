"""Dimensions and layout of the lip balm task.

Station frame: metres from the midpoint between the arm bases, x forward, y to the robot's left, z up; heights are
above the table top (TOP).

The head is the MakeHuman base mesh (hm08, CC0, github.com/makehumancommunity/makehuman), cut below the jaw at the
neck and scaled to a 52 cm head circumference (the Mehron Practice Head's listed size). Its "MH frame" is the mesh's
own: x to the resident's left, y up the body, z out of the face, in metres before scaling.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                       # the station's table top

# -- the scale, board and towels ---------------------------------------------------------------------------------------
SCALE_C = (0.34, -0.13)            # the kitchen scale's centre
SCALE_SIZE = (0.18, 0.15, 0.020)   # platform 18 x 15 cm, 2 cm tall
BOARD_X = (0.25, 0.55)
BOARD_Y = (-0.38, 0.12)
BOARD_T = 0.006
TOWEL_X = (0.27, 0.52)             # folded hand towel, 25 x 45 cm, 1 cm thick
TOWEL_Y = (-0.33, 0.12)
TOWEL_T = 0.010
BED = SCALE_SIZE[2] + BOARD_T + TOWEL_T     # the towel's top: 3.6 cm
ROLL_R = 0.025                     # the rolled hand towel behind the head, 5 cm across
ROLL_X = 0.480
ROLL_Y = (-0.20, 0.08)

# -- the head ----------------------------------------------------------------------------------------------------------
HEAD_SCALE = 0.952                 # MakeHuman's 54.6 cm head circumference to 52 cm
FACE_ELEV = math.radians(45)       # the face points toward -x, 45 degrees above horizontal
NECK_CUT_MH = 0.600                # cut the body at this MH height (the neck's base, 10 cm across once scaled)
MOUTH_MH = np.array([0.0, 0.6608, 0.1555])  # the middle of the mouth (between the lips, on the surface), MH frame
MOUTH = np.array([0.300, -0.080, 0.1885])   # where it sits: the back of the head then rests on the towel


def head_rotation(elev: float = FACE_ELEV) -> np.ndarray:
    """Columns: the MH frame's x (resident's left), y (toward the crown) and z (out of the face) in the station frame."""
    n = np.array([-math.cos(elev), 0.0, math.sin(elev)])
    up = np.array([0.0, 1.0, 0.0])
    return np.stack([np.cross(up, n), up, n], 1)


HEAD_R = head_rotation()
FACE_N = HEAD_R[:, 2]              # the face's outward normal
LIP_LINE = HEAD_R[:, 0]            # corner to corner, toward the resident's left (the high corner)


def mh_to_world(p) -> np.ndarray:
    """MH-frame points (metres, unscaled) to the station frame (table-top heights)."""
    p = np.asarray(p, float)
    return (p - MOUTH_MH) * HEAD_SCALE @ HEAD_R.T + MOUTH


def world_to_mh(p) -> np.ndarray:
    p = np.asarray(p, float)
    return (p - MOUTH) @ HEAD_R / HEAD_SCALE + MOUTH_MH


# -- the lips, traced on the mesh (MH frame, mm, front view: x across, y up; symmetric about x = 0) --------------------
# the vermilion's outer border is the lip's sculpted edge: on the upper lip, the ridge of the white roll (the most
# forward point on each vertical line); on the lower lip, where the surface turns under toward the chin.
CORNER_X_MH = 25.5                 # the mouth corners
MOUTH_Y_MH = 661.0                 # the corners' height
UPPER_BORDER = [(0.0, 668.3), (2.2, 668.4), (5.3, 669.0), (8.0, 669.3), (10.9, 668.6), (14.4, 667.6),
                (17.8, 667.0), (20.9, 665.8), (23.9, 664.3), (25.5, 661.0)]
LIP_JOIN = [(0.0, 660.6), (5.0, 660.9), (10.0, 661.1), (15.0, 661.4), (20.0, 661.5), (23.0, 661.4), (25.5, 661.0)]
LOWER_BORDER = [(0.0, 649.0), (5.4, 649.4), (10.0, 649.8), (13.0, 650.2), (16.0, 650.8), (19.0, 651.8),
                (22.0, 653.6), (24.0, 656.0), (25.5, 661.0)]
CORNER_IN = 0.003                  # regions start 3 mm (world) in from each corner
N_REGIONS = 6                      # per lip
BAND = 0.002                       # the neutral skin band outside the border
SAMPLE = 0.001                     # sample spacing
MARK_R = 0.003                     # balm marks lip points within 3 mm of the tip's contact
LIP_FORCE = 0.3                    # N: the press that leaves balm on the lip
SKIN_FORCE = 0.1                   # N: any touch on the skin this hard marks it
COATED = 0.5                       # a region is coated at 50% of its points marked


def border_y(table, x_mm) -> np.ndarray:
    xs, ys = np.array(table).T
    return np.interp(np.abs(x_mm), xs, ys)


# -- the tray and the balm stick -----------------------------------------------------------------------------------------
TRAY_X = (0.12, 0.24)
TRAY_Y = (-0.04, 0.16)
TRAY_RIM = 0.015
TRAY_T = 0.002
STICK_C = (0.18, 0.06)             # the capped stick lies along y, cap toward +y
TUBE_R = 0.008                     # tube 16 mm across
TUBE_L = 0.049                     # the tube's body, base to the shoulder
NECK_R = 0.0072                    # the neck the cap slides over
NECK_L = 0.008
BULLET_R = 0.0055                  # bullet 11 mm across
BULLET_OUT = 0.004                 # wound out 4 mm past the tube's rim (the dome's tip)
CAP_L = 0.020                      # cap 20 mm long, over the neck and the bullet
CAP_R = 0.008
CAP_BORE = 0.0076                  # the cap's bore over the 7.2 mm neck (0.4 mm clearance)
CAP_TOP = 0.0015                   # the cap's closed end
CAP_SOLID_R = 0.0080               # the solid cap collider: the cap's own 8 mm radius (it must not dip into the tray)
CAP_WALL_COLLIDE = 0.0016          # the cap's collider wall: 1 mm thicker than the real 0.6 mm wall, outward
CAPPED_L = TUBE_L + CAP_L          # 69 mm
TUBE_MASS = 0.012                  # the tube, the twist base and the wax (4.25 g of balm)
BULLET_MASS = 0.003
CAP_MASS = 0.002
CAP_PULL = 4.0                     # N: the two-hand axial pull that takes the cap off (task.py)
SLIDE_K = 1000.0                   # N/m: the bullet's sprung slide (1 N/mm)
SLIDE_TRAVEL = 0.003


def tip_from_base() -> float:
    """The bullet's dome tip, along the stick from the tube's base end."""
    return TUBE_L + NECK_L + BULLET_OUT


# -- scoring -------------------------------------------------------------------------------------------------------------
SCALE_LIMIT_N = 0.100 * 9.81           # 100 g, as a force: 0.981 N
SCALE_AVG_STEPS = 5                # 0.25 s at 20 Hz
CAP_SEATED = 0.001                 # cap within 1 mm of fully on
HANDS_OFF = 0.03                   # grippers 3 cm or more from the stick and cap
STILL = 0.002                      # m/s
