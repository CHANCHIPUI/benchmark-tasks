"""Every size, place and constant of the ceiling-light-pair task (metres, kg, N, s; world frame of yam_pair's
side_by_side layout at 1.0 m: origin midway between the stations, x forward from the arms, y toward station A, z up;
heights are given above the table top TOP). Items marked "estimate" have no published source."""

from __future__ import annotations

import math

TOP = 0.7475                      # yam_pair TABLE_TOP
SEPARATION = 1.0                  # station origins at y +0.5 (A) and -0.5 (B)

# -- ceiling stand-in -----------------------------------------------------------------------------------------------
CEIL = TOP + 0.580                # panel underside
PANEL_T = 0.018                   # 23/32 in plywood
PANEL_X = (0.18, 0.78)
PANEL_Y = (-1.0, 1.0)
LEG = (0.038, 0.089)              # 2x4 legs, at the panel's corners
LEG_XY = ((0.20, 0.98), (0.76, 0.98), (0.20, -0.98), (0.76, -0.98))
PLUG_R = 0.025                    # 50 mm plug (2 in hole saw) at each screw spot
HOLE_OPEN = 0.005                 # half width of the collision opening over a pilot hole (the screw's way in)

# -- the light's bare housing (Commercial Electric WR484080LSL, lens and channel cover off) --------------------------
HOUSING_L, HOUSING_W, HOUSING_D = 1.218, 0.280, 0.055      # length from the spec sheet; width and depth estimates
HOUSING_MASS = 3.7                # estimate: 10.41 lb (4.72 kg) with lens per the spec sheet, lens about 1.0 kg
HOUSING_X = 0.42                  # centre line when mounted (the pencil outline: long sides at x 0.28 and 0.56)
SHEET_T = 0.0015                  # back plate collider (sheet steel 0.6 mm; thicker for contact robustness)
WALL_T = 0.003                    # walls, collider
TRIM_T, TRIM_H = 0.008, 0.024     # push-on rubber edge trim at the grip spots (estimate)
START_DROP = 0.10                 # on its stands, the back 100 mm below the ceiling

# keyholes (estimate): 1,016 mm apart along the housing (40 in), 100 mm apart across; the narrow slot is 6 mm wide
# and runs from 5 mm inside the screw spot to the 10 mm round end, which is outboard
KEY_DY = 0.508
KEY_X = (0.370, 0.470)
SLOT_W = 0.0060
SLOT_IN = 0.005                   # the slot's inner end, inboard of the screw spot
SLOT_TO_ROUND = 0.012             # screw spot to the round end's centre (outboard)
ROUND_R = 0.005
# the screws used: A's at its near slot, B's at its far slot (a diagonal pair through the centre of mass)
SCREW_XY = {"a": (0.370, +0.508), "b": (0.470, -0.508)}

# -- stands and screwdriver holders -----------------------------------------------------------------------------------
STAND_Y = (0.15, -0.15)
STAND_POST = (0.089, 0.038)       # 2x4 post, its wide face along x
CROSSBAR = (0.30, 0.089, 0.038)   # 2x4 laid flat, along x, with 10 mm foam on top
FOAM_T = 0.010
HOLDER_XY = {"a": (0.18, 0.06), "b": (0.18, -0.06)}
# the driver's bottom above the table: B's holder is 15 mm taller, so that B's hand, at the same height, grips its
# driver 15 mm lower (165 mm below the bit tip), which keeps B's wrist clear of the housing at the far slot
HOLDER_FLOOR = {"a": 0.190, "b": 0.205}
# each driver stands on a hardwood peg on top of its 2x4 riser; the fingers pass either side of the peg
PEG_W, PEG_H = 0.022, 0.030

# -- screwdriver (Bosch GO class, push to start) ------------------------------------------------------------------------
DRV_R = 0.018                     # grip, 36 mm across (estimate)
DRV_BODY_L = 0.150
DRV_COLLAR_R, DRV_COLLAR_L = 0.010, 0.020
DRV_BIT_HALF, DRV_BIT_L = 0.003, 0.025      # #2 square bit
DRV_TIP = DRV_BODY_L + DRV_COLLAR_L + DRV_BIT_L             # bottom to bit tip, 195 mm
DRV_GRIP = {"a": 0.045, "b": 0.030}   # grip point above the driver's bottom: 150 (A) and 165 (B) mm below the bit tip
DRV_MASS = 0.31                   # Bosch GO data sheet, with battery
RPM = 360.0
CLUTCH = 0.6                      # N m, lowest clutch setting (estimate; Bosch publishes only 2.5 / 5 N m maxima)
PUSH_START = 3.0                  # N, push that switches the motor on (estimate)
BITE_PUSH = 5.0                   # N, push that starts the screw in its pilot hole (estimate)
BITE_HOLD = 0.1                   # s
BITE_TILT = 8.0                   # degrees off vertical, at most

# -- screws (#8 x 3/4 in pan-head square-drive wood screw) -------------------------------------------------------------
SCREW_L = 0.01905                 # under the head
SHANK_R = 0.0021
HEAD_R, HEAD_H = 0.00395, 0.0030  # estimates
SCREW_MASS = 0.003
PITCH = 0.0254 / 15               # 15 threads per inch (estimate): 1.69 mm per turn
TRAVEL = SCREW_L - SHEET_T        # tip at the ceiling to the head on the housing's back
THREAD_DRAG = 0.25                # N m, running torque in the plywood (estimate)
THREAD_EFF = 0.115                # thread efficiency, so the clutch slips near 150 N of clamp (fidelity note)
THREAD_K, THREAD_C = 4.0e4, 40.0  # the thread's coupling drive (N/m, N s/m)
MAGNET_F = 3.0                    # N, the bit's magnet (per axis)
BIT_K, BIT_C = 5.0e3, 20.0        # the bit's contact with the head, a spring in task code (N/m, N s/m)
BIT_D = 40.0                      # N s/m, its damping on the driver
PUSH_MAX = 60.0                   # N, the most that spring passes
MAGNET_K, MAGNET_C = 3000.0, 6.0
MAGNET_RANGE = 0.006              # the head within 6 mm of the bit tip

# -- scoring -------------------------------------------------------------------------------------------------------------
SEAT_GAP = 0.0003                 # head within 0.3 mm of the housing's back
SEAT_F = 50.0                     # and pressing it with 50 N
FLUSH_GAP = 0.002                 # 2 mm card at the nearest long edge, in line with the screw
DROP_V = 0.30                     # a corner moving down at 0.3 m/s,
DROP_BELOW = 0.025                # more than 25 mm below where it is flush (a screw only just bitten leaves 17.5)
HANG_S = 3.0                      # no gripper contact for the last 3 s
HELD_F = 0.05


def subtract(rects, cut):
    """Rectangles (x0, x1, y0, y1) minus one rectangle, as non-overlapping rectangles."""
    cx0, cx1, cy0, cy1 = cut
    out = []
    for x0, x1, y0, y1 in rects:
        if cx1 <= x0 or cx0 >= x1 or cy1 <= y0 or cy0 >= y1:
            out.append((x0, x1, y0, y1))
            continue
        if y0 < cy0:
            out.append((x0, x1, y0, cy0))
        if cy1 < y1:
            out.append((x0, x1, cy1, y1))
        ya, yb = max(y0, cy0), min(y1, cy1)
        if x0 < cx0:
            out.append((x0, cx0, ya, yb))
        if cx1 < x1:
            out.append((cx1, x1, ya, yb))
    return out


def housing_mounted_z() -> float:
    """The housing origin's height when flush (origin at the back's top face centre)."""
    return CEIL


def screw_dir(station: str) -> float:
    return 1.0 if station == "a" else -1.0


def rot_z(a: float):
    import numpy as np

    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
