"""Dimensions and layout of the gel-comb task.

Station frame: metres from the midpoint between the arm bases, x forward (away from the arms), y to the robot's left,
z up; heights are above the table top (TOP).

Parts (sizes marked "est." are estimates, to be measured on the real parts before the trials; README lists them):
- Bio-Rad Mini-Sub Cell GT base (1704360), the "tank": clear moulded plastic, 25.5 x 9.2 x 5.6 cm (maker), long axis
  along y, black (cathode) end at +y, red (anode) end at -y. Tank frame: origin on the table under the base's middle,
  axes as the station's.
- Bio-Rad UV-transparent mini-gel tray, 7 x 7 cm (1704436), open at both ends. Tray frame: origin at the middle of the
  tray's bottom face, x along its run axis (the wells end at -x), y across it, z up. At the start its axes are the
  station's.
- The gel: 1% agarose, 70 x 70 x 5 mm, cast in the tray (its ends flush with the tray's open ends), 8 wells cast by
  the comb. Gel frame: origin at the middle of the gel's bottom face, axes as the tray's.
- Bio-Rad Mini-Sub 8-well, 1.5 mm comb (1704463): a bar with 8 teeth 5.54 x 1.5 mm. Comb frame: origin at the middle
  of the bar's bottom edge, x across its thickness, y along the bar, z up (the teeth point to -z).
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                         # the station's table top
PAD_TIP = 0.013                      # yam_arena: the pad tips are 13 mm past the tool point

# -- tray (UV-transparent acrylic, 7 x 7 cm) ---------------------------------------------------------------------------
TRAY_L = 0.070                       # along the run axis (open ends); the gel fills it end to end (est.)
TRAY_W = 0.078                       # across, outside (est., spec: about 7.8 cm)
FLOOR_T = 0.003                      # floor thickness (est.)
WALL_T = 0.004                       # side-wall thickness (est.)
WALL_H = 0.013                       # side walls' top above the tray's bottom: 10 mm above the floor (est.)
INNER_W = TRAY_W - 2 * WALL_T        # 70 mm: the gel's width
SLOT_X = (-0.027, 0.027)             # comb slots in each side wall, centred this far from the tray's middle (est.)
SLOT_W = 0.006                       # slot width along x (est.): 1 mm play each side on the 4 mm bar
SLOT_D = 0.003                       # slot depth below the wall's top (est.)
TRAY_MASS = 0.030                    # acrylic, est.

# -- gel (1% agarose in 1x TAE, 20 ml) ---------------------------------------------------------------------------------
GEL_L, GEL_W, GEL_T = TRAY_L, INNER_W, 0.005
GEL_MASS = 0.024
GEL_Z0 = FLOOR_T                     # the gel's bottom (on the floor) in the tray frame
N_WELLS = 8
TOOTH_W = 0.00554                    # Bio-Rad 1704463: 5.54 mm wide
TOOTH_T = 0.0015                     # 1.5 mm thick
PITCH = 0.008                        # tooth pitch (est.)
WELL_X = SLOT_X[0]                   # the wells' line (the comb's slots at the -x end)
WELL_Y = np.array([(3.5 - i) * PITCH for i in range(N_WELLS)])     # well 1 (robot's left, +y) .. well 8
CAST_DEPTH = 0.004                   # the teeth reach 4 mm into the 5 mm gel (1 mm above the floor)
POCKET_CLEAR = 0.0005                # the sim's pockets: 0.5 mm extra clearance per side (spec)
POCKET_X = TOOTH_T + 2 * POCKET_CLEAR
POCKET_Y = TOOTH_W + 2 * POCKET_CLEAR
GEL_FRICTION = 0.21                  # gel on acrylic: tan(slide angle 12 deg), provisional (pilot 2)

# -- comb (8-well, 1.5 mm, moulded plastic) ----------------------------------------------------------------------------
BAR_L, BAR_H, BAR_T = 0.075, 0.020, 0.004     # spec: about 7.5 x 2 x 0.4 cm
COMB_MASS = 0.008
COMB_Z0 = WALL_H - SLOT_D            # the bar's bottom edge rests on the slots' bottoms: 10 mm over the tray's bottom
TOOTH_L = COMB_Z0 - (GEL_Z0 + GEL_T - CAST_DEPTH)    # 6 mm teeth: tips 1 mm above the floor
COMB_START = np.array([WELL_X, 0.0, COMB_Z0])        # the comb's origin in the tray frame at the start

# -- the hold (the comb stuck in the set gel) ---------------------------------------------------------------------------
# An invisible catch over each end of the comb's bar, on a stiff vertical slide in the tray (task.py): it carries the
# comb's pull into the tray until the pull reaches F_REL, then lets go over RELEASE_S and parks inside the side wall.
CATCH_SIZE = (0.014, 0.0022, 0.003)  # x (covers the bar's tilt and drift), y, z
CATCH_Y = INNER_W / 2 + 0.0012       # its centre, over the bar's end (the bar ends at 37.5 mm; the wall's inner face 35)
CATCH_Z = COMB_Z0 + BAR_H + 0.0001 + CATCH_SIZE[2] / 2     # resting 0.1 mm over the bar's top edge
CATCH_PARK = -(CATCH_Z - 0.0055)     # slide travel that parks it inside the solid wall below the slot (z 4-7 mm)
CATCH_K = 4.0e4                      # N/m: 0.025 mm per 1 N

# -- tank (Bio-Rad Mini-Sub Cell GT base) --------------------------------------------------------------------------------
TANK_X = 0.38                        # its middle (reach check, dev/gate4.py)
TANK_Y = 0.0
BASE_L, BASE_W, BASE_H = 0.255, 0.092, 0.056     # maker
BASE_WALL = 0.003                    # wall thickness (est.)
BASE_FLOOR = 0.005                   # floor of the buffer chambers above the table (est.)
TANK_INNER_W = BASE_W - 2 * BASE_WALL            # 86 mm (spec: about 8.6 cm)
STAGE_L = 0.072                      # the gel stage along y (est.): the 70 mm tray sits on it with 1 mm to spare each end
STAGE_TOP = 0.031                    # 25 mm below the rim (est., spec: 2-3 cm)
GUARD_Y = (0.106, 0.121)             # each electrode wire guard's extent along |y| (est.)
GUARD_H = 0.016                      # its top above the table (est.)
WIRE_Y, WIRE_Z = 0.116, 0.009        # the platinum wire (0.25 mm) along x, inside the guard (est.)
PLUG_Y = 0.1235                      # banana plug on the end wall's top
PLUG_R, PLUG_L = 0.002, 0.044        # 4.4 cm gold-plated plug (maker)

# -- tray start, comb spot ----------------------------------------------------------------------------------------------
TRAY_X = 0.225                       # reach and two-hand clearance check (dev/gate3.py, gate4.py)
TRAY_Y = -0.06
TISSUE_X, TISSUE_Y = 0.33, 0.22      # folded lab tissue, 10 x 10 cm
TISSUE_SIZE = (0.10, 0.10, 0.002)

# -- scoring thresholds ---------------------------------------------------------------------------------------------------
# Provisional, set by the pilot (spec: each midway between the largest pilot perturbation that left every well intact
# and the smallest that tore one). A rule whose perturbation never tears a well in the pilot is removed.
D_TEAR = 0.0004                      # tooth offset across its thickness, relative to its well (m)
THETA_T = math.radians(4.0)          # comb tilt about its long axis
THETA_E = math.radians(20.0)         # comb end-first tilt in its own plane
V_TEAR = 0.030                       # a tooth's upward speed while it is in the gel (m/s)
PUSH_DOWN = 0.001                    # a tooth pushed below its cast depth (m)
F_REL = 1.0                          # N, the comb's release force (pilot 1); above the tray + gel weight (0.53 N)
F_DENT = 3.0                         # N, a pad or the comb on the gel's top surface (pilot 3)
F_END = 8.0                          # N, a pad on a gel end face (pilot 3)
# fixed by the spec
LIFT_MAX = 0.003                     # zero 1: tray or gel lifted by the comb
SLIDE_ZERO, SLIDE_DEFECT = 0.010, 0.003
DROP = 0.003                         # zero 3: free fall
TIP_MAX = math.radians(30.0)
COMB_CLEAR = 0.005                   # comb out: teeth 5 mm or more above the gel
CORNER_TOL = 0.001                   # seated: every corner within 1 mm of the stage
SQUARE_TOL = math.radians(5.0)
HANDS_OFF = 0.03

RELEASE_S = 0.3                      # the hold fades to nothing over 0.3 s once the pull reaches F_REL


def well_xy(i: int) -> np.ndarray:
    """Well i (0..7: wells 1..8) in the gel frame (x, y)."""
    return np.array([WELL_X, WELL_Y[i]])


def tray_corners() -> np.ndarray:
    """The tray's four bottom corners in the tray frame."""
    hx, hy = TRAY_L / 2, TRAY_W / 2
    return np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])


def stage_centre() -> np.ndarray:
    return np.array([TANK_X, TANK_Y, TOP + STAGE_TOP])
