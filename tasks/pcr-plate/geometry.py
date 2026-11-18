"""Dimensions and layout of the PCR plate task.

Station frame: metres from the midpoint between the arm bases, x forward (away from the arms), y to the robot's left,
z up; heights (h) are above the table top (TOP).

The thermal cycler is an Applied Biosystems GeneAmp PCR System 9700 with its 96-well sample block module. Its own
frame ("u, v, h"): u back from the middle of its front face, v to the robot's left, h up from the table. Its front faces
the arms. Sizes marked (est.) are estimates from the maker's overall dimensions (26 H x 28 W x 41 D cm) and manual
drawings; README lists which ones are measured on a unit before the real trials.

The plate is a MicroAmp Optical 96-Well Reaction Plate (N8010560, semi-skirted, 0.2 mL wells, 9 mm pitch) capped with a
MicroAmp 96-Well Full Plate Cover (N8010550), as one rigid body. Its frame: origin at the middle of the deck's top
face; x toward row A (the back), y toward column 1 (the left), z up. Placed the right way round its axes are the
station's: A1 is at the back left.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                        # the station's table top

# -- the thermal cycler (GeneAmp 9700, 96-well block) -----------------------------------------------------------------
FRONT_X = 0.140                     # its front face (set by the reach check, dev/gate*.py)
CENTRE_Y = 0.160                    # in front of the left arm: the block's top is at the edge of the arm's reach
WIDTH, DEPTH, HEIGHT = 0.28, 0.41, 0.26     # maker's overall size (W x D x H)
PANEL_U = 0.100                     # the sloped keypad and display panel at the front (est.)
PANEL_H = (0.085, 0.150)            # its height at the front edge and where it meets the deck (est.)
DECK_H = 0.160                      # the module's top surround round the block (est.)
BLOCK_RISE = 0.008                  # the block's top stands this far proud of the surround (est.)
BLOCK_TOP = DECK_H + BLOCK_RISE     # 16.8 cm (est.; measured on the unit for the README)
BLOCK_BODY = 0.020                  # the block's body below its top (its wells are 12.3 mm deep)
POCKET = BLOCK_BODY - BLOCK_RISE + 0.002    # the base's top is pocketed this deep round the block's body
BLOCK_U = 0.175                     # the block's centre behind the front face (est.)
BLOCK_SIZE = (0.0787, 0.121)        # the block's top, u x v: fits inside the plate's semi-skirt with 2.4 mm round it
RAIL_V = (0.125, 0.140)             # the cover's side rails (housings), each side
RAIL_H = DECK_H + 0.004             # their top: low rails along the surround's edges (est.)
# the heated cover: a carriage sliding along u on the rails, from back (open) to its stop (fully forward)
COVER_L, COVER_W = 0.160, 0.240    # along u, across v (est.)
COVER_UNDER = BLOCK_TOP + 0.013     # its underside and the raised platen: 4.8 mm over a seated plate's caps
COVER_TOP = 0.236                   # its top (the lever lies forward of it when down: overall height ~26 cm)
COVER_WALL = 0.008
COVER_FWD_U = BLOCK_U - COVER_L / 2     # its front face when fully forward (centred over the block)
COVER_BACK_U = 0.255                # its front face slid back (open): 3.7 cm behind the plate's back edge
COVER_TRAVEL = COVER_BACK_U - COVER_FWD_U   # 16.0 cm
COVER_STOP = 0.001                  # "fully forward": within 1 mm of the end stop
# the platen inside the cover, pressed down onto the plate by the lever's cam
PLATEN_SIZE = (0.130, 0.200, 0.008)
PLATEN_TRAVEL = 0.0058              # down 5.8 mm: 1 mm into a seated plate's caps
PLATEN_K = 40000.0                  # N/m: its spring (40 N on a seated plate, est.; measured for the README)
PLATEN_MAX = 80.0                   # N
# the lever: pivots on the cover's front top edge about v; up = its handle bar straight above the pivot, down = the
# bar forward of the cover at its top's height (joint angle 0 -> -90 deg)
LEVER_R = 0.070                     # pivot to the handle bar's axis (est.)
LEVER_ARM_V = 0.105                 # its two side arms
LEVER_BAR_R = 0.006                 # the handle bar, 12 mm across
LEVER_BAR_L = 0.220
LEVER_DOWN = -math.pi / 2
LEVER_STOP = math.radians(2.0)      # "down to its stop": within 2 degrees of it
LEVER_LATCH = math.radians(3.0)     # within 3 degrees of the stop it goes over centre and latches down
LEVER_CATCH = math.radians(0.5)     # latched: held within 0.5 degrees of its stop
# the lever's resistance (est.; the real pull is measured on the unit): until it latches, its drive pulls it back toward
# up like a spring, K (q_lever - LEVER_PRELOAD): 0.7 N m (10 N at the bar) at 45 degrees and 1.4 N m (20 N) at the
# stop. Latched, the drive holds it down (target LEVER_DOWN).
LEVER_K = 0.90                      # N m/rad
LEVER_PRELOAD = 0.0                 # rad: the spring's rest angle (up)
LEVER_DAMP = 0.3                    # N m s/rad
LEVER_UNLOCK = 0.001                # the lever is blocked until the cover is within 1 mm of its stop


def cover_front_u(q_slide: float) -> float:
    """The cover's front face for the slide joint's position (0 = back, -COVER_TRAVEL = fully forward)."""
    return COVER_BACK_U + q_slide


def lever_bar_uh(q_slide: float, q_lever: float) -> tuple[float, float]:
    """The handle bar's axis (u, h) for the slide and lever joint positions."""
    return cover_front_u(q_slide) + LEVER_R * math.sin(q_lever), COVER_TOP + LEVER_R * math.cos(q_lever)


def platen_drop(q_lever: float) -> float:
    """How far the cam has lowered the platen at this lever angle."""
    s = min(1.0, max(0.0, q_lever / LEVER_DOWN))
    return PLATEN_TRAVEL * s * s * (3 - 2 * s)


# -- the plate (MicroAmp N8010560 + full plate cover N8010550) ---------------------------------------------------------
PLATE_X, PLATE_Y = 0.0855, 0.1278   # outer footprint: 85.5 x 127.8 mm (ANSI/SBS footprint)
PITCH = 0.009
ROWS, COLS = 8, 12
A1_X = PLATE_X / 2 - 0.01124        # A1's centre: 11.24 mm from the back edge, 14.38 mm from the left edge
A1_Y = PLATE_Y / 2 - 0.01438
DECK_T = 0.0012                     # deck thickness (est.)
SKIRT_H = 0.0060                    # semi-skirt depth below the deck's top (est.)
SKIRT_COLLIDE = 0.003               # the skirt's collider thickness (the real wall is ~1 mm; thin colliders let pads sink)
WELL_R_TOP = 0.0032                 # a well's outside radius under the deck (est.)
WELL_R_TIP = 0.0013                 # at its tip (est.)
WELL_TIP = 0.0155                   # tip below the deck's top (est.)
CAP_H = 0.005                       # the full plate cover's caps stand 5 mm over the deck (est.)
CAP_MARGIN = 0.0045                 # the cap mat covers the wells to 4.5 mm beyond the outer wells' centres
CUT = 0.005                         # the cut corner at A12 (back right; ABI-style)
PLATE_MASS = 0.025                  # plate + cover, empty (est.)


def well_xy(row: int, col: int) -> np.ndarray:
    """A well's centre in the plate frame (row 0 = A, col 0 = column 1)."""
    return np.array([A1_X - row * PITCH, A1_Y - col * PITCH])


WELLS = np.array([well_xy(r, c) for r in range(ROWS) for c in range(COLS)])
DECK_CORNERS = np.array([[sx * PLATE_X / 2, sy * PLATE_Y / 2, 0.0] for sx in (1, -1) for sy in (1, -1)])

# -- the block and the support base: conical wells matching the plate's, 0.3 mm round them ----------------------------
HOLE_CLEAR = 0.0003
SEAT = 0.0032                       # a seated plate's deck top over the block's top (its tips on the well bottoms)
HOLE_DEPTH = WELL_TIP - SEAT        # 12.3 mm


def hole_radius(d: float) -> float:
    """The block well's radius at depth d below the block's top (a seated well's outside + HOLE_CLEAR)."""
    s = d + SEAT                    # depth below a seated deck's top
    k = (WELL_R_TOP - WELL_R_TIP) / (WELL_TIP - DECK_T)
    return WELL_R_TOP - max(0.0, s - DECK_T) * k + HOLE_CLEAR


SEATED_DECK_H = BLOCK_TOP + SEAT    # a seated plate's deck top over the table
ENGAGED_TOL = 0.001                 # engaged: every deck corner within 1 mm of seated
HIGH = 0.005                        # not engaged (zero 2): any deck corner 5 mm or more above seated (on the ridges: 12 mm)
ORIENT_TOL = 0.0045                 # A1 within half a pitch of the block's back-left well

# the MicroAmp Splash-Free 96-Well Support Base (4312063): a pedestal with the same wells, on a wider foot (est.)
SUPPORT_X = 0.200                   # to the robot's left of the cycler, beside its front half (in the top camera's view,
SUPPORT_Y = 0.380                   # clear of the left arm's ready pose)
SUPPORT_TOP = 0.020                 # its top over the table (est.)
SUPPORT_FOOT = (0.098, 0.140, 0.003)
SUPPORT_TOP_SIZE = BLOCK_SIZE          # its pedestal's top (with the wells), x by y

# -- scoring -------------------------------------------------------------------------------------------------------------
DROP = 0.010                        # a free fall of more than 1 cm is a drop
HOLD_F = 0.3                        # N: a hand holds the plate with both its pads pressing on it this hard
REST_SPEED, REST_SPIN = 0.01, 0.3   # m/s, rad/s: at rest
REST_STEPS = 10                     # 0.5 s at rest counts as set down
HANDS_OFF = 0.010                   # grippers 1 cm or more from the plate, cover and lever
COVER_MOVED = 0.001                 # the cover "first moves": 1 mm along its slide
LEVER_MOVED = math.radians(1.0)     # the lever "first moves": 1 degree down
MOVED_STEP = (0.0003, math.radians(0.5))   # per control step: cover / lever moving (zero 3, while a hand holds the plate)
