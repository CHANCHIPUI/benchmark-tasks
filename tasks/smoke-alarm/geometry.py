"""Dimensions and layout of the smoke-alarm task.

Station frame: metres from the midpoint between the arm bases, x forward (away from the arms), y to the robot's left,
z up. Heights below are above the table top (TOP) unless a name says _Z (world height).

The alarm is a First Alert SMI100 battery smoke alarm (listed 5.58 in across, 1.99 in thick, 0.43 lb). It starts face
up: its back on the table, its front cover up. Sizes marked "estimate" are not published by the makers.

Alarm frame: origin on the alarm's axis at the middle of its back (on the table when it lies face up), z toward the
front cover. The back is a flat disc; the front cover has a flat middle and curves down to the side band at the rim.
The battery compartment is front access, as on the real unit: a tray on a hinge at the rim swings out sideways, its
pocket open toward the front, and the front cover over the compartment's hinge end (the shroud) still covers the
hinge end of the battery when the compartment is open.
Tray frame (the compartment): origin on the hinge axis at the alarm frame's height 0; shut, its axes are the alarm
frame's. The tray runs along -x from the hinge (the hinge at its far end, the free end and its latch toward the
robot); +y is its inner side (toward the alarm's centre); +z is toward the front (up when the alarm lies face up). The
hinge turns it about +z: opening swings the free end out, toward the robot's right (-y).
The contacts are at the free end (the latch end) and a battery lies with its snaps toward them: its bottom end goes
toward the hinge, under the shroud.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                        # the station's table top (world z)
PHYSICS_HZ = 240                    # PhysX rate (yam_arena's default)

# -- the alarm ---------------------------------------------------------------------------------------------------------
ALARM_X, ALARM_Y = 0.330, 0.030     # its axis (lying face up on the box)
# the small cardboard box the alarm lies on (estimate): its rim overhangs the box's side toward the robot, so a hand
# can pinch the rim across the alarm's thickness
BOX_X = (0.289, 0.385)              # x extent
BOX_Y = (-0.040, 0.100)             # y extent
BOX_H = 0.050                       # height
ALARM_R = 0.071                     # 5.58 in across
ALARM_H = 0.051                     # 1.99 in thick
ALARM_MASS = 0.195                  # 0.43 lb (body, compartment and latch together)
TRAY_MASS = 0.016                   # (estimate)
TAB_MASS = 0.002                    # (estimate)
BODY_MASS = ALARM_MASS - TRAY_MASS - TAB_MASS
BODY_COM_Z = 0.026                  # from the back (estimate)
FLAT_R = 0.055                      # the front cover's flat middle, out to this radius (estimate)
RIM_Z = 0.012                       # the cover curves down from its flat middle by this much to the side band (estimate)
BACK_T = 0.0035                     # the back: a solid plate under the compartment (estimate)
TORSION_R = 0.008                   # the alarm's colliders' torsional patch radius (the fingertips' flat side on the band)
NOTCH_Y = -0.0315                   # the compartment's slot in the body: y below this ...
NOTCH_X = (-0.056, 0.032)           # ... between these x (alarm frame), from the back plate up to the shroud
REC_X = -0.010                      # the front cover is open over the slot for x below this: the latch's finger recess

# -- the battery compartment (a tray on a vertical hinge at the rim, the robot's right) ------------------------------
HINGE = np.array([0.030, -0.06435])  # the hinge axis in the alarm frame (on the rim, at the compartment's far end)
TRAY_L = 0.060                      # hinge to free end (estimate)
FLOOR_Z0 = BACK_T + 0.0005          # the tray's floor, underside
FLOOR_T = 0.0015
FLOOR_TOP = FLOOR_Z0 + FLOOR_T      # 5.5 mm: the battery's bed
WALL_H = 0.008                      # pocket walls (estimate)
WALL_TOP = FLOOR_TOP + WALL_H       # 13.5 mm
WALL_T = 0.002
SHROUD_CLEAR = 0.009                # the shroud's underside this far above a battery lying in the pocket (estimate)
SHROUD_Z = FLOOR_TOP + 0.0175 + SHROUD_CLEAR   # 32 mm: the front cover's underside over the compartment
POCKET_L = 0.0495                   # from the contact block's face to the hinge-end wall (estimate)
POCKET_W = 0.0275                   # (estimate)
POCKET_Y0 = 0.00235                 # the pocket's outer side (tray frame); the outer skirt beyond it follows the rim
POCKET_Y1 = POCKET_Y0 + POCKET_W    # its inner side
POCKET_YC = (POCKET_Y0 + POCKET_Y1) / 2
NEAR_WALL = (-0.002, 0.0)           # the hinge-end wall (tray frame x)
BLOCK_FACE = NEAR_WALL[0] - POCKET_L  # the contact block's face toward the pocket (-51.5 mm), at the free end
BLOCK_X = (BLOCK_FACE - 0.005, BLOCK_FACE)
BLOCK_TOP = FLOOR_TOP + 0.009       # (estimate)
LEAD_IN = (0.0015, 0.0025)          # the pocket's side walls lean out 1.5 mm over their top 2.5 mm (estimate)
SLOT_DEPTH = 0.003                  # the contacts' slots, open on top, into the block from its face
SWING = math.radians(55.0)          # the hinge's stop (estimate, from the maker's photo of the open compartment)
# the latch: a tab on the free end's outer face, standing out past the rim beside the cover's finger recess; pressed
# down 1.5 mm it frees the compartment
TAB_X = (-0.070, -0.065)            # tray frame: a hook just beyond the free end
TAB_Y = (-0.006, 0.000)             # 6 out from the free end's outer face (estimate)
TAB_Z = (WALL_TOP + 0.0025, SHROUD_Z + 0.002)   # its top 2 mm above the shroud's underside (estimate)
TAB_TRAVEL = 0.002
TAB_K = 1000.0                      # N/m: 0.5 N preload, 2 N at 1.5 mm (estimate)
TAB_PRELOAD = 0.0005
TAB_RELEASE = 0.0015                # pressed this far it frees the compartment (estimate)
TAB_DAMP = 1.0
HINGE_DAMP = 0.004                  # N m s/rad
HINGE_DRAG = 0.005                  # N m: the hinge's friction (task code, stick-slip), so it stays where it is left (estimate)
HINGE_DRAG_K = 5.0                  # N m/rad: the stick spring
HUMP = math.radians(2.5)            # the catch: over the last 2.5 degrees shut, it pushes back ...
LATCH_AT = math.radians(1.0)        # ... until the compartment is past it, within 1 degree (latched)
HUMP_F = 2.0                        # N at the free end at the catch (estimate)
HUMP_K = HUMP_F * TRAY_L / (HUMP - LATCH_AT)   # N m/rad
OPENED = math.radians(45.0)         # "opened" for the 0.25 tier

# -- the 9 V batteries (Energizer 522: 48.5 x 26.5 x 17.5 mm with the terminals, 45 g) ------------------------------
BATT_L, BATT_W, BATT_T = 0.0485, 0.0265, 0.0175
TERM_L = 0.0026                     # the snaps stand this far off the battery's top
BODY_L = BATT_L - TERM_L            # 45.9 mm of case
BATT_MASS = 0.045
TERM_PITCH = 0.0127                 # the two snaps' centres (12.45 to 12.95)
POS_R, NEG_R = 0.0028, 0.0042       # + stud 5.6 across, - snap 8.4 across
POS_SLOT, NEG_SLOT = 0.0070, 0.0100  # the contacts' slot widths (+ is narrow: the - snap can't enter it)
TAPE_W = 0.010                      # the old battery's red tape band, round its bottom end
# battery frame: origin at the case's centre; +x toward the terminals, +z out of the label face, + stud at +y
POS_Y, NEG_Y = TERM_PITCH / 2, -TERM_PITCH / 2

# seated + to + in the tray: snaps toward the block (-x_tray), label up (+z_tray); the battery's +y is then -y_tray,
# so the + stud and the + contact are on the outer side (toward the rim)
SEAT_ROT = np.diag([-1.0, -1.0, 1.0])   # the seated battery's axes in the tray frame
SEAT = np.array([BLOCK_FACE + BODY_L / 2, POCKET_YC, FLOOR_TOP + BATT_T / 2])
POS_SLOT_Y = POCKET_YC - POS_Y
NEG_SLOT_Y = POCKET_YC - NEG_Y

# -- the new battery on its torn-open card --------------------------------------------------------------------------
CARD = (0.100, 0.150, 0.001)        # x, y, thickness (estimate)
CARD_X, CARD_Y = 0.220, -0.150
NEW_X, NEW_Y = CARD_X, CARD_Y       # the battery's case centre; label up, terminals toward the robot (-x)
NEW_YAW = math.pi

# -- the snap (task code), estimates ---------------------------------------------------------------------------------
SNAP_F = 6.0                        # N of press on the battery (the tray floor's push) that snaps it, seated
SNAP_GAP = 0.0015                   # the case's top within this of the block's face (terminals in their slots)
SNAP_FLAT = 0.0010                  # both ends of the case within this of the floor
SNAP_SIDE = 0.0010                  # off the pocket's centre line by at most this
SNAP_ANG = math.radians(5.0)
CLAMP_F = 6.0                       # N: snapped, held down on its contacts (3 N at each end) ...
CLAMP_AX = 2.0                      # ... and onto them
UNSNAP_LIFT = 0.0015                # an end lifted this far off the floor unsnaps it
UNSNAP_GAP = 0.0025                 # or the case drawn this far off the block

# -- the mounting bracket (the SMI100's own twist-on bracket), under a shelf board standing in for the ceiling ------
BRACKET_X, BRACKET_Y = 0.380, 0.260  # the ring's centre
MOUNT_H = 0.170                     # the ring's face (underside), above the table: the alarm's back meets it here
RING_R = 0.055                      # 110 across (estimate)
RING_T = 0.010                      # (estimate)
SPACER = 0.090                      # a pine block 90 square and 30 thick between the ring and the board
SPACER_T = 0.030
BOARD_W, BOARD_T = 0.160, 0.018     # the shelf board: 160 wide, 18 thick
BOARD_DIR = math.radians(15.0)      # it runs this way (from +x) from the ring's centre ...
BOARD_SPAN = (-0.085, 0.200)        # ... from 85 behind it to 200 beyond (285 long)
BOARD_Z = MOUNT_H + RING_T + SPACER_T  # its underside above the table (210)
POST = 0.035                        # the board's square post, under its far end
POST_AT = 0.180                     # along BOARD_DIR
ENTRY_STEP = math.radians(60.0)     # the alarm sits on the bracket in any of six positions 60 degrees apart
RING_YAW = 0.0                      # the ring is turned so that one of the six is at this yaw of the alarm (face down)
ENTRY_TOL = math.radians(8.0)       # (estimate)
LOCK_TWIST = math.radians(20.0)     # turned this far clockwise (seen from below) it locks (estimate)
UNLOCK_BACK = math.radians(10.0)    # turned back this far from the lock it comes free
SEAT_GAP = 0.0015                   # the alarm's back within this of the ring's face ...
SEAT_OFF = 0.003                    # ... its axis within this of the ring's ...
SEAT_TILT = math.radians(3.0)       # ... and level within this: seated
HOLD_F = 12.0                       # N: locked, task code holds it up against the ring (the lugs)
PULL_OFF = 0.003                    # locked, pulled this far off the ring it comes free

# -- scoring constants ------------------------------------------------------------------------------------------------
ON_TABLE_Z = 0.005                  # the alarm's lowest point this close to the table top, any side up: on the counter
HELD_F = 0.05                       # N: a contact force over this between a hand and the alarm: the hand touches it
TABLE_X, TABLE_Y = (-0.05, 0.75), (-0.60, 0.60)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tray_to_alarm(p, hinge: float) -> np.ndarray:
    """A point in the tray frame, in the alarm frame, with the compartment swung ``hinge`` rad out."""
    p = np.asarray(p, float)
    return np.r_[HINGE, 0.0] + rot_z(hinge) @ p


def cover_z(r: float) -> float:
    """The front cover's height above the back at radius r: flat to FLAT_R, then a quarter-ellipse down to the rim."""
    if r <= FLAT_R:
        return ALARM_H
    s = min(1.0, (r - FLAT_R) / (ALARM_R - FLAT_R))
    return ALARM_H - RIM_Z * (1.0 - math.sqrt(max(0.0, 1.0 - s * s)))


def rim_y(x: float) -> float:
    """The rim's y (alarm frame, on the robot's right) at x."""
    return -math.sqrt(max(0.0, ALARM_R ** 2 - x ** 2))


def skirt_outline(n: int = 16) -> list:
    """The compartment's outer side: the rim's arc from the hinge to the free end, in the tray frame (x, y)."""
    xs = np.linspace(HINGE[0] - TRAY_L, HINGE[0], n)
    return [(float(x - HINGE[0]), float(rim_y(x) - HINGE[1])) for x in xs]


def front_frame(p, R):
    """The alarm's pose in the frame the bracket uses: origin on its axis at its front's middle, z toward its back
    (the back's middle is then origin + z * ALARM_H). From the alarm frame's pose (p, R)."""
    R = np.asarray(R, float)
    return np.asarray(p, float) + R[:, 2] * ALARM_H, R @ np.diag([1.0, -1.0, -1.0])
