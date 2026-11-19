"""Dimensions and layout of the plug-in timer task.

Station frame: metres from the midpoint between the arm bases, x forward (away from the arms), y to the robot's left,
z up. Heights below are above the table top (TOP) unless a name says _Z (world height).

The timer is a GE 24-Hour Plug-In Mechanical Timer, model 15119 (Jasco): 48 captive segments of 30 minutes round the
dial's edge, a fixed white centre with a pointer, an OUTLET ON / TIMER ON slide switch on its top edge. Listed 2.4 x
2.4 x 2.3 in, 3.2 oz. Sizes marked (est.) are not published; the submitted text calls them estimates.

Dial frame: origin on the dial axis at the dial's top face, z up; at dial angle 0 its axes are the station's.
Times are hours after midnight (14.5 = 2:30 PM). Seen from above, the printed times run clockwise backwards (11, 10,
9 ... as on the maker's figure 1), so turning the dial clockwise (negative angle about +z) brings later times to the
pointer, which is fixed at +x (the timer's top edge, away from the robot).
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                        # the station's table top (world z)
PHYSICS_HZ = 480                    # PhysX rate (yam_arena's default is 240)

# -- power strip: 6 outlets in a row and an on/off switch at one end, lying flat, sockets up, length along y ------------
STRIP_L, STRIP_W, STRIP_H = 0.280, 0.050, 0.038     # (est.)
STRIP_MASS = 0.40                   # (est.)
SOCKET_PITCH = 0.038                # (est.)
SOCKET_1 = -0.081                   # socket 1 (next to the switch end, -y) from the strip's centre along y
SOCKET = 3                          # the timer is in the third socket from the switch end
STRIP_X = 0.300
STRIP_Y = -(SOCKET_1 + (SOCKET - 1) * SOCKET_PITCH)   # so the timer's socket is at y 0
STRIP_SWITCH_Y = -STRIP_L / 2 + 0.020                 # the strip's rocker switch (off), from its centre

# -- the timer ----------------------------------------------------------------------------------------------------------
TIMER_X, TIMER_Y = STRIP_X, 0.000   # the dial's axis
HOUSING = (0.061, 0.061, 0.032)     # x (top edge +x), y (outlet +y), z: the white case on the strip (est. depth)
HOUSING_Z0 = STRIP_H                # its back on the strip's top
FACE_H = HOUSING_Z0 + HOUSING[2]    # the housing's face (0.070)
TIMER_MASS = 0.091                  # 3.2 oz
# the dial: a black wheel on the face, ribbed side, segment ring round its top edge
RIB_R = 0.0258                      # the ribbed side's radius (est.)
DIAL_TOP_H = FACE_H + 0.008         # the dial's top face, 8 mm proud of the housing (est.)
DIAL_MASS = 0.015                   # (est.)
SEG_N = 48
SEG_R_OUT = 0.0252                  # the segments' outer ends: 3.0 mm wide + 0.3 mm gap = 3.3 mm pitch here (est.)
SEG_R_IN = 0.0202                   # 5 mm long radially (est.)
SEG_W_OUT, SEG_GAP_OUT = 0.0030, 0.0003            # at the outer end (est.)
SEG_ANG = 2 * math.pi / SEG_N                      # 7.5 deg per segment (half an hour)
SEG_ANG_W = SEG_ANG * SEG_W_OUT / (SEG_W_OUT + SEG_GAP_OUT)   # the segment's angular width (wedge)
SEG_TRAVEL = 0.0020                 # up: 2 mm proud of the dial top; down: flush (est.)
SEG_MASS = 0.005                    # for the solver (the real part is lighter)
SEG_TOP_H = DIAL_TOP_H + SEG_TRAVEL  # raised segments' tops (0.080)
CAP_R = 0.0120                      # the fixed white centre (est.)
CAP_H = DIAL_TOP_H + 0.0025         # its top
NUM_R = (CAP_R + 0.0005, SEG_R_IN - 0.0003)        # the printed ring (hours, night band)
# segment detent (task code), estimates
SEG_K_UP = 3000.0                   # N/m: an up segment is held by a spring, 3 N at 1 mm
SEG_LATCH = 0.0010                  # pushed this far it latches down (3 N)
SEG_K_DOWN = 3000.0                 # N/m: a latched segment is held down at -SEG_TRAVEL (on its stop)
SEG_DAMP = 2.0                      # N s/m
# the dial's drag (task code, stick-slip), estimates
DIAL_DRAG = 0.03                    # N m to start it turning
DIAL_K = 5.0                        # N m/rad: the stick spring
DIAL_DAMP = 0.002                   # N m s/rad
# the slide switch on the top edge (+x face): two stops 4 mm apart along y, TIMER ON at -y
SWITCH_Z = FACE_H - 0.006           # its knob's centre height
SWITCH_KNOB = (0.0035, 0.008, 0.004)               # x (proud of the face), y, z
SWITCH_STOPS = (0.0, 0.004)         # joint position: TIMER ON (start), OUTLET ON (+y) (est.)
SWITCH_K = 400.0                    # N/m to its stop; flips past the middle (est.)
# the timer's own outlet on its +y side (visual), the lamp's plug in it
OUTLET_Z = HOUSING_Z0 + HOUSING[2] / 2

# times and angles
START_TIME = 9.0                    # the pointer at 9:00 AM (dial angle 0)
NOW = 14.5                          # 2:30 PM
DEG_PER_H = 15.0
TARGET_ANGLE = -math.radians(DEG_PER_H * (NOW - START_TIME))   # -82.5 deg (clockwise)
TIME_TOL = math.radians(3.75)       # a quarter of an hour
CCW_TOL = math.radians(15.0)        # a counterclockwise turn beyond this (an hour) is not allowed
ON_SEG, OFF_SEG = 36, 46            # 6:00 PM .. 10:30 PM are segments 36..45 (segment i covers i/2 .. i/2 + 0.5 h)
TARGETS = tuple(range(ON_SEG, OFF_SEG))


def seg_angle_dial(i: int) -> float:
    """Segment i's centre angle in the dial frame (CCW from the dial's +x), rad."""
    return math.radians(DEG_PER_H * (0.5 * i + 0.25 - START_TIME))


def seg_angle_world(i: int, dial: float) -> float:
    return dial + seg_angle_dial(i)


def pointer_time(dial: float) -> float:
    """The time at the pointer for a dial angle (rad, unwrapped), hours in [0, 24)."""
    return (START_TIME - math.degrees(dial) / DEG_PER_H) % 24.0


def seg_point(i: int, dial: float, r: float, h: float = SEG_TOP_H) -> np.ndarray:
    """A point on segment i's centre line at radius r, height h above the table (world)."""
    a = seg_angle_world(i, dial)
    return np.array([TIMER_X + r * math.cos(a), TIMER_Y + r * math.sin(a), TOP + h])


def fmt_time(t: float) -> str:
    h, m = int(t) % 24, int(round((t % 1) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{(h - 1) % 12 + 1}:{m:02d} {'AM' if h < 12 else 'PM'}"


# -- the pen: a ballpoint (its ink run out), 1.0 mm ball, standing point down in a mug ---------------------------------
PEN_L = 0.150                       # (est.)
PEN_R = 0.004                       # 8 across (est.)
PEN_CONE = 0.015                    # the tapered point (est.)
PEN_BALL_R = 0.0005                 # the 1.0 mm ball
PEN_MASS = 0.005                    # (est.)
# mug: ceramic, 95 tall, 80 across (est.)
MUG_X, MUG_Y = 0.220, -0.200
MUG_H, MUG_R, MUG_WALL, MUG_FLOOR = 0.095, 0.040, 0.004, 0.006
MUG_MASS = 0.30
PEN_LEAN_DIR = math.radians(180.0)  # the pen leans its top toward the robot (-x): on the mug's near rim

# -- the lamp (visual and a static collider), IKEA LAMPAN ------------------------------------------------------------------
LAMP_X, LAMP_Y = 0.550, 0.300
LAMP_H, LAMP_BASE_R, LAMP_SHADE_R = 0.290, 0.065, 0.095

# -- scoring constants ----------------------------------------------------------------------------------------------------
CLEAR = 0.020                       # grippers and pen this far from the timer at the end
RAISED = 0.003                      # the timer raised more than this: pulled out of its socket
HOLD_STEPS = 20                     # 1 s at 20 Hz
TOUCH_F = 1e-3                      # N: a contact carrying force is a touch
PUSH_DOWN = 0.0005                  # a gripper moving a segment this far down pushed it


def pen_rest_pose():
    """The pen's resting pose in the mug, from geometry: its ball on the floor against the wall opposite the lean, its
    barrel on the rim on the lean side. Returns (tip position, unit axis from tip to top), world."""
    d = np.array([math.cos(PEN_LEAN_DIR), math.sin(PEN_LEAN_DIR), 0.0])   # horizontal lean direction
    rin = MUG_R - MUG_WALL
    tip = np.array([MUG_X, MUG_Y, TOP + MUG_FLOOR + PEN_BALL_R]) - d * (rin - 0.003)
    rim = np.array([MUG_X, MUG_Y, TOP + MUG_H]) + d * rin
    # the axis through tip at angle a from vertical (toward d), its barrel (radius PEN_R) touching the rim's inner edge
    v = rim - tip
    hx, hz = float(v @ d), float(v[2])
    best = None
    for a in np.linspace(0.0, math.radians(60), 6001):
        u = np.array([math.sin(a), math.cos(a)])
        dist = abs(hx * u[1] - hz * u[0])
        along = hx * u[0] + hz * u[1]
        if along > 0 and (best is None or abs(dist - PEN_R) < best[0]):
            best = (abs(dist - PEN_R), a)
    a = best[1]
    axis = d * math.sin(a) + np.array([0.0, 0.0, math.cos(a)])
    return tip, axis
