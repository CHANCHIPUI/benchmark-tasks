"""Dimensions and layout of the tourniquet task: a training thigh lying across the table with a windlass tourniquet
already strapped round it, and a pressure gauge on the band's bladder.

Station frame: metres from the midpoint between the arm bases, x forward, y to the robot's left, z up; heights here
are above the table top (TOP), and task.py adds TOP. The windlass rod's angle THETA is measured about the vertical
from +x (across the leg, pointing away from the arms) toward +y (along the leg, toward the hip).
"""

from __future__ import annotations

import math

TOP = 0.7475                      # the station's table top

# -- the training thigh: a firm foam cylinder lying along y, hip end on the robot's left (+y) -----------------------------
LEG_X = 0.280
LEG_R = 0.0635                    # 12.7 cm across
LEG_L = 0.40
LEG_Z = LEG_R + 0.002             # its axis: resting on two chocks, 2 mm off the table
CHOCK_Y = (0.15, -0.15)
CHOCK = (0.03, 0.04, 0.035)       # wooden chocks: x, y, height
CHOCK_IN = math.sqrt(LEG_R ** 2 - (LEG_Z - CHOCK[2]) ** 2)   # their inner faces this far from the axis (5.6 cm)

# -- the tourniquet: band and bladder round the thigh, the windlass on top ---------------------------------------------
BAND_Y = 0.0
BAND_W = 0.038
BAND_T = 0.006                    # strap and the bladder under it
BLADDER_W = 0.050                 # the pressure bladder under the band: 5 cm wide, 42 cm long, wrapped once round the
BLADDER_L = 0.42                  # thigh (12.7 cm across, so 39.9 cm round) with 2 cm of overlap
BLADDER_T = 0.0025
BAND_TOP = LEG_Z + LEG_R + BAND_T
HUB = (0.030, 0.022, 0.0075)      # the twisted strap under the rod's middle (visual)
ROD_R = 0.006
ROD_L = 0.150
ROD_Z = BAND_TOP + HUB[2] + ROD_R  # the rod's axis, free
ROD_MASS = 0.03
TILT_MAX = math.radians(12)       # the strap lets either end of the rod be pushed down this far
THETA0 = math.radians(-120)      # the rod starts 30 degrees off lying along the thigh (its +x end toward the arms)

# -- the clip: a C-shaped snap clip on the knee side; the rod's end goes down into it only when the rod lies along the leg ----
# Two flexible walls stand either side of a channel 0.5 mm wider than the rod each side. Near the top, each wall has a
# lip 1 mm inward, so the opening (11 mm) is narrower than the rod (12 mm): pushed down, the rod spreads the walls 0.5 mm
# and snaps past the lips, which then sit over it above its centre line and hold it down. Above the lips a 45 degree
# lead-in guides the rod into the channel. All heights are above the clip's floor (the top of its base plate).
CLIP_Y = -0.062                   # the clip's middle, along the leg from the windlass
CLIP_GAP = 0.0005                 # clear either side of the rod in the channel (13 mm wide)
PRONG_T = 0.0015                  # each wall's thickness across the rod (PETG)
PRONG_L = 0.008                   # its length along the rod
LIP_IN = 0.001                    # each lip reaches this far into the channel: an 11 mm opening for the 12 mm rod
LIP_Z = (0.007, 0.008)            # the lips' underside and top: above the seated rod's centre line (6 mm)
PRONG_TOP = ROD_Z - ROD_R - 0.0015  # the rod sweeps just over the clip (the walls' tops)
PLATE = (0.019, 0.010, 0.003)     # the clip's base plate glued on the thigh: the channel's floor
HOOK_FLOOR = LEG_Z + LEG_R + PLATE[2]
HOOK_DIP = ROD_Z - ROD_R - HOOK_FLOOR   # how far the rod's centre line comes down to seat on the clip's floor
LOCK_R = abs(CLIP_Y) + PLATE[1] / 2      # the tilted rod seats on the floor's far edge (6.7 cm out): depths are taken there

# -- mechanics: the band unwinds the rod unless something holds it; twisting either way tightens -----------------------------
K_TWIST = 0.06                    # N m per rad of twist
TWIST_DAMPING = 0.03              # N m s / rad
TILT_K = 1.5                      # N m / rad, the strap holding the rod level
TILT_D = 0.02
P_SNUG = 60.0                     # mmHg under the band as strapped
P_PER_DEG = 1.0                   # mmHg per degree of twist
GREEN = 250.0                     # the gauge's green arc starts here
GAUGE_MAX = 300.0
LOCKED_DEPTH = HOOK_DIP - 0.0005  # the rod counts as locked with its centre line this far down at LOCK_R (10 mm):
                                  # within 0.5 mm of the floor, its centre line under the lips at the clip's outer end
UNLOCK_DEPTH = HOOK_DIP + ROD_R - 0.008   # and comes out when lifted until its centre line is 8 mm up, above the lips
LOCK_ALIGN = math.radians(1.5)    # in the channel the rod lies within half a degree of along the thigh


def pressure(twist_deg: float) -> float:
    return P_SNUG + P_PER_DEG * abs(twist_deg)


# -- the gauge: an aneroid dial on a stand beyond the thigh, tilted up toward the arms ----------------------------------------
GAUGE_XY = (0.430, -0.150)
GAUGE_Z = 0.120                   # the dial's centre
GAUGE_TILT = math.radians(50)     # the dial faces this far up from facing the arms
DIAL_R = 0.060                    # 12 cm across
NEEDLE_SWEEP = math.radians(270)  # 0 to GAUGE_MAX


def needle_angle(p: float) -> float:
    """Clockwise from straight up, seen on the dial: 0 at -135 degrees, GAUGE_MAX at +135 (a stop holds it there)."""
    return -NEEDLE_SWEEP / 2 + NEEDLE_SWEEP * min(max(p, 0.0), GAUGE_MAX) / GAUGE_MAX


# -- scoring ---------------------------------------------------------------------------------------------------------------
CLEAR = 0.01                      # let go: every part of both arms at least 1 cm from the rod and the clip
