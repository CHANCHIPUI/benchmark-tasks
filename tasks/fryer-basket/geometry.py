"""Dimensions and layout of the fryer task: an Avantco F100 countertop electric fryer (10 lb, 10.5 x 15.75 x 11.5 in,
a 9.5 x 12 x 6 in tank) on the station table, as supplied, with its 177FBASKET wire basket (9.5 x 7.25 x 4 in, a
9.5 in coated handle and a hook on the end opposite the handle) of fries down in the tank. The fryer's own basket
hanger is its control head's top front edge: the hook goes over it, and the basket's back rests against the head's
front face. Sizes not in the maker's sheets (the deck and hanger heights, the head's depth) are measured from the
maker's product photos, scaled by the listed overall sizes.

Station frame: metres from the midpoint between the arm bases, x forward, y to the robot's left, z up; heights are
above the table top (TOP). The fryer stands with its front toward the arms; "u" is the distance back from its front
face (along +x) and "v" is across it (along +y) from its centreline.
"""

from __future__ import annotations

import math

TOP = 0.7475                       # the station's table top

# -- the fryer (Avantco F100) ------------------------------------------------------------------------------------------
FRONT_X = 0.270                    # its front face
CENTRE_Y = 0.050                   # its centreline (a little toward the left arm, which works it in the demo)
WIDTH = 0.267                      # 10.5 in
DEPTH = 0.400                      # 15.75 in
HEIGHT = 0.292                     # 11.5 in, to the top of the control head
DECK = 0.190                       # the top of the body, round the tank's opening
VAT_U = (0.015, 0.300)             # the tank's open top, front to back (the head covers the rest of its 12 in)
VAT_HALF = 0.1205                  # and half its width (9.5 in)
VAT_FLOOR = DECK - 0.152           # the tank is 6 in deep
RACK = VAT_FLOOR + 0.010           # the bottom grate (crumb screen) the basket rests on
OIL = 0.110                        # the fill level: 10 lb of oil, about 4.9 L, 7.2 cm deep in the tank
HEAD_U = 0.300                     # the control head's front face; it runs back to the rear, from the deck up to HEAD_TOP
HEAD_TOP = 0.284                   # the head's flat top; its rolled front edge stands above it, to the listed HEIGHT
ROD_U = HEAD_U + 0.008             # the hanger: the head's rolled top front edge, a 8 mm round lip along it
ROD_Z = HEAD_TOP + 0.004
KNOB_V = 0.119                     # the thermostat dial on the head's red panel, at its right (seen from the front: -v),
KNOB_R = 0.013                     # beside a hung basket; it collides
ROD_R = 0.004
ROD_HALF = 0.100                   # its half-length across

# -- the basket (Avantco 177FBASKET: wire mesh, coated handle, hook) ---------------------------------------------------------
BASKET = (0.241, 0.184, 0.102)     # length front to back (u), width across (v), height: 9.5 x 7.25 x 4 in
WALL = 0.002                       # its walls as collision shells
BASKET_MASS = 0.40                 # kg with the handle: estimated (the maker lists no net weight)
HANDLE_STEM = 0.050                # the handle: a bare wire riser this tall from the front wall's top (clearing the
#                                   tank's front rim), then
HANDLE_LEN = 0.191                 # the coated grip, forward and up (9.5 in in all with the riser)
HANDLE_PITCH = math.radians(40)
HANDLE = (0.022, 0.008)            # the coated grip: width and thickness (a flat bar)
STEM_T = 0.006                     # the riser's wire, square for collisions
GRIP_S = 0.070                     # the demo holds the grip this far along from the riser's top: its 6.8 cm finger
#                                   plates then end 3.6 cm short of the bare riser, wholly on the coating
HOOK_REACH = 0.020                 # the hook: a tab running back from the back wall's top this far,
HOOK_LIP = 0.009                   # then a lip this far down from the tab's top, behind the hanger lip
HOOK_T = 0.003
HOOK_W = 0.060                     # its width across
REST_U = 0.262                     # the basket's back wall, down in the tank on the grate
HUNG_U = HEAD_U - 0.0005           # and hanging, its back against the head's face
FRIES_N = 30
FRY = (0.010, 0.070, 0.010)        # each fry: 1 x 1 cm, 7 cm long (7 cm3), lying across the basket
FRY_MASS = 0.0077                  # kg (1.1 g/cm3, like potato): 230 g in all, a half-pound batch (the F100's manual
#                                   allows up to 1.98 lb a pot)

# -- scoring -------------------------------------------------------------------------------------------------------------
DRAIN_CLEAR = 0.020                # out of the oil: the basket's bottom at least 2 cm above the oil's surface
HANDS_OFF = 0.020                  # let go: every part of both arms at least 2 cm from the basket for the last 1 s
HOOK_SEAT = 0.003                  # hung: the rod in the hook's notch (wall, tab, lip); tolerance at the tab
TOUCH = 0.002                      # touching: part of the robot within 2 mm of the wire basket or its hook
RIM_REST = 0.003                   # resting on the fryer: a floor corner outside the tank's opening, at most this far above
#                                   the surface under it (the deck, or the head's top)
FRY_TOL = 0.005                    # a fry counts as in the basket with its middle up to 5 mm outside the walls or floor
SPILL_SOME = 3                     # up to this many fries out of the basket still earns 0.25


def basket_origin(u_back: float, bottom: float) -> tuple[float, float, float]:
    """The basket body's origin (the middle of its floor) in the station frame, for its back wall at ``u_back`` and
    its floor's underside at height ``bottom``."""
    return (FRONT_X + u_back - BASKET[0] / 2, CENTRE_Y, TOP + bottom)
