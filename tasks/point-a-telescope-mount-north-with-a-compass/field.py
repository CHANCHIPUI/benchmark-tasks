"""The magnetic field the compass needle follows. Pure NumPy; computed, not simulated (physics engines have no magnetism).

Positions are metres in the station frame (x forward, y to the robot's left), heights above the table top. Headings are
radians anticlockwise from +x seen from above; a bearing is degrees clockwise from +x (bearing = -heading).

The field is Earth's horizontal field, 20 uT toward north (bearing 115 degrees; declination taken as zero), plus the
field of the steel toolbox, modelled as one point magnetic dipole at its centre:
    B = mu0/(4 pi) * (3 (m . r^) r^ - m) / |r|^3.
The arms, the mount and the table are not magnetic in this model.
"""

from __future__ import annotations

import math

import numpy as np

NORTH_BEARING = 115.0                         # degrees clockwise from straight ahead
NORTH = math.radians(-NORTH_BEARING)          # heading, rad
EARTH_UT = 20.0                               # horizontal field, microtesla
BOX_LO = np.array([0.37, 0.14, 0.0])          # toolbox outline (m)
BOX_HI = np.array([0.53, 0.46, 0.11])
DIPOLE_AT = (BOX_LO + BOX_HI) / 2             # its centre
DIPOLE_HEADING = math.radians(114.0)          # the dipole's direction: bearing 246, horizontal
DIPOLE_AM2 = 0.3283                           # moment, A m^2: tuned so the compass reads 50 degrees off at its start
MU0_4PI = 1e-7


def earth() -> np.ndarray:
    return EARTH_UT * np.array([math.cos(NORTH), math.sin(NORTH), 0.0])


def dipole(p) -> np.ndarray:
    """The toolbox's field at p (m), in microtesla."""
    m = DIPOLE_AM2 * np.array([math.cos(DIPOLE_HEADING), math.sin(DIPOLE_HEADING), 0.0])
    r = np.asarray(p, float) - DIPOLE_AT
    d = float(np.linalg.norm(r))
    rh = r / d
    return MU0_4PI * (3 * rh * float(m @ rh) - m) / d**3 * 1e6


def field(p) -> np.ndarray:
    """Earth's horizontal field plus the toolbox's dipole, at p (m), in microtesla."""
    return earth() + dipole(p)


def reading_heading(p) -> float:
    """Where a level needle at p points: the heading of the field's horizontal part (rad)."""
    b = field(p)
    return math.atan2(b[1], b[0])


def error_deg(p) -> float:
    """How far a level compass at p reads off north, degrees anticlockwise (+) or clockwise (-)."""
    d = math.degrees(reading_heading(p) - NORTH)
    return (d + 180.0) % 360.0 - 180.0


def box_distance(p) -> float:
    q = np.clip(np.asarray(p, float), BOX_LO, BOX_HI)
    return float(np.linalg.norm(np.asarray(p, float) - q))
