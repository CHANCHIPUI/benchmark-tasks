"""Offline static estimate (URDF FK and inertials; no simulator) of each outer arm's joint torques while it holds its
end of the housing flush: the arm's own weight plus a downward load at the tool point equal to the hand's push on the
housing. The pose is the demo's pinch pose at the end wall (scripted.py: support point, approach elevated 60 degrees,
fingers along y), solved with the demo's IK. The per-step log keeps only peak torques, so this gives the sustained
level. The load here is vertical. The log's hand figures are net contact forces, which include sideways force (B's
outer hand reaches 42 N late in its hold while its J2 peak stays at 17.8 N m), so they are not all vertical push.
Usage: python dev/hold_torque.py"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import geometry as g  # noqa: E402
import scripted  # noqa: E402
from yam_arena import YAM_URDF, ik  # noqa: E402
from yam_arena.constants import ARM_JOINTS  # noqa: E402
from yam_arena.station import BASES  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402

INERT = {}
for link in ET.parse(YAM_URDF).getroot().findall("link"):
    i = link.find("inertial")
    if i is not None and link.get("name") != "base":
        INERT[link.get("name")] = (float(i.find("mass").get("value")),
                                   np.array([float(v) for v in i.find("origin").get("xyz").split()]))


def potential(q, load):
    fk = forward_kinematics(YAM_URDF, {**{n: float(v) for n, v in zip(ARM_JOINTS, q)}, "joint7": 0.0, "joint8": 0.0})
    v = sum(m * 9.81 * (fk[n][:3, :3] @ c + fk[n][:3, 3])[2] for n, (m, c) in INERT.items())
    return v + load * ik.tcp_pose(np.asarray(q, float))[0][2]


def torques(q, load):
    out = []
    for i in range(6):
        d = np.zeros(6)
        d[i] = 1e-5
        out.append((potential(q + d, load) - potential(q - d, load)) / 2e-5)
    return np.abs(out)


L, D, WT = g.HOUSING_L, g.HOUSING_D, g.WALL_T
for name, side, m, loads in (("A outer (left)", "left", 1.0, (0.0, 15.0, 20.0)),
                             ("B outer (right)", "right", -1.0, (0.0, 15.0, 20.0))):
    base = np.array(BASES[side])
    origin_y = 0.5 * m                                     # the station's origin, world y
    p = np.array([g.HOUSING_X + 0.10, m * (L / 2 - WT / 2) - origin_y, g.CEIL - D + 0.014])
    best = None
    for f in ((0, 1, 0), (0, -1, 0)):
        R = scripted.rotm(scripted.elevated(scripted.OUTER_ELEV), f)
        for q0 in ([0, 1.0, 1.0, -0.6, 0, 0], [0.3 * m, 1.5, 1.2, 0.3, 0, 0], [0.5 * m, 1.8, 1.8, 0, 0, 0],
                   [-0.4 * m, 2.2, 2.0, 1.2, 0, 0], [0.0, 2.0, 2.0, 0.5, 0, -1.5], [0.0, 2.0, 2.0, 0.5, 0, 1.5]):
            q, e, r = ik.solve(p, ik.matrix_to_quat_xyzw(R), q_init=q0, base_pos=base, iters=400)
            if best is None or e + 0.05 * r < best[1] + 0.05 * best[2]:
                best = (q, e, r)
    q, e, r = best
    print(f"{name}: tool point {np.round(p * 1000).tolist()} mm (station frame), IK error {e * 1000:.1f} mm, "
          f"q {np.round(q, 2).tolist()}")
    for load in loads:
        print(f"  hand load {load:4.0f} N: |torque| J1-J6 {np.round(torques(q, load), 1).tolist()} N m")
