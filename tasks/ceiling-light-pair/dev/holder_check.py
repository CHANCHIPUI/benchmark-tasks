"""Offline as-built check (URDF FK + DLS IK + the URDF's collision hulls; no simulator), each station's inner arm in
its own frame (B mirrored into A's, y -> -y):
(1) the pick: tool point at the driver's grip point (A: 45 mm above its bottom, which stands 190 above the table on
    its peg; B: 30 mm, 205), approach level and turned toward the holder, fingers closed on the 36 mm grip: clearance
    of the fingers and gripper to the riser's top and to the 22 mm peg;
(2) the drive: the driver upright, its screw's tip at the ceiling (bite) and seated: clearance of the arm to the
    housing's near long wall, far wall, end wall and back, each as a thin box."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import geometry as g  # noqa: E402
import layout_check as L  # noqa: E402
import scripted  # noqa: E402
from yam_arena import analysis, ik  # noqa: E402
from yam_arena.constants import ARM_JOINTS  # noqa: E402

TOP, CEIL = g.TOP, g.CEIL
base = np.array(L.RIGHT)
print("(1) pick from the holder, fingers closed on the driver")
for s in "ab":
    hx, hy = g.HOLDER_XY["a"]                      # B mirrored onto A's holder place
    hl = np.array([hx, hy - 0.5])
    fl = TOP + g.HOLDER_FLOOR[s]
    yaw = math.atan2(hl[1] - base[1], hl[0] - base[0])
    R = scripted.rotm(scripted.elevated(0.0, math.degrees(yaw)), (math.sin(yaw), -math.cos(yaw), 0.0))
    p = np.array([hl[0], hl[1], fl + g.DRV_GRIP[s]])
    q, e, r = ik.solve(p, ik.matrix_to_quat_xyzw(R), q_init=[-0.64, 0.02, 0.25, -0.23, 0.0, 0.0], base_pos=base, iters=300)
    qq = {n: float(v) for n, v in zip(ARM_JOINTS, q)}
    qq["joint7"] = qq["joint8"] = -0.030
    hulls = analysis.world_hulls(analysis.fk_link_poses_world(qq, base), ["tip_left", "tip_right", "gripper"])
    rt = fl - g.PEG_H
    dr = analysis.hulls_to_box(hulls, np.array([hl[0], hl[1], (TOP + rt) / 2]), np.eye(3), np.array([0.089, 0.089, rt - TOP]))
    Rp = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
    ds = analysis.hulls_to_box(hulls, np.array([hl[0], hl[1], rt + g.PEG_H / 2]), Rp, np.array([g.PEG_W, g.PEG_W, g.PEG_H]))
    print(f"  {s.upper()}: on its peg {g.HOLDER_FLOOR[s] * 1000:.0f} above the table, grip {g.DRV_GRIP[s] * 1000:.0f} above the driver's "
          f"bottom ({(g.DRV_TIP - g.DRV_GRIP[s]) * 1000:.0f} below the bit tip), tool point {(p[2] - TOP) * 1000:.0f} up: "
          f"IK error {e * 1000:.1f} mm; riser top {dr[0] * 1000:.1f} mm ({dr[1]}), peg {ds[0] * 1000:.1f} mm ({ds[1]})")
print("(2) at the slot (screw tip at the ceiling / seated)")
W, Lh, D, WT = g.HOUSING_W, g.HOUSING_L, g.HOUSING_D, g.WALL_T
X_F, yend = g.HOUSING_X, Lh / 2 - 0.5
walls = {"near wall": ((X_F - W / 2 + WT / 2, -0.5, CEIL - D / 2), (WT, Lh, D)),
         "far wall": ((X_F + W / 2 - WT / 2, -0.5, CEIL - D / 2), (WT, Lh, D)),
         "end wall": ((X_F, yend - WT / 2, CEIL - D / 2), (W, WT, D)),
         "back": ((X_F, -0.5, CEIL - g.SHEET_T / 2), (W, Lh, g.SHEET_T))}
for s, x_k in (("a", 0.37), ("b", 0.47)):
    below = g.DRV_TIP - g.DRV_GRIP[s]
    for label, d in (("bite", below + g.SCREW_L), ("seated", below + g.SHEET_T)):
        p = np.array((x_k, 0.008, CEIL - d))
        yaw = float(np.degrees(np.arctan2(p[1] - base[1], p[0] - base[0])))
        q, e, r = min((L.solve(p, L.rot(0, yaw, roll), base) for roll in (0.0, 180.0)), key=lambda t: t[1] + 0.05 * t[2])
        arm = analysis.world_hulls(analysis.fk_link_poses_world({n: float(v) for n, v in zip(ARM_JOINTS, q)}, base),
                                   ["link1", "link2", "link3", "link4", "link5", "gripper"])
        out = []
        for wn, (c, sz) in walls.items():
            dd, ln = analysis.hulls_to_box(arm, np.array(c), np.eye(3), np.array(sz))
            out.append(f"{wn} {dd * 1000:.1f} ({ln})")
        print(f"  {s.upper()} slot x {x_k * 1000:.0f}, gripped {below * 1000:.0f} below the bit tip, {label}: IK error "
              f"{e * 1000:.1f} mm; " + ", ".join(out))
