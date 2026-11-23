"""Offline build checks (URDF FK + DLS IK + collision hulls; no simulator), ceiling 580 above the table, housing
centre line x 420. Driver gripped 165 mm below its bit tip (bite: screw tip at the ceiling; seated: 17.5 mm higher).
(1) A's driver at its near slot (370, +508), B's at its far slot (470, -508), both in their own station frames.
(2) Driver pick from its holder, approach horizontal (driver vertical), grip heights 150-190 mm.
(3) Inner hand pushing up under the near rim at y +/-300 (start and flush) and at y 0, closed fingers."""
import numpy as np
import layout_check as L

H, X_F = 0.58, 0.42
TOP = L.TOP
T_BITE, T_SEAT = 0.165 + 0.019, 0.165 + 0.019 - 0.0175
rows = []
# A: inner arm is A-right, slot world (0.37, +0.508) -> local y +0.008; B: inner arm is B-left; mirror B into A's frame:
# B's slot world (0.47, -0.508) -> B local y -0.008; mirrored (y -> -y) it is A-right reaching local +0.008 at x 0.47.
for name, x_k in (('A near slot x 370', 0.37), ('B far slot x 470 (mirrored)', 0.47)):
    for label, d in (('bite', T_BITE), ('seated', T_SEAT)):
        rows.append((f'{name}, {label}', L.RIGHT, (x_k, 0.008, TOP + H - d), (0,), None, True))
for gz in (0.15, 0.17, 0.19):
    rows.append((f'pick, grip {gz*1000:.0f} up, holder (220, 380)', L.RIGHT, (0.22, 0.38 - 0.5, TOP + gz), (0,), None, False))
rim_start, rim_flush = TOP + H - 0.10 - 0.055, TOP + H - 0.055
for label, y, z in (('inner push, rim y 300, start', 0.30, rim_start), ('inner push, rim y 300, flush', 0.30, rim_flush),
                    ('inner push, rim y 0, flush', 0.0, rim_flush)):
    rows.append((label, L.RIGHT, (X_F - L.HOUSING['W'] / 2 + 0.006, y - 0.5, z - 0.016), (60, 45, 30), 0, True))
for label, base, p, elevs, yaw, mounted in rows:
    if yaw is None:
        yaw = float(np.degrees(np.arctan2(p[1] - base[1], p[0] - base[0])))
    r = L.best_pose(np.array(p), base, elevs, yaw, H, X_F, mounted)
    if r is None:
        print(f'{label:48s} UNREACHED', flush=True)
    else:
        el, _, q, (dp, lp, dh, lh) = r
        print(f'{label:48s} elev {el:3d} yaw {yaw:5.1f} panel clr {dp*1000:6.1f} mm ({lp}) housing box clr {dh*1000:6.1f} mm ({lh}) '
              f'q={np.round(q, 2).tolist()}', flush=True)
