"""Offline r2 checks (URDF FK + DLS IK + collision hulls; no simulator), station A's frame, ceiling 580 above the table,
housing centre line x 420. (1) Start grips with the housing's back 100 mm below the ceiling. (2) The inner arm driving
at the near keyhole (x 370, station A) and at the far keyhole (x 470, as station B does at its end, mirrored).
(3) Reach from a lone station's arms to the far end's slot."""
import numpy as np
import layout_check as L
from yam_arena import station

H, X_F = 0.58, 0.42
TOP = L.TOP
y_end = L.HOUSING['L'] / 2 - L.A_ORIGIN_Y
start_back = H - 0.10
rows = [
    ('outer: end wall, back 100 below', L.LEFT, (X_F, y_end, TOP + start_back - L.HOUSING['D'] + 0.025), (60, 45, 30), 0, False),
    ('inner: near rim y 300, back 100 below', L.RIGHT, (X_F - L.HOUSING['W'] / 2 - 0.01, 0.30 - L.A_ORIGIN_Y, TOP + start_back - L.HOUSING['D'] + 0.01), (60, 45, 30), 0, False),
]
for x_k in (0.37, 0.47):
    for label, dz in (('tip in pilot hole', 0.019), ('seated', 0.004)):
        rows.append((f'inner: driver at x {x_k:.2f}, {label}', L.RIGHT, (x_k, L.KEY_Y - L.A_ORIGIN_Y, TOP + H - L.DRIVER_TIP - dz), (0,), None, True))
for label, base, p, elevs, yaw, mounted in rows:
    if yaw is None:
        yaw = float(np.degrees(np.arctan2(p[1] - base[1], p[0] - base[0])))
    r = L.best_pose(np.array(p), base, elevs, yaw, H, X_F, mounted)
    if r is None:
        print(f'{label:40s} UNREACHED', flush=True)
    else:
        el, _, q, (dp, lp, dh, lh) = r
        print(f'{label:40s} elev {el:3d} yaw {yaw:5.1f} panel clr {dp*1000:6.1f} mm ({lp}) housing clr {dh*1000:6.1f} mm ({lh})', flush=True)
# lone-station reach: shoulder (link2 origin, 114 mm above the base) to the far slot and far end wall
for name, base in (('A left (outer)', L.LEFT), ('A right (inner)', L.RIGHT)):
    sh = np.array(base) + (0.02, 0.0, 0.114)
    for tname, t in (('far slot (x 470, y -508)', (0.47, -0.508 - L.A_ORIGIN_Y, TOP + H)), ('far end wall (y -609)', (X_F, -0.609 - L.A_ORIGIN_Y, TOP + H - 0.03))):
        print(f'{name:16s} shoulder to {tname}: {np.linalg.norm(np.array(t) - sh):.3f} m', flush=True)
