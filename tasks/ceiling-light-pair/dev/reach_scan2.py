"""Offline reach scan (URDF FK + DLS IK, no simulator). For TCP targets in one arm's station frame (x forward,
y left, heights above the table top), the largest approach elevation (0 = horizontal, 90 = straight up) the arm
reaches, approach turned toward the target in yaw, fingers closing horizontally."""
import sys, numpy as np
from yam_arena import station, ik
TOP = station.TABLE_TOP
def rot(elev_deg, yaw_deg, roll_deg=0.0):
    e, y = np.radians(elev_deg), np.radians(yaw_deg)
    z = np.array([np.cos(e) * np.cos(y), np.cos(e) * np.sin(y), np.sin(e)])
    x = np.array([-np.sin(y), np.cos(y), 0.0])      # fingers close horizontally, across the approach
    if roll_deg:
        r = np.radians(roll_deg); yv = np.cross(z, x); x = np.cos(r) * x + np.sin(r) * yv
    yv = np.cross(z, x)
    return ik.matrix_to_quat_xyzw(np.column_stack([x, yv, z]))
INITS = [[0, 1.0, 1.0, -0.6, 0, 0], [0, 1.6, 1.2, 0.3, 0, 0], [0, 2.2, 1.6, 1.0, 0, 0], [0, 1.2, 0.4, 0.8, 0, 0]]
def solve(p, q_rot, base):
    best = None
    for q0 in INITS:
        q, e, r = ik.solve(p, q_rot, q_init=q0, base_pos=base, iters=250)
        if best is None or e + 0.05 * r < best[1] + 0.05 * best[2]:
            best = (q, e, r)
    return best
def max_elev(p, base, yaw=None):
    if yaw is None:
        yaw = np.degrees(np.arctan2(p[1] - base[1], p[0] - base[0]))
    for elev in (90, 75, 60, 45, 30, 15, 0):
        q, e, r = solve(p, rot(elev, yaw), base)
        if e < 1e-3 and r < 0.01:
            return elev, q
    return None, None
if __name__ == '__main__':
    base = station.BASES['left']
    dys = [float(v) for v in sys.argv[1:]] or [0.0]
    for dy in dys:
        print(f'dy={dy:+.2f} from the arm base; cells: max approach elevation (deg), "--" unreachable')
        zs = np.arange(0.20, 0.61, 0.05)
        print('  z_above_table:', ' '.join(f'{z:5.2f}' for z in zs))
        for x in np.arange(0.20, 0.51, 0.05):
            row = []
            for z in zs:
                el, _ = max_elev((x, base[1] + dy, TOP + z), base)
                row.append('   -- ' if el is None else f'{el:5d} ')
            print(f'  x={x:4.2f}      ', ''.join(row), flush=True)
