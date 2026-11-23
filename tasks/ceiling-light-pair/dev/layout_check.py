"""Offline layout and reach check (URDF FK + DLS IK + the URDF's collision hulls; no simulator).

Station A's frame (yam_pair side_by_side, separation 1.0 m: A's origin is world (0, +0.5)); B mirrors it.
For each candidate (ceiling height H above the table, housing centre line x_f), solve every arm pose the scripted
demo needs and report position/rotation error and the arm's clearance to the ceiling panel and the housing.
"""
import itertools
import sys

import numpy as np
from yam_arena import analysis, ik, station
from yam_arena.constants import ARM_JOINTS

TOP = station.TABLE_TOP
A_ORIGIN_Y = 0.5                       # world y of station A's origin
LEFT, RIGHT = station.BASES['left'], station.BASES['right']   # A-left is the outer arm, A-right the inner one
HOUSING = dict(L=1.218, W=0.280, D=0.055)                      # estimates (lens off)
KEY_Y, KEY_DX = 0.508, -0.050          # keyhole used: 508 mm from the middle, 50 mm toward the robots
PANEL = dict(x0=0.18, x1=0.78, t=0.018)
DRIVER_TIP = 0.135                      # TCP to bit tip; the screw adds 32 mm
SCREW = 0.032


def rot(elev, yaw, roll=0.0):
    """Approach elevated `elev` deg and turned `yaw` deg; fingers close horizontally across the approach (roll 0 or
    180, the same grip for the symmetric fingers)."""
    e, y, r = np.radians(elev), np.radians(yaw), np.radians(roll)
    z = np.array([np.cos(e) * np.cos(y), np.cos(e) * np.sin(y), np.sin(e)])
    x0 = np.array([-np.sin(y), np.cos(y), 0.0])
    x = np.cos(r) * x0 + np.sin(r) * np.cross(z, x0)
    return ik.matrix_to_quat_xyzw(np.column_stack([x, np.cross(z, x), z]))


INITS = [[0, 1.0, 1.0, -0.6, 0, 0], [0, 1.6, 1.2, 0.3, 0, 0], [0, 1.3, 2.2, 0.6, 0, 0], [0, 1.6, 2.6, 1.0, 0, 0],
         [0, 1.0, 2.0, 0.8, 0, 0], [0, 1.8, 1.8, 0.0, 0, 0], [0, 2.2, 2.4, 0.4, 0, 0]]


def solve(p, q_rot, base):
    best = None
    yaw0 = float(np.arctan2(p[1] - base[1], p[0] - base[0]))
    for q0 in INITS:
        q0 = [yaw0] + list(q0[1:5]) + [0.0]
        q, e, r = ik.solve(p, q_rot, q_init=q0, base_pos=base, iters=300)
        if best is None or e + 0.05 * r < best[1] + 0.05 * best[2]:
            best = (q, e, r)
    return best


def clearance(q, base, H, x_f, mounted=True):
    poses = analysis.fk_link_poses_world({n: float(v) for n, v in zip(ARM_JOINTS, q)}, base)
    arm = analysis.world_hulls(poses, ['link1', 'link2', 'link3', 'link4', 'link5', 'gripper'])
    panel_c = np.array([(PANEL['x0'] + PANEL['x1']) / 2, 0.0, TOP + H + PANEL['t'] / 2])
    d_panel, l_panel = analysis.hulls_to_box(arm, panel_c, np.eye(3), (PANEL['x1'] - PANEL['x0'], 3.0, PANEL['t']))
    top = TOP + H if mounted else TOP + H - 0.20
    hc = np.array([x_f, -A_ORIGIN_Y, top - HOUSING['D'] / 2])
    d_h, l_h = analysis.hulls_to_box(arm, hc, np.eye(3), (HOUSING['W'], HOUSING['L'], HOUSING['D']))
    return d_panel, l_panel, d_h, l_h


def best_pose(p, base, elevs, yaw, H, x_f, mounted=True):
    errs = []
    for el in elevs:
        q, e, r = min((solve(p, rot(el, yaw, roll), base) for roll in (0.0, 180.0)), key=lambda t: t[1] + 0.05 * t[2])
        errs.append(f'{el}:{e * 1000:.0f}mm/{np.degrees(r):.0f}deg')
        if e > 1.5e-3 or r > 0.02:
            continue
        c = clearance(q, base, H, x_f, mounted)
        errs[-1] += f'/panel{c[0] * 1000:.0f}'
        if c[0] > 0.01:
            return (el, 0.0, q, c)
    print('      (failed:', ' '.join(errs), ')', flush=True)
    return None


def check(H, x_f):
    y_end = HOUSING['L'] / 2 - A_ORIGIN_Y            # end wall in A's frame
    rows = []
    for label, base, p, elevs, yaw, mounted in [
        ('outer: end wall, on its stands', LEFT, (x_f, y_end, TOP + H - 0.20 - HOUSING['D'] + 0.025), (60, 45, 30, 15, 0), 0, False),
        ('outer: end wall, flush', LEFT, (x_f, y_end, TOP + H - 0.035), (85, 75, 60, 45, 30), 0, True),
        ('inner: near rim, on its stands', RIGHT, (x_f - HOUSING['W'] / 2 - 0.01, 0.30 - A_ORIGIN_Y, TOP + H - 0.20 - HOUSING['D'] + 0.01), (60, 45, 30, 15), 0, False),
        ('inner: near rim, flush', RIGHT, (x_f - HOUSING['W'] / 2 - 0.01, 0.30 - A_ORIGIN_Y, TOP + H - HOUSING['D'] + 0.01), (75, 60, 45, 30), 0, True),
        ('inner: driver, screw tip at ceiling', RIGHT, (x_f + KEY_DX, KEY_Y - A_ORIGIN_Y, TOP + H - DRIVER_TIP - SCREW), (0,), None, True),
        ('inner: driver, screw seated', RIGHT, (x_f + KEY_DX, KEY_Y - A_ORIGIN_Y, TOP + H - DRIVER_TIP - 0.004), (0,), None, True),
        ('inner: middle support (after its screw)', RIGHT, (x_f - HOUSING['W'] / 2 - 0.01, 0.0 - A_ORIGIN_Y, TOP + H - HOUSING['D'] + 0.01), (75, 60, 45, 30), 0, True),
    ]:
        if yaw is None:
            yaw = float(np.degrees(np.arctan2(p[1] - base[1], p[0] - base[0])))
        r = best_pose(np.array(p), base, elevs, yaw, H, x_f, mounted)
        if r is None:
            rows.append(f'    {label:42s} UNREACHED')
        else:
            el, _, q, (dp, lp, dh, lh) = r
            rows.append(f'    {label:42s} elev {el:3d} yaw {yaw:5.1f}  panel clr {dp * 1000:6.1f} mm ({lp})  '
                        f'housing clr {dh * 1000:6.1f} mm ({lh})  q={np.round(q, 2).tolist()}')
    return rows


if __name__ == '__main__':
    Hs = [float(v) for v in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.55, 0.58]
    xs = [float(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.38, 0.42]
    for H, x_f in itertools.product(Hs, xs):
        print(f'H={H:.2f} (ceiling underside above the table), housing centre line x_f={x_f:.2f}', flush=True)
        for row in check(H, x_f):
            print(row, flush=True)
