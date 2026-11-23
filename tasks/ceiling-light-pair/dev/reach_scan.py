"""Offline reach scan (URDF FK + DLS IK, no simulator): which TCP points one YAM arm reaches with the
gripper's approach axis pointing straight up, in its station frame (x forward, y left, z above the table top)."""
import numpy as np
from yam_arena import station, ik
from yam_arena.urdf import axis_angle_matrix
TOP = station.TABLE_TOP
def up_rot(finger_axis):
    # approach +z up; fingers close along finger_axis ('x' or 'y')
    z = np.array([0, 0, 1.0])
    x = np.array([1.0, 0, 0]) if finger_axis == 'x' else np.array([0, 1.0, 0])
    y = np.cross(z, x)
    return ik.matrix_to_quat_xyzw(np.column_stack([x, y, z]))
def solve(p, rot, base, inits=None):
    best = None
    for q0 in (inits or [[0, 1.0, 1.0, -0.6, 0, 0], [0, 1.5, 1.5, 0.5, 0, 0], [0, 0.8, 0.6, 1.2, 0, 0], [0, 2.0, 2.0, 1.4, 0, 0]]):
        q, e, r = ik.solve(p, rot, q_init=q0, base_pos=base, iters=300)
        if best is None or e + 0.05 * r < best[1] + 0.05 * best[2]:
            best = (q, e, r)
    return best
if __name__ == '__main__':
    base = station.BASES['left']
    for fa in ('x', 'y'):
        rot = up_rot(fa)
        print('approach up, fingers along', fa)
        print('  z_above_table:', ' '.join(f'{z:5.2f}' for z in np.arange(0.15, 0.61, 0.05)))
        for x in np.arange(0.20, 0.51, 0.05):
            row = []
            for z in np.arange(0.15, 0.61, 0.05):
                for dy in (0.0,):
                    q, e, r = solve((x, base[1] + dy, TOP + z), rot, base)
                    row.append('  ok ' if e < 1e-3 and r < 0.01 else f'{e*1000:4.0f} ')
            print(f'  x={x:4.2f}', ''.join(row))
