"""Offline workspace map by random joint sampling (URDF FK, no simulator): for one YAM arm, the largest approach
elevation (deg above horizontal) seen in each (x, |dy|, z) cell, station frame, heights above the table top."""
import collections
import sys

import numpy as np
from yam_arena import ik, station

rng = np.random.default_rng(0)
base = np.array(station.BASES['left'])
TOP = station.TABLE_TOP
best = collections.defaultdict(lambda: -99.0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
for _ in range(N):
    q = rng.uniform(ik.LOWER, ik.UPPER)
    p, R = ik.tcp_pose(q, base)
    a = R[:, 2]
    elev = float(np.degrees(np.arcsin(np.clip(a[2], -1, 1))))
    cell = (round(p[0] / 0.05), round(abs(p[1] - base[1]) / 0.1), round((p[2] - TOP) / 0.05))
    if elev > best[cell]:
        best[cell] = elev
np.save('ws_best.npy', np.array([[*k, v] for k, v in best.items()]))
for dy in (0, 1, 2, 3):
    print(f'|dy|={dy / 10:.1f} m: max approach elevation (deg) by x (rows) and height above table (cols)')
    zs = range(4, 14)
    print('        ' + ''.join(f'{z * 0.05:6.2f}' for z in zs))
    for x in range(3, 12):
        row = ''
        for z in zs:
            v = best.get((x, dy, z))
            row += '    --' if v is None else f'{v:6.0f}'
        print(f'  x={x * 0.05:.2f}' + row)
