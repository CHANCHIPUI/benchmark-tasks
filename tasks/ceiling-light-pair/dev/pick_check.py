"""Offline check: the inner arm (A-right) takes its screwdriver from its holder (station A frame), approach
horizontal, fingers closing horizontally, for a few holder positions and grip heights above the table."""
import numpy as np
from layout_check import solve, rot, clearance, RIGHT, TOP
for x in (0.18, 0.22):
    for y_world in (0.33, 0.38):
        for z in (0.10, 0.14):
            p = np.array([x, y_world - 0.5, TOP + z])
            yaw = float(np.degrees(np.arctan2(p[1] - RIGHT[1], p[0] - RIGHT[0])))
            res = [(el, solve(p, rot(el, yaw, roll), RIGHT)) for el in (0, -15) for roll in (0.0, 180.0)]
            el, (q, e, r) = min(res, key=lambda t: t[1][1] + 0.05 * t[1][2])
            print(f'holder x={x:.2f} y={y_world:+.2f} grip z={z:.2f}: elev {el} yaw {yaw:5.1f} err {e*1000:.1f} mm {np.degrees(r):.1f} deg q={np.round(q,2).tolist()}', flush=True)
