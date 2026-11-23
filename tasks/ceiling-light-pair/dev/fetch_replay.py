"""Offline replay of a station's demo script with perfect joint tracking and the housing at its place, to the
screwdriver pick: the inner arm's joints along the fetch and its links' lowest clearance to the holder and stand.
Usage: python dev/fetch_replay.py a|b"""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geometry as g  # noqa: E402
import scripted  # noqa: E402
from yam_arena import YAM_URDF  # noqa: E402
from yam_arena.constants import ARM_JOINTS  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402
from yam_pair.demo import StationView, controller  # noqa: E402
from yam_pair.layout import Layout  # noqa: E402

s = sys.argv[1]
layout = Layout("side_by_side", separation=1.0)
t = [0.0]


def objects():
    z = g.CEIL - g.START_DROP if t[0] < 7.0 else g.CEIL
    h = NS(pos=np.array([g.HOUSING_X, 0.0, z]), quat_xyzw=np.array([0, 0, 0, 1.0]))
    sc = NS(joints={"a_pz": g.TRAVEL, "b_pz": 0.0 if s == "a" else 0.0})
    if s == "b":
        sc.joints["a_pz"] = g.TRAVEL
    drv = {f"driver_{k}": NS(pos=np.array([x, y, g.TOP + g.HOLDER_FLOOR[k]]), quat_xyzw=np.array([0, 0, 0, 1.0]))
           for k, (x, y) in g.HOLDER_XY.items()}
    return {"housing": h, "screws": sc, **drv}


ctl = controller(scripted.station_script(s, ""), s, layout)
cmd = np.r_[ctl.ctx.left.command(), ctl.ctx.right.command()]
inner = "right" if s == "a" else "left"
qs = []
for k in range(1600):
    t[0] = (k + 1) / 20.0
    view = StationView(station=s, t=t[0], state=cmd, layout=layout, _objects=objects)
    cmd = ctl.act(view)
    st = ctl.ctx.flags.get("station")
    q = cmd[7:13] if inner == "right" else cmd[0:6]
    qs.append((t[0], st.phase if st else "", q.copy()))
    if st and st.phase.startswith("to slot"):
        break
base = scripted.ik  # noqa
arm = ctl.ctx.right if inner == "right" else ctl.ctx.left
hx, hy = g.HOLDER_XY[s]
for tt, ph, q in qs:
    if ph != "fetch driver" or round(tt * 20) % 5:
        continue
    fk = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
    pts = {n: ctl.ctx.world(f[:3, 3] + arm.base) for n, f in fk.items() if n.startswith("link") or n == "gripper"}
    low = min(pts.items(), key=lambda kv: kv[1][2])
    near = min(pts.items(), key=lambda kv: np.hypot(kv[1][0] - hx, kv[1][1] - hy) if kv[1][2] < g.TOP + 0.25 else 9)
    print(f"t {tt:5.2f} q {np.round(q, 2).tolist()} lowest {low[0]} z {low[1][2] - g.TOP:.3f} "
          f"nearest-holder {near[0]} {np.round((near[1] - [hx, hy, g.TOP]) * 1000).tolist()}")
