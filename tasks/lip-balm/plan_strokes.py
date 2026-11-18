"""Plan the demo's two strokes offline: for points every 2 mm along each lip's centre line, a stick axis within 10
degrees of the lip's local inward normal (so the bullet, not the tube's rim, meets the lip), a hand turn about the
stick that the right arm can reach with its links clear of the face, and joint angles that change smoothly from point
to point.

    python plan_strokes.py          # writes assets/stroke_plan.npz (needs yam_arena, not the simulator)

The right hand holds the stick in a pen grip, as demo.py takes it: fingers along the stick, closed on the tube's base
end, the tool point PEN_S from the base end.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("task", HERE / "task.py")
task = importlib.util.module_from_spec(spec)
sys.modules["task"] = task
spec.loader.exec_module(task)

import demo  # noqa: E402
import geometry as g  # noqa: E402
from demokit import Arm, turn  # noqa: E402
from yam_arena import constants as C, ik, station  # noqa: E402

MAX_OFF_NORMAL = math.radians(10)
MIN_CLEAR = 0.004                           # the arm's collision hulls at least 4 mm from the head
REF = np.array([math.sin(math.radians(20)), 0.0, -math.cos(math.radians(20))])   # a comfortable stick axis
SPINS = np.radians(np.arange(0, 360, 15))
STEP = 2                                    # mm between planned points
BASE_R = np.array(station.BASES["right"], float)
SEEDS = [np.array(C.READY, float)] + [np.array(q, float) for q in Arm.SEEDS]
SEEDS += [q + np.array([0, 0, 0, 0, 0, d]) for q in list(SEEDS) for d in (-1.5, 1.5)]


def held():
    a = np.array([0.0, 1.0, 0.0])
    Rp = demo.pen_frame(a)
    p = np.array(task._BASE)
    return demo.Held(p + a * demo.PEN_S, Rp, p, a)


_TREE = {}


def clearance(q) -> float:
    """The right arm's collision hulls' smallest distance to the head's surface (the fingers closed on the tube)."""
    from scipy.spatial import cKDTree
    from yam_arena import analysis as A

    if not _TREE:
        import trimesh
        h = np.load(HERE / "build" / "head.npz", allow_pickle=True)
        pts = trimesh.Trimesh(h["V"], h["F"], process=False).sample(40000) + task.HEAD_POS
        _TREE["t"] = cKDTree(pts)
    qd = dict(zip(C.ARM_JOINTS, q)) | {k: -0.0076 for k in C.FINGER_JOINTS}
    return min(float(_TREE["t"].query(h.vertices)[0].min()) for _, h in A.world_hulls(A.fk_link_poses_world(qd, BASE_R)))


def smooth_normals(P, N, r=0.002):
    out = []
    for p in P:
        m = np.linalg.norm(P - p, axis=1) <= r
        n = N[m].mean(0)
        out.append(n / np.linalg.norm(n))
    return np.array(out)


def axis_for(n):
    """The stick axis: the inward normal, turned toward REF by as much as MAX_OFF_NORMAL allows."""
    a0 = -n
    ang = math.acos(float(np.clip(a0 @ REF, -1, 1)))
    if ang < 1e-6:
        return a0
    t = min(ang, MAX_OFF_NORMAL)
    w = np.cross(a0, REF)
    return turn(w / np.linalg.norm(w), t) @ a0


def frame(a, spin):
    """Stick frame (y = axis) turned ``spin`` about the axis from a fixed reference."""
    hint = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return turn(a, spin) @ demo.stick_frame(a, np.cross(hint, a))


def solve(tcp, R, q0s):
    best = None
    for q0 in q0s:
        q, e, r = ik.solve(tcp, ik.matrix_to_quat_xyzw(R), q_init=q0, base_pos=BASE_R, iters=150)
        if e < 0.0005 and r < math.radians(0.5) and (best is None or e + r < best[1]):
            best = (q, e + r)
    return None if best is None else best[0]


def plan(P, N, H):
    N = smooth_normals(P, N)
    idx = list(range(0, len(P), STEP)) + ([len(P) - 1] if (len(P) - 1) % STEP else [])
    layers = []
    for i in idx:
        a = axis_for(N[i])
        c = P[i] + N[i] * g.BULLET_R                  # the dome's centre touching the lip
        cands = []
        for spin in SPINS:
            S = frame(a, spin)
            tcp, Rt = H.tcp_for(c - a * task.DOME_C, S)
            q = solve(tcp, Rt, SEEDS if not layers else [l[2] for l in layers[-1]][:6] + SEEDS[:3])
            if q is not None:
                cl = clearance(q)
                if cl >= MIN_CLEAR:
                    cands.append((spin, a, q, cl))
        layers.append(cands)
        print(f"point {i:3d}: axis {np.round(a, 2)}  feasible spins {len(cands)}"
              f"{'  clearance up to %.1f mm' % (max(c[3] for c in cands) * 1000) if cands else ''}", flush=True)
    # dynamic programming: the smoothest joint path
    cost = [np.zeros(len(layers[0]))]
    back = []
    for k in range(1, len(layers)):
        prev, cur = layers[k - 1], layers[k]
        if not cur or not prev:
            raise SystemExit(f"no reachable hand turn at planned point {k}")
        D = np.array([[np.abs(c[2] - p[2]).max() for p in prev] for c in cur])
        pen = np.array([0.02 / max(c[3], 1e-3) for c in cur])[:, None]      # prefer more room from the face
        tot = D + 0.05 * pen + cost[-1][None, :]
        back.append(tot.argmin(1))
        cost.append(tot.min(1))
    j = int(np.argmin(cost[-1]))
    path = [j]
    for b in reversed(back):
        j = int(b[j])
        path.append(j)
    path.reverse()
    chosen = [layers[k][j] for k, j in enumerate(path)]
    jump = max(np.abs(chosen[k][2] - chosen[k - 1][2]).max() for k in range(1, len(chosen)))
    print(f"largest joint change between planned points: {math.degrees(jump):.1f} deg", flush=True)
    return (np.array([P[i] for i in idx]), np.array([N[i] for i in idx]), np.array([c[1] for c in chosen]),
            np.array([c[0] for c in chosen]), np.array([c[2] for c in chosen]))


def main():
    H = held()
    lines = demo.lip_lines()
    out = {}
    for name in ("upper", "lower"):
        P, N = lines[name]
        pts, nrm, axes, spins, qs = plan(P, N, H)
        out.update({f"{name}_pts": pts, f"{name}_nrm": nrm, f"{name}_axis": axes, f"{name}_spin": spins,
                    f"{name}_q": qs})
    np.savez(HERE / "assets" / "stroke_plan.npz", **out)
    print("wrote assets/stroke_plan.npz", flush=True)


if __name__ == "__main__":
    main()
