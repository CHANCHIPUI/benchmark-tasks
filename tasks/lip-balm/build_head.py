"""Build the manikin head from the MakeHuman base mesh (hm08, CC0).

    curl -L -o build/dl_makehuman/base.obj \\
        https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/3dobjs/base.obj
    python build_head.py build/dl_makehuman/base.obj

Writes build/head.npz (the head's triangle mesh in the station frame, relative to the head body's origin, which is the
middle of the mouth) and assets/lips.npz (the scoring samples, same frame):
- lip_pts, lip_region: points every 1 mm on both lips' red part; region 0-5 upper lip, 6-11 lower lip, each from the
  resident's right corner to their left; -1 within 3 mm of a corner (lip, not scored);
- skin_pts: points every 1 mm on the skin round the mouth, outside the 2 mm band round the lips' border;
- band_pts: the band itself (neutral).
build_assets.py turns head.npz into assets/head.usda.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

import geometry as g

HERE = Path(__file__).resolve().parent
EYE_GROUPS = ("helper-l-eye", "helper-r-eye")


def read_obj(path):
    V, faces, grp = [], {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "):
                grp = line.split()[1]
            elif line.startswith("f ") and (grp == "body" or grp in EYE_GROUPS):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                faces.setdefault(grp, []).extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return np.array(V) * 0.1, {k: np.array(v) for k, v in faces.items()}   # MakeHuman units are decimetres


def compact(V, F):
    used = np.unique(F)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def cap_boundary(V, F):
    """Close the neck's cut with a fan round its centroid."""
    m = trimesh.Trimesh(V, F, process=False)
    edges = m.edges_sorted[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    # keep only the boundary edges at the cut (lowest), not the mouth slit or eyes
    low = V[edges].mean(1)[:, 1] < g.NECK_CUT_MH + 0.02
    loop = edges[low]
    c = V[np.unique(loop)].mean(0)
    V2 = np.vstack([V, c])
    ci = len(V)
    # orient the fan's faces outward (downward, -y in MH)
    new = []
    for a, b in loop:
        n = np.cross(V[a] - c, V[b] - c)
        new.append([a, b, ci] if n[1] < 0 else [b, a, ci])
    return V2, np.vstack([F, new])


def main(obj_path):
    V, groups = read_obj(obj_path)
    body = groups["body"]
    keep = (V[body][:, :, 1] > g.NECK_CUT_MH).all(1)
    Vh, Fh = compact(V, body[keep])
    Vh, Fh = trimesh.remesh.subdivide_loop(Vh, Fh, iterations=1)
    Vh, Fh = cap_boundary(Vh, Fh)
    eyes = [compact(V, groups[k]) for k in EYE_GROUPS if k in groups]
    head_mh = trimesh.Trimesh(Vh, Fh, process=False)

    # -- scoring samples, by casting rays onto the face from the front (MH frame) -------------------------------------
    step = g.SAMPLE / g.HEAD_SCALE * 1000          # mm in the MH frame per 1 mm on the head
    xs = np.arange(-40, 40 + 1e-9, step)
    ys = np.arange(630, 690 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    origins = np.c_[X / 1000, Y / 1000, np.full(len(X), 0.40)]
    loc, ri, ti = head_mh.ray.intersects_location(origins, np.tile([0, 0, -1.0], (len(X), 1)), multiple_hits=False)
    X, Y = X[ri], Y[ri]
    nrm = head_mh.face_normals[ti] @ g.HEAD_R.T          # outward surface normals, station frame
    inside = np.abs(X) < g.CORNER_X_MH
    up = inside & (Y >= g.border_y(g.LIP_JOIN, X)) & (Y <= g.border_y(g.UPPER_BORDER, X))
    lo = inside & (Y <= g.border_y(g.LIP_JOIN, X)) & (Y >= g.border_y(g.LOWER_BORDER, X))
    lip = up | lo
    P = g.mh_to_world(loc) - g.MOUTH                # station frame, relative to the head's origin
    # regions: equal widths across the mouth, between points 3 mm in from the corners (world mm)
    half = g.CORNER_X_MH * g.HEAD_SCALE / 1000 - g.CORNER_IN
    u = X * g.HEAD_SCALE / 1000                    # across the lip, world metres, toward the resident's left
    k = np.floor((u + half) / (2 * half) * g.N_REGIONS).astype(int)
    k[(u < -half) | (u >= half)] = -1
    region = np.where(up, k, np.where(k >= 0, k + g.N_REGIONS, -1))
    lip_pts, lip_region, lip_nrm = P[lip], region[lip], nrm[lip]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(lip_pts).query(P[~lip])
    band = d <= g.BAND
    skin_pts, band_pts = P[~lip][~band], P[~lip][band]

    Vw = g.mh_to_world(head_mh.vertices) - g.MOUTH
    eyes_w = [(g.mh_to_world(v) - g.MOUTH, f) for v, f in eyes]
    np.savez(HERE / "build" / "head.npz", V=Vw, F=head_mh.faces, eyes=np.array(eyes_w, dtype=object), allow_pickle=True)
    np.savez(HERE / "assets" / "lips.npz", lip_pts=lip_pts.astype(np.float32), lip_region=lip_region.astype(np.int8),
             lip_nrm=lip_nrm.astype(np.float32),
             skin_pts=skin_pts.astype(np.float32), band_pts=band_pts.astype(np.float32))
    counts = np.bincount(lip_region[lip_region >= 0], minlength=12)
    print("head", len(Vw), "verts", len(head_mh.faces), "faces; lips", len(lip_pts), "skin", len(skin_pts),
          "band", len(band_pts))
    print("points per region", counts.tolist())
    for name, sel in (("upper", up), ("lower", lo)):
        h = [np.ptp(Y[sel & (np.abs(X - x0) < 1)]) * g.HEAD_SCALE for x0 in (0, 10, 20)]
        print(name, "height mm at x 0/10/20:", np.round(h, 1))
    print("corner to corner mm:", round(2 * g.CORNER_X_MH * g.HEAD_SCALE, 1))


if __name__ == "__main__":
    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "assets").mkdir(exist_ok=True)
    main(sys.argv[1])
