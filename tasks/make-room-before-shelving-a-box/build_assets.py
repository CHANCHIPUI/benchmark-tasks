"""Writes assets/*.usda and assets/textures/*.png from layout.py. Plain Python with NumPy and Pillow; no simulator.

    python build_assets.py

- ``room.usda``: the deck, its legs and bracing, the guardrail, the shelving unit, the office chair and the seated
  mannequin, as one fixed body with box colliders.
- ``soup_case.usda``, ``new_case.usda``: the cases, each a printed box mesh over one box collider.

The mannequin's mesh is MakeHuman's base mesh (CC0), posed seated here. Fetch its files once into ``mh/`` beside this
file (they are not shipped):

    mkdir -p mh && cd mh
    B=https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data
    curl -L -O $B/3dobjs/base.obj
    for e in african asian caucasian; do curl -L -O $B/targets/macrodetails/$e-male-young.target; done
    curl -L -O $B/targets/macrodetails/universal-male-young-averagemuscle-averageweight.target
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import usdkit as u  # noqa: E402

ASSETS = HERE / "assets"
TEX = ASSETS / "textures"
MH = Path(os.environ.get("MH_DIR", HERE / "mh"))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def srgb(c):
    """sRGB 0..1 to linear, for UsdPreviewSurface colours."""
    c = np.asarray(c, float)
    return tuple(float(x) for x in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


LOOKS = {
    "deck": u.Look(color=srgb((0.50, 0.53, 0.56)), roughness=0.8),          # grey floor paint on plywood
    "timber": u.Look(color=srgb((0.80, 0.66, 0.46)), roughness=0.8),        # untreated pine
    "chrome": u.Look(color=srgb((0.80, 0.81, 0.83)), metallic=0.9, roughness=0.25),
    "foot": u.Look(color=srgb((0.15, 0.15, 0.16)), roughness=0.6),
    "chair": u.Look(color=srgb((0.10, 0.10, 0.11)), roughness=0.7),
    "chair_metal": u.Look(color=srgb((0.35, 0.36, 0.38)), metallic=0.6, roughness=0.4),
    "skin": u.Look(color=srgb((0.90, 0.74, 0.62)), roughness=0.45),         # painted fibreglass
    "shirt": u.Look(color=srgb((0.86, 0.94, 0.12)), roughness=0.8),          # a high-visibility yellow work shirt
    "trousers": u.Look(color=srgb((0.22, 0.22, 0.24)), roughness=0.85),
    "shoes": u.Look(color=srgb((0.06, 0.05, 0.05)), roughness=0.5),
    "hair": u.Look(color=srgb((0.16, 0.11, 0.07)), roughness=0.9),             # a short dark wig
    "soup_case": u.Look(texture="soup_case.png", roughness=0.85),
    "new_case": u.Look(texture="rice_case.png", roughness=0.85),
}


# -- textures -------------------------------------------------------------------------------------------------------------
CELL = 512
# atlas cells (column, row) for each face: +x far end, -x near end, +y left side, -y right side, +z top, -z bottom
CELLS = {"+x": (0, 0), "-x": (1, 0), "+y": (2, 0), "-y": (0, 1), "+z": (1, 1), "-z": (2, 1)}


def _kraft(w, h, seed):
    from PIL import Image

    rng = np.random.default_rng(seed)
    base = np.array([176, 136, 92], float)
    img = base + rng.normal(0, 5, (h, w, 1)) + 6 * np.sin(np.arange(w)[None, :, None] / 3.0)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _arrows(draw, x, y, s, color):
    for dx in (0, int(s * 0.9)):
        draw.polygon([(x + dx, y), (x + dx + s // 2, y - s), (x + dx + s, y)], fill=color)
        draw.rectangle([x + dx + s // 3, y, x + dx + 2 * s // 3, y + s], fill=color)


def textures() -> None:
    from PIL import Image, ImageDraw, ImageFont

    TEX.mkdir(parents=True, exist_ok=True)
    big, mid, small = ImageFont.truetype(FONT, 64), ImageFont.truetype(FONT, 40), ImageFont.truetype(FONT_REG, 30)
    for name, seed in (("soup_case", 1), ("new_case", 2)):
        atlas = Image.new("RGB", (3 * CELL, 2 * CELL))
        for face, (cx, cy) in CELLS.items():
            cell = _kraft(CELL, CELL, seed * 10 + cx + 3 * cy)
            d = ImageDraw.Draw(cell)
            side = face in ("+y", "-y", "-x", "+x")
            if name == "soup_case" and side:
                d.rectangle([20, 150, CELL - 20, 370], fill=(196, 36, 40))
                d.rectangle([20, 150, CELL - 20, 185], fill=(240, 236, 226))
                d.text((CELL // 2, 270), "SOUP", font=big, fill=(250, 246, 238), anchor="mm")
                d.text((CELL // 2, 335), "12 x 10.75 OZ", font=mid, fill=(250, 246, 238), anchor="mm")
                d.text((CELL // 2, 440), "CONDENSED", font=small, fill=(60, 40, 25), anchor="mm")
            elif name == "soup_case" and face == "+z":
                d.line([(0, CELL // 2), (CELL, CELL // 2)], fill=(150, 112, 74), width=10)     # the flaps' seam
                d.text((CELL // 2, 140), "SOUP", font=big, fill=(170, 30, 34), anchor="mm")
            elif name == "new_case" and face in ("+y", "-y"):
                # the hand hole: the cell spans the face's length (x) and height (z)
                u0 = 0.5 - L.HOLE_L / 2 / L.CASE[0]
                u1 = 0.5 + L.HOLE_L / 2 / L.CASE[0]
                v_top = 0.5 + L.HOLE_TOP / L.CASE[2]
                v_bot = 0.5 + L.HOLE_BOT / L.CASE[2]
                box_ = [int(u0 * CELL), int((1 - v_top) * CELL), int(u1 * CELL), int((1 - v_bot) * CELL)]
                d.rounded_rectangle(box_, radius=40, fill=(22, 16, 12))
                d.rectangle([20, 20, 200, 150], fill=(245, 243, 236))
                d.text((110, 60), "RICE", font=mid, fill=(30, 30, 30), anchor="mm")
                d.text((110, 115), "1.5 KG", font=small, fill=(30, 30, 30), anchor="mm")
                _arrows(d, 330, 100, 60, (20, 20, 20))
                d.text((CELL // 2, 440), "THIS SIDE UP", font=mid, fill=(20, 20, 20), anchor="mm")
            elif name == "new_case" and side:
                d.rectangle([30, 60, CELL - 30, 250], fill=(245, 243, 236))
                d.text((CELL // 2, 115), "RICE", font=big, fill=(30, 30, 30), anchor="mm")
                d.text((CELL // 2, 195), "1.5 KG", font=mid, fill=(30, 30, 30), anchor="mm")
                _arrows(d, 150, 380, 90, (20, 20, 20))
                d.text((CELL // 2, 470), "THIS SIDE UP", font=mid, fill=(20, 20, 20), anchor="mm")
            elif name == "new_case" and face == "+z":
                d.line([(0, CELL // 2), (CELL, CELL // 2)], fill=(150, 112, 74), width=10)
                d.text((CELL // 2, 150), "THIS SIDE UP", font=mid, fill=(20, 20, 20), anchor="mm")
                d.text((CELL // 2, 380), "RICE", font=big, fill=(30, 30, 30), anchor="mm")
            atlas.paste(cell, (cx * CELL, cy * CELL))
        atlas.save(TEX / LOOKS[name].texture)


def case_mesh(size) -> u.Mesh:
    """A box with each face mapped to its cell of the atlas, text upright on the sides and reading from outside."""
    hx, hy, hz = (s / 2 for s in size)
    faces = {   # face: four corners counter-clockwise from outside, as (u, v) = (0,0) (1,0) (1,1) (0,1) of the cell
        "+x": [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],
        "-x": [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)],
        "+y": [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)],
        "-y": [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],
        "+z": [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
        "-z": [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)],
    }
    v, f, uv, nrm = [], [], [], []
    for face, corners in faces.items():
        k = len(v)
        v += corners
        cx, cy = CELLS[face]
        cuv = [((cx + a) / 3.0, 1.0 - (cy + 1 - b) / 2.0) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))]
        n = np.cross(np.subtract(corners[1], corners[0]), np.subtract(corners[2], corners[0]))
        n = n / np.linalg.norm(n)
        for tri in ((0, 1, 2), (0, 2, 3)):
            f.append([k + t for t in tri])
            uv += [cuv[t] for t in tri]
            nrm += [n] * 3
    m = u.Mesh(np.array(v, float), np.array(f))
    m.uv = np.array(uv)
    m.normals = np.array(nrm)
    return m


def n_colliders():
    """N's walls as boxes, with the two hand holes cut out of the long side walls, and the rice bags as one block
    filling the box below the holes. Returns (size, centre) pairs."""
    hx, hy, hz = (v / 2 for v in L.CASE)
    t = L.WALL
    out = [((2 * hx, 2 * hy, t), (0, 0, -hz + t / 2)), ((2 * hx, 2 * hy, t), (0, 0, hz - t / 2))]
    for sx in (-1, 1):
        out.append(((t, 2 * hy - 2 * t, 2 * hz - 2 * t), (sx * (hx - t / 2), 0, 0)))
    xl, zb, zt = L.HOLE_L / 2, L.HOLE_BOT, L.HOLE_TOP
    for sy in (-1, 1):
        y = sy * (hy - t / 2)
        ix = hx - t
        out += [((2 * ix, t, zb - (-hz + t)), (0, y, (zb + (-hz + t)) / 2)),          # below the hole
                ((2 * ix, t, (hz - t) - zt), (0, y, (zt + hz - t) / 2)),              # above it
                ((ix - xl, t, zt - zb), (-(ix + xl) / 2, y, (zt + zb) / 2)),          # to its near side
                ((ix - xl, t, zt - zb), ((ix + xl) / 2, y, (zt + zb) / 2))]           # to its far side
    top = zb - 0.005
    out.append(((2 * hx - 2 * t, 2 * hy - 2 * t, top - (-hz + t)), (0, 0, (top + (-hz + t)) / 2)))   # the rice
    return out


def cases() -> None:
    for name in ("soup_case", "new_case"):
        cols = ([u.collider(f"col{k:02d}", size, c, name) for k, (size, c) in enumerate(n_colliders())] if name == "new_case"
                else [u.collider("collider", L.CASE, (0, 0, 0), name)])
        u.write_usda(str(ASSETS / f"{name}.usda"), name, [("box", case_mesh(L.CASE), name)], LOOKS, cols,
                     mass=L.N_KG if name == "new_case" else L.SOUP_KG, friction=(L.CASE_MU, L.CASE_MU), solver_iters=16)


# -- the mannequin ----------------------------------------------------------------------------------------------------
def read_obj(path):
    V, groups, grp = [], {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "):
                grp = line.split()[1]
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                groups.setdefault(grp, []).extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return np.array(V), {k: np.array(v) for k, v in groups.items()}


def apply_target(V, path, w):
    off = np.zeros_like(V)
    for line in open(path):
        if line[:1].isdigit():
            t = line.split()
            off[int(t[0])] += np.array([float(x) for x in t[1:4]])
    return V + w * off


def _rotate(V, centre, axis, ang):
    """Rotate points V about the line through centre along axis, by per-point angles ang (rad)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    P = V - centre
    a = np.asarray(ang, float)[:, None]
    return centre + P * np.cos(a) + (P @ K.T) * np.sin(a) + np.outer(P @ axis, axis) * (1 - np.cos(a))


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mannequin():
    """A seated adult male mannequin from MakeHuman's base mesh, its back to the deck (facing +x), the top of its head at
    L.HEAD_TOP. Returns {look: Mesh} in the world frame, the seat height and the colliders' boxes (lo, hi)."""
    V, groups = read_obj(MH / "base.obj")
    for e in ("african", "asian", "caucasian"):
        V = apply_target(V, MH / f"{e}-male-young.target", 1.0 / 3.0)
    V = apply_target(V, MH / "universal-male-young-averagemuscle-averageweight.target", 1.0)
    F = groups["body"]
    idx = np.unique(F)
    lo, hi = V[idx, 1].min(), V[idx, 1].max()
    V = (V - [0.0, lo, 0.0]) * (L.STATURE / (hi - lo))       # metres, feet at y 0; x her left, y up, z forward

    def j(name):
        return V[np.unique(groups[f"joint-{name}"])].mean(0)

    J0 = {n: j(n) for n in ("neck", "head", "l-shoulder", "r-shoulder", "l-hand", "r-hand", "l-elbow", "r-elbow",
                            "l-upper-leg", "r-upper-leg", "l-knee", "r-knee", "l-ankle", "r-ankle", "pelvis")}
    V0 = V.copy()
    # regions, from the standing A pose
    cent = V0[F].mean(1)
    hand_x = abs(J0["l-hand"][0]) - 0.03
    region = np.full(len(F), "shirt", dtype=object)
    region[cent[:, 1] < J0["pelvis"][1] + 0.06] = "trousers"
    region[cent[:, 1] < J0["l-ankle"][1] + 0.04] = "shoes"
    region[cent[:, 1] > J0["neck"][1] + 0.02] = "skin"
    region[(np.abs(cent[:, 0]) > hand_x) & (cent[:, 1] > J0["l-knee"][1])] = "skin"
    # short hair: the skin of the crown, the back and the sides of the head above the ears
    head_v = V0[np.unique(F[region == "skin"])]
    head_v = head_v[head_v[:, 1] > J0["neck"][1] + 0.05]
    y_top, z_mid = head_v[:, 1].max(), (head_v[:, 2].min() + head_v[:, 2].max()) / 2
    hair = (region == "skin") & (cent[:, 1] > y_top - 0.115) & ((cent[:, 2] < z_mid + 0.01) | (cent[:, 1] > y_top - 0.045))
    hair &= ~((cent[:, 2] > z_mid - 0.03) & (cent[:, 1] < y_top - 0.085))          # leave the ears and temples bare
    region[hair] = "hair"
    # arms down to the sides, as the A pose's arms turned about each shoulder
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        sh, hand, el = J0[f"{side}-shoulder"], J0[f"{side}-hand"], J0[f"{side}-elbow"]
        dvec = hand - sh
        n = np.linalg.norm(dvec)
        want = np.array([sgn * math.sin(math.radians(8.0)), -math.cos(math.radians(8.0)), 0.03])
        u0, u1 = dvec / n, want / np.linalg.norm(want)
        axis = np.cross(u0, u1)
        axis /= np.linalg.norm(axis)
        ang = math.acos(float(np.clip(u0 @ u1, -1, 1)))
        t = (V - sh) @ u0
        w = np.clip((t + 0.02) / 0.06, 0, 1)
        bones = [(sh, el), (el, hand), (hand, hand + 0.15 * (hand - el) / np.linalg.norm(hand - el))]
        dist = np.min([np.linalg.norm(V - (a + np.clip(((V - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)[:, None] * (b - a)),
                                      axis=1) for a, b in bones], axis=0)
        f_ = np.clip((0.14 - dist) / 0.03, 0, 1)
        w = _smooth(w) * _smooth(f_)
        V = _rotate(V, sh, axis, ang * w)
    # seated: knees bent 90 degrees, then hips bent 90 degrees (legs forward, shins down)
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        leg = (V0[:, 0] * sgn > 0.0) & (V0[:, 1] < J0["pelvis"][1] + 0.05) & (np.abs(V0[:, 0]) < hand_x)
        knee = J0[f"{side}-knee"]
        wk = _smooth((knee[1] + 0.04 - V0[:, 1]) / 0.08) * leg
        V = _rotate(V, knee, (1.0, 0.0, 0.0), math.radians(90.0) * wk)
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        leg = (V0[:, 0] * sgn > -0.01) & (np.abs(V0[:, 0]) < hand_x)
        hip = J0[f"{side}-upper-leg"]
        wh = _smooth((hip[1] + 0.02 - V0[:, 1]) / 0.12) * leg
        V = _rotate(V, hip + np.array([0.0, 0.0, -0.02]), (1.0, 0.0, 0.0), math.radians(-90.0) * wh)
    # to the world: forward -> +x, her left -> +y, up -> +z
    W = np.stack([V[:, 2], V[:, 0], V[:, 1]], axis=1)
    head = np.unique(F[(region == "skin") | (region == "hair")].reshape(-1))
    head = head[W[head, 2] > np.percentile(W[idx, 2], 80)]
    top = head[np.argmax(W[head, 2])]
    W = W + (np.asarray(L.HEAD_TOP) - W[top])
    seat_z = float(np.percentile(W[idx, 2][np.abs(W[idx, 1] - L.HEAD_TOP[1]) < 0.12], 0.5))
    meshes = {}
    fn = np.cross(W[F[:, 1]] - W[F[:, 0]], W[F[:, 2]] - W[F[:, 0]])
    vn = np.zeros_like(W)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    for look in ("skin", "shirt", "trousers", "shoes", "hair"):
        Fl = F[region == look]
        keep = np.unique(Fl)
        remap = -np.ones(len(W), int)
        remap[keep] = np.arange(len(keep))
        meshes[look] = u.Mesh(W[keep] + (0.004 * vn[keep] if look == "hair" else 0.0), remap[Fl])
    hip_w = (W[np.unique(groups["joint-l-upper-leg"])].mean(0) + W[np.unique(groups["joint-r-upper-leg"])].mean(0)) / 2
    hx, hy, hz = L.HEAD_TOP
    boxes = [((hx - 0.10, hy - 0.08, hz - 0.24), (hx + 0.08, hy + 0.08, hz)),               # head
             ((hip_w[0] - 0.12, hy - 0.20, seat_z + 0.05), (hip_w[0] + 0.12, hy + 0.20, hz - 0.30)),   # torso
             ((hip_w[0] - 0.05, hy - 0.20, seat_z), (hip_w[0] + 0.45, hy + 0.20, seat_z + 0.17))]     # thighs
    shirt = man_back = W[np.unique(F[region == "shirt"])]
    band = shirt[(shirt[:, 2] > seat_z + 0.10) & (shirt[:, 2] < seat_z + 0.50)]
    back_x = float(band[:, 0].min()) - 0.005
    return meshes, seat_z, float(hip_w[0]), boxes, back_x


def smooth_normals(m: u.Mesh) -> u.Mesh:
    v, f = m.v, m.f
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    m.normals = vn[f].reshape(-1, 3)
    m.uv = np.zeros((len(f) * 3, 2))
    return m


# -- the room -----------------------------------------------------------------------------------------------------------
def bx(lo, hi) -> u.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return u.box(hi - lo, (lo + hi) / 2)


def office_chair(cx, cy, seat_z, back_x):
    """A gas-lift office chair facing +x, its seat cushion's top at seat_z and its backrest's front at back_x; returns
    {look: [Mesh]} and collider boxes."""
    out = {"chair": [], "chair_metal": []}
    out["chair"].append(u.box((0.46, 0.48, 0.07), (back_x + 0.03 + 0.23, cy, seat_z - 0.035), bevel=0.015, round_xy=0.04))
    out["chair"].append(u.box((0.06, 0.44, 0.44), (back_x - 0.03, cy, seat_z + 0.33), bevel=0.012, round_xy=0.03))
    out["chair_metal"].append(bx((back_x - 0.05, cy - 0.025, seat_z - 0.06), (back_x - 0.03, cy + 0.025, seat_z + 0.15)))
    out["chair_metal"].append(bx((back_x - 0.05, cy - 0.025, seat_z - 0.08), (cx, cy + 0.025, seat_z - 0.06)))
    floor = L.LOWER_Z
    out["chair_metal"].append(u.cylinder(0.025, seat_z - 0.07 - (floor + 0.12), (cx, cy, (seat_z - 0.07 + floor + 0.12) / 2)))
    for k in range(5):
        a = math.radians(90 + 72 * k)
        tip = np.array([cx + 0.29 * math.cos(a), cy + 0.29 * math.sin(a), floor + 0.08])
        out["chair"].append(u.tube([(cx, cy, floor + 0.11), tuple(tip)], 0.018, 8))
        out["chair"].append(u.cylinder(0.025, 0.05, (tip[0], tip[1], floor + 0.03), axis="y"))
    cols = [((back_x + 0.03, cy - 0.24, seat_z - 0.07), (back_x + 0.49, cy + 0.24, seat_z)),
            ((back_x - 0.06, cy - 0.22, seat_z + 0.11), (back_x, cy + 0.22, seat_z + 0.55))]
    return out, cols


def shelving():
    """The chrome wire unit: four posts with levelling feet, and the two fitted shelves, each a wire deck on a front
    and back truss. Returns {look: [Mesh]}."""
    out = {"chrome": [], "foot": []}
    x0, x1 = L.SHELF_X
    y0, y1 = L.SHELF_Y
    for px, py in L.UNIT_POSTS:
        out["chrome"].append(u.cylinder(L.POST_R, L.POST_H - 0.03, (px, py, L.H(0.03 + (L.POST_H - 0.03) / 2)), segments=20))
        out["foot"].append(u.cylinder(0.016, 0.03, (px, py, L.H(0.015)), segments=16))
    for top in (L.WORK_TOP, L.TOP_SHELF):
        zt = L.H(top)
        for y in np.arange(y0 + 0.02, y1 - 0.01, 0.0254):            # deck wires, front to back
            out["chrome"].append(bx((x0 + 0.004, y - 0.002, zt - 0.004), (x1 - 0.004, y + 0.002, zt)))
        for x in (x0 + 0.004, x1 - 0.004):                             # truss: top and bottom bars, zigzag
            out["chrome"].append(bx((x - 0.004, y0, zt - 0.008), (x + 0.004, y1, zt)))
            out["chrome"].append(bx((x - 0.003, y0, zt - L.SHELF_T), (x + 0.003, y1, zt - L.SHELF_T + 0.006)))
            ys = np.linspace(y0 + 0.02, y1 - 0.02, 25)
            for a, b in zip(ys[:-1], ys[1:]):
                za, zb = (zt - 0.006, zt - L.SHELF_T + 0.003) if len(out["chrome"]) % 2 else (zt - L.SHELF_T + 0.003, zt - 0.006)
                out["chrome"].append(u.tube([(x, a, za), (x, b, zb)], 0.002, 6))
        for y in np.linspace(y0 + 0.30, y1 - 0.30, 3):                  # under-supports, front to back
            out["chrome"].append(bx((x0, y - 0.003, zt - 0.012), (x1, y + 0.003, zt - 0.004)))
        for px, py in L.UNIT_POSTS:                                     # corner collars
            out["chrome"].append(u.cylinder(L.POST_R + 0.006, 0.05, (px, py, zt - 0.02), segments=20))
    return out


def room() -> dict:
    parts: dict[str, list] = {k: [] for k in LOOKS}
    cols = []
    dz = L.DECK_Z
    # the deck: plywood top on a 2x6 frame, six 4x4 legs, diagonals and a stretcher
    deck_lo, deck_hi = (L.DECK_X0, -L.DECK_HY, dz - L.PLY), (L.EDGE_X, L.DECK_HY, dz)
    parts["deck"].append(bx(deck_lo, deck_hi))
    cols.append(((L.DECK_X0, -L.DECK_HY, dz - L.PLY - L.FRAME_H), deck_hi))
    fz0, fz1 = dz - L.PLY - L.FRAME_H, dz - L.PLY
    for x in (L.DECK_X0, L.EDGE_X - 0.038):
        parts["timber"].append(bx((x, -L.DECK_HY, fz0), (x + 0.038, L.DECK_HY, fz1)))
    for y in (-L.DECK_HY, L.DECK_HY - 0.038):
        parts["timber"].append(bx((L.DECK_X0, y, fz0), (L.EDGE_X, y + 0.038, fz1)))
    legs = [(x, y) for x in (L.DECK_X0 + 0.038, L.EDGE_X - 0.038 - L.LEG) for y in (-L.DECK_HY + 0.038, -L.LEG / 2, L.DECK_HY - 0.038 - L.LEG)]
    for x, y in legs:
        parts["timber"].append(bx((x, y, L.LOWER_Z), (x + L.LEG, y + L.LEG, fz1)))
    for x in (L.DECK_X0 + 0.038 + L.LEG, L.EDGE_X - 0.038 - L.LEG - 0.038):
        parts["timber"].append(bx((x, -L.DECK_HY + 0.04, L.LOWER_Z + 0.10), (x + 0.038, L.DECK_HY - 0.04, L.LOWER_Z + 0.189)))
    for xa in (L.EDGE_X - 0.038 - L.LEG - 0.02,):
        for ya, yb in ((-L.DECK_HY + 0.10, -0.06), (0.06, L.DECK_HY - 0.10)):
            parts["timber"].append(u.tube([(xa, ya, L.LOWER_Z + 0.20), (xa, yb, fz0 - 0.02)], 0.03, 4))
    # the guardrail: posts outside the rims, rails on their inner faces, a toeboard on the drop side
    for y in L.DROP_POST_Y:
        lo, hi = (L.POST_X[0], y - L.POST_W / 2, fz0), (L.POST_X[1], y + L.POST_W / 2, L.H(L.TOP_RAIL_Z[1]))
        parts["timber"].append(bx(lo, hi))
        cols.append((lo, hi))
    for z0, z1 in (L.TOP_RAIL_Z, L.MID_RAIL_Z):
        lo, hi = (L.RAIL_X[0], -L.DECK_HY, L.H(z0)), (L.RAIL_X[1], L.DECK_HY, L.H(z1))
        parts["timber"].append(bx(lo, hi))
        cols.append((lo, hi))
    lo, hi = (L.TOE_X[0], -L.DECK_HY, L.H(L.TOE_Z[0])), (L.TOE_X[1], L.DECK_HY, L.H(L.TOE_Z[1]))
    parts["timber"].append(bx(lo, hi))
    cols.append((lo, hi))
    for sy in (-1.0, 1.0):                                               # the two ends
        y = sy * (L.DECK_HY + 0.019)
        for x in (L.DECK_X0 + 0.02, (L.DECK_X0 + L.EDGE_X) / 2, L.EDGE_X - 0.02):
            parts["timber"].append(bx((x - 0.0445, y - 0.019, fz0), (x + 0.0445, y + 0.019, L.H(L.TOP_RAIL_Z[1]))))
        for z0, z1 in (L.TOP_RAIL_Z, L.MID_RAIL_Z):
            parts["timber"].append(bx((L.DECK_X0, y - 0.057 * sy - 0.019, L.H(z0)), (L.EDGE_X, y - 0.057 * sy + 0.019, L.H(z1))))
    xb = L.DECK_X0 - 0.019                                               # the back, with its lift-out section
    for y in (-L.DECK_HY + 0.05, -0.6, 0.6, L.DECK_HY - 0.05):
        parts["timber"].append(bx((xb - 0.019, y - 0.0445, fz0), (xb + 0.019, y + 0.0445, L.H(L.TOP_RAIL_Z[1]))))
    for z0, z1 in (L.TOP_RAIL_Z, L.MID_RAIL_Z):
        parts["timber"].append(bx((xb + 0.019, -L.DECK_HY, L.H(z0)), (xb + 0.057, L.DECK_HY, L.H(z1))))
    # the shelving unit
    for look, ms in shelving().items():
        parts[look] += ms
    x0, x1 = L.SHELF_X
    y0, y1 = L.SHELF_Y
    for top in (L.WORK_TOP, L.TOP_SHELF):
        cols.append(((x0, y0, L.H(top - L.SHELF_T)), (x1, y1, L.H(top))))
    for px, py in L.UNIT_POSTS:
        cols.append(((px - L.POST_R, py - L.POST_R, L.H(0.0)), (px + L.POST_R, py + L.POST_R, L.H(L.POST_H))))
    # the coworker: the seated mannequin on its chair, built facing +x, then turned about the head's top
    man, seat_z, hip_x, man_cols, back_x = mannequin()
    chair, chair_cols = office_chair(hip_x + 0.10, L.HEAD_TOP[1], seat_z, back_x)
    c, sn = math.cos(math.radians(L.COWORKER_YAW)), math.sin(math.radians(L.COWORKER_YAW))
    Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    pivot = np.array([L.HEAD_TOP[0], L.HEAD_TOP[1], 0.0])

    def turn(m):
        m.v = (m.v - pivot) @ Rz.T + pivot
        return m

    for look, m in man.items():
        parts[look].append(smooth_normals(turn(m)))
    for look, ms in chair.items():
        parts[look] += [turn(m) for m in ms]
    for lo, hi in man_cols + chair_cols:
        P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        P = (P - pivot) @ Rz.T + pivot
        cols.append((P.min(0), P.max(0)))
    return {"parts": parts, "cols": cols, "seat_z": seat_z}


def write_room() -> None:
    r = room()
    meshes = []
    for look, ms in r["parts"].items():
        if not ms:
            continue
        smooth = [m for m in ms if getattr(m, "normals", None) is not None]
        flat = [m for m in ms if getattr(m, "normals", None) is None]
        if flat:
            meshes.append((look, u.join(*flat), look))
        for k, m in enumerate(smooth):
            meshes.append((f"{look}_body{k}", m, look))
    colliders = []
    for k, (lo, hi) in enumerate(r["cols"]):
        lo, hi = np.asarray(lo), np.asarray(hi)
        colliders.append(u.collider(f"col{k:02d}", hi - lo, (lo + hi) / 2, "room"))
    u.write_usda(str(ASSETS / "room.usda"), "room", meshes, LOOKS, colliders, mass=None, friction=(L.ROOM_MU, L.ROOM_MU),
                 kinematic=True)
    print(f"room: {len(meshes)} meshes, {len(colliders)} colliders, seat {r['seat_z'] - L.LOWER_Z:.3f} above the floor below")


def hand_hole_template() -> None:
    """A 1:1 cutting template for N's hand holes (print at 100 %): the long side panel's outline, 11 5/8 x 5 5/8 in, and
    the 4 x 1-1/4 in hole with round ends, centred along it, its top edge 2 in below the panel's top edge."""
    W, Hh = 11.625, 5.625
    hl, hh, top = 4.0, 1.25, 2.0
    x0, y0 = (W - hl) / 2, top
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W + 1}in" height="{Hh + 1.6}in" viewBox="-0.5 -0.5 {W + 1} {Hh + 1.6}">
  <rect x="0" y="0" width="{W}" height="{Hh}" fill="none" stroke="black" stroke-width="0.02"/>
  <rect x="{x0}" y="{y0}" width="{hl}" height="{hh}" rx="{hh / 2}" ry="{hh / 2}" fill="none" stroke="black" stroke-width="0.03"/>
  <line x1="{W / 2}" y1="-0.3" x2="{W / 2}" y2="{Hh + 0.3}" stroke="black" stroke-width="0.01" stroke-dasharray="0.1,0.1"/>
  <text x="0" y="{Hh + 0.45}" font-size="0.22" font-family="sans-serif">Hand-hole template, print at 100 %: long side panel 11 5/8 x 5 5/8 in (top edge up). Cut the slot, 4 x 1 1/4 in, top 2 in below the top edge.</text>
  <text x="0" y="{Hh + 0.8}" font-size="0.22" font-family="sans-serif">Check: this bar is 4 in long.</text>
  <line x1="4" y1="{Hh + 0.75}" x2="8" y2="{Hh + 0.75}" stroke="black" stroke-width="0.04"/>
</svg>
"""
    (HERE / "hand_hole_template.svg").write_text(svg)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    textures()
    cases()
    write_room()
    hand_hole_template()
