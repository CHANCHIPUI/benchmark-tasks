"""Geometry of the guitar and the capo, and the USD files built from it (into assets/, at load time). Pure NumPy. Sizes
in metres, in the station frame (x forward from the arm bases, y to the robot's left), heights above the table top.

Guitar: a full-size right-handed dreadnought steel-string acoustic (650 mm scale, 104 cm long) lying on its back on a
foam work mat, strings up, its neck along y in a padded neck rest, headstock toward the robot's right (-y). Frets are
placed by the equal-temperament rule, 650 x (1 - 2^(-n/12)) mm from the nut. The low E string is farthest from the arms.

Capo: a rigid red anodised aluminium yoke with a black rubber pad, sized from common 6-string capos: a 70 mm bar over
the strings and two side arms that straddle the neck. A real capo also closes under the neck with a spring, cam or
screw; this one rests on the strings by its weight.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as uk

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

# -- guitar ---------------------------------------------------------------------------------------------------------
X = 0.30  # the neck's centre line
NUT_Y = -0.27  # the nut's body-side face: distances along the neck are measured from here, toward +y
SCALE = 0.650
MAT_T = 0.010  # foam work mat under the body
BODY_TOP = 0.110  # the soundboard
FB_T = 0.006
FB_TOP = BODY_TOP + FB_T  # the fretboard's top, level along the whole neck
FRET_H, FRET_W = 0.0012, 0.0027  # fret wire crown: height above the fretboard, width along the neck
STRING_TOP = 0.119  # the strings' tops over the neck (3 mm above the fretboard)
DOT_D = 0.008


def fret_d(n: int) -> float:
    """Fret n's centre line, metres from the nut."""
    return SCALE * (1.0 - 2.0 ** (-n / 12.0))


def neck_w(d: float) -> float:
    """Fretboard width d metres from the nut: 43 mm at the nut, 54 mm at the 12th fret."""
    return 0.043 + 0.011 * d / fret_d(12)


def string_x(k: int, d: float) -> float:
    """String k's (1 = high E, nearest the arms; 6 = low E) centre line at d from the nut: 35 mm apart (1st to 6th,
    centre to centre) at the nut, 54 mm at the saddle."""
    spread = 0.035 + (0.054 - 0.035) * d / SCALE
    return X + (k - 3.5) / 5.0 * spread


FRETS = [fret_d(n) for n in range(1, 21)]
BODY_Y0 = NUT_Y + fret_d(14)  # where the neck meets the body (14 frets clear)
BODY_L = 0.508
FB_END = NUT_Y + fret_d(20) + 0.007
SADDLE_Y = NUT_Y + SCALE
SOUNDHOLE = (X, BODY_Y0 + 0.143, 0.050)  # centre x, y and radius
DOT_SPACES = [3, 5, 7, 9, 15, 17]  # single dots in these fret spaces; two at the 12th

# Dreadnought outline: half width at s metres from the body's neck end (upper bout 29.2 cm, waist 27 cm, lower 39.7 cm).
_HALF = [(0.0, 0.0), (0.002, 0.045), (0.008, 0.085), (0.02, 0.112), (0.04, 0.131), (0.07, 0.142), (0.10, 0.146),
         (0.13, 0.144), (0.16, 0.139), (0.19, 0.1355), (0.215, 0.135), (0.24, 0.138), (0.27, 0.150), (0.30, 0.166),
         (0.33, 0.180), (0.36, 0.191), (0.39, 0.1975), (0.41, 0.1985), (0.43, 0.197), (0.45, 0.190), (0.47, 0.176),
         (0.485, 0.158), (0.496, 0.133), (0.503, 0.100), (0.5065, 0.060), (0.508, 0.0)]


def body_outline(inset: float = 0.0) -> np.ndarray:
    """The body's outline in xy, counter-clockwise seen from above."""
    s = np.array([p[0] for p in _HALF])
    hw = np.array([p[1] for p in _HALF])
    fine = np.linspace(0.0, BODY_L, 90)
    h = np.interp(fine, s, hw)
    h = np.convolve(np.pad(h, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # smooth the table's corners
    h[0] = h[-1] = 0.0
    y = BODY_Y0 + fine
    near = np.c_[X - np.maximum(h - inset, 0.0), y]  # x < X side, from the neck end to the tail
    far = np.c_[X + np.maximum(h - inset, 0.0), y][::-1]
    pts = np.vstack([near[1:-1], [[X, BODY_Y0 + BODY_L - inset]], far[1:-1], [[X, BODY_Y0 + inset]]])
    # counter-clockwise seen from above (x right, y up): the near side runs with +y
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    return pts if area > 0 else pts[::-1]


def _loft(sections) -> uk.Mesh:
    """A closed solid through cross-sections [(y, [(x, z), ...]), ...], each ring with the same number of points,
    counter-clockwise seen from +y."""
    rings = [np.c_[np.asarray(p)[:, 0], np.full(len(p), y), np.asarray(p)[:, 1]] for y, p in sections]
    n = len(rings[0])
    v = np.vstack(rings + [rings[0].mean(axis=0)[None], rings[-1].mean(axis=0)[None]])
    f = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, e = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            f += [(a, e, c), (a, c, b)]
    c0, c1 = len(rings) * n, len(rings) * n + 1
    last = (len(rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        f += [(c0, i, j), (c1, last + j, last + i)]
    return uk.Mesh(v, np.array(f))


def _neck_profile(w: float, depth: float, top: float, n: int = 14) -> list:
    """The neck's cross-section under the fretboard: a flat top `w` wide and a rounded C below, `depth` deep."""
    pts = [(X + w / 2, top), (X - w / 2, top)]
    for k in range(1, n):
        a = math.pi * k / n
        pts.append((X - w / 2 * math.cos(a), top - depth * math.sin(a) ** 0.8))
    return pts


NECK_BACK = {"nut": 0.016, "body": 0.019}  # neck depth below the fretboard


def neck_depth(d: float) -> float:
    """The neck's depth below the fretboard (to the lowest point of its back), d metres from the nut, along the neck."""
    return NECK_BACK["nut"] + (NECK_BACK["body"] - NECK_BACK["nut"]) * max(d, 0.0) / fret_d(14)


def neck_back_z(d: float) -> float:
    return BODY_TOP - neck_depth(d)


def _neck() -> uk.Mesh:
    ys = list(np.linspace(NUT_Y - 0.006, BODY_Y0 - 0.04, 8)) + [BODY_Y0 - 0.03, BODY_Y0 - 0.02, BODY_Y0 - 0.01, BODY_Y0 - 0.002]
    secs = []
    for y in ys:
        d = y - NUT_Y
        depth = neck_depth(d)
        if y > BODY_Y0 - 0.045:  # the heel deepens toward the body's back
            t = (y - (BODY_Y0 - 0.045)) / 0.043
            depth += (BODY_TOP - MAT_T - 0.012 - depth) * t ** 1.6
        secs.append((y, _neck_profile(neck_w(max(d, 0.0)), depth, BODY_TOP)))
    return _loft(secs)


def _headstock(z_off: float, thick: float) -> tuple[uk.Mesh, np.ndarray]:
    """The headstock, tilted 14 degrees down from the nut toward -y. Returns the mesh and its 8 corner points."""
    u = np.array([0.0, 0.012, 0.03, 0.06, 0.12, 0.165, 0.180, 0.185])
    hw = np.array([0.0225, 0.026, 0.033, 0.037, 0.040, 0.041, 0.036, 0.020])
    near = np.c_[X - hw, -u]
    far = np.c_[X + hw, -u][::-1]
    outline = np.vstack([far, near])  # in (x, -u): counter-clockwise seen from above
    area = 0.5 * np.sum(outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1])
    if area < 0:
        outline = outline[::-1]
    m = uk.prism(outline, -thick, 0.0, bevel=0.0008)
    m = m.turned(14.0, "x")  # tip the far end (-y) down
    shift = np.array([0.0, NUT_Y - 0.006, FB_TOP - 0.004 + z_off])
    corners = np.array([[X + sx * 0.041, -uu, zz] for sx in (-1, 1) for uu in (0.0, 0.185) for zz in (-thick, 0.0)])
    c = math.cos(math.radians(14.0)), math.sin(math.radians(14.0))
    rot = np.array([[1, 0, 0], [0, c[0], -c[1]], [0, c[1], c[0]]])
    return m.moved(shift), corners @ rot.T + shift


HEAD_THICK = 0.014
SIDE = {-1: "near", 1: "far"}  # prim-name suffixes for the two sides of the neck


def head_point(u: float, across: float, up: float = 0.0) -> np.ndarray:
    """A point on the headstock's face, u metres along it from the nut, `across` from its centre line."""
    t = math.radians(14.0)
    return np.array([X + across, NUT_Y - 0.006 - u * math.cos(t), FB_TOP - 0.004 - u * math.sin(t) + up])


TUNER_U = (0.055, 0.095, 0.135)  # posts along the headstock, three a side


def _guitar() -> str:
    looks = {
        "mat": uk.Look(color=(0.055, 0.058, 0.062), roughness=0.95),
        "mahogany": uk.Look(texture="mahogany.png", color=(1, 1, 1), roughness=0.4, tile=0.12),
        "top": uk.Look(texture="top.png", color=(1, 1, 1), roughness=0.35,
                       origin=(X - 0.20, BODY_Y0, 0.0), tile_uv=(0.40, BODY_L)),
        "fretboard": uk.Look(texture="fretboard.png", color=(1, 1, 1), roughness=0.55,
                             origin=(X - 0.032, NUT_Y - 0.001, 0.0), tile_uv=(0.064, FB_END - NUT_Y + 0.002)),
        "rosewood": uk.Look(texture="rosewood.png", color=(1, 1, 1), roughness=0.5, tile=0.08),
        "bone": uk.Look(color=(0.86, 0.83, 0.74), roughness=0.45),
        "fret": uk.Look(color=(0.90, 0.90, 0.87), metallic=0.5, roughness=0.3),
        "steel": uk.Look(color=(0.80, 0.80, 0.80), metallic=0.9, roughness=0.25),
        "bronze": uk.Look(color=(0.78, 0.55, 0.30), metallic=0.85, roughness=0.3),
        "chrome": uk.Look(color=(0.85, 0.85, 0.86), metallic=0.95, roughness=0.15),
        "head_face": uk.Look(color=(0.012, 0.012, 0.014), roughness=0.18),
        "pin": uk.Look(color=(0.90, 0.88, 0.82), roughness=0.4),
        "beech": uk.Look(texture="beech.png", color=(1, 1, 1), roughness=0.6, tile=0.06),
        "felt": uk.Look(color=(0.035, 0.035, 0.04), roughness=1.0),
    }
    parts = []
    cols = []
    R = "Guitar"
    # mat and body
    mat_y0, mat_y1 = BODY_Y0 - 0.02, min(BODY_Y0 + BODY_L + 0.03, 0.64)
    parts.append(("mat", uk.box((0.46, mat_y1 - mat_y0, MAT_T), (X, (mat_y0 + mat_y1) / 2, MAT_T / 2), bevel=0.002), "mat"))
    cols.append(uk.collider("col_mat", (0.46, mat_y1 - mat_y0, MAT_T), (X, (mat_y0 + mat_y1) / 2, MAT_T / 2), R))
    outline = body_outline()
    parts.append(("body", uk.prism(outline, MAT_T, BODY_TOP - 0.0012, bevel=0.003), "mahogany"))
    parts.append(("top", uk.prism(body_outline(0.0006), BODY_TOP - 0.0012, BODY_TOP), "top"))
    # body colliders: convex hulls of its upper and lower parts, split at the waist
    s_cut = 0.215
    for name, lo, hi in (("upper", 0.0, s_cut), ("lower", s_cut, BODY_L)):
        sel = outline[(outline[:, 1] >= BODY_Y0 + lo - 1e-9) & (outline[:, 1] <= BODY_Y0 + hi + 1e-9)]
        sel = sel[:: max(1, len(sel) // 24)]
        pts = np.vstack([np.c_[sel, np.full(len(sel), MAT_T)], np.c_[sel, np.full(len(sel), BODY_TOP)]])
        cols.append(uk.hull_collider(f"col_body_{name}", pts, R))
    # neck, heel and fretboard
    parts.append(("neck", _neck(), "mahogany"))
    fb = np.array([[X - neck_w(0) / 2, NUT_Y], [X + neck_w(0) / 2, NUT_Y],
                   [X + neck_w(FB_END - NUT_Y) / 2, FB_END], [X - neck_w(FB_END - NUT_Y) / 2, FB_END]])
    fb_out = np.array([fb[0], fb[1], fb[2], fb[3]])
    parts.append(("fretboard", uk.prism(fb_out, BODY_TOP, FB_TOP, bevel=0.0006), "fretboard"))
    # the neck: convex hulls of its rounded back and the fretboard over it, in 3 cm lengths, so a capo's jaw meets the
    # back where it is drawn
    ys = list(np.arange(NUT_Y - 0.006, BODY_Y0 - 0.045, 0.03)) + [BODY_Y0 - 0.045]
    for k, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        pts = []
        for y in (y0, y1):
            d = max(y - NUT_Y, 0.0)
            w = neck_w(d)
            prof = [(X + w / 2, FB_TOP), (X - w / 2, FB_TOP)] + _neck_profile(w, neck_depth(d), BODY_TOP, n=10)
            pts += [(px, y, pz) for px, pz in prof]
        cols.append(uk.hull_collider(f"col_neck_{k:02d}", pts, R))
    cols.append(uk.collider("col_fretboard_ext", (neck_w(FB_END - NUT_Y), FB_END - BODY_Y0, FB_T),
                            (X, (BODY_Y0 + FB_END) / 2, BODY_TOP + FB_T / 2), R))
    cols.append(uk.collider("col_heel", (neck_w(fret_d(14)), 0.04, BODY_TOP - MAT_T - 0.01),
                            (X, BODY_Y0 - 0.02, (BODY_TOP + MAT_T + 0.01) / 2), R))
    # the strings as one flat surface at their tops, over the neck: nut to the 14th fret, in three widths
    for k, (y0, y1) in enumerate(((NUT_Y, NUT_Y + 0.12), (NUT_Y + 0.12, NUT_Y + 0.24), (NUT_Y + 0.24, FB_END))):
        w = neck_w(min(y1, FB_END) - NUT_Y) - 0.001
        cols.append(uk.collider(f"col_strings_{k}", (w, y1 - y0, STRING_TOP - FB_TOP),
                                (X, (y0 + y1) / 2, (STRING_TOP + FB_TOP) / 2), R))
    # frets and nut
    for n, d in enumerate(FRETS, start=1):
        w = neck_w(d) - 0.0008
        parts.append((f"fret_{n:02d}", uk.box((w, FRET_W, FRET_H), (X, NUT_Y + d, FB_TOP + FRET_H / 2), bevel=0.0004), "fret"))
    nut_h = STRING_TOP - 0.0005 - (BODY_TOP - 0.004)
    parts.append(("nut", uk.box((0.043, 0.006, nut_h), (X, NUT_Y - 0.003, BODY_TOP - 0.004 + nut_h / 2), bevel=0.0008), "bone"))
    cols.append(uk.collider("col_nut", (0.043, 0.006, nut_h), (X, NUT_Y - 0.003, BODY_TOP - 0.004 + nut_h / 2), R))
    # headstock: a black face over mahogany, three tuners a side
    face, _ = _headstock(0.0, 0.0012)
    wood, corners = _headstock(-0.0012, HEAD_THICK - 0.0012)
    parts += [("head_face", face, "head_face"), ("head_wood", wood, "mahogany")]
    cols.append(uk.hull_collider("col_head", corners, R))
    side_cols = []
    for side in (-1, 1):
        for i, u in enumerate(TUNER_U):
            p = head_point(u, side * 0.024)
            parts.append((f"post_{SIDE[side]}{i}", uk.cylinder(0.0032, 0.011, center=p + [0, 0, 0.0055], segments=14), "chrome"))
            parts.append((f"bush_{SIDE[side]}{i}", uk.cylinder(0.0055, 0.002, center=p + [0, 0, 0.001], segments=16), "chrome"))
            hw = 0.041 if u < 0.15 else 0.036
            q = head_point(u, side * (hw + 0.008), up=-HEAD_THICK / 2)
            parts.append((f"shaft_{SIDE[side]}{i}", uk.cylinder(0.0016, 0.016, center=q, axis="x", segments=10), "chrome"))
            b = head_point(u, side * (hw + 0.021), up=-HEAD_THICK / 2)
            parts.append((f"button_{SIDE[side]}{i}", uk.box((0.006, 0.016, 0.013), b, bevel=0.002, round_xy=0.0025), "chrome"))
            side_cols.append(b)
    for side in (-1, 1):
        pts = np.array([p for p in side_cols if np.sign(p[0] - X) == side])
        lo, hi = pts.min(axis=0) - [0.006, 0.010, 0.008], pts.max(axis=0) + [0.006, 0.010, 0.008]
        cols.append(uk.collider(f"col_buttons_{SIDE[side]}", hi - lo, (lo + hi) / 2, R))
    # bridge, saddle and pins
    parts.append(("bridge", uk.box((0.155, 0.030, 0.009), (X, SADDLE_Y + 0.004, BODY_TOP + 0.0045), bevel=0.002, round_xy=0.006), "rosewood"))
    parts.append(("saddle", uk.box((0.075, 0.003, 0.0035), (X, SADDLE_Y, BODY_TOP + 0.009 + 0.00175), bevel=0.0006), "bone"))
    cols.append(uk.collider("col_bridge", (0.155, 0.030, 0.0125), (X, SADDLE_Y + 0.004, BODY_TOP + 0.00625), R))
    for k in range(1, 7):
        px = string_x(k, SCALE)
        parts.append((f"pin_{k}", uk.cylinder(0.0028, 0.003, center=(px, SADDLE_Y + 0.012, BODY_TOP + 0.0105), segments=12, bevel=0.0008), "pin"))
    # strings: from the saddle over the nut to the posts
    gauges = {1: 0.0003, 2: 0.0004, 3: 0.0006, 4: 0.0008, 5: 0.00107, 6: 0.00135}
    for k in range(1, 7):
        r = gauges[k] / 2
        p_saddle = (string_x(k, SCALE), SADDLE_Y, BODY_TOP + 0.0125 - r)
        p_nut = (string_x(k, 0.0), NUT_Y - 0.003, STRING_TOP - 0.0005 - r)
        side = -1 if k <= 3 else 1
        u = TUNER_U[(k - 1) % 3] if k <= 3 else TUNER_U[(6 - k) % 3]
        post = head_point(u, side * 0.024, up=0.004)
        parts.append((f"string_{k}", uk.tube([p_saddle, p_nut, post], max(r, 0.0002), sides=6), "steel" if k <= 2 else "bronze"))
    # neck rest under the 8th fret: a beech block with felt-lined cheeks either side of the neck
    rest_y = NUT_Y + 0.5 * (fret_d(7) + fret_d(9))
    under = BODY_TOP - (NECK_BACK["nut"] + 0.0025)
    parts.append(("rest", uk.box((0.10, 0.05, under), (X, rest_y, under / 2), bevel=0.003), "beech"))
    for side in (-1, 1):
        parts.append((f"rest_cheek_{SIDE[side]}", uk.box((0.02, 0.05, 0.015), (X + side * 0.038, rest_y, under + 0.0075), bevel=0.003), "felt"))
    cols.append(uk.collider("col_rest", (0.10, 0.05, under), (X, rest_y, under / 2), R))
    # Written as a rigid body (the loader needs one); the task spawns it without a mass, which makes it kinematic: fixed.
    return uk.write_usda(str(OUT / "guitar.usda"), R, parts, looks, cols, mass=2.2, friction=(0.6, 0.5))


# -- capo -----------------------------------------------------------------------------------------------------------
# A cam-lever capo, as a free articulation of three links. Frame coordinates: origin at the middle of the pad's
# underside (on the string tops when fitted), x across the neck toward the far side, y along the neck, z up.
# Frame: a rubber-lined bar over the strings and a spine down the near side of the neck. Jaw: a padded bar under the
# neck, sliding up and down the spine (joint "jaw", along z; 0 = against the neck's back at the 4th fret). Lever: a
# rocker on the bar (joint "lever", about y; 0 = closed, handle flat; -20 degrees = open, handle up and tail flat).
PAD = ((-0.027, -0.006, 0.0), (0.027, 0.006, 0.004))
BAR = ((-0.0365, -0.006, 0.004), (0.0295, 0.006, 0.012))
SPINE = ((-0.0365, -0.006, -0.048), (-0.0305, 0.006, 0.004))
BAR_W = 0.012
JAW_X = (-0.0305, 0.0255)  # the jaw's span across the neck, from the spine's inner face
JAW_T, JAW_PAD_T = 0.006, 0.003
JAW_OPEN = -0.010  # m below its clamped place
JAW_LIMITS = (-0.012, 0.006)
PIVOT = (-0.0225, 0.0, 0.014)  # the lever's hinge: 14 mm in from the spine's outer face, mid-thickness on the bar
HANDLE_L, TAIL_L, LEVER_W, LEVER_T = 0.036, 0.013, 0.008, 0.004
TAIL_BEND = 20.0  # degrees
LEVER_OPEN = -math.radians(20.0)
MASS = {"frame": 0.030, "jaw": 0.013, "lever": 0.004}
CAPO_MASS = sum(MASS.values())
CAPO_FRICTION = (0.9, 0.8)  # rubber on steel strings and on the neck's finish; anodised aluminium on the gripper pads
D4 = 0.5 * (fret_d(3) + fret_d(4))  # where the capo starts: the middle of the 4th fret space
JAW_Z0 = -(STRING_TOP - neck_back_z(D4))  # the jaw pad's top, in frame coordinates, clamped at the 4th fret


def neck_thickness(d: float) -> float:
    """From the string tops to the lowest point of the neck's back, d from the nut."""
    return STRING_TOP - neck_back_z(d)


def _box_inertia(m: float, lo, hi) -> tuple:
    sx, sy, sz = np.asarray(hi, float) - np.asarray(lo, float)
    return (m / 12 * (sy * sy + sz * sz), m / 12 * (sx * sx + sz * sz), m / 12 * (sx * sx + sy * sy))


def _bx(lo, hi, bevel=0.0006, round_xy=0.0006) -> uk.Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return uk.box(hi - lo, (lo + hi) / 2, bevel=bevel, round_xy=round_xy)


def _col(name, lo, hi, root, indent="        ") -> str:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    text = uk.collider(name, hi - lo, (lo + hi) / 2, root)
    return text.replace("\n    ", "\n" + indent).replace("contactOffset = 0.004", "contactOffset = 0.002")


def _hull(name, pts, root, indent="        ") -> str:
    text = uk.hull_collider(name, pts, root)
    return text.replace("\n    ", "\n" + indent).replace("contactOffset = 0.004", "contactOffset = 0.002")


def _tail_corners() -> np.ndarray:
    """The lever's tail in lever coordinates: from the pivot toward -x, bent up TAIL_BEND degrees."""
    a = math.radians(TAIL_BEND)
    u = np.array([-math.cos(a), 0.0, math.sin(a)])  # along the tail
    n = np.array([math.sin(a), 0.0, math.cos(a)])  # its thickness
    pts = []
    for t in (0.0, TAIL_L):
        for h in (-LEVER_T / 2, LEVER_T / 2):
            for w in (-LEVER_W / 2, LEVER_W / 2):
                pts.append(t * u + h * n + np.array([0.0, w, 0.0]))
    return np.array(pts)


def _body(name, pos, mass, com, inertia, colliders, visuals, root_api=False) -> str:
    api = ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"' if root_api else ""
    extra = """
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 4
        float physxArticulation:sleepThreshold = 0
        float physxArticulation:stabilizationThreshold = 0""" if root_api else ""
    return f"""
    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"{api}]
    )
    {{
        double3 xformOp:translate = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {mass:.5f}
        point3f physics:centerOfMass = ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})
        float3 physics:diagonalInertia = ({inertia[0]:.3e}, {inertia[1]:.3e}, {inertia[2]:.3e})
        float physxRigidBody:maxDepenetrationVelocity = 1
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0{extra}
{colliders}

        def Xform "visual"
        {{{visuals}
        }}
    }}"""


def _capo() -> str:
    root = "Capo"
    looks = {
        "anodised": uk.Look(color=(0.62, 0.03, 0.03), metallic=0.35, roughness=0.38),
        "rubber": uk.Look(color=(0.02, 0.02, 0.022), roughness=0.9),
        "steel": uk.Look(color=(0.80, 0.80, 0.81), metallic=0.9, roughness=0.25),
    }
    ind = "            "

    def meshes(parts):
        return "".join(uk._mesh_usda(n, m, look, root, looks[look], 35.0, indent=ind) for n, m, look in parts)

    # frame
    frame_parts = [("pad", _bx(*PAD, bevel=0.0005), "rubber"), ("bar", _bx(*BAR), "anodised"), ("spine", _bx(*SPINE), "anodised")]
    for s_ in (-1, 1):  # the lever's hinge lugs either side of the lever
        lo = (PIVOT[0] - 0.004, s_ * (LEVER_W / 2 + 0.0002), 0.012)
        hi = (PIVOT[0] + 0.004, s_ * (BAR_W / 2), 0.0175)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        frame_parts.append((f"lug_{'a' if s_ < 0 else 'b'}", _bx(lo, hi, bevel=0.0004, round_xy=0.0004), "anodised"))
    frame_parts.append(("hinge_pin", uk.cylinder(0.0012, BAR_W + 0.001, center=PIVOT, axis="y", segments=12), "steel"))
    frame_cols = "".join([_col("col_pad", *PAD, root), _col("col_bar", *BAR, root), _col("col_spine", *SPINE, root)])
    com_f = (-0.013, 0.0, -0.0026)
    lo_all, hi_all = np.array([-0.0365, -0.006, -0.048]), np.array([0.0295, 0.006, 0.012])
    frame = _body("frame", (0, 0, 0), MASS["frame"], com_f, _box_inertia(MASS["frame"], lo_all, hi_all), frame_cols,
                  meshes(frame_parts), root_api=True)
    # jaw: link origin at the top of its pad, the middle of its span
    jx = 0.5 * (JAW_X[0] + JAW_X[1])
    jl = JAW_X[1] - JAW_X[0]
    pad_lo, pad_hi = (-jl / 2 + 0.001, -BAR_W / 2, -JAW_PAD_T), (jl / 2, BAR_W / 2, 0.0)
    bar_lo, bar_hi = (-jl / 2, -BAR_W / 2, -JAW_PAD_T - JAW_T), (jl / 2, BAR_W / 2, -JAW_PAD_T)
    jaw_parts = [("jaw_pad", _bx(pad_lo, pad_hi, bevel=0.0005), "rubber"), ("jaw_bar", _bx(bar_lo, bar_hi), "anodised")]
    jaw_cols = _col("col_jaw_pad", pad_lo, pad_hi, root) + _col("col_jaw_bar", bar_lo, bar_hi, root)
    jaw = _body("jaw", (jx, 0.0, JAW_Z0), MASS["jaw"], (0, 0, -0.005), _box_inertia(MASS["jaw"], bar_lo, pad_hi),
                jaw_cols, meshes(jaw_parts))
    # lever: link origin at the hinge
    h_lo, h_hi = (0.0, -LEVER_W / 2, -LEVER_T / 2), (HANDLE_L, LEVER_W / 2, LEVER_T / 2)
    tail = _tail_corners()
    tail_mesh = uk.Mesh(tail, np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                                         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]))
    vol = lambda m: float(np.sum(np.einsum("ij,ij->i", m.v[m.f[:, 0]], np.cross(m.v[m.f[:, 1]], m.v[m.f[:, 2]]))) / 6)  # noqa: E731
    if vol(tail_mesh) < 0:
        tail_mesh = uk.Mesh(tail_mesh.v, tail_mesh.f[:, ::-1])
    lever_parts = [("handle", _bx(h_lo, h_hi, bevel=0.0006, round_xy=0.0015), "anodised"), ("tail", tail_mesh, "anodised"),
                   ("knuckle", uk.cylinder(0.0028, LEVER_W, center=(0, 0, 0), axis="y", segments=14), "anodised")]
    lever_cols = _col("col_handle", h_lo, h_hi, root) + _hull("col_tail", tail, root)
    lever = _body("lever", PIVOT, MASS["lever"], (0.012, 0, 0), _box_inertia(MASS["lever"], (-0.012, -0.004, -0.002), (0.036, 0.004, 0.002)),
                  lever_cols, meshes(lever_parts))
    joints = f"""
        def PhysicsPrismaticJoint "jaw" (
            prepend apiSchemas = ["PhysicsDriveAPI:linear", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Z"
            rel physics:body0 = </{root}/frame>
            rel physics:body1 = </{root}/jaw>
            point3f physics:localPos0 = ({jx:.6f}, 0, {JAW_Z0:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {JAW_LIMITS[0]:.6f}
            float physics:upperLimit = {JAW_LIMITS[1]:.6f}
            uniform token drive:linear:physics:type = "force"
            float drive:linear:physics:stiffness = 4000
            float drive:linear:physics:damping = 30
            float drive:linear:physics:maxForce = 40
            float physxJoint:armature = 0.00001
        }}

        def PhysicsRevoluteJoint "lever" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Y"
            rel physics:body0 = </{root}/frame>
            rel physics:body1 = </{root}/lever>
            point3f physics:localPos0 = ({PIVOT[0]:.6f}, {PIVOT[1]:.6f}, {PIVOT[2]:.6f})
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = -20
            float physics:upperLimit = 0
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:stiffness = 0.003
            float drive:angular:physics:damping = 0.0001
            float drive:angular:physics:maxForce = 0.5
            float physxJoint:armature = 0.00001
        }}"""
    used = sorted({p[2] for p in frame_parts + jaw_parts + lever_parts})
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{uk.physics_material("PhysicsMaterial", *CAPO_FRICTION)}

    def Scope "Looks"
    {{{"".join(uk._material_usda(m, root, looks[m]) for m in used)}
    }}
{frame}{jaw}{lever}

    def Scope "joints"
    {{{joints}
    }}
}}
"""
    path = OUT / "capo.usda"
    OUT.mkdir(exist_ok=True)
    rel = Path(__import__("os").path.relpath(HERE / "textures", OUT)).as_posix()
    text = text.replace(f"@{(HERE / 'textures').as_posix()}/", f"@{rel}/")
    path.write_text(text)
    return str(path)


def build() -> None:
    _guitar()
    _capo()


if __name__ == "__main__":
    build()
    print("wrote", sorted(p.name for p in OUT.glob("*.usda")))
    print("frets 1-5 (mm):", [round(f * 1000, 1) for f in FRETS[:5]], "body", round(BODY_Y0, 4), "to", round(BODY_Y0 + BODY_L, 4))
    print("neck width at fret 2:", round(neck_w(fret_d(2)) * 1000, 1), "mm; strings 1/6 at fret 2:",
          round(string_x(1, fret_d(2)), 4), round(string_x(6, fret_d(2)), 4))
