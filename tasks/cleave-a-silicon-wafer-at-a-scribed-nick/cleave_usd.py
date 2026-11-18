"""The task's objects as plain-text USD: a 100 mm silicon wafer and its pre-cut piece pairs, the pocket stage with its
notch, the black lab tray, the pen-style diamond scriber in its stand, the 3-point wafer-cleaving pliers (an
articulation: two crossed halves on a sprung pivot, each carrying a plastic jaw insert) on their stand, and a hidden
shelf under the table where unused piece pairs wait.

Visuals are meshes with real proportions; colliders are separate plain boxes, cylinders and convex prisms (purpose
"guide"). Units: metres. Each file's origin is stated with its builder.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import usdkit as uk
from usdkit import Look

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"

# -- the wafer (SEMI M1 100 mm, 525 um, one 32.5 mm primary flat) -----------------------------------------------------
# Wafer frame: origin at the wafer's centre on its bottom face, +z out of the polished face, the flat toward -x.
WAFER_R = 0.050
WAFER_T = 0.001          # drawn and collided 1 mm thick (a 525 um box is squeezed through by the pliers' jaws in
                         # PhysX); the mass is the real 525 um wafer's
WAFER_T_REAL = 0.000525
FLAT_HALF = 0.01625
FLAT_D = math.sqrt(WAFER_R ** 2 - FLAT_HALF ** 2)       # 47.29 mm: the flat's distance from the centre
TARGET_Y = 0.020                                       # the target cleave line (world y 0) in the wafer frame
LINES_MM = list(range(-6, 7))                          # pre-cut lines, 1 mm apart, within 6 mm of the target
TICK_L, TICK_W = 0.004, 0.0006
SI_DENSITY = 2330.0
GAP = 0.00005                                          # each piece stops this short of its line (visual hairline)

# -- the stage --------------------------------------------------------------------------------------------------------
PLATE_R, PLATE_T = 0.060, 0.008
STAGE_H = 0.120                                        # plate top above the bench
COL_R = 0.030
NOTCH_Y = (0.005, 0.045)                               # wafer frame: world y -15 to +25 mm
NOTCH_X_END = -math.sqrt(WAFER_R ** 2 - TARGET_Y ** 2) + 0.010   # 10 mm in from the rim at the tick
SHIM_T = 0.0005          # the pocket: 0.5 mm shim for the 1 mm drawn wafer
SHIM_T_REAL = 0.00025
POCKET_GAP = 0.0003
TAB_ANGLES = (35.0, -35.0)                             # hold-down tabs on the far rim (wafer frame, from +x)
TAB_W, TAB_OVER, TAB_CLEAR, TAB_LIP = 0.010, 0.0015, 0.0002, 0.0015

# -- the tray -------------------------------------------------------------------------------------------------------
TRAY_IN = (0.300, 0.400)                               # inside, x by y
TRAY_WALL, TRAY_H, TRAY_FLOOR = 0.004, 0.025, 0.003

# -- the scriber (EMS 62107-ST) and its stand ----------------------------------------------------------------------
# Scriber frame: origin at the diamond point, +z along the pen.
PEN_L, PEN_R = 0.140, 0.0045
PEN_TIP_R = 0.0005
PEN_CONE_L = 0.010                                     # brass collet cone from the point to the barrel
PEN_MASS = 0.015
STAND = (0.040, 0.040, 0.125)                          # PLA stand; the hole is 11 mm square, 25 mm deep
STAND_HOLE, STAND_DEPTH = 0.011, 0.025

# -- the pliers (CleanBreak-type 6 in wafer-cleaving pliers, 3-point plastic jaws) ---------------------------------
# Pliers frame: origin at the pivot, +x toward the jaws, +y along the pivot axis, +z up when they lie on edge.
JAW_X = (0.018, 0.028)                                 # jaw inserts, 10 mm long
JAW_W = 0.016                                          # across the cleave line
RIDGE_W, PAD_W, PAD_Y = 0.001, 0.002, 0.006
GAP_REST = 0.006                                       # ridge top to pad bottoms, at rest
SLEEVE_X = (-0.124, -0.092)                            # rubber grips at the handle ends
SLEEVE_Z = 0.025                                       # their centres, above and below the axis, at rest
SLEEVE_T, SLEEVE_W = 0.010, 0.012
GRIP_X = -0.108                                        # where the demo holds them (sleeve centre)
HEAD_W = 0.012
PLIERS_MASS = 0.120
CONTACT_R = 0.0255                                     # the wafer's contact centre on the jaws, 6 mm engaged
JAW_TILT = math.asin((GAP_REST / 2 - 0.0005) / CONTACT_R) if False else 0.0979   # rad, each jaw at rest
CLOSE_LIMIT = -2 * 0.0979                              # rad: the closing stop, set at the (drawn) wafer's thickness:
                                                       # the faces close parallel, 1 mm apart

LOOKS = {
    "wafer": Look(color=(0.30, 0.31, 0.36), metallic=0.85, roughness=0.06),
    "wafer_edge": Look(color=(0.22, 0.22, 0.25), metallic=0.6, roughness=0.3),
    "ink": Look(color=(0.01, 0.01, 0.01), roughness=0.7),
    "petg": Look(color=(0.80, 0.81, 0.80), roughness=0.45),
    "shim": Look(color=(0.85, 0.78, 0.45), roughness=0.2, opacity=0.6),
    "tray": Look(color=(0.025, 0.025, 0.028), roughness=0.55),
    "alu": Look(color=(0.72, 0.73, 0.75), metallic=0.9, roughness=0.32),
    "knurl": Look(color=(0.45, 0.46, 0.48), metallic=0.9, roughness=0.6),
    "brass": Look(color=(0.78, 0.60, 0.30), metallic=0.9, roughness=0.3),
    "diamond": Look(color=(0.9, 0.9, 0.95), metallic=0.2, roughness=0.05),
    "pla_grey": Look(color=(0.30, 0.32, 0.35), roughness=0.5),
    "pla_blue": Look(color=(0.10, 0.25, 0.55), roughness=0.5),
    "steel": Look(color=(0.55, 0.56, 0.58), metallic=0.9, roughness=0.35),
    "grip": Look(color=(0.65, 0.06, 0.05), roughness=0.7),
    "jaw": Look(color=(0.92, 0.92, 0.90), roughness=0.4),
    "shelf": Look(color=(0.2, 0.2, 0.2), roughness=0.9),
}


# -- geometry helpers -------------------------------------------------------------------------------------------------
def _quat(axis, deg):
    a = math.radians(deg) / 2
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    return (math.cos(a), *(math.sin(a) * ax))


def clip(poly, n, c):
    """Sutherland-Hodgman: the part of convex polygon `poly` [k,2] where n . p >= c."""
    out = []
    P = np.asarray(poly, float)
    n = np.asarray(n, float)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        da, db = a @ n - c, b @ n - c
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out)


def wafer_outline(n=96) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.c_[WAFER_R * np.cos(t), WAFER_R * np.sin(t)]
    return clip(circ, (1, 0), -FLAT_D)


def plate_parts(n=96):
    """The plate's outline split into three convex parts around the notch (wafer frame, xy)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.c_[PLATE_R * np.cos(t), PLATE_R * np.sin(t)]
    above = clip(circ, (0, 1), NOTCH_Y[1])
    below = clip(circ, (0, -1), -NOTCH_Y[0])
    mid = clip(clip(clip(circ, (0, 1), NOTCH_Y[0]), (0, -1), -NOTCH_Y[1]), (1, 0), NOTCH_X_END)
    return [above, below, mid]


def poly_area(p) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def _fmt(a, d=6):
    return ", ".join("(" + ", ".join(f"{x:.{d}f}" for x in row) + ")" for row in np.atleast_2d(a))


def prism_hull(name, poly, z0, z1, root, offset=0.0005, material="PhysicsMaterial", indent="    ") -> str:
    """A convex prism collider (convex hull of the polygon swept from z0 to z1)."""
    p = np.asarray(poly, float)
    n = len(p)
    pts = np.vstack([np.c_[p, np.full(n, z0)], np.c_[p, np.full(n, z1)]])
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    text = f"""
    def Mesh "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        uniform token physics:approximation = "convexHull"
        int physxConvexHullCollision:hullVertexLimit = 255
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        int[] faceVertexCounts = [{", ".join(str(len(f)) for f in faces)}]
        int[] faceVertexIndices = [{", ".join(str(i) for f in faces for i in f)}]
        point3f[] points = [{_fmt(pts)}]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def box_col(name, size, center, root, rot=None, offset=0.001, material="PhysicsMaterial", indent="    ") -> str:
    sx, sy, sz = size
    cx, cy, cz = center
    w, qx, qy, qz = rot if rot is not None else (1.0, 0.0, 0.0, 0.0)
    text = f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        double size = 1
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({cx:.6f}, {cy:.6f}, {cz:.6f})
        quatf xformOp:orient = ({w:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def cyl_col(name, radius, height, center, root, axis="Z", offset=0.001, material="PhysicsMaterial", indent="    ") -> str:
    text = f"""
    def Cylinder "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        uniform token axis = "{axis}"
        double radius = {radius:.6f}
        double height = {height:.6f}
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def sphere_col(name, radius, center, root, offset=0.0003, material="PhysicsMaterial", indent="    ") -> str:
    text = f"""
    def Sphere "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        double radius = {radius:.6f}
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def _meshes(parts, root, indent="        ") -> str:
    return "".join(uk._mesh_usda(p[0], p[1], p[2], root, LOOKS[p[2]], 35.0, indent=indent, hidden=len(p) > 3 and p[3])
                   for p in parts)


def _looks(parts, root) -> str:
    return "".join(uk._material_usda(m, root, LOOKS[m]) for m in sorted({p[2] for p in parts}))


def _write(name: str, text: str) -> str:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.usda"
    path.write_text(text)
    return str(path.relative_to(HERE))


def _body(name, root, parts, colliders, mass=None, friction=(0.6, 0.5), solver=16, report=False, extra_mat="",
          depen=1.0, hidden=False) -> str:
    if True:
        kin = mass is None
        mass = 1.0 if kin else mass
        api = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"' + (', "PhysxContactReportAPI"' if report else "")
        head = f""" (
    prepend apiSchemas = [{api}]
)"""
        rep = "\n    float physxContactReport:threshold = 0" if report else ""
        rep += "\n    bool physics:kinematicEnabled = 1" if kin else ""
        body = f"""
    float physics:mass = {mass:.6f}
    int physxRigidBody:solverPositionIterationCount = {solver}
    int physxRigidBody:solverVelocityIterationCount = 4
    float physxRigidBody:maxDepenetrationVelocity = {depen}
    float physxRigidBody:sleepThreshold = 0
    float physxRigidBody:stabilizationThreshold = 0{rep}
"""
    vis = '\n        token visibility = "invisible"' if hidden else ""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"{head}
{{{body}{uk.physics_material("PhysicsMaterial", *friction)}{extra_mat}
{colliders}

    def Scope "Looks"
    {{{_looks(parts, root)}
    }}

    def Xform "visual"
    {{{vis}{_meshes(parts, root)}
    }}
}}
"""
    return _write(name, text)


def _flat_prism(poly, z0, z1) -> uk.Mesh:
    """A closed prism mesh from a convex polygon (visual)."""
    p = np.asarray(poly, float)
    n = len(p)
    v = np.vstack([np.c_[p, np.full(n, z0)], np.c_[p, np.full(n, z1)], [[*p.mean(0), z0]], [[*p.mean(0), z1]]])
    f = []
    cb, ct = 2 * n, 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (n + i, n + j, ct), (j, i, cb)]
    return uk.Mesh(v, np.array(f))


# -- the wafer and its pieces -----------------------------------------------------------------------------------------
def _tick_poly():
    x0 = -math.sqrt(WAFER_R ** 2 - TARGET_Y ** 2)
    return np.array([[x0, TARGET_Y - TICK_W / 2], [x0 + TICK_L, TARGET_Y - TICK_W / 2],
                     [x0 + TICK_L, TARGET_Y + TICK_W / 2], [x0, TARGET_Y + TICK_W / 2]])


def _chord(poly, y):
    """The x-interval of convex polygon `poly` along the line at y (None if it misses)."""
    xs = []
    P = np.asarray(poly)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1]:
            xs.append(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def strip_edges():
    """Strip boundaries (wafer frame y): 1 mm apart on the pre-cut lines, 4 mm apart elsewhere."""
    lo, hi = TARGET_Y + LINES_MM[0] / 1000.0, TARGET_Y + LINES_MM[-1] / 1000.0
    e = list(np.arange(-WAFER_R, lo - 1e-9, 0.004)) + [TARGET_Y + k / 1000.0 for k in LINES_MM] + \
        list(np.arange(hi + 0.004, WAFER_R + 1e-9, 0.004))
    return sorted(set(round(v, 6) for v in e + [WAFER_R]))


def strips(poly):
    """The polygon's collider as boxes (x0, x1, y0, y1): each strip spans the polygon's chord at both its edges."""
    P = np.asarray(poly)
    ylo, yhi = P[:, 1].min(), P[:, 1].max()
    edges = [ylo] + [e for e in strip_edges() if ylo + 1e-6 < e < yhi - 1e-6] + [yhi]
    out = []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        c0, c1 = _chord(P, y0 + 1e-7), _chord(P, y1 - 1e-7)
        if c0 is None or c1 is None:
            continue
        x0, x1 = max(c0[0], c1[0]), min(c0[1], c1[1])
        if x1 - x0 > 0.001 and y1 - y0 > 0.0002:
            out.append((x0, x1, y0, y1))
    return out


def _wafer_file(name, poly, tick, hidden=False) -> str:
    root = "Wafer"
    parts = [("face", _flat_prism(poly, 0.0, WAFER_T), "wafer")]
    if tick is not None and len(tick) >= 3:
        parts.append(("tick", _flat_prism(tick, WAFER_T, WAFER_T + 0.00002), "ink"))
    cols = "".join(box_col(f"col{i}", (x1 - x0, y1 - y0, WAFER_T), ((x0 + x1) / 2, (y0 + y1) / 2, WAFER_T / 2), root,
                           offset=0.0002) for i, (x0, x1, y0, y1) in enumerate(strips(poly)))
    mass = poly_area(poly) * WAFER_T_REAL * SI_DENSITY
    return _body(name, root, parts, cols, mass=mass, friction=(0.35, 0.3), solver=64, report=True,
                 depen=0.05, hidden=hidden)


def sfx(k: int) -> str:
    """A line's name: m3 for 3 mm toward -y, p3 toward +y, 0 for the target line."""
    return f"m{-k}" if k < 0 else (f"p{k}" if k > 0 else "0")


def wafer_files() -> dict:
    out = {"wafer": _wafer_file("wafer", wafer_outline(), _tick_poly())}
    o = wafer_outline()
    tk = _tick_poly()
    for k in LINES_MM:
        y = TARGET_Y + k / 1000.0
        cpn = clip(o, (0, 1), y + GAP)
        main = clip(o, (0, -1), -(y - GAP))
        out[f"cpn_{sfx(k)}"] = _wafer_file(f"cpn_{sfx(k)}", cpn, clip(tk, (0, 1), y + GAP), hidden=True)
        out[f"main_{sfx(k)}"] = _wafer_file(f"main_{sfx(k)}", main, clip(tk, (0, -1), -(y - GAP)), hidden=True)
    return out


# -- the stage (origin: bench, under the wafer's centre) ------------------------------------------------------------------
def _ring_boxes(prefix, root, z0, z1, width=0.003, skip=None):
    """The pocket's shim as boxes round the wafer outline + POCKET_GAP, skipping where `skip(x, y)` is true."""
    cols, parts = "", []
    o = wafer_outline(96)
    n = len(o)
    c = np.zeros(2)
    for i in range(n):
        a, b = o[i], o[(i + 1) % n]
        mid = (a + b) / 2
        d = b - a
        L = float(np.linalg.norm(d))
        nrm = np.array([d[1], -d[0]]) / L          # outward for a counter-clockwise outline
        if nrm @ (mid - c) < 0:
            nrm = -nrm
        centre = mid + nrm * (POCKET_GAP + width / 2)
        if skip is not None and skip(*centre):
            continue
        yaw = math.degrees(math.atan2(d[1], d[0]))
        size = (L + 0.0004, width, z1 - z0)
        cols += box_col(f"{prefix}{i}", size, (centre[0], centre[1], (z0 + z1) / 2), root, rot=_quat((0, 0, 1), yaw),
                        offset=0.0003)
        parts.append((f"{prefix}{i}", uk.box(size).turned(yaw, "z").moved((centre[0], centre[1], (z0 + z1) / 2)), "shim"))
    return cols, parts


def stage() -> str:
    root = "Stage"
    top = STAGE_H
    parts, cols = [], ""
    for i, p in enumerate(plate_parts()):
        parts.append((f"plate{i}", _flat_prism(p, top - PLATE_T, top), "petg"))
        cols += prism_hull(f"col_plate{i}", p, top - PLATE_T, top, root, offset=0.0005)
    col_h = top - PLATE_T - TRAY_FLOOR
    parts.append(("column", uk.cylinder(COL_R, col_h, center=(0, 0, TRAY_FLOOR + col_h / 2), segments=48), "petg"))
    cols += cyl_col("col_column", COL_R, col_h, (0, 0, TRAY_FLOOR + col_h / 2), root)

    def in_notch(x, y):
        return NOTCH_Y[0] - 0.002 < y < NOTCH_Y[1] + 0.002 and x < NOTCH_X_END + 0.002

    c2, p2 = _ring_boxes("shim", root, top, top + SHIM_T, skip=in_notch)
    # two hold-down tabs on the far rim: a post outside the pocket and a lip 0.2 mm over the face, reaching 1.5 mm in
    for i, ang in enumerate(TAB_ANGLES):
        a = math.radians(ang)
        u = np.array([math.cos(a), math.sin(a)])
        rot = _quat((0, 0, 1), ang)
        post_r0, post_r1 = WAFER_R + POCKET_GAP + 0.0005, WAFER_R + 0.0065
        lip_r0, lip_r1 = WAFER_R - TAB_OVER, post_r0
        z_lip0 = top + WAFER_T + TAB_CLEAR
        for nm, r0, r1, z0, z1 in ((f"tab{i}_post", post_r0, post_r1, top, z_lip0 + TAB_LIP),
                                   (f"tab{i}_lip", lip_r0, lip_r1, z_lip0, z_lip0 + TAB_LIP)):
            c = u * (r0 + r1) / 2
            size = (r1 - r0, TAB_W, z1 - z0)
            cols += box_col(f"col_{nm}", size, (c[0], c[1], (z0 + z1) / 2), root, rot=rot, offset=0.0003)
            parts.append((nm, uk.box(size, bevel=0.0002).turned(ang, "z").moved((c[0], c[1], (z0 + z1) / 2)), "petg"))
    return _body("stage", root, parts + p2, cols + c2, mass=None, friction=(0.5, 0.45))


# -- the tray (origin: bench, centre) --------------------------------------------------------------------------------
def tray() -> str:
    root = "Tray"
    ix, iy = TRAY_IN
    w, h, f = TRAY_WALL, TRAY_H, TRAY_FLOOR
    ox, oy = ix + 2 * w, iy + 2 * w
    boxes = [("floor", (ox, oy, f), (0, 0, f / 2)),
             ("wall_n", (ox, w, h), (0, iy / 2 + w / 2, h / 2)), ("wall_s", (ox, w, h), (0, -iy / 2 - w / 2, h / 2)),
             ("wall_e", (w, iy, h), (ix / 2 + w / 2, 0, h / 2)), ("wall_w", (w, iy, h), (-ix / 2 - w / 2, 0, h / 2))]
    parts = [(n, uk.box(s, center=c, bevel=0.0008), "tray") for n, s, c in boxes]
    cols = "".join(box_col(f"col_{n}", s, c, root) for n, s, c in boxes)
    return _body("tray", root, parts, cols, mass=None, friction=(0.5, 0.45))


# -- the scriber and its stand -------------------------------------------------------------------------------------------
def scriber() -> str:
    root = "Scriber"
    cone = uk.lathe([(0.0, 0.0), (PEN_TIP_R, 0.0003), (0.0016, 0.004), (0.0035, PEN_CONE_L), (0.0, PEN_CONE_L)], 24)
    point = uk.lathe([(0.0, 0.0), (PEN_TIP_R, 0.0003), (0.0007, 0.0012), (0.0, 0.0012)], 12)
    barrel = uk.lathe([(0.0, PEN_CONE_L), (PEN_R, PEN_CONE_L), (PEN_R, PEN_L - 0.003), (PEN_R - 0.0015, PEN_L), (0.0, PEN_L)], 32)
    knurl = uk.cylinder(PEN_R + 0.0002, 0.035, center=(0, 0, 0.032), segments=32)
    clip_ = uk.box((0.0015, 0.004, 0.040), center=(PEN_R + 0.0006, 0, 0.110), bevel=0.0003)
    parts = [("point", point, "diamond"), ("cone", cone, "brass"), ("barrel", barrel, "alu"), ("knurl", knurl, "knurl"),
             ("clip", clip_, "alu")]
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = lambda r, z: np.c_[r * np.cos(t), r * np.sin(t), np.full(16, z)]  # noqa: E731
    cone_pts = np.vstack([ring(0.0012, 0.0015), ring(0.0035, PEN_CONE_L)])
    cols = (sphere_col("col_point", PEN_TIP_R, (0, 0, PEN_TIP_R), root, material="PointMaterial")
            + uk.hull_collider("col_cone", cone_pts, root, material="PointMaterial").replace("0.004\n", "0.0005\n")
            + cyl_col("col_barrel", PEN_R, PEN_L - PEN_CONE_L, (0, 0, (PEN_L + PEN_CONE_L) / 2), root, offset=0.001))
    return _body("scriber", root, parts, cols, mass=PEN_MASS, friction=(0.8, 0.7), solver=32,
                 extra_mat=uk.physics_material("PointMaterial", 0.15, 0.12))   # diamond point on silicon


def stand() -> str:
    """The scriber's PLA stand (origin: bench, centre): a block with an 11 mm square hole, 25 mm deep."""
    root = "Stand"
    sx, sy, sz = STAND
    hb = sz - STAND_DEPTH
    hw = STAND_HOLE / 2
    side = (sx - STAND_HOLE) / 2
    boxes = [("base", (sx, sy, hb), (0, 0, hb / 2)),
             ("w_e", (side, sy, STAND_DEPTH), (hw + side / 2, 0, hb + STAND_DEPTH / 2)),
             ("w_w", (side, sy, STAND_DEPTH), (-hw - side / 2, 0, hb + STAND_DEPTH / 2)),
             ("w_n", (STAND_HOLE, side, STAND_DEPTH), (0, hw + side / 2, hb + STAND_DEPTH / 2)),
             ("w_s", (STAND_HOLE, side, STAND_DEPTH), (0, -hw - side / 2, hb + STAND_DEPTH / 2))]
    parts = [(n, uk.box(s, center=c, bevel=0.0006), "pla_blue") for n, s, c in boxes]
    cols = "".join(box_col(f"col_{n}", s, c, root) for n, s, c in boxes)
    return _body("stand", root, parts, cols, mass=None, friction=(0.5, 0.45))


# -- the pliers -----------------------------------------------------------------------------------------------------------
def _seg_box(p0, p1, t, w):
    """(size, centre, quat wxyz, yaw-free pitch deg) of a box from p0 to p1 in the xz plane, t thick in z, w in y."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(d[0], d[1]))
    pitch = math.degrees(math.atan2(d[1], d[0]))
    c = (p0 + p1) / 2
    # rotation about +y by -pitch takes +x to (cos, 0, sin)
    return (L, w, t), (c[0], 0.0, c[1]), _quat((0, 1, 0), -pitch), pitch


def pliers_geometry():
    """Each link's boxes as (name, size, centre, quat wxyz, look) in the pliers frame (rest pose)."""
    g0, g1 = -GAP_REST / 2, GAP_REST / 2
    jx0, jx1 = JAW_X
    jl, jc = jx1 - jx0, (jx0 + jx1) / 2
    q0 = (1.0, 0.0, 0.0, 0.0)
    sl, sc = SLEEVE_X[1] - SLEEVE_X[0], (SLEEVE_X[0] + SLEEVE_X[1]) / 2
    links = {}
    # B: lower handle + upper jaw arm (the articulation root)
    hb = _seg_box((-0.010, -0.006), (SLEEVE_X[1], -SLEEVE_Z), 0.009, 0.010)
    ab = _seg_box((0.004, 0.006), (jx0, g1 + 0.004), 0.006, HEAD_W)
    links["b_frame"] = [("handle", hb[0], hb[1], hb[2], "steel"), ("sleeve", (sl, SLEEVE_W, SLEEVE_T), (sc, 0, -SLEEVE_Z), q0, "grip"),
                        ("arm", ab[0], ab[1], ab[2], "steel"),
                        ("head", (0.018, HEAD_W, 0.010), (0.0, 0.0, 0.004), q0, "steel")]
    # A: upper handle + lower jaw arm
    ha = _seg_box((-0.010, 0.006), (SLEEVE_X[1], SLEEVE_Z), 0.009, 0.010)
    aa = _seg_box((0.004, -0.006), (jx0, g0 - 0.004), 0.006, HEAD_W)
    links["a_frame"] = [("handle", ha[0], ha[1], ha[2], "steel"), ("sleeve", (sl, SLEEVE_W, SLEEVE_T), (sc, 0, SLEEVE_Z), q0, "grip"),
                        ("arm", aa[0], aa[1], aa[2], "steel"),
                        ("head", (0.018, HEAD_W, 0.010), (0.0, 0.0, -0.004), q0, "steel")]
    # jaw inserts: the lower carries the ridge, the upper the two pads. Their faces are parallel when the jaws close
    # onto the (drawn 1 mm) wafer; at rest each is turned JAW_TILT about the pivot, opening toward the tips.
    h = WAFER_T / 2
    low = [("block", (jl, JAW_W, 0.004), (jc, 0, -h - 0.001 - 0.002)), ("ridge", (jl, RIDGE_W, 0.001), (jc, 0, -h - 0.0005))]
    up = [("block", (jl, JAW_W, 0.004), (jc, 0, h + 0.001 + 0.002)), ("pad_l", (jl, PAD_W, 0.001), (jc, PAD_Y, h + 0.0005)),
          ("pad_r", (jl, PAD_W, 0.001), (jc, -PAD_Y, h + 0.0005))]
    links["jaw_lower"] = [(n, sz, tuple(_ry(JAW_TILT) @ np.array(c)), _quat((0, 1, 0), math.degrees(JAW_TILT)), "jaw")
                          for n, sz, c in low]
    links["jaw_upper"] = [(n, sz, tuple(_ry(-JAW_TILT) @ np.array(c)), _quat((0, 1, 0), -math.degrees(JAW_TILT)), "jaw")
                          for n, sz, c in up]
    del g0, g1
    return links


def _ry(a):
    """Rotation about +y by a (rad): +x turns toward -z."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ridge_point(r=None):
    """The ridge's top at distance r (default: the contact centre) along the jaw, jaw_lower link frame (rest pose)."""
    r = CONTACT_R if r is None else r
    return _ry(JAW_TILT) @ np.array([r, 0.0, -WAFER_T / 2])


def pad_point(r=None):
    r = CONTACT_R if r is None else r
    return _ry(-JAW_TILT) @ np.array([r, 0.0, WAFER_T / 2])


LINK_MASS = {"b_frame": 0.050, "a_frame": 0.050, "jaw_lower": 0.010, "jaw_upper": 0.010}
LINK_COM = {"b_frame": (-0.035, 0, -0.008), "a_frame": (-0.035, 0, 0.008), "jaw_lower": (0.023, 0, -0.006),
            "jaw_upper": (0.023, 0, 0.006)}
PIVOT_K = 1.08          # N m/rad: the return spring (6 N at the grips at rest, about 8 N closed on a wafer)
PIVOT_D = 0.02          # N m s/rad
SPRING_TARGET = 0.60    # rad beyond the open stop (0): the spring's preload


def pliers() -> str:
    root = "Pliers"
    links = pliers_geometry()
    ind = "        "

    def body(name, root_api=False, report=False):
        boxes = links[name]
        mass = LINK_MASS[name]
        com = LINK_COM[name]
        api = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"'
        if root_api:
            api += ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"'
        extra = """
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 4
        float physxArticulation:sleepThreshold = 0
        float physxArticulation:stabilizationThreshold = 0""" if root_api else ""
        mat = "JawMaterial" if name.startswith("jaw") else ("GripMaterial" if name.endswith("frame") else "PhysicsMaterial")
        cols = ""
        for bn, size, c, q, look in boxes:
            m = "GripMaterial" if look == "grip" else ("JawMaterial" if look == "jaw" else "PhysicsMaterial")
            cols += box_col(f"col_{bn}", size, c, root, rot=q, offset=0.0005 if look == "jaw" else 0.001, material=m,
                            indent=ind)
        del mat
        parts = []
        for bn, size, c, q, look in boxes:
            w, x, y, z = q
            pitch = 2 * math.degrees(math.atan2(y, w))
            m = uk.box(size, bevel=min(size) * 0.15).turned(pitch, "y").moved(c)
            parts.append((bn, m, look))
        L = max(abs(c[0]) for _, _, c, _, _ in boxes) + 0.02
        inertia = (mass * 0.02 ** 2 / 3, mass * L ** 2 / 6, mass * L ** 2 / 6)
        return f"""
    def Xform "{name}" (
        prepend apiSchemas = [{api}]
    )
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {mass:.6f}
        point3f physics:centerOfMass = ({com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f})
        float3 physics:diagonalInertia = ({inertia[0]:.3e}, {inertia[1]:.3e}, {inertia[2]:.3e})
        float physxRigidBody:maxDepenetrationVelocity = 0.5
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0{extra}
{cols}

        def Xform "visual"
        {{{_meshes(parts, root, indent="            ")}
        }}
    }}"""

    pivot = uk.cylinder(0.009, HEAD_W + 0.004, center=(0, 0, 0), axis="y", segments=24)
    looks_parts = [(bn, None, look) for boxes in links.values() for bn, _, _, _, look in boxes]
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{uk.physics_material("PhysicsMaterial", 0.6, 0.5)}{uk.physics_material("GripMaterial", 1.0, 0.9)}{uk.physics_material("JawMaterial", 0.5, 0.4)}

    def Scope "Looks"
    {{{_looks(looks_parts, root)}
    }}

    def Xform "pivot_visual"
    {{{_meshes([("pin", pivot, "steel")], root)}
    }}
{body("b_frame", root_api=True)}{body("a_frame")}{body("jaw_lower")}{body("jaw_upper")}

    def Scope "joints"
    {{
        def PhysicsRevoluteJoint "pivot" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "Y"
            rel physics:body0 = </{root}/b_frame>
            rel physics:body1 = </{root}/a_frame>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = {math.degrees(CLOSE_LIMIT):.4f}
            float physics:upperLimit = 0
            float physxJoint:armature = 0.00001
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:targetPosition = {math.degrees(SPRING_TARGET):.4f}
            float drive:angular:physics:stiffness = {PIVOT_K * math.pi / 180:.7f}
            float drive:angular:physics:damping = {PIVOT_D * math.pi / 180:.7f}
            float drive:angular:physics:maxForce = 5
        }}

        def PhysicsFixedJoint "fix_upper" (
        )
        {{
            rel physics:body0 = </{root}/b_frame>
            rel physics:body1 = </{root}/jaw_upper>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}

        def PhysicsFixedJoint "fix_lower" (
        )
        {{
            rel physics:body0 = </{root}/a_frame>
            rel physics:body1 = </{root}/jaw_lower>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}
    }}
}}
"""
    return _write("pliers", text)


# -- the pliers' stand (origin: bench, under the pivot; the pliers lie on edge along x) ---------------------------------
CRADLE_AXIS_Z = 0.120     # the pliers' pivot axis above the bench
CRADLE_SLOT = 0.020       # slot width (the parts it holds are 12 mm thick)
CRADLE_POSTS = ((0.010, "front"), (-0.060, "rear"))   # post centres along the pliers (pliers frame x)


def _bottom_at(x):
    """The pliers' lowest point (pliers frame z) at x, at rest: the lower jaw arm in front, the lower handle behind."""
    if x >= 0:
        return -GAP_REST / 2 - 0.004 - 0.003      # under the lower jaw arm
    t = (x - (-0.010)) / (SLEEVE_X[1] - (-0.010))
    return -0.006 + t * (-SLEEVE_Z + 0.006) - 0.0045 / math.cos(math.atan2(SLEEVE_Z - 0.006, 0.082))


def cradle() -> str:
    """Two slotted PLA posts on a base plate: each post's slot floor meets the pliers' lowest edge; the slot walls
    rise 12 mm either side (20 mm apart)."""
    root = "Cradle"
    parts, cols = [], ""
    wall_t, post_l = 0.005, 0.016
    base_t = 0.004
    for x, nm in CRADLE_POSTS:
        floor = CRADLE_AXIS_Z + _bottom_at(x) - 0.0005
        span = CRADLE_SLOT + 2 * wall_t
        bx = [(f"{nm}_post", (post_l, span, floor - TRAY_FLOOR), (x, 0, TRAY_FLOOR + (floor - TRAY_FLOOR) / 2)),
              (f"{nm}_wl", (post_l, wall_t, 0.012), (x, CRADLE_SLOT / 2 + wall_t / 2, floor + 0.006)),
              (f"{nm}_wr", (post_l, wall_t, 0.012), (x, -CRADLE_SLOT / 2 - wall_t / 2, floor + 0.006))]
        for n, s, c in bx:
            parts.append((n, uk.box(s, center=c, bevel=0.0005), "pla_grey"))
            cols += box_col(f"col_{n}", s, c, root)
    base = ("base", (0.100, 0.040, base_t), (-0.025, 0, TRAY_FLOOR + base_t / 2))
    parts.append((base[0], uk.box(base[1], center=base[2], bevel=0.0008), "pla_grey"))
    cols += box_col("col_base", base[1], base[2], root)
    return _body("cradle", root, parts, cols, mass=None, friction=(0.5, 0.45))


# -- the hidden shelf (under the table: where unused piece pairs wait) ---------------------------------------------------
SHELF = (0.90, 1.10, 0.02)


def shelf() -> str:
    root = "Shelf"
    parts = [("slab", uk.box(SHELF), "shelf")]
    cols = box_col("col_slab", SHELF, (0, 0, 0), root)
    return _body("shelf", root, parts, cols, mass=None, friction=(0.8, 0.7), hidden=True)


def build() -> dict:
    out = wafer_files()
    out.update(stage=stage(), tray=tray(), scriber=scriber(), stand=stand(), pliers=pliers(), cradle=cradle(),
               shelf=shelf())
    return out


# -- printable parts as STL (python cleave_usd.py --stl): mm, z up, each part in its own frame ------------------------
def _stl(path, meshes):
    tris = []
    for m in meshes:
        for f in m.f:
            tris.append(m.v[f] * 1000.0)
    with open(path, "w") as out:
        out.write("solid part\n")
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n = n / max(np.linalg.norm(n), 1e-12)
            out.write(f"facet normal {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n outer loop\n")
            for v in t:
                out.write(f"  vertex {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            out.write(" endloop\nendfacet\n")
        out.write("endsolid part\n")


def _meshes_of(text_builder_parts):
    return [p[1] for p in text_builder_parts]


def export_stl(outdir: Path) -> list:
    """The stage (plate, column and hold-down tabs; print the shim from 0.25 mm polyester with the same outline), the
    scriber's stand, the pliers' stand and the tick gauge."""
    outdir.mkdir(exist_ok=True)
    top = STAGE_H
    stage_parts = [_flat_prism(p, top - PLATE_T, top) for p in plate_parts()]
    col_h = top - PLATE_T - TRAY_FLOOR
    stage_parts.append(uk.cylinder(COL_R, col_h, center=(0, 0, TRAY_FLOOR + col_h / 2), segments=64))
    for ang in TAB_ANGLES:
        a = math.radians(ang)
        u = np.array([math.cos(a), math.sin(a)])
        post_r0, post_r1 = WAFER_R + POCKET_GAP + 0.0005, WAFER_R + 0.0065
        z_lip0 = top + WAFER_T_REAL + SHIM_T_REAL - SHIM_T_REAL + TAB_CLEAR
        for r0, r1, z0, z1 in ((post_r0, post_r1, top, z_lip0 + TAB_LIP), (WAFER_R - TAB_OVER, post_r0, z_lip0, z_lip0 + TAB_LIP)):
            c = u * (r0 + r1) / 2
            stage_parts.append(uk.box((r1 - r0, TAB_W, z1 - z0)).turned(ang, "z").moved((c[0], c[1], (z0 + z1) / 2)))
    sx, sy, sz = STAND
    hb, hw, side = sz - STAND_DEPTH, STAND_HOLE / 2, (sx - STAND_HOLE) / 2
    stand_parts = [uk.box((sx, sy, hb), center=(0, 0, hb / 2)),
                   uk.box((side, sy, STAND_DEPTH), center=(hw + side / 2, 0, hb + STAND_DEPTH / 2)),
                   uk.box((side, sy, STAND_DEPTH), center=(-hw - side / 2, 0, hb + STAND_DEPTH / 2)),
                   uk.box((STAND_HOLE, side, STAND_DEPTH), center=(0, hw + side / 2, hb + STAND_DEPTH / 2)),
                   uk.box((STAND_HOLE, side, STAND_DEPTH), center=(0, -hw - side / 2, hb + STAND_DEPTH / 2))]
    o = wafer_outline(128)
    x0 = -math.sqrt(WAFER_R ** 2 - TARGET_Y ** 2)
    gauge = [_flat_prism(clip(o, (0, 1), TARGET_Y + TICK_W / 2), 0, 0.002),
             _flat_prism(clip(o, (0, -1), -(TARGET_Y - TICK_W / 2)), 0, 0.002),
             _flat_prism(clip(clip(clip(o, (0, 1), TARGET_Y - TICK_W / 2), (0, -1), -(TARGET_Y + TICK_W / 2)), (1, 0), x0 + TICK_L), 0, 0.002)]
    cradle_parts = []
    wall_t, post_l, base_t = 0.005, 0.016, 0.004
    for x, _ in CRADLE_POSTS:
        floor = CRADLE_AXIS_Z + _bottom_at(x) - 0.0005
        span = CRADLE_SLOT + 2 * wall_t
        cradle_parts += [uk.box((post_l, span, floor - TRAY_FLOOR), center=(x, 0, TRAY_FLOOR + (floor - TRAY_FLOOR) / 2)),
                         uk.box((post_l, wall_t, 0.012), center=(x, CRADLE_SLOT / 2 + wall_t / 2, floor + 0.006)),
                         uk.box((post_l, wall_t, 0.012), center=(x, -CRADLE_SLOT / 2 - wall_t / 2, floor + 0.006))]
    cradle_parts.append(uk.box((0.100, 0.040, base_t), center=(-0.025, 0, TRAY_FLOOR + base_t / 2)))
    out = []
    for name, meshes in (("stage", stage_parts), ("scriber_stand", stand_parts), ("tick_gauge", gauge),
                         ("pliers_stand", cradle_parts)):
        _stl(outdir / f"{name}.stl", meshes)
        out.append(name)
    return out


if __name__ == "__main__":
    import sys

    if "--stl" in sys.argv:
        print(export_stl(HERE / "cad"))
    for k, v in build().items():
        print(k, v)
