"""Scene geometry for tapping an M6 hole in a robot tool adapter plate: the vise with its parallels, the adapter
plate, the countersink seat, the wrench stand, and the T-handle tap wrench with its tap (a fixed-base articulation
whose six joints leave the wrench free until the thread rule drives them).

Writes plain-text USD into build/ when imported (`build()`); visual meshes and simple colliders are separate.
All sizes in metres. Every object's frame is described where it is written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usdkit as uk  # noqa: E402
from usdkit import Look  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"

# -- the tap and wrench (wrench frame: origin at the tap's tip, +z up the tap's axis, T-bar along x) -----------------
TIP_R = 0.0023            # the plug chamfer's point
MAJOR_R = 0.0030          # M6 major radius
MINOR_R = 0.00255
CHAMFER_L = 0.004         # 4 threads of chamfer
THREAD_L = 0.0254
SHANK_R = 0.00324         # 6.48 mm shank
TAP_OUT = 0.0555          # the tap stands this far out of the wrench's body
BODY_R = 0.010
BODY_L = 0.065
BAR_R = 0.0032            # 6.4 mm T-bar
BAR_L = 0.100
BAR_Z = TAP_OUT + BODY_L - 0.007          # 0.1135: the T-bar's axis above the tip
BAND_Z = (0.0135, 0.0155)                 # the red band, above the tip
WRENCH_MASS = 0.082                       # wrench 70 g + tap 12 g
WRENCH_COM = (0.0, 0.0, 0.086)
WRENCH_I = (7.8e-5, 9.4e-5, 1.8e-5)

# -- the plate (plate frame: origin on the top face at the marked hole's axis) ------------------------------------------
PLATE_T = 0.006
DISC_R = 0.040
HOLE_R = 0.0025           # 5.0 mm drilled
CSK_R = 0.0035            # countersunk 90 deg to 7.0 mm
CLEAR_HALF = 0.004        # the collider's clearance hole: 8 mm square (the tap's 6 mm thread passes freely)
TOOL_PCD_R = 0.0315       # ISO 9409-1-63-4-M6 tool holes
FLANGE_PCD_R = 0.025      # ISO 9409-1-50-4-M6 screw holes
# the seat: the countersink and the drilled hole as the tap meets them before it bites (a separate kinematic body)
SEAT_BORE_R = 0.0026      # octagon's corner radius slightly above 2.5 mm so the 4.6 mm point fits its flats
SEAT_OUT_R = 0.0039

# -- the stand (stand frame: origin on the table under the hole's axis) -----------------------------------------------
STAND = (0.040, 0.040, 0.060)
STAND_HOLE_R = 0.006
STAND_CHAMFER_R = 0.012
STAND_CHAMFER_D = 0.006

LOOKS = {
    "hss": Look(color=(0.62, 0.63, 0.65), metallic=0.95, roughness=0.22),
    "hss_dark": Look(color=(0.33, 0.34, 0.36), metallic=0.9, roughness=0.35),
    "band": Look(color=(0.75, 0.04, 0.03), roughness=0.55),
    "knurl": Look(color=(0.80, 0.81, 0.84), metallic=0.9, roughness=0.55, texture="knurl.png", tile=0.008),
    "steel": Look(color=(0.60, 0.61, 0.63), metallic=0.95, roughness=0.28),
    "alu": Look(color=(1.0, 1.0, 1.0), metallic=0.9, roughness=0.34, texture="brushed.png", tile=0.03),
    "alu_cut": Look(color=(0.88, 0.89, 0.91), metallic=0.95, roughness=0.18),
    "hole": Look(color=(0.015, 0.015, 0.017), roughness=0.9),
    "marker": Look(color=(0.03, 0.16, 0.62), roughness=0.5),
    "cast": Look(color=(0.14, 0.20, 0.30), metallic=0.3, roughness=0.55, texture="hammertone.png", tile=0.05),
    "cast_dark": Look(color=(0.06, 0.07, 0.08), metallic=0.4, roughness=0.5),
    "jawsteel": Look(color=(0.30, 0.31, 0.33), metallic=0.85, roughness=0.45),
    "parallel": Look(color=(0.50, 0.51, 0.53), metallic=0.95, roughness=0.2),
    "pla_grey": Look(color=(0.50, 0.52, 0.55), roughness=0.55, texture="pla.png", tile=0.0064),
    "rubber": Look(color=(0.02, 0.02, 0.02), roughness=0.8),
}


def _fmt(a, d=6):
    return ", ".join("(" + ", ".join(f"{x:.{d}f}" for x in row) + ")" for row in np.atleast_2d(a))


def hull_col(name, pts, root, offset=0.0005, material="PhysicsMaterial", indent="    ") -> str:
    """A convex-hull collider from points (the hull's faces are left to PhysX: a point cloud as one polygon list)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    p = np.asarray(pts, float)
    h = ConvexHull(p)
    faces = []
    c = p.mean(0)
    for simp in h.simplices:
        a, b, d = p[simp]
        n = np.cross(b - a, d - a)
        faces.append(list(simp) if n @ (a - c) > 0 else [simp[0], simp[2], simp[1]])
    text = f"""
    def Mesh "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxConvexHullCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/{material}>
        uniform token purpose = "guide"
        uniform token physics:approximation = "convexHull"
        int physxConvexHullCollision:hullVertexLimit = 64
        float physxCollision:contactOffset = {offset}
        float physxCollision:restOffset = 0
        int[] faceVertexCounts = [{", ".join("3" for _ in faces)}]
        int[] faceVertexIndices = [{", ".join(str(i) for f in faces for i in f)}]
        point3f[] points = [{_fmt(p)}]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def box_col(name, size, center, root, offset=0.001, material="PhysicsMaterial", indent="    ") -> str:
    sx, sy, sz = size
    cx, cy, cz = center
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
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}"""
    return text.replace("\n    ", "\n" + indent)


def cyl_col(name, radius, height, center, root, axis="Z", offset=0.0008, material="PhysicsMaterial", indent="    ") -> str:
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


def _kinematic(name, root, parts, colliders, friction=(0.6, 0.5)) -> str:
    """A fixed object: a kinematic rigid body (the harness's contact sensing needs a rigid body on fixed objects)."""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = 1
    bool physics:kinematicEnabled = 1
    float physxRigidBody:sleepThreshold = 0
    float physxRigidBody:stabilizationThreshold = 0
{uk.physics_material("PhysicsMaterial", *friction)}
{colliders}

    def Scope "Looks"
    {{{_looks(parts, root)}
    }}

    def Xform "visual"
    {{{_meshes(parts, root)}
    }}
}}
"""
    return _write(name, text)


def circle(r, n=48, z=None):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.c_[r * np.cos(t), r * np.sin(t)]
    return p if z is None else np.c_[p, np.full(n, z)]


def annulus(r0, r1, z, n=48) -> uk.Mesh:
    """A flat ring (decal) facing up at height z."""
    a, b = circle(r0, n, z), circle(r1, n, z)
    v = np.vstack([a, b])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i)]
    return uk.Mesh(v, np.array(f))


def disc(r, z, n=32) -> uk.Mesh:
    v = np.vstack([[0.0, 0.0, z], circle(r, n, z)])
    f = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return uk.Mesh(v, np.array(f))


# -- the wrench and tap -----------------------------------------------------------------------------------------------
def tap_profile():
    prof = [(0.0, 0.0), (TIP_R, 0.0), (MAJOR_R, CHAMFER_L)]
    z = CHAMFER_L
    while z < THREAD_L - 0.0006:
        prof += [(MINOR_R, z + 0.0005), (MAJOR_R, z + 0.001)]
        z += 0.001
    prof += [(SHANK_R * 0.82, THREAD_L + 0.001), (SHANK_R, THREAD_L + 0.003), (SHANK_R, TAP_OUT + 0.004),
             (0.0, TAP_OUT + 0.004)]
    return prof


def wrench_parts():
    parts = []
    tap = uk.lathe(tap_profile(), 20)
    parts.append(("tap", tap, "hss"))
    # four flutes along the thread: dark strips
    for k in range(4):
        a = 45.0 + 90.0 * k
        flute = uk.box((0.0012, 0.0016, THREAD_L - 0.002), bevel=0.0003).moved((MAJOR_R - 0.0004, 0, THREAD_L / 2 + 0.0005))
        parts.append((f"flute{k}", flute.turned(a, "z"), "hss_dark"))
    band = uk.lathe([(0.0, BAND_Z[0]), (MAJOR_R + 0.00008, BAND_Z[0]), (MAJOR_R + 0.00008, BAND_Z[1]),
                     (0.0, BAND_Z[1])], 24)
    parts.append(("band", band, "band"))
    z0 = TAP_OUT
    nose = uk.lathe([(0.0, z0), (0.0052, z0), (BODY_R, z0 + 0.008), (BODY_R, z0 + 0.010), (0.0, z0 + 0.010)], 28)
    parts.append(("nose", nose, "steel"))
    knurl = uk.lathe([(0.0, z0 + 0.010), (BODY_R, z0 + 0.010), (BODY_R, z0 + BODY_L - 0.016), (0.0, z0 + BODY_L - 0.016)], 28)
    parts.append(("knurl", knurl, "knurl"))
    head = uk.lathe([(0.0, z0 + BODY_L - 0.016), (BODY_R, z0 + BODY_L - 0.016), (BODY_R, z0 + BODY_L - 0.0015),
                     (BODY_R - 0.0015, z0 + BODY_L), (0.0, z0 + BODY_L)], 28)
    parts.append(("head", head, "steel"))
    # knurl lines: thin dark rings for texture at Dream-1's size
    for i in range(9):
        zz = z0 + 0.012 + i * 0.0042
        ring = uk.lathe([(0.0, zz), (BODY_R + 0.00006, zz), (BODY_R + 0.00006, zz + 0.0012), (0.0, zz + 0.0012)], 28)
        parts.append((f"kr{i}", ring, "hss_dark"))
    bar = uk.cylinder(BAR_R, BAR_L - 0.004, center=(0, 0, BAR_Z), axis="x", segments=16)
    parts.append(("bar", bar, "steel"))
    for s in (-1, 1):
        knob = uk.cylinder(BAR_R + 0.0006, 0.004, center=(s * (BAR_L / 2 - 0.002), 0, BAR_Z), axis="x", segments=16,
                           bevel=0.0008)
        parts.append((f"knob{'p' if s > 0 else 'm'}", knob, "steel"))
    return parts


def wrench_colliders(root, indent):
    cone = np.vstack([circle(TIP_R, 16, 0.0), circle(MAJOR_R, 16, CHAMFER_L)])
    cols = hull_col("col_chamfer", cone, root, offset=0.0005, material="TapMaterial", indent=indent)
    cols += cyl_col("col_thread", MAJOR_R, THREAD_L - CHAMFER_L, (0, 0, (THREAD_L + CHAMFER_L) / 2), root,
                    offset=0.0005, material="TapMaterial", indent=indent)
    cols += cyl_col("col_shank", SHANK_R, TAP_OUT - THREAD_L, (0, 0, (TAP_OUT + THREAD_L) / 2), root, offset=0.0005,
                    material="TapMaterial", indent=indent)
    cols += cyl_col("col_body", BODY_R, BODY_L, (0, 0, TAP_OUT + BODY_L / 2), root, material="KnurlMaterial",
                    indent=indent)
    cols += cyl_col("col_bar", BAR_R, BAR_L, (0, 0, BAR_Z), root, axis="X", offset=0.0008, material="BarMaterial",
                    indent=indent)
    return cols


JOINTS = [("j_x", "Prismatic", "X", "anchor", "l_x"), ("j_y", "Prismatic", "Y", "l_x", "l_y"),
          ("j_z", "Prismatic", "Z", "l_y", "l_z"), ("j_a", "Revolute", "X", "l_z", "l_a"),
          ("j_b", "Revolute", "Y", "l_a", "l_b"), ("j_s", "Revolute", "Z", "l_b", "wrench")]
JOINT_NAMES = [j[0] for j in JOINTS]
LIMITS = {"j_x": (-0.6, 0.6), "j_y": (-0.6, 0.6), "j_z": (-0.3, 0.6), "j_a": (-170.0, 170.0), "j_b": (-85.0, 85.0),
          "j_s": None}


KNURL_MAT = """
    def Material "KnurlMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {
        float physics:staticFriction = 1.3
        float physics:dynamicFriction = 1.2
        float physics:restitution = 0
        uniform token physxMaterial:frictionCombineMode = "max"
    }
"""


def wrench() -> str:
    """The wrench and tap on a chain of three slides and three hinges from a fixed anchor at the tap's starting tip
    position: x, y, z slides move the tip; hinges about x and y tilt the tap about its tip; the last hinge turns it
    about its own axis. With no drives the wrench is a free body (the five links between weigh 2 g in all)."""
    root = "TapWrench"
    ind = "        "
    parts = wrench_parts()

    def dummy(name, art=False):
        api = '"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"'
        if art:
            api += ', "PhysicsArticulationRootAPI", "PhysxArticulationAPI"'
        extra = """
        bool physxArticulation:enabledSelfCollisions = 0
        int physxArticulation:solverPositionIterationCount = 32
        int physxArticulation:solverVelocityIterationCount = 4
        float physxArticulation:sleepThreshold = 0
        float physxArticulation:stabilizationThreshold = 0""" if art else ""
        return f"""
    def Xform "{name}" (
        prepend apiSchemas = [{api}]
    )
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 0.0004
        float3 physics:diagonalInertia = (1e-8, 1e-8, 1e-8)
        point3f physics:centerOfMass = (0, 0, 0)
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0{extra}
    }}"""

    links = dummy("anchor", art=True) + "".join(dummy(n) for n in ("l_x", "l_y", "l_z", "l_a", "l_b"))
    wbody = f"""
    def Xform "wrench" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {WRENCH_MASS:.5f}
        point3f physics:centerOfMass = ({WRENCH_COM[0]:.5f}, {WRENCH_COM[1]:.5f}, {WRENCH_COM[2]:.5f})
        float3 physics:diagonalInertia = ({WRENCH_I[0]:.3e}, {WRENCH_I[1]:.3e}, {WRENCH_I[2]:.3e})
        float physxRigidBody:maxDepenetrationVelocity = 0.5
        float physxRigidBody:sleepThreshold = 0
        float physxRigidBody:stabilizationThreshold = 0
{wrench_colliders(root, ind)}

        def Xform "visual"
        {{{_meshes(parts, root, indent="            ")}
        }}
    }}"""
    joints = ""
    for name, kind, axis, b0, b1 in JOINTS:
        lim = LIMITS[name]
        drive = "linear" if kind == "Prismatic" else "angular"
        lim_txt = "" if lim is None else f"""
            float physics:lowerLimit = {lim[0]}
            float physics:upperLimit = {lim[1]}"""
        joints += f"""
        def Physics{kind}Joint "{name}" (
            prepend apiSchemas = ["PhysicsDriveAPI:{drive}", "PhysxJointAPI"]
        )
        {{
            uniform token physics:axis = "{axis}"
            rel physics:body0 = </{root}/{b0}>
            rel physics:body1 = </{root}/{b1}>
            point3f physics:localPos0 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0){lim_txt}
            float physxJoint:armature = 0.00001
            float physxJoint:maxJointVelocity = 1000000
            uniform token drive:{drive}:physics:type = "force"
            float drive:{drive}:physics:targetPosition = 0
            float drive:{drive}:physics:stiffness = 0
            float drive:{drive}:physics:damping = 0
        }}
"""
    text = f"""#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}"
{{{uk.physics_material("PhysicsMaterial", 0.5, 0.4)}{uk.physics_material("TapMaterial", 0.25, 0.2)}{uk.physics_material("BarMaterial", 1.0, 0.9)}{KNURL_MAT}

    def Scope "Looks"
    {{{_looks(parts, root)}
    }}
{links}{wbody}

    def Scope "joints"
    {{{joints}
    }}
}}
"""
    return _write("tapwrench", text)


# -- the adapter plate (fixed), the seat (kinematic, moved away when the tap bites) ----------------------------------
DISC_C = np.array([0.0, TOOL_PCD_R])      # the disc's centre in the plate frame: the marked hole is on the -y side


def plate() -> str:
    """Plate frame: origin on the top face at the marked hole's axis; the disc's centre is 31.5 mm toward +y."""
    root = "AdapterPlate"
    cx, cy = DISC_C
    parts = [("disc", uk.lathe([(0.0, -PLATE_T), (DISC_R - 0.0005, -PLATE_T), (DISC_R, -PLATE_T + 0.0005),
                                (DISC_R, -0.0005), (DISC_R - 0.0005, 0.0), (0.0, 0.0)], 72).moved((cx, cy, 0)), "alu")]
    zt = 0.00004
    # the hole pattern, as seen from above: angles about the disc's centre, from +x
    # tool holes (63 mm circle) at -90 (marked), 0, 90, 180; robot-side screw holes (50 mm circle) 22.5 deg off them
    tool = [-90.0, 0.0, 90.0, 180.0]
    screws = [a + 22.5 for a in (-90.0, 0.0, 90.0, 180.0)]
    dowel = -90.0 + 22.5 + 45.0
    for i, a in enumerate(tool):
        p = (cx + TOOL_PCD_R * math.cos(math.radians(a)), cy + TOOL_PCD_R * math.sin(math.radians(a)))
        parts.append((f"th{i}", disc(HOLE_R, zt, 24).moved((*p, 0)), "hole"))
        parts.append((f"tc{i}", annulus(HOLE_R, CSK_R, zt, 24).moved((*p, 0)), "alu_cut"))
    for i, a in enumerate(screws):
        p = (cx + FLANGE_PCD_R * math.cos(math.radians(a)), cy + FLANGE_PCD_R * math.sin(math.radians(a)))
        parts.append((f"sh{i}", disc(0.0033, zt, 24).moved((*p, 0)), "hole"))
        parts.append((f"sc{i}", annulus(0.0033, 0.0062, zt, 32).moved((*p, 0)), "alu_cut"))
    p = (cx + FLANGE_PCD_R * math.cos(math.radians(dowel)), cy + FLANGE_PCD_R * math.sin(math.radians(dowel)))
    parts.append(("dowel", disc(0.003, zt, 24).moved((*p, 0)), "hole"))
    parts.append(("dowelc", annulus(0.003, 0.0034, zt, 24).moved((*p, 0)), "alu_cut"))
    parts.append(("mark", annulus(0.0047, 0.0060, 2 * zt, 40), "marker"))
    # colliders: the disc's square bounds less an 8 mm square clearance hole at the marked hole
    h = CLEAR_HALF
    x0, x1, y0, y1 = cx - DISC_R, cx + DISC_R, cy - DISC_R, cy + DISC_R
    zc, t = -PLATE_T / 2, PLATE_T
    cols = box_col("col_l", (-h - x0, y1 - y0, t), ((x0 - h) / 2, (y0 + y1) / 2, zc), root)
    cols += box_col("col_r", (x1 - h, y1 - y0, t), ((x1 + h) / 2, (y0 + y1) / 2, zc), root)
    cols += box_col("col_f", (2 * h, -h - y0, t), (0, (y0 - h) / 2, zc), root)
    cols += box_col("col_b", (2 * h, y1 - h, t), (0, (y1 + h) / 2, zc), root)
    return _kinematic("plate", root, parts, cols, friction=(0.5, 0.45))


def seat() -> str:
    """The countersink and the drilled hole's wall, as eight convex wedges (origin: the hole's axis on the top face).
    It stands in for the uncut hole until the tap bites; then the task moves it away and the thread rule holds the tap."""
    root = "Seat"
    cols = ""
    n = 8
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        pts = []
        for a in (a0, a1):
            c, s = math.cos(a), math.sin(a)
            for r, z in ((CSK_R, 0.0), (SEAT_BORE_R, -(CSK_R - SEAT_BORE_R)), (SEAT_BORE_R, -PLATE_T),
                         (SEAT_OUT_R, -PLATE_T), (SEAT_OUT_R, 0.0)):
                pts.append((r * c, r * s, z))
        cols += hull_col(f"col_w{k}", pts, root, offset=0.0004)
    parts = [("seatmark", disc(0.0001, -PLATE_T + 0.0001, 6), "hole")]
    return _kinematic("seat", root, parts, cols, friction=(0.3, 0.25))


# -- the vise and parallels (vise frame: origin on the table under the marked hole) -------------------------------------
JAW_X = (-0.040, 0.040)          # inner faces, about the disc's centre line (x 24.0 and 32.0 cm in the world)
JAW_TOP = 0.073
BED_Z = 0.035


def vise() -> str:
    root = "Vise"
    cy = TOOL_PCD_R               # the jaws are centred on the disc's centre (y)
    parts, cols = [], ""
    bx0, bx1 = -0.090, 0.170       # base along x (world 19 to 45 cm)
    by = 0.075
    parts.append(("base", uk.box((bx1 - bx0, 2 * by, 0.018), bevel=0.004).moved(((bx0 + bx1) / 2, cy, 0.009)), "cast"))
    parts.append(("bed", uk.box((bx1 - bx0 - 0.03, 0.070, BED_Z - 0.018), bevel=0.002)
                  .moved(((bx0 + bx1) / 2 + 0.01, cy, (BED_Z + 0.018) / 2)), "cast"))
    for s in (-1, 1):
        parts.append((f"rail{'p' if s > 0 else 'm'}", uk.box((bx1 - bx0 - 0.02, 0.012, 0.006), bevel=0.001)
                      .moved(((bx0 + bx1) / 2, cy + s * 0.041, BED_Z - 0.003)), "cast_dark"))
        parts.append((f"slot{'p' if s > 0 else 'm'}", uk.box((0.040, 0.012, 0.0006)).moved(((bx0 + bx1) / 2, cy + s * 0.062, 0.0182)),
                      "cast_dark"))
    cols += box_col("col_base", (bx1 - bx0, 2 * by, BED_Z), ((bx0 + bx1) / 2, cy, BED_Z / 2), root)
    jw, jd = 0.102, 0.025
    for name, x_in, sgn in (("fixed", JAW_X[0], -1), ("moving", JAW_X[1], 1)):
        xc = x_in + sgn * jd / 2
        parts.append((f"{name}_jaw", uk.box((jd - 0.004, jw + 0.006, JAW_TOP - 0.012 - BED_Z + 0.004), bevel=0.003)
                      .moved((xc + sgn * 0.002, cy, (BED_Z - 0.004 + JAW_TOP - 0.012) / 2)), "cast"))
        parts.append((f"{name}_plate", uk.box((0.004, jw, 0.020), bevel=0.0006).moved((x_in + sgn * 0.002, cy, JAW_TOP - 0.010)),
                      "jawsteel"))
        for i, dy in enumerate((-0.03, 0.03)):
            parts.append((f"{name}_screw{i}", uk.cylinder(0.0025, 0.0006, center=(x_in + sgn * 0.0003, cy + dy, JAW_TOP - 0.010),
                                                          axis="x", segments=12), "cast_dark"))
        cols += box_col(f"col_{name}", (jd, jw, JAW_TOP - BED_Z), (xc, cy, (BED_Z + JAW_TOP) / 2), root)
    # screw and handle, away from the robot
    parts.append(("screw", uk.cylinder(0.008, 0.11, center=(JAW_X[1] + 0.025 + 0.055, cy, 0.030), axis="x", segments=20),
                  "steel"))
    parts.append(("nut", uk.box((0.03, 0.05, 0.03), bevel=0.004).moved((bx1 - 0.015, cy, 0.030)), "cast"))
    hx = bx1 + 0.012
    parts.append(("hub", uk.cylinder(0.012, 0.022, center=(hx, cy, 0.030), axis="x", segments=20), "steel"))
    parts.append(("handle", uk.cylinder(0.005, 0.14, center=(hx + 0.004, cy, 0.030), axis="y", segments=16), "steel"))
    for s in (-1, 1):
        parts.append((f"ball{'p' if s > 0 else 'm'}", uk.lathe([(0.0, -0.008), (0.006, -0.006), (0.008, 0.0), (0.006, 0.006), (0.0, 0.008)], 16)
                      .moved((hx + 0.004, cy + s * 0.074, 0.030)), "steel"))
    # parallels: on edge along y, against each jaw, under the plate
    for name, x in (("par_f", JAW_X[0] + 0.0015), ("par_b", JAW_X[1] - 0.0015)):
        z0, z1 = BED_Z, JAW_TOP - PLATE_T
        parts.append((name, uk.box((0.003, 0.150, z1 - z0), bevel=0.0004).moved((x, cy, (z0 + z1) / 2)), "parallel"))
        cols += box_col(f"col_{name}", (0.003, 0.150, z1 - z0), (x, cy, (z0 + z1) / 2), root)
    return _kinematic("vise", root, parts, cols, friction=(0.6, 0.5))


# -- the stand (origin: table under its hole) --------------------------------------------------------------------------
def stand() -> str:
    root = "Stand"
    sx, sy, sz = STAND
    hz = sz - STAND_CHAMFER_D
    n = 48
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sq = np.c_[np.clip(np.cos(t) * 2, -1, 1), np.clip(np.sin(t) * 2, -1, 1)] * (sx / 2)
    parts = [("block", uk.ring_prism(sq, circle(STAND_HOLE_R, n), 0.0, hz), "pla_grey"),
             ("top", uk.ring_prism(sq, circle(STAND_CHAMFER_R, n), hz, sz), "pla_grey"),
             ("chamfer", uk.lathe([(STAND_CHAMFER_R, sz), (STAND_HOLE_R, hz)], n), "pla_grey")]
    cols = ""
    a, b, h = sx / 2, STAND_HOLE_R, hz
    cols += box_col("col_l", (a - b, sx, h), (-(a + b) / 2, 0, h / 2), root)
    cols += box_col("col_r", (a - b, sx, h), ((a + b) / 2, 0, h / 2), root)
    cols += box_col("col_f", (2 * b, a - b, h), (0, -(a + b) / 2, h / 2), root)
    cols += box_col("col_b", (2 * b, a - b, h), (0, (a + b) / 2, h / 2), root)
    c = STAND_CHAMFER_R
    cols += box_col("col_tl", (a - c, sx, sz - hz), (-(a + c) / 2, 0, (hz + sz) / 2), root)
    cols += box_col("col_tr", (a - c, sx, sz - hz), ((a + c) / 2, 0, (hz + sz) / 2), root)
    cols += box_col("col_tf", (2 * c, a - c, sz - hz), (0, -(a + c) / 2, (hz + sz) / 2), root)
    cols += box_col("col_tb", (2 * c, a - c, sz - hz), (0, (a + c) / 2, (hz + sz) / 2), root)
    for k in range(8):
        a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
        pts = []
        for ang in (a0, a1):
            cc, ss = math.cos(ang), math.sin(ang)
            for r, z in ((STAND_CHAMFER_R + 0.0005, sz), (STAND_HOLE_R + 0.0003, hz), (STAND_CHAMFER_R + 0.003, hz),
                         (STAND_CHAMFER_R + 0.003, sz)):
                pts.append((r * cc, r * ss, z))
        cols += hull_col(f"col_c{k}", pts, root, offset=0.0005)
    return _kinematic("stand", root, parts, cols, friction=(0.5, 0.45))


def build() -> dict:
    return {"tapwrench": wrench(), "plate": plate(), "seat": seat(), "vise": vise(), "stand": stand()}


if __name__ == "__main__":
    print(build())
